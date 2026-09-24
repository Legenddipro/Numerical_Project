"""Tests for src/montecarlo.py. Owner: Asikur Rahman.

Small sweeps only -- two loose tolerances, a handful of initial conditions --
so the whole module runs in seconds. Gate G6 is checked on a slightly larger
slice at the bottom.
"""

import csv
from statistics import median

import pytest

from src import montecarlo
from src.montecarlo import (BASELINE_ROOTFINDER, CSV_COLUMNS, SOLVERS,
                            expected_row_count, run_single, run_sweep)
from src.problems import make_pendulum, sample_initial_conditions
from src.rootfind import ROOT_FINDERS
from src.tableaus import BS3

SMALL = dict(tableau_names=("BS3",),
             method_names=("baseline", "naive", "fsalr", "rfsal"),
             rootfinder_names=("newton", "toms748"),
             tolerances=(1e-3, 1e-4))


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def ics():
    return sample_initial_conditions(3, seed=7, regime="both")


@pytest.fixture(scope="module")
def serial_csv(ics, tmp_path_factory):
    path = tmp_path_factory.mktemp("mc") / "serial.csv"
    run_sweep(ics, str(path), n_workers=1, seed=7, **SMALL)
    return path


def test_registry_has_all_four_methods():
    assert set(SOLVERS) == {"baseline", "naive", "fsalr", "rfsal"}


def test_run_single_dispatches(ics):
    problem = make_pendulum()
    for method in SOLVERS:
        r = run_single(problem, BS3, 1e-4, method, "toms748")
        assert r.method == method
        assert r.nf > 0


def test_sweep_covers_full_cross_product(serial_csv, ics):
    """Rows = ICs x tableaus x tolerances x (1 + (methods-1) x finders)."""
    rows = read_rows(serial_csv)
    assert len(rows) == expected_row_count(len(ics), **SMALL)
    assert len(rows) == 3 * 1 * 2 * (1 + 3 * 2)
    combos = {(r["ic_index"], r["tableau"], r["tolerance"], r["method"],
               r["rootfinder"]) for r in rows}
    assert len(combos) == len(rows)


def test_baseline_not_duplicated_across_rootfinders(serial_csv):
    rows = [r for r in read_rows(serial_csv) if r["method"] == "baseline"]
    assert {r["rootfinder"] for r in rows} == {BASELINE_ROOTFINDER}
    keys = [(r["ic_index"], r["tableau"], r["tolerance"]) for r in rows]
    assert len(keys) == len(set(keys)) == 3 * 2


def test_histories_excluded_from_csv(serial_csv):
    with open(serial_csv) as fh:
        header = fh.readline().strip().split(",")
    assert header == list(CSV_COLUMNS)
    assert not any("history" in col for col in header)


def test_rows_are_consistent_with_solver_invariants(serial_csv):
    """Every successful row satisfies gate G3 and the relaxed rows hold eta."""
    s = BS3.n_stages
    for r in read_rows(serial_csv):
        assert r["status"] == "ok", r
        attempts = int(r["n_accept"]) + int(r["n_reject"])
        expected = 2 + (s - 1) * attempts
        if r["method"] == "naive":
            expected += int(r["n_accept"])
        if r["method"] == "rfsal":
            expected -= int(r["n_relaxation_failures"])
        assert int(r["nf"]) == expected
        if r["method"] != "baseline":
            assert float(r["energy_drift_max"]) < 1e-12
            assert float(r["gamma_min"]) >= 0.8
            assert float(r["gamma_max"]) <= 1.2
        else:
            assert r["gamma_min"] == ""


def test_sweep_is_reproducible_from_seed(serial_csv, ics, tmp_path):
    again = tmp_path / "again.csv"
    run_sweep(ics, str(again), n_workers=1, seed=7, **SMALL)
    assert again.read_bytes() == serial_csv.read_bytes()


def test_parallel_matches_serial(serial_csv, ics, tmp_path):
    """n_workers>1 must write byte-identical output."""
    parallel = tmp_path / "parallel.csv"
    run_sweep(ics, str(parallel), n_workers=3, seed=7, **SMALL)
    assert parallel.read_bytes() == serial_csv.read_bytes()


def test_task_seed_depends_on_index_not_worker(serial_csv):
    rows = read_rows(serial_csv)
    by_ic = {}
    for r in rows:
        by_ic.setdefault(r["ic_index"], set()).add(r["task_seed"])
    assert all(len(v) == 1 for v in by_ic.values())
    assert len({next(iter(v)) for v in by_ic.values()}) == len(by_ic)


def test_solver_failure_is_recorded_not_raised(ics, tmp_path):
    """tol = 0 makes initial_step_size raise ValueError -- a real failure path,
    not a mock. It must become failure rows while the other runs complete."""
    path = tmp_path / "failing.csv"
    config = dict(SMALL, tolerances=(1e-3, 0.0))
    run_sweep(ics, str(path), n_workers=1, seed=7, **config)
    rows = read_rows(path)
    assert len(rows) == expected_row_count(len(ics), **config)

    failed = [r for r in rows if r["status"] == "failed"]
    ok = [r for r in rows if r["status"] == "ok"]
    assert failed and ok
    assert all(float(r["tolerance"]) == 0.0 for r in failed)
    assert all(r["error_type"] == "ValueError" for r in failed)
    assert all(r["nf"] == "" for r in failed)


def test_arithmetic_error_is_also_caught(ics, tmp_path, monkeypatch):
    """ADDENDUM G: not only RuntimeError. A solver raising ArithmeticError
    (the controller's failure mode) must be recorded, too."""
    def broken(*args, **kwargs):
        raise ArithmeticError("dt factor is not finite")

    monkeypatch.setitem(montecarlo.SOLVERS, "fsalr", broken)
    path = tmp_path / "arith.csv"
    run_sweep(ics[:1], str(path), n_workers=1, seed=7, **SMALL)
    rows = read_rows(path)
    failed = [r for r in rows if r["status"] == "failed"]
    assert {r["method"] for r in failed} == {"fsalr"}
    assert {r["error_type"] for r in failed} == {"ArithmeticError"}
    assert len(rows) == expected_row_count(1, **SMALL)


def test_unknown_names_rejected_before_running(ics, tmp_path):
    with pytest.raises(KeyError):
        run_sweep(ics, str(tmp_path / "x.csv"), method_names=("rk4",))
    with pytest.raises(KeyError):
        run_sweep(ics, str(tmp_path / "x.csv"), rootfinder_names=("secant",))


def test_regime_recorded_per_row(serial_csv):
    for r in read_rows(serial_csv):
        energy = float(r["energy0"])
        assert r["regime"] == ("rotation" if energy > 1.0 else "libration")


def test_estimate_runtime_is_positive_and_scales():
    small = montecarlo.estimate_runtime(10, **SMALL)
    assert small > 0
    # Linear in N and inversely proportional to workers, by construction.
    assert montecarlo.estimate_runtime(10, n_workers=2, **SMALL) < small * 1.5


def test_gate_g6_gamma_clusters_and_tightens(tmp_path):
    """Gate G6: gamma sits at 1 + O(dt**(p-1)), and tightening the tolerance
    pulls it toward 1.

    Checked on the median over initial conditions of each run's mean
    |gamma - 1|, which must fall monotonically with tolerance for every
    relaxed method and both tableaus. Outliers are not trimmed; the median is
    what the claim is about, and the max is reported by the sweep for anyone
    investigating the tail.
    """
    ics = sample_initial_conditions(6, seed=11, regime="libration")
    tols = (1e-3, 1e-5, 1e-7)
    path = tmp_path / "g6.csv"
    run_sweep(ics, str(path), tableau_names=("BS3", "DP5"),
              method_names=("fsalr", "rfsal"), rootfinder_names=("toms748",),
              tolerances=tols, n_workers=1, seed=11)
    rows = read_rows(path)
    assert all(r["status"] == "ok" for r in rows)

    for tableau in ("BS3", "DP5"):
        for method in ("fsalr", "rfsal"):
            medians = []
            for tol in tols:
                values = [float(r["gamma_absdev_mean"]) for r in rows
                          if r["tableau"] == tableau and r["method"] == method
                          and float(r["tolerance"]) == tol]
                medians.append(median(values))
            assert medians[0] > medians[1] > medians[2], (tableau, method, medians)
            assert medians[0] < 0.2          # inside the bracket, near 1
