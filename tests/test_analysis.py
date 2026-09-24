"""Step 9 checks: plotted statistics, numerical claims, and output contracts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis import (load_results, work_precision_data, work_precision_diagram,
                          energy_drift_plot, rootfinder_comparison, monte_carlo_summary)
from src.montecarlo import run_sweep, run_single
from src.problems import make_pendulum
from src.tableaus import BS3


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    path = tmp_path_factory.mktemp("analysis") / "runs.csv"
    run_sweep([(1.5, 1.0), (.8, .4), (2.5, .5)], str(path),
              tableau_names=["BS3", "DP5"], tolerances=[1e-5, 1e-6],
              rootfinder_names=["newton", "toms748", "bisection"])
    return path


def test_load_results_warns_on_incomplete_sweep(sweep, tmp_path):
    df = load_results(sweep)
    path = tmp_path / "missing.csv"
    df.iloc[:-1].to_csv(path, index=False)
    with pytest.warns(UserWarning, match="Incomplete sweep"):
        load_results(path)
    # An entire missing dimension is detectable with an explicit expectation.
    df[df.tableau == "BS3"].to_csv(path, index=False)
    with pytest.warns(UserWarning, match="Incomplete sweep"):
        load_results(path, expected=dict(n_initial_conditions=3,
            tableau_names=["BS3", "DP5"], tolerances=[1e-5, 1e-6],
            rootfinder_names=["newton", "toms748", "bisection"]))


def test_work_precision_ordering_matches_paper(sweep):
    df = load_results(sweep)
    # Headline comparison on libration, away from loose-tolerance outliers.
    for tableau in ("BS3", "DP5"):
        data = work_precision_data(df[df.regime == "libration"], tableau)
        for _, g in data.groupby("tolerance"):
            g = g.set_index("method")
            assert g.error_final.idxmax() == "baseline"
            assert g.nf.idxmax() == "naive"
            for method in ("fsalr", "rfsal"):
                assert g.loc[method, "nf"] == pytest.approx(g.loc["baseline", "nf"], rel=.1)
                assert g.loc[method, "error_final"] == pytest.approx(g.loc["naive", "error_final"], rel=.5)


def test_rootfinder_table_reports_expected_iteration_ordering(sweep, tmp_path):
    path = tmp_path / "finders.csv"
    rootfinder_comparison(load_results(sweep), path)
    table = pd.read_csv(path)
    means = table.groupby("rootfinder").iterations_per_accepted_step.mean()
    assert means["newton"] < means["toms748"] < means["bisection"]


def test_energy_drift_far_below_baseline(sweep):
    # Addendum E replaces the obsolete indistinguishability assertion.
    df = load_results(sweep)
    keys = ["ic_index", "tableau", "tolerance"]
    baseline = df[df.method == "baseline"][keys + ["energy_drift_max"]]
    pairs = df[df.method != "baseline"].merge(baseline, on=keys, suffixes=("", "_baseline"))
    assert (pairs.energy_drift_max < pairs.energy_drift_max_baseline * .01).all()


def test_summary_splits_by_regime_and_retains_outliers(sweep, tmp_path):
    df = load_results(sweep)
    path = tmp_path / "summary.csv"
    monte_carlo_summary(df, path)
    table = pd.read_csv(path)
    assert set(table.regime) == {"libration", "rotation"}
    assert table.n_runs.sum() == len(df)
    assert table.gamma_absdev_max_max.max() == pytest.approx(df.gamma_absdev_max.max())
    assert table.groupby("regime").n_runs.sum().to_dict() == df.groupby("regime").size().to_dict()


def test_failed_runs_counted_without_polluting_metrics(sweep, tmp_path):
    df = load_results(sweep)
    df.loc[0, "status"] = "failed"
    df.loc[0, ["nf", "error_final"]] = np.nan
    path = tmp_path / "failed.csv"
    df.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="1 failed runs"):
        loaded = load_results(path)
    data = work_precision_data(loaded, "BS3")
    assert data.n_failed.sum() == 1
    assert data.n_runs.sum() == len(loaded[(loaded.tableau == "BS3") &
        ((loaded.method == "baseline") | (loaded.rootfinder == "toms748"))])


def test_duplicate_rows_rejected(sweep, tmp_path):
    df = load_results(sweep)
    path = tmp_path / "duplicate.csv"
    pd.concat([df, df.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_results(path)


def test_outputs_written_to_requested_path(sweep, tmp_path):
    df = load_results(sweep)
    figure = tmp_path / "nested" / "work.png"
    work_precision_diagram(df, "BS3", figure)
    saved = pd.read_csv(figure.with_suffix(".csv"))
    expected = work_precision_data(df, "BS3")
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    problem = make_pendulum()
    results = [run_single(problem, BS3, 1e-5, method, "toms748", save_history=True)
               for method in ("baseline", "naive", "fsalr", "rfsal")]
    energy_drift_plot(results, tmp_path / "energy.png")
    assert figure.stat().st_size > 1000
    assert (tmp_path / "energy.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="save_history"):
        energy_drift_plot([results[0]._replace(t_history=())], tmp_path / "bad.png")
    assert not plt.get_fignums()


def test_aggregation_and_band_values(sweep):
    df = load_results(sweep)
    data = work_precision_data(df, "BS3", aggregate="mean")
    row = data[(data.regime == "libration") & (data.method == "baseline")].iloc[0]
    raw = df[(df.regime == row.regime) & (df.method == row.method) &
             (df.tableau == row.tableau) & (df.tolerance == row.tolerance)]
    assert row.nf == raw.nf.mean()
    assert row.error_final_q10 == raw.error_final.quantile(.1)
    assert row.error_final_q90 == raw.error_final.quantile(.9)
    with pytest.raises(ValueError, match="aggregate"):
        work_precision_data(df, "BS3", aggregate="invalid")
