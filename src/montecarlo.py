"""Monte Carlo sweep harness.

Runs every combination of method, integrator, tolerance, root-finder, and
initial condition, and writes one row per run to CSV. Deliberately does no
analysis: raw results are expensive to regenerate, so they are produced once,
stored, and read repeatedly by `analysis.py`.

What each row holds
-------------------
The configuration (initial condition, regime, method, tableau, tolerance,
root-finder), a status column, every scalar field of `SolverResult`, and a few
per-run summaries that would otherwise need the histories: the maximum energy
drift over the run and statistics of ``|gamma - 1|`` (gate G6). The histories
themselves are computed inside the worker, reduced to those numbers, and
dropped -- never written.

Wall-clock time is deliberately *not* a column. It differs between runs of the
same configuration, and the sweep's reproducibility check is that the same seed
gives byte-identical rows. `estimate_runtime` is where timing lives.

Running it
----------
From the repository root::

    python -m src.montecarlo --n 50 --seed 0 --workers 4 --out results/mc_n50.csv

Owner: Asikur Rahman.
"""

import argparse
import csv
import os
import time
from functools import partial
from math import fsum, isfinite, nan
from multiprocessing import get_context
from typing import Callable, Iterable, Sequence

import numpy as np

from .contracts import Problem, SolverResult, State, Tableau
from .problems import make_pendulum, pendulum_entropy, regime_of, sample_initial_conditions
from .rootfind import ROOT_FINDERS, RootFinder
from .solvers.classic import solve_baseline, solve_fsalr, solve_naive
from .solvers.rfsal import solve_rfsal
from .tableaus import TABLEAUS

Solver = Callable[..., SolverResult]

SOLVERS: dict[str, Solver] = {
    "baseline": solve_baseline,
    "naive": solve_naive,
    "fsalr": solve_fsalr,
    "rfsal": solve_rfsal,
}
"""Name -> solver entry point. Expected keys: ``"baseline"``, ``"naive"``,
``"fsalr"``, ``"rfsal"``. All four share a signature, so the sweep calls them
through this registry without special cases."""

DEFAULT_TOLERANCES: tuple[float, ...] = (
    1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
)
"""Matches the authors' sweep, so our work-precision diagrams span the same
range as theirs."""


def run_single(problem: Problem,
               tableau: Tableau,
               tol: float,
               method: str,
               rootfinder_name: str,
               *,
               save_history: bool = False) -> SolverResult:
    """Run one solver on one problem at one tolerance.

    A thin dispatch over `SOLVERS` and the root-finder registry, kept separate
    so it can be called directly when debugging a single configuration without
    going through the sweep machinery.

    Exceptions from the solver are allowed to propagate here; `run_sweep` is
    where they are caught and recorded as failures.

    Returns
    -------
    SolverResult
    """
    solver = SOLVERS[method]
    if method == "baseline":
        finder: RootFinder | None = None
    else:
        finder = ROOT_FINDERS[rootfinder_name]
    return solver(problem, tableau, tol, rootfinder=finder,
                  save_history=save_history)


def run_sweep(initial_conditions: Sequence[State],
              output_path: str,
              *,
              tableau_names: Sequence[str] = ("BS3", "DP5"),
              method_names: Sequence[str] = ("baseline", "naive", "fsalr", "rfsal"),
              rootfinder_names: Sequence[str] = ("toms748",),
              tolerances: Sequence[float] = DEFAULT_TOLERANCES,
              t_span: tuple[float, float] = (0.0, 10.0),
              n_workers: int = 1,
              seed: int = 0) -> None:
    """Run the full cross product and write results to `output_path` as CSV.

    One row per completed run, carrying the configuration that produced it
    alongside every field of `SolverResult` except the histories, which are
    omitted because storing them for a thousand samples would cost gigabytes.

    A run that raises is recorded as a row with a failure flag rather than
    aborting the sweep: with random initial conditions some configurations are
    expected to fail, and how often they do is itself a result.

    `baseline` uses no root-finder, so it is run once per (tableau, tolerance,
    initial condition) rather than once per root-finder. Running it repeatedly
    would inflate its apparent share of the compute and produce duplicate
    identical rows.

    The sweep is embarrassingly parallel over initial conditions; `n_workers`
    above 1 distributes them with `multiprocessing`. Each worker must derive
    its own seed from `seed` so the run stays reproducible regardless of worker
    count.

    Start with fifty initial conditions as a smoke test and inspect the output
    before scaling to a thousand. Discovering a bug after generating the full
    sweep wastes the whole run.

    Rows come out in a fixed order -- initial condition, then tableau,
    tolerance, method, root-finder, in the order given -- whatever `n_workers`
    is, so serial and parallel runs write identical files.

    About the seed. The solvers are deterministic, so nothing inside a run
    draws random numbers; the randomness is in *which* initial conditions were
    sampled, and that seed belongs to `sample_initial_conditions`. `seed` here
    is recorded in every row for provenance, together with a per-initial-
    condition ``task_seed`` derived from it by ``numpy.random.SeedSequence``
    with the condition's index as spawn key. That derivation depends on the
    index alone, never on which worker ran it, so any future stochastic
    component seeded from ``task_seed`` stays reproducible under any
    `n_workers`.
    """
    for name in method_names:
        if name not in SOLVERS:
            raise KeyError(f"unknown method {name!r}; expected one of {sorted(SOLVERS)}")
    for name in rootfinder_names:
        if name not in ROOT_FINDERS:
            raise KeyError(f"unknown root-finder {name!r}; expected one of "
                           f"{sorted(ROOT_FINDERS)}")
    for name in tableau_names:
        if name not in TABLEAUS:
            raise KeyError(f"unknown tableau {name!r}; expected one of {sorted(TABLEAUS)}")

    configs = sweep_configurations(tableau_names, method_names,
                                   rootfinder_names, tolerances)
    tasks = [(index, (float(u[0]), float(u[1])), _task_seed(seed, index))
             for index, u in enumerate(initial_conditions)]
    worker = partial(_run_initial_condition, configs=configs,
                     t_span=tuple(t_span), seed=seed)

    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rows in _map(worker, tasks, n_workers):
            writer.writerows(rows)
            fh.flush()


def estimate_runtime(n_initial_conditions: int,
                     *,
                     tableau_names: Sequence[str] = ("BS3", "DP5"),
                     method_names: Sequence[str] = ("baseline", "naive", "fsalr", "rfsal"),
                     rootfinder_names: Sequence[str] = ("toms748",),
                     tolerances: Sequence[float] = DEFAULT_TOLERANCES,
                     n_workers: int = 1) -> float:
    """Predict sweep wall-clock time in seconds from a short calibration run.

    Times a handful of representative configurations, then scales by the size
    of the cross product. Exists so that nobody launches an overnight job by
    accident when a parameter was set an order of magnitude too high.

    Calibrates on the authors' initial condition plus one near-separatrix
    libration state, which is where step sizes -- and so run times -- are
    largest; the estimate therefore leans pessimistic, which is the useful
    direction for a guard against accidental overnight jobs. The reference
    solution each initial condition needs is timed and included.

    Returns
    -------
    float
        Estimated seconds.
    """
    calibration = ((1.5, 1.0), (1.35, 1.2))
    configs = sweep_configurations(tableau_names, method_names,
                                   rootfinder_names, tolerances)
    start = time.perf_counter()
    for index, u0 in enumerate(calibration):
        _run_initial_condition((index, u0, 0), configs=configs,
                               t_span=(0.0, 10.0), seed=0)
    per_condition = (time.perf_counter() - start) / len(calibration)
    workers = max(1, min(n_workers, n_initial_conditions))
    return per_condition * n_initial_conditions / workers


# ---------------------------------------------------------------------------
# Sweep internals
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    # configuration
    "ic_index", "omega0", "theta0", "energy0", "regime", "seed", "task_seed",
    "t_start", "t_end",
    "method", "tableau", "tolerance", "rootfinder",
    # outcome
    "status", "error_type", "error_message",
    # SolverResult scalars (u_final split into components)
    "t_final", "omega_final", "theta_final", "error_final",
    "nf", "n_accept", "n_reject", "n_root_iterations", "n_relaxation_failures",
    # summaries of the histories, which are never written
    "energy_drift_final", "energy_drift_max",
    "gamma_min", "gamma_max", "gamma_absdev_mean", "gamma_absdev_max",
    "mean_dt",
)
"""Column order of the output CSV. No history columns, by design."""

BASELINE_ROOTFINDER = "none"
"""What the ``rootfinder`` column holds for baseline, which uses none."""


def sweep_configurations(tableau_names: Sequence[str],
                         method_names: Sequence[str],
                         rootfinder_names: Sequence[str],
                         tolerances: Sequence[float]
                         ) -> tuple[tuple[str, float, str, str], ...]:
    """Every (tableau, tolerance, method, root-finder) run for one initial
    condition, in output order.

    Baseline appears once per (tableau, tolerance) with root-finder
    ``"none"``, not once per root-finder.
    """
    configs = []
    for tableau_name in tableau_names:
        for tol in tolerances:
            for method in method_names:
                if method == "baseline":
                    configs.append((tableau_name, float(tol), method,
                                    BASELINE_ROOTFINDER))
                else:
                    for finder in rootfinder_names:
                        configs.append((tableau_name, float(tol), method, finder))
    return tuple(configs)


def expected_row_count(n_initial_conditions: int,
                       *,
                       tableau_names: Sequence[str] = ("BS3", "DP5"),
                       method_names: Sequence[str] = ("baseline", "naive", "fsalr", "rfsal"),
                       rootfinder_names: Sequence[str] = ("toms748",),
                       tolerances: Sequence[float] = DEFAULT_TOLERANCES) -> int:
    """Rows a complete sweep writes: the cross product, minus baseline's
    root-finder dimension. `analysis.load_results` can compare against this
    to warn on an incomplete sweep."""
    return n_initial_conditions * len(sweep_configurations(
        tableau_names, method_names, rootfinder_names, tolerances))


def _task_seed(seed: int, index: int) -> int:
    """Per-initial-condition seed, a function of (seed, index) only."""
    state = np.random.SeedSequence(seed, spawn_key=(index,)).generate_state(1)
    return int(state[0])


def _map(worker, tasks, n_workers: int) -> Iterable[list[dict]]:
    """Ordered map, serial or over a process pool.

    ``imap`` (not ``imap_unordered``) keeps output order independent of which
    worker finishes first. ``spawn`` rather than ``fork`` so behaviour is the
    same on Linux, macOS and Windows, the three machines this team uses.
    """
    if n_workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield worker(task)
        return
    with get_context("spawn").Pool(processes=n_workers) as pool:
        yield from pool.imap(worker, tasks, chunksize=1)


def _run_initial_condition(task: tuple[int, State, int],
                           *,
                           configs: Sequence[tuple[str, float, str, str]],
                           t_span: tuple[float, float],
                           seed: int) -> list[dict]:
    """All configurations for one initial condition. Runs inside a worker.

    Builds the Problem -- and so the DOP853 reference, the expensive part --
    once, and reuses it for every configuration.
    """
    index, u0, task_seed = task
    base = {
        "ic_index": index,
        "omega0": repr(u0[0]),
        "theta0": repr(u0[1]),
        "energy0": repr(pendulum_entropy(u0)),
        "regime": regime_of(u0),
        "seed": seed,
        "task_seed": task_seed,
        "t_start": repr(float(t_span[0])),
        "t_end": repr(float(t_span[1])),
    }

    try:
        problem = make_pendulum(u0=u0, t_span=t_span)
    except Exception as exc:          # noqa: BLE001 -- recorded, not raised
        return [_failure_row(base, cfg, exc) for cfg in configs]

    rows = []
    for cfg in configs:
        tableau_name, tol, method, finder = cfg
        try:
            result = run_single(problem, TABLEAUS[tableau_name], tol, method,
                                finder, save_history=True)
        except Exception as exc:      # noqa: BLE001 -- ADDENDUM G: catch broadly
            # RuntimeError (step collapse, max_steps), ArithmeticError (the
            # controller), ValueError (initial_step_size) all end up here. A
            # failure is a result, not a reason to abort the sweep.
            rows.append(_failure_row(base, cfg, exc))
            continue
        rows.append(_success_row(base, cfg, result, problem))
    return rows


def _config_fields(cfg) -> dict:
    tableau_name, tol, method, finder = cfg
    return {"method": method, "tableau": tableau_name,
            "tolerance": repr(tol), "rootfinder": finder}


def _failure_row(base: dict, cfg, exc: BaseException) -> dict:
    row = dict.fromkeys(CSV_COLUMNS, "")
    row.update(base)
    row.update(_config_fields(cfg))
    row["status"] = "failed"
    row["error_type"] = type(exc).__name__
    # One line, bounded: messages include run state and can be long.
    row["error_message"] = " ".join(str(exc).split())[:300]
    return row


def _success_row(base: dict, cfg, result: SolverResult, problem: Problem) -> dict:
    eta0 = problem.entropy(problem.u0)
    drifts = [abs(e - eta0) for e in result.entropy_history]
    gammas = result.gamma_history

    row = dict.fromkeys(CSV_COLUMNS, "")
    row.update(base)
    row.update(_config_fields(cfg))
    row.update({
        "status": "ok",
        "t_final": repr(result.t_final),
        "omega_final": repr(result.u_final[0]),
        "theta_final": repr(result.u_final[1]),
        "error_final": repr(result.error_final),
        "nf": result.nf,
        "n_accept": result.n_accept,
        "n_reject": result.n_reject,
        "n_root_iterations": result.n_root_iterations,
        "n_relaxation_failures": result.n_relaxation_failures,
        "energy_drift_final": repr(abs(problem.entropy(result.u_final) - eta0)),
        "energy_drift_max": repr(max(drifts) if drifts else nan),
        "mean_dt": repr((result.t_final - problem.t_span[0]) / result.n_accept
                        if result.n_accept else nan),
    })
    if gammas:
        absdev = [abs(g - 1.0) for g in gammas]
        row.update({
            "gamma_min": repr(min(gammas)),
            "gamma_max": repr(max(gammas)),
            "gamma_absdev_mean": repr(fsum(absdev) / len(absdev)),
            "gamma_absdev_max": repr(max(absdev)),
        })
    return row


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.montecarlo",
        description="Monte Carlo sweep over random pendulum initial conditions.")
    parser.add_argument("--n", type=int, default=50,
                        help="number of initial conditions (default 50: smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regime", choices=("libration", "rotation", "both"),
                        default="both",
                        help="sampling regime; each row records its own regime "
                             "so the two are never pooled silently")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--finders", nargs="+", default=sorted(ROOT_FINDERS))
    parser.add_argument("--methods", nargs="+", default=list(SOLVERS))
    parser.add_argument("--tableaus", nargs="+", default=list(TABLEAUS))
    parser.add_argument("--tols", nargs="+", type=float, default=list(DEFAULT_TOLERANCES))
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--estimate-only", action="store_true",
                        help="print the runtime estimate and exit")
    args = parser.parse_args(argv)

    kwargs = dict(tableau_names=args.tableaus, method_names=args.methods,
                  rootfinder_names=args.finders, tolerances=args.tols)
    estimate = estimate_runtime(args.n, n_workers=args.workers, **kwargs)
    rows = expected_row_count(args.n, **kwargs)
    print(f"{args.n} initial conditions, {rows} runs, {args.workers} workers: "
          f"estimated {estimate / 60:.1f} min")
    if args.estimate_only:
        return

    out = args.out or f"results/mc_n{args.n}_seed{args.seed}_{args.regime}.csv"
    ics = sample_initial_conditions(args.n, seed=args.seed, regime=args.regime)
    start = time.perf_counter()
    run_sweep(ics, out, t_span=(0.0, args.t_end), n_workers=args.workers,
              seed=args.seed, **kwargs)
    print(f"wrote {out} in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
