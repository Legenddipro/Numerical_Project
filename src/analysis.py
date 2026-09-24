"""Analysis and figures for the FSAL relaxation experiments.

Reads the raw CSV written by `montecarlo.py` and produces the work-precision
diagrams and comparison tables used in the report. This module performs no
integration: an expensive sweep is run once, then its results can be inspected
and its figures regenerated without repeating the numerical solves.

Figures and tables
------------------
Work-precision diagrams plot final solution error against RHS evaluations on
logarithmic axes, with one series per method and one point per tolerance.
The comparison asks whether FSAL-R and R-FSAL achieve naive relaxation's
accuracy at approximately baseline's cost. Points show the median across
initial conditions by default; error bands and horizontal bars show the
10th--90th percentile spread in error and cost. Each figure has a companion
CSV containing the exact plotted statistics and successful/failed run counts.

The root-finder table compares Newton, bisection and Algorithm 748 by iteration
cost, failure rates and achieved energy drift. Iterations per accepted step
include work spent on rejected attempts. Newton may legitimately drift more
than the bracketing methods, so the relevant check is that all three preserve
energy far better than baseline, not that their drifts are identical.

Monte Carlo summaries retain means, medians, quantiles and extrema for cost,
error, drift and the available gamma statistics. Libration and rotation are
reported separately throughout: their failure mechanisms differ, and pooling
them would hide that distinction. Gamma summaries describe per-run history
statistics from the CSV, not a reconstructed distribution of individual steps.

Energy-drift plots are a separate diagnostic interface. `energy_drift_plot`
accepts `SolverResult` objects from runs made with ``save_history=True``;
the sweep CSV does not contain the time and entropy histories it needs. Its
symmetric-log axis retains exact zero drift alongside the baseline's drift.

Completeness and failures
-------------------------
`load_results` checks the CSV schema and run identities, rejects duplicate
configurations, and warns about incomplete sweeps and failed runs. By default,
completeness is inferred from the observed configuration dimensions. Supply
its ``expected`` argument to detect entirely missing dimensions or trailing
initial conditions, which cannot be inferred from the remaining rows alone.

Failed runs remain in run counts and failure rates; numerical distributions
use successful runs only. Relaxation retries are reported separately from
failed integrations, and their rates use counters from completed runs because
failure rows contain no partial solver counters. Extrema are retained rather
than trimming outliers that may reveal a method's limits.

Running it
----------
From the repository root::

    python -m src.analysis results/mc_n1000_seed0.csv --out results/analysis

The CLI writes work-precision figures and their companion CSVs for each
observed tableau/root-finder combination, plus ``rootfinder_comparison.csv``
and ``monte_carlo_summary.csv``. Energy plots must be requested separately
with saved histories. Generated outputs belong in ``results/`` and are not
committed to the repository.

Owner: Tousif Fahmeed Quadir.
"""

import argparse
from pathlib import Path
from typing import Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import SolverResult
from .montecarlo import CSV_COLUMNS, expected_row_count

KEYS = ["regime", "tableau", "tolerance", "method", "rootfinder"]
LABELS = {"baseline": "Baseline", "naive": "Naive", "fsalr": "FSAL-R", "rfsal": "R-FSAL"}
METRICS = ["nf", "error_final", "energy_drift_max", "gamma_min", "gamma_max",
           "gamma_absdev_mean", "gamma_absdev_max", "mean_dt"]


def load_results(path: str, *, expected: dict | None = None):
    """Validate a CSV, retaining failed runs and warning about missing runs.

    Completeness is inferred from observed dimensions per seed/time span. Supply
    ``expected`` (expected_row_count keyword arguments, including
    n_initial_conditions) to detect entirely absent dimensions or trailing ICs;
    those cannot be inferred from a CSV without a sweep manifest.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Sweep contains no rows")
    config = ["seed", "t_start", "t_end", "ic_index", "tableau", "tolerance", "method", "rootfinder"]
    if df[config + ["regime", "status"]].isna().any().any():
        raise ValueError("Missing configuration or status")
    if not df.status.isin(["ok", "failed"]).all():
        raise ValueError("Unknown run status")
    if not df.method.isin(LABELS).all() or not df.regime.isin(["libration", "rotation"]).all():
        raise ValueError("Unknown method or regime")
    if not (df.loc[df.method == "baseline", "rootfinder"] == "none").all():
        raise ValueError("Baseline must use rootfinder='none'")
    if df.duplicated(config).any():
        raise ValueError("Duplicate run configurations")
    numeric = ["tolerance", "nf", "n_accept", "n_reject", "n_root_iterations",
               "n_relaxation_failures", "error_final", "energy_drift_max"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    ok = df.status == "ok"
    if not np.isfinite(df.loc[ok, numeric].to_numpy(dtype=float)).all():
        raise ValueError("Successful rows contain nonfinite metrics")
    if (df.loc[ok, numeric] < 0).any().any() or (df.tolerance <= 0).any():
        raise ValueError("Negative metrics or nonpositive tolerance")
    for _, block in df.groupby(["seed", "t_start", "t_end"]):
        dims = dict(n_initial_conditions=block.ic_index.nunique(),
                    tableau_names=block.tableau.unique(), method_names=block.method.unique(),
                    rootfinder_names=block.loc[block.method != "baseline", "rootfinder"].unique(),
                    tolerances=block.tolerance.unique())
        count = expected_row_count(**(expected if expected is not None else dims))
        if len(block) != count or block.groupby("ic_index").size().nunique() != 1:
            warnings.warn(f"Incomplete sweep: observed {len(block)} rows; expected {count}. "
                          "Expectation uses observed dimensions unless explicitly supplied.",
                          UserWarning, stacklevel=2)
    failed = int((~ok).sum())
    if failed:
        warnings.warn(f"Sweep contains {failed} failed runs out of {len(df)}; "
                      "metric summaries use successful runs only.", UserWarning, stacklevel=2)
    return df


def _destination(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def work_precision_data(df, tableau_name, *, rootfinder_name="toms748", aggregate="median"):
    """Exact plotted data, split by regime; bands are the 10th–90th percentiles.

    Counts include failed runs so survivorship is visible in the companion CSV.
    """
    if aggregate not in ("median", "mean"):
        raise ValueError("aggregate must be 'median' or 'mean'")
    selected = df[(df.tableau == tableau_name) &
                  ((df.method == "baseline") | (df.rootfinder == rootfinder_name))]
    rows = []
    for key, group in selected.groupby(KEYS, sort=True):
        good = group[group.status == "ok"]
        row = dict(zip(KEYS, key))
        row.update(n_runs=len(group), n_success=len(good), n_failed=len(group)-len(good))
        for metric in ("nf", "error_final"):
            row[metric] = getattr(good[metric], aggregate)()
            row[metric + "_q10"] = good[metric].quantile(.1)
            row[metric + "_q90"] = good[metric].quantile(.9)
        rows.append(row)
    if not rows:
        raise ValueError("No matching work-precision configurations")
    return pd.DataFrame(rows)


def work_precision_diagram(df, tableau_name: str, output_path: str, *,
                           rootfinder_name: str = "toms748", aggregate: str = "median") -> None:
    """Log-log work/precision panels by regime, with 10–90% error bands.

    Horizontal bars show the corresponding RHS-count spread. Save exact plotted
    statistics beside the figure as <figure>.csv, including failures and counts.
    """
    data = work_precision_data(df, tableau_name, rootfinder_name=rootfinder_name, aggregate=aggregate)
    regimes = sorted(data.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.5), squeeze=False)
    try:
        for ax, regime in zip(axes[0], regimes):
            for method, group in data[data.regime == regime].groupby("method", sort=False):
                group = group.sort_values("tolerance", ascending=False)
                valid = (group.nf > 0) & (group.error_final > 0)
                if not valid.all():
                    warnings.warn("Nonpositive or unavailable work-precision points omitted from log axes")
                g = group[valid]
                line, = ax.plot(g.nf, g.error_final, "o-", label=LABELS[method])
                ax.fill_between(g.nf, g.error_final_q10.clip(lower=np.finfo(float).tiny),
                                g.error_final_q90, alpha=.15, color=line.get_color())
                ax.hlines(g.error_final, g.nf_q10, g.nf_q90, color=line.get_color(), alpha=.4)
            ax.set(xscale="log", yscale="log", xlabel="RHS evaluations", ylabel="Final solution error",
                   title=f"{tableau_name} · {regime}")
            ax.grid(True, which="both", alpha=.2)
            ax.legend()
        fig.suptitle(f"{aggregate.title()} across initial conditions; 10–90% spread · {rootfinder_name}")
        fig.tight_layout()
        path = _destination(output_path)
        fig.savefig(path, dpi=180)
        data.to_csv(path.with_suffix(".csv"), index=False)
    finally:
        plt.close(fig)


def energy_drift_plot(results: Sequence[SolverResult], output_path: str) -> None:
    """Plot absolute invariant drift from save_history=True runs.

    A symmetric-log axis preserves exact zeros instead of assigning fake drift.
    """
    if not results:
        raise ValueError("At least one result is required")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        for result in results:
            if not result.t_history or len(result.t_history) != len(result.entropy_history):
                raise ValueError("Aligned time and entropy histories required; use save_history=True")
            drift = np.abs(np.asarray(result.entropy_history) - result.entropy_history[0])
            ax.plot(result.t_history, drift,
                    label=f"{LABELS[result.method]} ({result.tableau_name}, tol={result.tolerance:g})")
        ax.set_yscale("symlog", linthresh=1e-16)
        ax.set(xlabel="Time", ylabel="Absolute energy drift", title="Energy conservation")
        ax.grid(True, alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(_destination(output_path), dpi=180)
    finally:
        plt.close(fig)


def _summary(df, metrics):
    rows = []
    for key, group in df.groupby(KEYS, sort=True):
        good = group[group.status == "ok"]
        row = dict(zip(KEYS, key))
        row.update(n_runs=len(group), n_success=len(good), n_failed=len(group)-len(good),
                   run_failure_rate=1-len(good)/len(group))
        # Failed runs have no partial counters in the source CSV. These rates
        # therefore refer only to completed runs, not unobserved failed attempts.
        attempts = (good.n_accept + good.n_reject).sum()
        accepted = good.n_accept.sum()
        row.update(relaxation_failures=good.n_relaxation_failures.sum(min_count=1),
                   relaxation_failure_rate=(good.n_relaxation_failures.sum()/attempts
                                            if attempts else np.nan),
                   runs_with_relaxation_failure_rate=(good.n_relaxation_failures.gt(0).mean()),
                   iterations_per_accepted_step=(good.n_root_iterations.sum()/accepted
                                                 if accepted else np.nan),
                   mean_iterations_per_run=good.n_root_iterations.mean())
        for metric in metrics:
            values = good[metric].dropna()
            for stat in ("mean", "median", "min", "max"):
                row[f"{metric}_{stat}"] = getattr(values, stat)() if len(values) else np.nan
            for q in (.1, .9):
                row[f"{metric}_q{int(q*100)}"] = values.quantile(q)
        rows.append(row)
    return pd.DataFrame(rows)


def rootfinder_comparison(df, output_path: str) -> None:
    """CSV by regime/method/tableau/tolerance/finder, including failure rates.

    Iterations per accepted step = total iterations / total accepted steps,
    including work spent on rejected attempts. Relaxation failure rate uses all
    attempts of completed runs. Newton drift can exceed bracketing drift (plan
    addendum E); it should remain far below baseline, not be forced equal.
    """
    _summary(df[df.method != "baseline"], ["energy_drift_max"]).to_csv(
        _destination(output_path), index=False)


def monte_carlo_summary(df, output_path: str) -> None:
    """CSV distributions per regime/configuration, preserving extrema/outliers.

    Gamma columns summarize per-run history statistics, not pooled step-level
    samples: the sweep deliberately does not store those histories.
    """
    _summary(df, METRICS).to_csv(_destination(output_path), index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args(argv)
    df = load_results(args.csv)
    out = Path(args.out)
    for tableau in sorted(df.tableau.unique()):
        finders = sorted(df.loc[df.method != "baseline", "rootfinder"].unique()) or ["toms748"]
        for finder in finders:
            work_precision_diagram(df, tableau, str(out / f"work_precision_{tableau}_{finder}.png"),
                                   rootfinder_name=finder)
    rootfinder_comparison(df, str(out / "rootfinder_comparison.csv"))
    monte_carlo_summary(df, str(out / "monte_carlo_summary.csv"))
    print(f"Analyzed {len(df)} runs; figures and tables in {out}")


if __name__ == "__main__":
    main()
