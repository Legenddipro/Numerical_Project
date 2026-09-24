# Project plan (parallel)

Sequencing and phase criteria. Who owns which file, the verification gates, and
the known implementation traps live in [CONTRIBUTING.md](../CONTRIBUTING.md) and
are referenced rather than repeated here.

> **Two people can work at once in this plan.** If you would rather have exactly
> one person working at a time, each step verified before handoff, use
> [PLAN_SEQUENTIAL.md](PLAN_SEQUENTIAL.md) instead — it is self-contained and
> needs no other document. Pick one plan and stick to it; do not mix them.

## What we are building

Bleecke & Ranocha (2026) show that relaxation — which conserves a physical
invariant exactly — normally costs one extra right-hand-side evaluation per step
when combined with FSAL, and that two new schemes, **FSAL-R** and **R-FSAL**,
remove that cost. We re-implement both in Python and extend the study in three
directions.

| Contribution | Type | What it adds |
|---|---|---|
| Nonlinear pendulum | Application | The paper's headline experiment is a PDE (BBM). We test a conservative ODE instead |
| Three root-finders for γ | Extension | The paper uses one bracketing method. We benchmark Newton, bisection, and that method |
| Monte Carlo sweep | Exploratory | The paper uses one fixed initial condition. We report distributions over many random ones |

## Decisions already made

Settled, and changing any of them means revisiting work:

- **Python, not Julia.** The paper's own solver loops depend on Julia only for
  Butcher coefficients and a PID controller — roughly thirty lines to
  reimplement. Our problem is two-dimensional, so none of the PDE machinery
  applies. A Julia install is still needed once, to generate reference numbers
  for gate G5.
- **`scipy.integrate.solve_ivp` cannot be used.** It exposes no FSAL cache and
  no pluggable step-size controller. Relaxation is surgery on the solver's
  internals, so the loops are hand-written — as the authors' are.
- **State as plain floats, not NumPy arrays.** At two components NumPy's
  per-call dispatch overhead dwarfs the arithmetic; it would be roughly 20×
  slower.
- **Three root-finders, not four.** Golden-section search was considered and
  dropped: it is a minimiser rather than a root-finder, so it would have to
  operate on `|r(γ)|`, and the reformulation adds a confounding variable
  without adding a comparison anyone would act on.
- **One shared `solve_relaxation_parameter`.** No solver assembles its own
  residual. See trap 6 in CONTRIBUTING.md for why.

## Phases

### Phase 0 — Foundation *(essentially complete)*

Repository scaffolded, and every module now carries function signatures, type
hints and docstrings with no bodies — the interface contract.

**Remaining:** the team reviews that contract together and changes anything it
disagrees with. Cheap now, expensive once bodies exist. Treat the signatures as
fixed afterwards.

**Exit criterion:** every owner can state what their module receives and returns
without asking anyone.

### Phase 1 — Independent modules *(parallel)*

| Who | What |
|---|---|
| Lead | `problems.py` — pendulum RHS, invariant, gradient, DOP853 reference, IC sampler |
| Abir | `tableaus.py`, `stepping.py` — coefficients, PID controller, error norm, initial step |
| Asikur | `rootfind.py` — three finders plus `solve_relaxation_parameter`; also begins the Julia install, which is independent of everything here |
| Shadman, Tousif | `docs/ALGORITHMS.md` — pseudocode transcription of the four solver loops from the authors' Julia source |

The transcription is real work and touches no code file. Writing the algorithms
out precisely before porting them is what prevents translation errors that are
painful to diagnose later.

**Exit criterion:** gate G1.

### Phase 2 — Solvers

Shadman builds baseline → naive → FSAL-R in `solvers/classic.py`. Tousif builds
R-FSAL in `solvers/rfsal.py`. Both depend on Phase 1 having merged.

**Exit criteria:** gates G2, G3, G4.

G2 first and alone. If convergence order is wrong, the tableau or the stage loop
is broken and nothing downstream is worth running.

### Phase 3 — Cross-validation against Julia

Asikur runs the authors' code on the fixed initial condition (ω = 1.5, θ = 1.0
over t ∈ [0, 10]) and checks a reference table into the repo. The lead writes the
test that reads it.

Budget roughly 45–75 minutes and 3–4 GB for the Julia install; pin to **1.9.3**,
matching the authors' `Manifest.toml`, and set `PYTHON=/usr/bin/python3` before
instantiating so PyCall does not download its own Conda.

**Exit criterion:** gate G5. This runs *before* the Monte Carlo sweep — debugging
a wrong port after generating a thousand samples of bad data is the failure mode
the gate exists to prevent.

### Phase 4 — Monte Carlo production

Asikur runs the sweep: 4 methods × 2 integrators × 7 tolerances × 3 root-finders
× N initial conditions, which is about ten configurations per initial condition
once baseline's root-finder dimension collapses.

Run **N = 50 first** and inspect the output before scaling to N = 1000.

**Exit criterion:** gate G6.

### Phase 5 — Analysis and report

Tousif produces work-precision diagrams, drift plots, the root-finder table and
the Monte Carlo summaries. The lead assembles the report. Everyone writes the
section covering their own component.

## Timing budget

About 5,000 steps per method per initial condition across the tolerance sweep, so
roughly 50,000 steps per initial condition over all configurations.

| Implementation | µs/step | 100 ICs | 1,000 ICs |
|---|---|---|---|
| Plain Python floats | ~15 | ~1.5 min | **~12 min** |
| NumPy 2-element arrays | ~80 | ~7 min | ~70 min |
| Python + Numba | ~0.3 | ~2 s | ~15 s |

Ratios are reliable; absolutes are within a factor of two or three. The sweep is
embarrassingly parallel over initial conditions, so `multiprocessing.Pool`
divides these by core count. Numba is only worth reaching for beyond ~10,000
samples.

## Risks

**A subtly wrong port that passes its own tests.** The reason gate G5 exists, and
the reason it precedes the expensive sweep.

**Relaxation failures under random initial conditions.** The bracket γ ∈ [0.8,
1.2] can fail, and in the authors' code the retry path is unreachable (trap 1).
Implement the retry correctly from the start; the failure rate is itself a
result.

**Both pendulum regimes in one plot.** Energy above 1 means the bob rotates over
the top rather than swinging. Decide deliberately whether to sample it, and never
pool the regimes silently.

**Interface churn after Phase 2 starts.** Mitigated by settling the contract in
Phase 0 while nothing depends on it.
