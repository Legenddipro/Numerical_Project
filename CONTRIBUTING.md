# Working agreement

This document is the single source of truth for who owns what.

**We are working sequentially**, following [docs/PLAN_SEQUENTIAL.md](docs/PLAN_SEQUENTIAL.md):
exactly one person implements at a time, on one shared branch, and each step is verified
before handoff. That plan is self-contained — this file records ownership, gates and
traps; the plan records the order and the checks.

## The one rule

**One owner per file. Two people never edit the same file.**

If you need a change in someone else's file, ask them — do not edit it yourself. The one
exception is the interface contract (step 0), which the whole team agrees together in a
single session.

Working sequentially does not make this rule redundant. It means nobody *should* be in
your file at the same time as you; the ownership rule means that even if the sequence
slips, two people are not editing the same lines.

## File ownership

| Owner | Files | Steps |
|---|---|---|
| Prachurja Dhar (lead) | `src/contracts.py`, `docs/ALGORITHMS.md`, `src/problems.py`, repo config, `README.md` | 1, 2, 10 (report) |
| MD. Abir Hossain | `src/tableaus.py`, `src/stepping.py`, Julia reference table | 3, 4 |
| MD. Shadman Shafie | `src/rootfind.py`, `src/solvers/classic.py` | 5, 6 |
| Asikur Rahman | `src/solvers/rfsal.py`, `src/montecarlo.py` | 7, 8 |
| Tousif Fahmeed Quadir | `src/analysis.py` | 9 |

The pairings are deliberate: whoever writes the root-finder then writes the solvers that
call it, and whoever writes R-FSAL then writes the sweep that runs it. Each person keeps
their context instead of handing it over mid-thought.

Everyone owns `tests/test_<their_module>.py` for their own modules.
`tests/test_gates.py` holds the cross-cutting checks and belongs to the lead, since
those span several owners' code.

`docs/ALGORITHMS.md` — the pseudocode transcription of the four solver loops — is step 1,
deliberately placed before any implementation. Its author ends up the one person who
knows all four loops cold without having written the code, which is the closest thing to
a reviewer this workflow retains once pull requests are gone.

### Why `src/contracts.py` exists

Both solver modules need the `SolverResult` type. If it lived in
`solvers/classic.py`, then `solvers/rfsal.py` — a different owner's file — would
have to import from it, coupling two people's work for no reason. Putting every
shared type in one lead-owned module keeps the dependency graph one-directional:
everything imports from `contracts`, and `contracts` imports from nothing.

Changing `contracts.py` affects every module, so changes there are discussed
first rather than made unilaterally.

## Branching

**One shared branch (`main`). No feature branches, no pull requests.** That is only safe
because exactly one person is working at a time.

1. `git pull` before you start — never work from a stale copy.
2. Implement only your step's files.
3. Run your step's Verify block. Every check must pass.
4. `git add`, `git commit`, `git push`.
5. Post your Handoff line. **Only then** does the next person start.

Never `--force` on `main`. It is the one operation that can destroy someone else's
committed work, and nothing in this workflow requires it.

Two consequences of the single branch worth stating plainly:

- **Starting before the handoff causes direct collisions**, not a merge to resolve later.
  The sequential discipline is what prevents conflicts, so it is load-bearing rather
  than merely polite.
- **Dropping pull requests removes the reviewer.** The Verify block is now the only
  gate, so run every check rather than assuming it passes.

## Verification gates

Work does not advance past a gate until its check passes. Gates exist to stop errors
compounding across phases — a tableau typo caught at G2 costs ten minutes; the same typo
found after the Monte Carlo runs costs the project.

| Gate | When | Pass condition |
|---|---|---|
| **G1** | after `rootfind.py` | All three methods return the same root on toy test functions |
| **G2** | after baseline solver | Fixed step, halve `h` → error drops by ≈8 (BS3) / ≈32 (DP5) |
| **G3** | after all four solvers | The RHS-count identity below holds **exactly** for every run |
| **G4** | after relaxation works | baseline energy drifts visibly; relaxed variants hold the invariant to ~1e-15 |
| **G5** | before Monte Carlo | Python RHS counts match the Julia reference **exactly**; errors match to several digits |
| **G6** | during Monte Carlo | γ values cluster at 1 + O(Δt^(p-1)); outliers investigated, not ignored |

### Gate G3 in detail

With `s` the stage count and `c0` the startup cost, every run must satisfy, exactly:

```
baseline, FSAL-R, R-FSAL:   nf == c0 + (s-1) * (n_accept + n_reject)
naive:                      nf == c0 + (s-1) * (n_accept + n_reject) + n_accept
```

Every *attempt* costs `s-1` evaluations, rejected attempts included, because stage 1
is always served from the FSAL cache and a rejection restarts from the same `u_n`.
Naive pays one extra evaluation on accepted steps only — that `+ n_accept` term **is**
the inefficiency the paper removes, and here it is checkable as exact integer
arithmetic rather than by eyeballing a ratio.

**This gate is a within-run identity, not a cross-run comparison.** An earlier draft
of this document demanded that baseline, FSAL-R and R-FSAL report *identical* total
RHS counts. That is wrong, and a correct implementation would fail it. Relaxation
shifts both the state and the time (`t_γ = t_n + γ·Δt`), so the variants follow
slightly different trajectories from the first step onward; their step-size
controllers then make different accept/reject decisions and their step counts
legitimately differ.

For the cross-method comparison, use `nf / n_accept` and position on the
work-precision diagram — approximately equal, which is what the paper actually
claims.

`c0` is not a constant anyone remembers. `initial_step_size` **returns** its own call
count — 2 when the heuristic runs, 1 when `dt` is supplied — and solvers initialise `nf`
from it. A hardcoded value would go stale the moment that heuristic changed, and the
resulting G3 failure would look like a solver bug rather than a bookkeeping one.

**A wrinkle for G5:** the authors' Julia does *not* count the two evaluations inside
`ode_determine_initdt` when `dt` is auto-selected, though it *does* count the single one
when `dt` is supplied. We count all of them. So compare `nf − c0` against their numbers,
not `nf` — otherwise G5 fails as a constant off-by-two that looks like a real bug.

G3 and G5 are the strongest checks available — RHS counts are integers, so there is no
"close enough" to hide behind.

## Known traps

Carried over from analysis of the authors' Julia implementation. Do not rediscover these
the hard way.

1. **Dead failure path.** In all three Julia solvers the sequence
   `if γ < eps(...) ... else relaxation_flag = true` unconditionally resets the failure
   flag, so the "relaxation failed → halve step and retry" branch never executes. The
   typo `integrator *= 0.5` (should be `integrator.dt *= 0.5`) survives only because it
   is unreachable. **Random initial conditions will reach this path.** Implement the retry
   correctly from the start.

2. **The bracket γ ∈ [0.8, 1.2] can fail.** Needs a documented fallback. How often it
   fails is itself a Monte Carlo result worth reporting.

3. **The trivial root γ = 0** always satisfies the relaxation equation. Bracketing methods
   must exclude it; Newton started at γ₀ = 1 converges to the correct root naturally.

4. **State as plain floats, not NumPy arrays.** For a 2-component state, NumPy's per-call
   dispatch overhead (~1 µs) dwarfs the arithmetic (~50 ns). NumPy would be roughly 20×
   slower here.

5. **Never commit `results/`.** Generated data files are the second-largest source of
   merge conflicts after `__pycache__`.

6. **Solve for γ through `solve_relaxation_parameter` only.** No solver assembles its
   own residual or calls a finder directly. This project compares FSAL schemes, so
   everything that is not an FSAL scheme must be held identical between them —
   otherwise a difference in the results could come from the scheme under study or
   from one solver having been configured with a different tolerance, and the two
   could not be separated afterwards.

