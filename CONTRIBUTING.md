# Working agreement

This document is the single source of truth for who owns what. Its purpose is to make
merge conflicts structurally impossible rather than merely unlikely.

## The one rule

**One owner per file. Two people never edit the same file.**

If you need a change in someone else's file, open an issue or ask them — do not edit it
yourself. The only exception is the Phase 0 interface contract, which the whole team
agrees together in a single session.


## File ownership

*Proposed assignment — confirm or swap in the Phase 0 session, then treat as fixed.*

| Owner | Files | Notes |
|---|---|---|
| Prachurja Dhar (lead) | `src/problems.py`, repo config, `README.md` | Also: all PR reviews and merges, Julia reference generation, final report assembly |
| MD. Abir Hossain | `src/tableaus.py`, `src/stepping.py` | Self-contained numerics; no dependencies; can start immediately |
| Asikur Rahman | `src/rootfind.py`, `src/montecarlo.py` | Root-finders take plain callables — testable with no ODE code present |
| MD. Shadman Shafie | `src/solvers/classic.py` | baseline, naive, FSAL-R — three structurally-related loops |
| Tousif Fahmeed Quadir | `src/solvers/rfsal.py`, `src/analysis.py` | R-FSAL is the hardest single solver; plotting once results exist |

Everyone owns `tests/test_<their_module>.py` for their own modules.

## Branching

- `main` is protected. No direct pushes, ever.
- Branch per person per task: `feature/<name>-<module>`, e.g. `feature/abir-tableaus`.
- Open a Pull Request; the lead reviews and merges.
- Rebase or merge `main` into your branch before requesting review.

## Verification gates

Work does not advance past a gate until its check passes. Gates exist to stop errors
compounding across phases — a tableau typo caught at G2 costs ten minutes; the same typo
found after the Monte Carlo runs costs the project.

| Gate | When | Pass condition |
|---|---|---|
| **G1** | after `rootfind.py` | All four methods return the same root on toy test functions |
| **G2** | after baseline solver | Fixed step, halve `h` → error drops by ≈8 (BS3) / ≈32 (DP5) |
| **G3** | after all four solvers | baseline, FSAL-R, R-FSAL have **identical integer** RHS counts; naive ≈ 4/3× (BS3), 7/6× (DP5) |
| **G4** | after relaxation works | baseline energy drifts visibly; relaxed variants hold the invariant to ~1e-15 |
| **G5** | before Monte Carlo | Python RHS counts match the Julia reference **exactly**; errors match to several digits |
| **G6** | during Monte Carlo | γ values cluster at 1 + O(Δt^(p-1)); outliers investigated, not ignored |

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

4. **Golden-section search is a minimizer, not a root-finder.** It must operate on
   `|r(γ)|`, not `r(γ)`. Minimizing `r(γ)²` instead limits γ accuracy to ~sqrt(eps) ≈ 1e-8
   and visibly degrades energy conservation — run that as a deliberate side-experiment,
   not as the primary implementation. Golden section also fails *silently* when no root
   is bracketed, so always post-check that `|r(γ*)|` is genuinely near zero.

5. **State as plain floats, not NumPy arrays.** For a 2-component state, NumPy's per-call
   dispatch overhead (~1 µs) dwarfs the arithmetic (~50 ns). NumPy would be roughly 20×
   slower here.

6. **Never commit `results/`.** Generated data files are the second-largest source of
   merge conflicts after `__pycache__`.
