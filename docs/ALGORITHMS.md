# Algorithm reference

Pseudocode transcription of the four time-stepping variants, taken from the authors'
Julia implementation (`ranocha/2023_FSAL_relaxation`, `code/code.jl`).

**Purpose.** Writing the algorithms out precisely *before* porting them prevents
translation errors that are hard to diagnose later. This document is a Phase 1
deliverable and is written by the two solver owners.

**Status:** not yet written.

## To transcribe

| Variant | Julia source | Target module |
|---|---|---|
| Baseline (no relaxation) | `solve_naive!` with `relaxation = false` | `src/solvers/classic.py` |
| Naive relaxation + FSAL | `solve_naive!` with `relaxation = true` | `src/solvers/classic.py` |
| FSAL-R | `solve_fsalr!` | `src/solvers/classic.py` |
| R-FSAL | `solve_rfsal!` | `src/solvers/rfsal.py` |

## Points to capture carefully

- Exactly which stages call the right-hand side, and which are reused from cache.
- Where `nf` (the RHS counter) is incremented — this drives gate G3.
- The interpolation used by FSAL-R for the next step's first stage.
- The extrapolation used by R-FSAL for the embedded solution, and the fact that R-FSAL
  skips stage `s` entirely in the main loop.
- Accept/reject ordering relative to relaxation, which differs between FSAL-R and R-FSAL.
- The end-of-integration special case (`relaxation_at_last_step`).
