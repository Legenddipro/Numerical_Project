# Algorithm reference

Pseudocode transcription of the four time-stepping variants from the authors' Julia
implementation (`2023_FSAL_relaxation/code/code.jl`).

**Why this exists.** Porting a 150-line Julia loop while reading it means doing two hard
things at once — working out what the algorithm does, and saying it in Python. When the
result misbehaves you cannot tell which half went wrong. This document does the first
half once, carefully, so the implementation steps only have to do the second.


**This is not a faithful copy.** Three places where the source is wrong or inconsistent
are marked **DEVIATION** and describe what our implementation should do instead. Read
those before writing any code.

| variant | Julia source | our module |
|---|---|---|
| baseline | `solve_naive!`, `relaxation = false` | `solvers/classic.py` |
| naive | `solve_naive!`, `relaxation = true` | `solvers/classic.py` |
| FSAL-R | `solve_fsalr!` | `solvers/classic.py` |
| R-FSAL | `solve_rfsal!` | `solvers/rfsal.py` |

---

## Notation

| symbol | meaning |
|---|---|
| `u_prev` | state at the start of the step. After a relaxed step this *is* the relaxed point — there is no separate "unrelaxed current state" |
| `t`, `dt` | current time and trial step size |
| `k[1..s]` | stage derivatives; `s` = number of stages (4 for BS3, 7 for DP5) |
| `fsal_cache` | `f` evaluated at `u_prev`, carried over from the previous step |
| `A, b, b̂, c` | Butcher coefficients; `b̂` = embedded weights |
| `nf` | RHS evaluation counter |

Two structural facts used throughout, both verified in step 3:

- **`b[s] = 0`** for every explicit FSAL method. The last stage contributes nothing to
  the propagated solution.
- **`b̂[s] ≠ 0`** (1/8 for BS3). The last stage *does* contribute to the embedded
  solution. This asymmetry is the only reason the last stage is computed at all, and it
  is what R-FSAL exploits.

---

## Shared scaffolding

Every variant wraps this.

```
INITIALISE:
    t        ← t_start
    u_prev   ← u0
    dt, f0, c0 ← initial_step_size(problem, abstol, reltol, order)
    fsal_cache ← f0
    nf ← c0   ;  n_accept ← 0  ;  n_reject ← 0
    # c0 is REPORTED, not remembered: 2 when the heuristic ran (it probes the
    # right-hand side twice to estimate curvature), 1 when dt was supplied.

MAIN LOOP:  while t < t_end:
    if t + dt > t_end:  dt ← t_end - t          # never step past the end
    ... one attempt (variant-specific) ...
    if dt < 1e-14:  abort — step size collapsed
```

```
STAGES(first..last):                    # compute k[first..last]
    k[1] ← fsal_cache                   # NO RHS call — this is the FSAL saving
    for i = 2 .. last:
        y ← u_prev + dt · Σ_{j<i} A[i][j] · k[j]
        k[i] ← f(t + c[i]·dt, y)
        nf ← nf + 1
```

```
RELAX(u_prev, u_new) -> (γ, ok):        # src/rootfind.py, shared by all variants
    r(γ)  ← η(u_prev + γ·(u_new − u_prev)) − η(u_prev)
    solve r(γ) = 0 on the bracket [0.8, 1.2]
    ok    ← converged AND |r(γ)| ≤ residual_tol
    # NO RHS calls here — r evaluates η, never f.
    # γ = 0 always solves this; the bracket excludes it.
```

---

## 1. Baseline — no relaxation

```
ATTEMPT:
    STAGES(1..s)                                   # s−1 RHS calls
    u_new ← u_prev + dt · Σ_i b[i]·k[i]
    u_emb ← u_prev + dt · Σ_i b̂[i]·k[i]

    err    ← error_norm(u_new, u_prev, u_emb, abstol, reltol)
    factor ← controller.dt_factor(err, order)

    if controller.accept(factor):
        controller.on_accept()
        n_accept ← n_accept + 1
        t          ← t + dt
        u_prev     ← u_new
        fsal_cache ← k[s]              # FSAL reuse: k[s] IS f(u_new). Free.
    else:
        controller.on_reject()
        n_reject ← n_reject + 1

    dt ← dt · factor
```

> **DEVIATION 1 — the source does not count accepted baseline steps.**
> In `solve_naive!` the `relaxation = false` branch (lines 1215–1219) advances `t`,
> copies `u_prev` and refreshes the cache, but never calls `accept_step!(controller)`
> and never increments `naccept`.
>
> Two consequences: the reported `n_accept` is **0** for every baseline run, and the PID
> controller's error history never shifts, so baseline silently runs on a degenerate
> controller that only ever sees the current error while the other two slots stay frozen
> at their initial value.
>
> **Ours must do both**, as written above. Otherwise gate G3's identity is unusable for
> baseline, and baseline's step sequence is not comparable with the relaxation variants'.
> When comparing against the Julia table at gate G5, compare `nf` for baseline — not
> `n_accept`.

---

## 2. Naive — relaxation after the error test

Identical to baseline up to the acceptance decision.

```
ATTEMPT:
    STAGES(1..s)                                   # s−1 RHS calls
    u_new ← u_prev + dt · Σ_i b[i]·k[i]
    u_emb ← u_prev + dt · Σ_i b̂[i]·k[i]

    err    ← error_norm(u_new, u_prev, u_emb, abstol, reltol)
    factor ← controller.dt_factor(err, order)

    if NOT controller.accept(factor):
        controller.on_reject() ; n_reject += 1 ; dt ← dt · factor
        return                                     # error test failed, nothing else runs

    γ, ok ← RELAX(u_prev, u_new)

    if ok:
        controller.on_accept()                     # only NOW is the step truly accepted
        n_accept ← n_accept + 1
        u_γ ← u_prev + γ·(u_new − u_prev)
        t   ← (t + dt)  if this step lands on t_end  else  (t + γ·dt)
        u_prev ← u_γ

        fsal_cache ← f(t, u_γ)                     # ◄── THE EXTRA RHS CALL
        nf ← nf + 1

        dt ← dt · factor
    else:
        controller.on_reject()
        n_reject ← n_reject + 1
        dt ← dt · 0.5                              # halved, NOT scaled by factor
```

The single extra call is the whole point of this variant: it exists to show the cost the
other two remove. The cached `k[s]` is `f(u_new)`, but integration continues from `u_γ`,
so the cache belongs to a point that was just abandoned.

Note the two-stage acceptance — error test, *then* relaxation. This is why
`controller.dt_factor()` and `controller.on_accept()` are separate calls: a step can
pass the error test and still be rejected afterwards.

---

## 3. FSAL-R — interpolate the cache

Byte-for-byte identical to naive except for one line.

```
        fsal_cache ← k[1] + γ·(k[s] − k[1])        # ◄── interpolation. NO RHS call.
```

Both `k[1]` and `k[s]` are already in hand from this step's stages, so the replacement
costs arithmetic rather than a function evaluation. Lemma 1 bounds the error at
`O(dt^(p+1))`, small enough to leave the order of accuracy unchanged.

The authors expose a switch (`interpolate_FSAL = false`) that uses `k[s]` unmodified
instead. Keep it — it shows what the interpolation actually buys, at identical cost.

> **DEVIATION 2 — the relaxation-failure branch is unreachable in the source.**
> All three Julia solvers end their γ-solve with
> ```julia
> if γ < eps(typeof(γ)); relaxation_flag = false; else; relaxation_flag = true; end
> ```
> which unconditionally overwrites the failure flag set above it. Since γ is either 1.0
> or a root inside [0.8, 1.2], it is never below machine epsilon, so `relaxation_flag`
> is **always true** and the "halve the step and retry" branch never executes.
>
> That dead branch is also where `integrator *= 0.5` sits — a typo for
> `integrator.dt *= 0.5` that would raise a `MethodError` if it ever ran. It survives
> precisely because it cannot be reached.
>
> **Ours must implement the retry properly.** `ok = false` from `RELAX` means reject the
> step and halve `dt` — never proceed with γ = 1. Random initial conditions in the Monte
> Carlo sweep *will* reach this path.

---

## 4. R-FSAL — relax before the error test

Restructured rather than patched at the end.

```
ATTEMPT:
    STAGES(1..s−1)                                 # s−2 RHS calls — stage s SKIPPED
    u_new ← u_prev + dt · Σ_{i=1}^{s−1} b[i]·k[i]  # nothing missing: b[s] = 0

    if not relax_main:  u_unrelaxed ← u_new        # variant switch, saved for later

    γ, ok ← RELAX(u_prev, u_new)

    if NOT ok:
        n_reject ← n_reject + 1
        dt ← dt · 0.5
        return                                     # note: no RHS call was made

    u_γ   ← u_prev + γ·(u_new − u_prev)
    t_new ← (t + dt)  if this step lands on t_end  else  (t + γ·dt)

    f_γ ← f(t_new, u_γ)                            # ◄── the step's ONE relaxed call
    nf  ← nf + 1

    # reconstruct the stage we never computed, for the embedded solution
    if interpolate_fsal:
        k[s] ← fsal_cache + (1/γ)·(f_γ − fsal_cache)      # extrapolation
    else:
        k[s] ← f_γ

    # embedded solution, built from relaxed quantities
    scale ← (γ·dt) if relax_embedded else dt
    u_emb ← u_prev + scale · Σ_i b̂[i]·k[i]

    target ← u_γ if relax_main else u_unrelaxed
    err    ← error_norm(target, u_prev, u_emb, abstol, reltol)
    factor ← controller.dt_factor(err, order)

    if controller.accept(factor):
        controller.on_accept()
        n_accept ← n_accept + 1
        t          ← t_new
        u_prev     ← u_γ
        fsal_cache ← f_γ                # EXACT — no approximation here
    else:
        controller.on_reject()
        n_reject ← n_reject + 1

    dt ← dt · factor
```

`f_γ` does **two** jobs: it is the next step's first stage (exactly — the trajectory
really does continue from `u_γ`), and via the `1/γ` extrapolation it stands in for the
`f(u_new)` the embedded solution needs.

So the approximation sits in the opposite place from FSAL-R:

| | exact | approximated |
|---|---|---|
| **FSAL-R** | error estimate | FSAL cache |
| **R-FSAL** | FSAL cache | error estimate |

**Why `f(u_new)` is needed at all**, given the trajectory continues from `u_γ`: it is not
for the trajectory. It is for the embedded solution, whose weight `b̂[s]` is nonzero.
Without a value in that slot there is no error estimate and no adaptive stepping.

The factor is `1/γ` — it *stretches past* `u_γ` to reach `u_new`. FSAL-R's `γ` shrinks
in the other direction. Writing `γ` where `1/γ` belongs is the easiest way to get this
loop subtly wrong, and it will not show up as an obvious failure.

---

## RHS accounting

Per attempt — rejected attempts included, because stage 1 is always cached and a
rejection restarts from the same `u_prev`.

| variant | stage calls | extra | per attempt (BS3, s=4) | (DP5, s=7) |
|---|---|---|---|---|
| baseline | s−1 | — | 3 | 6 |
| naive | s−1 | +1 per **accepted** step | 3 (+1) | 6 (+1) |
| FSAL-R | s−1 | — | 3 | 6 |
| R-FSAL | s−2 | +1 at the relaxed point | 3 | 6 |

Giving the gate G3 identities:

```
baseline, FSAL-R:   nf = c0 + (s−1)·(n_accept + n_reject)
naive:              nf = c0 + (s−1)·(n_accept + n_reject) + n_accept
R-FSAL:             nf = c0 + (s−1)·(n_accept + n_reject) − n_relax_fail
```

> **DEVIATION 3 — R-FSAL carries a correction term.**
> A relaxation failure in R-FSAL returns *before* the `f_γ` call, so that attempt costs
> only `s−2`, not `s−1`. Naive and FSAL-R have no such gap: their relaxation runs after
> all stages are computed, so a failure there still cost the full `s−1`.
>
> When relaxation never fails — the normal case on well-behaved initial conditions —
> `n_relax_fail = 0` and all three identities collapse to the same form. Track the
> counter anyway: an unexplained G3 failure for R-FSAL alone is almost certainly this,
> not a bug in the FSAL logic.

---

## Hand-trace: one BS3 step

Walking the baseline/naive pseudocode above from `u_prev = (ω, θ) = (1.5, 1.0)`,
`dt = 0.4`, `g/L = 1`. These are the numbers your implementation must reproduce.

| quantity | value |
|---|---|
| `η(u_prev)` | `0.5846976941` |
| `k[1]` *(cached, free)* | `(-0.8414710, 1.5000000)` |
| `k[2]` | `(-0.9635582, 1.3317058)` |
| `k[3]` | `(-0.9853666, 1.2109325)` |
| `u_new` | `(1.1215519, 1.5261710)` |
| `η(u_new)` | `0.5843287805` — **drifted by −3.689e−4** |
| `k[4] = f(u_new)` | `(-0.9990045, 1.1215519)` |
| `γ` | `1.0036353183` |
| `u_γ` | `(1.1201761, 1.5280838)`, drift `0.0` |
| `t_γ` | `0.4014541` (not `0.4`) |
| `f(u_γ)` *(naive computes this)* | `(-0.9990880, 1.1201761)` |
| FSAL-R interpolation, `k[1] + γ·(k[4] − k[1])` | `(-0.9995771, 1.1201761)` |
| R-FSAL extrapolation, `k[1] + (1/γ)·(f(u_γ) − k[1])` | `(-0.9985170, 1.1215519)` |
| `u_emb` | `(1.1241401, 1.5257058)` |
| scaled error @ tol 1e-3 | `0.7436` → accept |

**The R-FSAL row is the one that catches the `γ` / `1/γ` swap.** Writing `γ` where
`1/γ` belongs gives `(-0.9996609, 1.1187953)` instead — different in the third decimal,
so the check discriminates clearly. This matters because that particular typo passes
gate G3 (the RHS count is unaffected) and gate G4 (relaxation still conserves energy
exactly). It corrupts only the error estimate, so it surfaces as "our numbers do not
quite match Julia" — a fuzzy failure with five candidate causes rather than a sharp one.
Check this number directly and the whole class of error is ruled out in a minute.

Two more checks worth noticing in that table:

**The FSAL-R interpolation is exact in the second component and off by 4.9e−4 in the
first.** That is not luck. `θ' = ω` is linear, and interpolation commutes with linear
maps; `ω' = −sin θ` is not. A correct implementation shows exactly this split — if both
components are off, or both exact, something is wrong.

**`t_γ ≠ t + dt`.** Relaxation moves the time as well as the state. Forgetting this is
a silent error: the solution stays plausible but is compared against the reference at
the wrong instant, which shows up only as a mysteriously large global error.
