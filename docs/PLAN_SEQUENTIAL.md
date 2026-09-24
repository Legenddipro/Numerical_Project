# Sequential plan

**One person works at a time.** Each step is finished, verified and handed off before
the next begins. This file is self-contained: everything needed to do and check a step
is here, including the reference numbers to check against.

(The parallel alternative is [PLAN.md](PLAN.md). Use one or the other, not both.)

---

## How a step works

Everyone works on **one shared branch** (`main`). No feature branches, no pull
requests. That only works because exactly one person is working at a time.

1. `git pull` — start from the previous person's finished work, never from a stale copy.
2. Implement only your step's files.
3. Run your step's **Verify** block. Every check must pass.
4. `git add`, `git commit`, `git push`.
5. Post the step's **Handoff** line to the group. **Only then** does the next person start.

Three rules that make the single branch safe:

- **Never start before the handoff.** With one shared branch, two people working at
  once will collide directly — the sequential discipline is what prevents conflicts, so
  it is load-bearing, not merely polite.
- **Always `git pull` first**, even if you are sure nothing changed.
- **Push as soon as your step passes.** Nobody downstream can begin until your work is
  on the remote.

And one consequence of dropping pull requests: **the Verify block is now the only
gate.** There is no reviewer to catch what it misses, so actually run every check rather
than assuming it passes. A bug caught at its own step costs minutes; the same bug found
three steps later costs a day, because by then you have to work out *which* module
introduced it.

## Assignment

| Step | Owner | Files | Depends on |
|---|---|---|---|
| 0 | everyone | contract review | — |
| 1 | Prachurja | `docs/ALGORITHMS.md` | 0 |
| 2 | Prachurja | `src/problems.py` | 0 |
| 3 | Abir | `src/tableaus.py`, `src/stepping.py` | 0 |
| 4 | Abir | Julia reference table → `tests/data/` | 3 |
| 5 | Shadman | `src/rootfind.py` | 0 |
| 6 | Shadman | `src/solvers/classic.py` | 1–5 |
| 7 | Asikur | `src/solvers/rfsal.py` | 6 |
| 8 | Asikur | `src/montecarlo.py` | 7 |
| 9 | Tousif | `src/analysis.py` | 8 |
| 10 | Prachurja | report | 9 |

Steps 5–6 and 7–8 are deliberately paired: whoever writes the root-finder then writes
the solvers that call it, and whoever writes R-FSAL then writes the sweep that runs it.
Each pair keeps its context instead of handing it over mid-thought.

Step 1 comes first because it is the understanding step — if transcribing the
algorithms reveals something surprising, it surfaces before anyone has built on a wrong
assumption. It also leaves its author as the one person who knows all four loops cold
without having written the code, which is the closest thing to a reviewer this workflow
has once pull requests are gone.

---

## The golden step

**Check your work against these numbers before anything else.** They are one BS3 step on
the pendulum from $u^0 = (\omega,\theta) = (1.5,\ 1.0)$ with $\Delta t = 0.4$ and
$g/L = 1$, computed and verified independently. Steps 1, 2, 3, 5, 6 and 7 all check
against some part of this table.

| quantity | value |
|---|---|
| $\eta(u^0)$ | `0.5846976941` |
| $k_1 = f(u^0)$ | `(-0.8414710, 1.5000000)` |
| $k_2$ | `(-0.9635582, 1.3317058)` |
| $k_3$ | `(-0.9853666, 1.2109325)` |
| $u^1$ | `(1.1215519, 1.5261710)` |
| $\eta(u^1)$ | `0.5843287805` (drift $-3.689\times10^{-4}$) |
| $k_4 = f(u^1)$ | `(-0.9990045, 1.1215519)` |
| $\gamma$ | `1.0036353183` |
| $u^1_\gamma$ | `(1.1201761, 1.5280838)` (drift `0.0`) |
| $t_\gamma$ | `0.4014541` |
| $f(u^1_\gamma)$ | `(-0.9990880, 1.1201761)` |
| FSAL-R interpolation, $k_1 + \gamma(k_4-k_1)$ | `(-0.9995771, 1.1201761)` |
| R-FSAL extrapolation, $k_1 + \tfrac1\gamma(f(u^1_\gamma)-k_1)$ | `(-0.9985170, 1.1215519)` |
| embedded $\widehat u^1$ | `(1.1241401, 1.5257058)` |
| scaled error @ tol `1e-3` | `0.7436` → accept |

The last two rows are the sharpest checks in the table. Getting $\gamma$ and $1/\gamma$
the wrong way round passes gates G3 and G4 untouched — the RHS count is unaffected and
relaxation still conserves energy exactly — and shows up only as a vague mismatch
against Julia. Checking these two numbers directly rules out the whole class in a minute.

---

## Step 0 — Contract review *(everyone, one sitting)*

Every module already has function names, arguments, type hints and docstrings with no
bodies. Read them together. Change anything you disagree with **now** — signatures are
cheap to change before bodies exist and expensive after.

Settle explicitly:

- **State is `(omega, theta)`, plain floats, not NumPy arrays.** At two components
  NumPy's ~1 µs per-call dispatch overhead dwarfs the ~50 ns of arithmetic. NumPy here
  is roughly 20× slower.
- **`c0`, the startup RHS count — settled.** We count every call to `f` honestly,
  including the two inside the initial-step heuristic, because the work-precision
  diagrams should reflect real cost. `initial_step_size` *returns* its own call count
  (2 when the heuristic runs, 1 when `dt` is supplied), so `c0` is reported rather than
  remembered and cannot go stale.

  The authors do **not** count those two when `dt` is auto-selected, but *do* count the
  single one when `dt` is given. So at step 4's comparison, compare `nf − c0` on both
  sides rather than `nf`. Fudging one side by a magic 2 would be worse than making the
  convention explicit.

**Handoff:** "Contract agreed. Step 1 can start."

---

## Step 1 — `docs/ALGORITHMS.md` *(Prachurja)*

### Build

Transcribe the four time-stepping loops out of the authors' Julia
(`../2023_FSAL_relaxation/code/code.jl`) into precise, language-neutral pseudocode:

| variant | Julia source |
|---|---|
| baseline | `solve_naive!` with `relaxation = false` |
| naive | `solve_naive!` with `relaxation = true` |
| FSAL-R | `solve_fsalr!` |
| R-FSAL | `solve_rfsal!` |

Porting a 150-line Julia loop *while reading it* means doing two hard things at once —
working out what the algorithm does, and saying it in Python. When the result misbehaves
you cannot tell which half went wrong. Writing the pseudocode first separates them, and
it is the cheapest insurance available against a mistranslated solver, which is the most
expensive kind of bug in this project.

Capture, for each loop:

- **Which stages call the RHS and which reuse the cache.** The first stage is never a
  fresh call except on the very first step.
- **Every place `nf` increments.** This drives gate G3; an off-by-one here stays
  invisible until the identity fails and then looks like a solver bug.
- **The interpolation and extrapolation formulas**, including which uses $\gamma$ and
  which uses $1/\gamma$ — FSAL-R interpolates with $\gamma$, R-FSAL extrapolates with
  $1/\gamma$, and swapping them is the easiest way to get R-FSAL subtly wrong.
- **Accept/reject ordering relative to relaxation**, which genuinely *differs* between
  FSAL-R and R-FSAL. FSAL-R runs the error test first and can then still fail relaxation;
  R-FSAL relaxes first, so a failed relaxation never reaches the error test.
- **The end-of-integration special case** (`relaxation_at_last_step`).

Flag, rather than faithfully copy, the known defect: in all three Julia solvers the
sequence `if γ < eps(...) ... else relaxation_flag = true` unconditionally resets the
failure flag, so the "relaxation failed → halve the step and retry" branch is
unreachable, and a typo survives inside it unnoticed. **Our version must implement that
retry properly**, so the pseudocode should describe the intended behaviour and note
where it departs from the source.

### Verify

The reference numbers in the document are already computed and cross-checked, so there
is no arithmetic to re-derive. Two cheap structural checks remain:

- **Count the RHS calls per attempt from the pseudocode alone**, ignoring the maths
  entirely. It must come to $s-1$ for baseline, FSAL-R and R-FSAL, and $s-1$ plus one
  per accepted step for naive. Two minutes, and it catches the error class that matters
  most: the RHS count *is* the paper's result, so a transcription that gets it wrong
  yields an implementation that cannot reproduce the finding.
- **Account for every `nf += 1` in the Julia source.** Each appears somewhere in the
  pseudocode, or is explicitly noted as excluded (the initial-step heuristic — see the
  `c0` discussion in step 0). This catches what was skipped without anyone noticing,
  which the first check cannot.

The remaining risk is not whether the document is *correct* but whether it is
*followable* — and its author is the one person who cannot test that, because they will
read their own shorthand as obvious. So that check is deferred to the people who depend
on it: steps 6 and 7 begin by reading it, and anything ambiguous goes back to the author
before any code is written.

**Handoff:** "ALGORITHMS.md written, RHS accounting checked. Step 2 can start."

---

## Step 2 — `src/problems.py` *(Prachurja, continuing)*

### Build

The pendulum $\theta'' + \frac{g}{L}\sin\theta = 0$ as a first-order system in
$u = (\omega, \theta)$:

$$\omega' = -\tfrac{g}{L}\sin\theta, \qquad \theta' = \omega$$
$$\eta(u) = \tfrac12\omega^2 - \tfrac{g}{L}\cos\theta, \qquad
\nabla\eta = \left(\omega,\ \tfrac{g}{L}\sin\theta\right)$$

Use $g/L = 1$. **Component order is `(omega, theta)`** — the authors' `du[2] = u[1]`
means their `u[2]` is the angle, so their initial condition `[1.5, 1.0]` is
$\omega=1.5,\ \theta=1.0$. Getting this backwards yields a system that still oscillates
plausibly and silently fails step 4.

Also build: the reference solution (`scipy.integrate.solve_ivp`, `DOP853`,
`rtol=1e-13, atol=1e-14`) and the initial-condition sampler.

The sampler must take a **mandatory seed** and a regime. With $g/L=1$, $E<1$ means
libration (the bob swings back and forth) and $E>1$ means rotation (it carries over the
top). These are qualitatively different and must never be pooled silently. Sample by
rejection: draw from a rectangle, discard the wrong regime, repeat.

### Verify

- $f(1.5,\ 1.0) = (-0.8414710,\ 1.5)$ and $\eta(1.5,\ 1.0) = 0.5846976941$
- Gradient matches a central finite difference of $\eta$ to ~1e-8
- Reference solution holds $\eta$ to ~1e-10 across $t\in[0,10]$
- Small-angle period → $2\pi$ as amplitude → 0 *(independent physics check, not just
  internal consistency)*
- Same seed gives identical samples; every libration sample has $E<1$

**Handoff:** "problems.py merged, all checks pass. Step 3 can start."

---

## Step 3 — `src/tableaus.py` + `src/stepping.py` *(Abir)*

### Build — tableaus

BS3 (4 stages, order 3) and DP5 (7 stages, order 5), transcribed from the authors'
`ButcherTableau` constructors. PID gains: **BS3 → (0.6, −0.2, 0.0)**, **DP5 → (0.7,
−0.4, 0.0)**.

### Build — stepping

PID controller, scaled error norm, initial step heuristic.

The controller uses the last **three** error estimates, not just the current one:

$$\text{dt\_factor} = \text{err}_1^{\beta_1/k}\cdot \text{err}_2^{\beta_2/k}\cdot
\text{err}_3^{\beta_3/k}$$

limited by $1 + \arctan(x-1)$, accepted when $\ge 0.81$. The error history shifts **only
on final acceptance**, not when the error test passes — because the relaxation variants
accept in two stages and a step passing the error test can still fail relaxation. Keep
`dt_factor()` and `on_accept()` as separate calls.

Error norm: scale each component by $\text{abstol} + \text{reltol}\cdot\max(|u_i|,
|u^{\text{prev}}_i|)$, then take the RMS. A value $\le 1$ means the step meets tolerance.

`initial_step_size` returns `(dt0, f(t0,u0), n_rhs_calls)`. It evaluates the RHS anyway
and every solver needs that value as its first cache entry; recomputing it would inflate
`nf` by exactly one, which step 6's identity check would see.

The third value is the startup count `c0` — 2 when the heuristic runs (it probes the
right-hand side twice to estimate curvature), 1 when `dt` is supplied. Reporting it
rather than letting solvers hardcode a constant means it cannot go stale if this
heuristic ever changes; a stale constant would surface as a gate G3 failure that looks
like a solver bug.

### Verify

Order conditions, exactly (use `fractions.Fraction`, not floats):

| condition | required | BS3 `b` | BS3 `b_embedded` |
|---|---|---|---|
| $\sum b_i$ | 1 | ✓ | ✓ |
| $\sum b_i c_i$ | 1/2 | ✓ | ✓ |
| $\sum b_i c_i^2$ | 1/3 | ✓ | **3/8 — must FAIL** |
| $\sum b_i a_{ij}c_j$ | 1/6 | ✓ | **3/16 — must FAIL** |

Those two failures are the point, not a bug: the embedded method is order 2 *because*
it misses the order-3 conditions. If it satisfied them, the two solutions would agree
too closely and their difference would estimate nothing.

FSAL identities, both tableaus: `A[-1] == b`, `c[-1] == 1`, `b[-1] == 0`,
`b_embedded[-1] != 0`. That last asymmetry is why the final stage is computed at all.

Controller: big error shrinks the step; the limiter bounds the factor even for a
near-zero error estimate; a zero estimate produces neither `inf` nor `nan`;
`on_reject()` leaves the history unchanged.

Against the golden step: with $k_1,k_2,k_3,k_4$ given above, your `b` weights must
reproduce $u^1$ and your `b_embedded` weights must reproduce $\widehat u^1$, and the
scaled error at tol `1e-3` must be `0.7436`.

**Handoff:** "tableaus.py and stepping.py merged, order conditions verified exactly.
Step 4 can start."

---

## Step 4 — Julia reference table *(Abir, continuing)*

### Build

Install Julia and generate the ground-truth numbers everything downstream is checked
against. Budget **45–75 minutes and 3–4 GB**, most of it waiting.

```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.9.3                      # pin: the Manifest.toml requires exactly this
sudo apt install python3-matplotlib
export PYTHON=/usr/bin/python3         # else PyCall downloads its own Conda (~400 MB)
cd ../2023_FSAL_relaxation/code
julia +1.9.3 --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then in `julia +1.9.3 --project=.`, after `include("code.jl")`, run the pendulum through
all four variants at each tolerance and record `nf`, final error, and entropy drift.
Use the authors' own initial condition (`[1.5, 1.0]`, $t\in[0,10]$).

Check the table into `tests/data/julia_reference.csv`.

### Verify

- The table has a row for every (method, tableau, tolerance) combination
- Drift is ~1e-15 for the three relaxation variants and visibly larger for baseline —
  if not, the Julia run is misconfigured and the table is worthless
- `nf` values are integers and naive's exceeds baseline's by roughly $s/(s-1)$

**Handoff:** "julia_reference.csv checked in. Step 5 can start."

*Doing this before the solvers means steps 6 and 7 each validate against Julia as they
are written, instead of all four being validated at once at the end. When something
disagrees you will know exactly which method introduced it.*

---

## Step 5 — `src/rootfind.py` *(Shadman)*

### Build

Each relaxation step solves one scalar equation:

$$r(\gamma) = \eta\big(u^n + \gamma(u^{n+1}-u^n)\big) - \eta(u^n) = 0$$

Three methods sharing one signature so solvers can swap them: **Newton** (from
$\gamma_0=1$; needs $dr/d\gamma = \nabla\eta(u^n+\gamma d)\cdot d$), **bisection**, and
**`scipy.optimize.toms748`** — which is the identical Algorithm 748 the paper uses, so
it is the baseline the other two are measured against.

Default bracket **$[0.8,\ 1.2]$**.

Two things that shape the design:

- **$\gamma = 0$ always solves the equation** — it means "do not move at all". The
  bracket is centred on 1 to exclude it.
- **Every result carries a `residual`**, checked independently of the method's own
  verdict. Newton has no bracketing guarantee: a near-zero derivative can throw an
  iterate anywhere and the method may still report success.

Also build `solve_relaxation_parameter` — the single entry point **both** solver modules
call. No solver assembles its own residual. This project compares FSAL schemes, so
everything that is not an FSAL scheme must be identical between them; otherwise a
difference in the results could come from the scheme under study or from one solver
having been given a different tolerance, and afterwards the two cannot be separated.

This is not hypothetical. A $\gamma$ differing in its 13th digit shifts the next error
estimate in its 13th digit, which can flip a borderline accept/reject, after which the
two runs take entirely different step sequences.

### Verify

- All three find the same root on toy functions (`x²−2`, `cos x − x`, …)
- Newton needs ~3–5 iterations; bisection ~45 on a width-0.4 bracket to 1e-14; toms748
  sits near Newton
- Bisection returns `converged=False` with no sign change — never a fabricated root
- Newton escaping the bracket returns `converged=False`, never a silent fallback to
  $\gamma=1$
- `make_residual_derivative` matches a finite difference of `make_residual`
- Against the golden step: with $u^0=(1.5,1.0)$ and $u^1=(1.1215519,\ 1.5261710)$, all
  three must return $\gamma = 1.0036353183$

**Handoff:** "rootfind.py merged, all three agree on γ = 1.0036353183. Step 6 can start."

---

## Step 6 — `src/solvers/classic.py` *(Shadman, continuing)*

**Start by reading [ALGORITHMS.md](ALGORITHMS.md) end to end.** It is the pseudocode for
exactly the three loops you are about to write, including three places where the
authors' Julia is wrong and ours must deviate. If anything in it is ambiguous, ask its
author before writing code — you are the first person to read it, so unclear passages
are found here or not at all.

Three loops sharing most of their structure. **Build and verify them in this order** —
each depends on the previous being correct.

### 5a — baseline (no relaxation)

Plain embedded RK with FSAL reuse. Stage 1 comes from the cache, never a fresh call;
after an accepted step the last stage is stashed for next time.

**Verify before going further — this is the most important check in the project:**

Fixed step (`adaptive=False`), no relaxation. Halve `dt`; the error must drop by
**≈ 8 for BS3** and **≈ 32 for DP5**. If this fails, the tableau or the stage loop is
wrong, and nothing built on top of it is worth running. Watch the step range: too small
and round-off dominates, flattening the observed order into a false failure.

Then the RHS identity, exactly:

$$\texttt{nf} = c_0 + (s-1)\times(\texttt{n\_accept} + \texttt{n\_reject})$$

Every *attempt* costs $s-1$ evaluations, rejections included, because stage 1 is always
cached and a rejected step restarts from the same $u^n$.

Finally: `nf` matches `julia_reference.csv` for baseline.

### 5b — naive relaxation

Ordinary step, error test, accept — *then* solve for $\gamma$, move to $u_\gamma$, and
compute $f(u_\gamma)$ with a **fresh RHS call**, because the cached value belongs to
$u^{n+1}$, a point you just abandoned.

That extra call is the inefficiency the whole paper removes. It must be visible:

$$\texttt{nf} = c_0 + (s-1)\times(\texttt{n\_accept} + \texttt{n\_reject}) + \texttt{n\_accept}$$

The trailing `n_accept` is one extra evaluation per accepted step, none on rejected ones.

Also verify: energy held to ~1e-15 across the whole run, and `nf` matches Julia's naive
row.

**Trap:** in the authors' Julia code the relaxation-failure path is unreachable — a flag
reset makes the "halve the step and retry" branch dead, which is why a typo survives
inside it unnoticed. Implement the retry properly. A `converged=False` from
`solve_relaxation_parameter` means **reject the step and halve `dt`**, never proceed with
$\gamma=1$.

### 5c — FSAL-R

Identical to naive until the relaxed point is formed. Instead of a fresh call,
interpolate between two stage values already in hand:

$$f(u_\gamma) \approx k_1 + \gamma\,(k_{\text{last}} - k_1)$$

**Verify:** `nf` satisfies the *baseline* identity (no `n_accept` term); energy still
~1e-15; `nf` matches Julia's FSAL-R row.

Against the golden step, the interpolation gives `(-0.9995771, 1.1201761)` against a
true `(-0.9990880, 1.1201761)`. Note the second component is **exact** — $\theta'=\omega$
is linear, and interpolation commutes with linear maps. The first component is off by
$4.9\times10^{-4}$ because $\omega'=-\sin\theta$ is not. That split is the theory made
visible and is a good sign your implementation is right.

**Handoff:** "classic.py merged; convergence order, both RHS identities, and Julia match
all verified. Step 7 can start."

---

## Step 7 — `src/solvers/rfsal.py` *(Asikur)*

**Start by reading the R-FSAL section of [ALGORITHMS.md](ALGORITHMS.md).** Flag anything
ambiguous to its author before coding. Pay particular attention to the extrapolation
factor: it is $1/\gamma$, not $\gamma$, and the golden step table has the number that
tells the two apart.

### Build

R-FSAL reorders the step rather than patching its end:

1. Compute stages $1..s-1$ **only**. Stage $s$ is skipped — legitimate because
   $b_s = 0$, so it never contributed to the propagated solution anyway.
2. Assemble $u^{n+1}$ from those stages and solve for $\gamma$ **immediately**, before
   any error test.
3. Make the step's single RHS call at the *relaxed* point: $f(u_\gamma)$.
4. That one value serves twice: **exactly** as the next step's cached first stage, and —
   via extrapolation with factor $1/\gamma$ — as the $f(u^{n+1})$ the embedded solution
   needs:

$$f(u^{n+1}) \approx f(u^n_\gamma) + \tfrac{1}{\gamma}\big(f(u^{n+1}_\gamma) - f(u^n_\gamma)\big)$$

5. Build the embedded solution from relaxed quantities, then apply error control.

**Why $f(u^{n+1})$ is needed at all**, given the trajectory continues from $u_\gamma$: it
is not for the trajectory. It is for the *error estimate*. The embedded weights have
$\widehat b_s \ne 0$ (for BS3, $1/8$), so without a value in that slot there is no
embedded solution, no error estimate, and no adaptive stepping.

So FSAL-R and R-FSAL are mirror images: **FSAL-R has an exact error estimate and an
approximated FSAL cache; R-FSAL has an exact FSAL cache and an approximated error
estimate.**

### Verify

- `nf` satisfies the baseline identity — $s-2$ stage calls plus one at the relaxed point
- Energy ~1e-15; `nf` matches Julia's R-FSAL row
- **Sharpest check:** on the harmonic oscillator (linear $f$), the $1/\gamma$
  extrapolation reproduces $f(u^{n+1})$ **exactly**, to round-off. Not a coincidence —
  a linear map commutes with the affine combination the formula performs. This fails
  loudly if the factor is $\gamma$ instead of $1/\gamma$.
- `interpolate_fsal`, `relax_embedded` and `relax_main` each change results
  independently, so all eight combinations are reachable

**Handoff:** "rfsal.py merged, all four methods match Julia. Step 8 can start."

---

## Step 8 — `src/montecarlo.py` *(Asikur, continuing)*

### Build

Sweep the cross product: 4 methods × 2 tableaus × 7 tolerances × 3 root-finders × N
initial conditions. Baseline uses no root-finder, so run it once per (tableau,
tolerance, IC) — running it once per finder would produce duplicate rows and inflate its
apparent share of the compute.

One CSV row per run. **Omit the histories** — keeping them for a thousand samples costs
gigabytes. A run that raises is recorded as a failure row, not allowed to abort the
sweep: with random initial conditions some failures are expected, and the rate is itself
a result.

**Run N = 50 first and look at the output** before scaling to 1000. Expect ~12 minutes
for N = 1000 in plain Python; `multiprocessing.Pool` divides that by core count, with
each worker deriving its seed from the master seed.

### Verify

- Row count equals the cross product, minus baseline's root-finder dimension
- Same seed → identical rows; `n_workers=4` → identical rows to `n_workers=1`
- A deliberately failing configuration appears as a failure row and the sweep continues
- **Gate G6:** $\gamma$ clusters at $1 + \mathcal O(\Delta t^{p-1})$, and tightening the
  tolerance visibly pulls it toward 1. Outliers get investigated, not trimmed — a
  stress test that never fails has not stressed anything.

**Handoff:** "Sweep complete, N=____, results/____.csv checked. Step 9 can start."

---

## Step 9 — `src/analysis.py` *(Tousif)*

### Build

- **Work-precision diagrams** — error against RHS calls, log-log, in the paper's Figure 1
  format. Each point aggregates over many initial conditions, so draw the spread as a
  band rather than discarding it.
- **Energy drift plots** — needs runs with `save_history=True`.
- **Root-finder table** — iterations per step, failure rate, drift achieved.
- **Monte Carlo summaries** — distributions split by regime, never pooled silently.

### Verify

The claim, asserted on the data behind the figure rather than left for a reader to
notice: at equal tolerance **baseline has the largest error**, **naive the largest RHS
count**, and **FSAL-R and R-FSAL match baseline's cost at naive's accuracy**.

Also: Newton < toms748 < bisection in mean iterations; drift indistinguishable across
the three finders (a difference there means some configuration leaked into the
comparison and the table is measuring that instead of the methods); `load_results` warns
on an incomplete sweep rather than quietly plotting around the gap.

> **Correction from step 6 — the drift claim above does not hold.** Newton's drift runs
> 10–100× the two bracketing methods', and legitimately so; asserting
> indistinguishability will fail a correct implementation. See ADDENDUM E below for the
> measured numbers and what to assert instead.

**Handoff:** "Figures and tables in results/. Step 10 can start."

---

## Step 10 — Report *(Prachurja, with everyone)*

Each person writes the section covering their own step; the lead assembles.

Cite the base paper (DOI `10.1007/s10915-025-03130-6`) **and** the authors'
reproducibility repository (DOI `10.5281/zenodo.10201246`) — their README explicitly
asks for the second when their implementations are used, and ours are ported from them.

Report what did **not** work as well as what did: relaxation failures, γ outliers,
regimes where a method struggled. A stress-testing project whose stress test found
nothing has not demonstrated robustness — it has demonstrated an untested range.

---

## Gate summary

| Gate | Step | Pass condition |
|---|---|---|
| G1 | 5 | All three root-finders agree on toy functions and on γ = 1.0036353183 |
| G2 | 6a | Halving `dt` cuts error by ≈8 (BS3) / ≈32 (DP5) |
| G3 | 6a, 6b, 6c, 7 | `nf == c0 + (s-1)(n_accept + n_reject)`, plus `+ n_accept` for naive |
| G4 | 6b | Baseline drifts; relaxed variants hold η to ~1e-15 |
| G5 | 6, 7 | `nf` matches `julia_reference.csv` exactly; errors to several digits |
| G6 | 8 | γ clusters at 1 + O(Δt^(p−1)) |

G3 and G5 are the strongest checks available: `nf` is an integer, so there is no
"close enough" to hide behind.

---

## Traps

1. **The relaxation-failure retry.** Dead code in the authors' Julia version; random
   initial conditions will reach it. Implement it properly (step 6b).
2. **The bracket $[0.8, 1.2]$ can fail.** Needs a documented fallback; the failure rate
   is a result.
3. **$\gamma = 0$ is always a root.** Bracketing methods must exclude it.
4. **Plain floats, not NumPy arrays**, for a 2-component state.
5. **Never commit `results/`.**
6. **Solve for γ only through `solve_relaxation_parameter`.** No solver builds its own
   residual.

---

## Addenda from steps 5–6

*Added by Shadman after implementing `rootfind.py` and `classic.py`. This is the lead's
file, so these are appended as marked addenda rather than edited into the plan above, and
they arrive in a commit of their own — revert that one commit and the document is exactly
as its author left it. The algorithmic counterparts (A–D) are in
[ALGORITHMS.md](ALGORITHMS.md).*

### Status

Steps 5 and 6 are done and pushed. Gates G1, G2, G3 and G4 pass; **G5 is outstanding and
cannot be run** — step 4 was skipped, so `tests/data/julia_reference.csv` does not exist.
Every step-6 check that names it was skipped for that reason and no other.

| Gate | Result |
|---|---|
| G1 | All three finders return γ = `1.0036353182810`. Newton 4 iterations, toms748 4, bisection 45 |
| G2 | BS3 error ratio 8.003, DP5 33.5 |
| G3 | Exact in all 70 (method × tableau × tolerance × finder) combinations; `nf` also checked against a call-counting spy |
| G4 | Baseline drifts 1.5e-2 … 8.7e-11; every relaxed variant holds η to ~1e-15 |
| G5 | **Not run — no reference table** |

Because G5 is the gate that stands between the port and the Monte Carlo sweep, step 8
should not be treated as unblocked merely because steps 6 and 7 pass their own checks.
Either step 4 gets revived before the sweep, or the report says plainly that the port was
never checked against the authors' numbers.

The cross-method gate checks live in `test_gates.py`, which is the lead's file and still
stubbed. What steps 5–6 could establish alone — G1, G2, G3, G4 and the golden step —
is checked in `test_rootfind.py` and `test_classic.py` instead, so a broken solver is not
handed to step 7 while waiting for someone else's test file.

### ADDENDUM E — step 9's root-finder table

Two corrections to the Verify block in step 9.

**Drift is not indistinguishable across the three finders**, and the difference is not a
leak in the comparison. Newton stops at its round-off floor (see ADDENDUM C in
ALGORITHMS.md), so its drift runs consistently 10–100× the two bracketing methods': at
BS3 with tolerance 1e-11, Newton reaches 2.0e-13 against 2.7e-15 for both bisection and
toms748. All three remain vastly better than baseline. Assert a *bound* — every finder
holds η far below baseline's drift — not an equivalence.

**`nf`, `n_accept` and `n_reject` are identical across all three finders**, so those
columns carry no signal at all. Mean iterations per step and attainable drift are the
whole table. Measured at step 6 over BS3 and DP5 at tolerances 1e-3 … 1e-11:

| finder | mean iterations / step |
|---|---|
| Newton | 2.8 – 4.6 |
| toms748 | 3.8 – 5.2 |
| bisection | 31.5 – 44.1 |

The expected `Newton < toms748 < bisection` ordering holds, but the first two are close
enough that a strict inequality could flip on individual configurations. Bisection's gap
is the headline number and is never in doubt.

### ADDENDUM F — for step 7

- **Call `solve_relaxation_parameter` with its default `bracket`, `xtol` and
  `residual_tol`.** Trap 6 says no solver builds its own residual; it does not say "do not
  retune the tolerances", and retuning them breaks the comparison just as effectively.
  `classic.py` passes `bracket` through from its own signature and never touches the other
  two.
- **Use the same end-of-span tolerance** — ADDENDUM D in ALGORITHMS.md.
- **Expect `n_relaxation_failures == 0`** on the authors' initial condition — ADDENDUM C.
- **Measure what the extrapolation buys directly, not through the global error.** For
  FSAL-R, `interpolate_fsal=False` does *not* give a reliably worse final error on this
  problem — sometimes it is slightly better, because both approximations sit below the
  method's own truncation term and the difference is lost in it. What the interpolation
  genuinely buys is one full order in the FSAL value itself: BS3 gains 2⁴ per halving
  against 2³, DP5 2⁶ against 2⁵, exactly Lemma 1. R-FSAL's `1/γ` extrapolation should be
  checked the same way, against `f(u_γ)` computed directly at fixed `dt`, rather than by
  looking at where the work-precision points land.

### ADDENDUM G — for step 8

Solvers raise on a run that cannot proceed, and **not only `RuntimeError`**: step-size
collapse and the `max_steps` bound raise `RuntimeError`, `PIDController.dt_factor` raises
`ArithmeticError`, and `initial_step_size` raises `ValueError`. The sweep's failure-row
handler has to catch broadly, or a configuration that fails in the controller will abort
the sweep instead of being recorded as the result it is.

### ADDENDUM H — for step 10

Two findings that belong in the report's "what did not work" section, which step 10 asks
for explicitly:

1. **Newton needed a stopping-rule fix to be usable at tight tolerances** (ADDENDUM C),
   while the paper's own choice, Algorithm 748, was robust untouched. That is a direct
   result for contribution 2, and it is the kind of thing a comparison study exists to
   find.
2. **Lemma 1's order gain is real but invisible in the global error** on this problem
   (ADDENDUM F). Worth reporting as a measurement that came out weaker than the theory
   suggests at first reading — the theory is about the FSAL value, not the solution.

### Housekeeping

Tests import `from src...`, so `pytest` must be run from the repository root. `README.md`
still says running is "to be documented once the implementation lands"; it now has, for
steps 1–3 and 5–6.
