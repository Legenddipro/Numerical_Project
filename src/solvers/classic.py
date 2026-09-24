"""Baseline, naive relaxation, and FSAL-R time-stepping loops.

These three share the great majority of their structure -- FSAL-R differs from
naive by only a handful of lines at the end of the step -- so they live in one
module under one owner. Splitting them across files would duplicate the shared
loop and invite the two copies to drift apart.

For the same reason they share one implementation here, `_integrate`, rather
than three near-identical loops: the authors' Julia keeps `solve_naive!` and
`solve_fsalr!` as separate 150-line functions differing in a single line, and
that single line is precisely the thing under study. Holding everything else
literally identical is what makes a difference between their results
attributable to the scheme.

All four solver entry points in the project take the same arguments and return
`SolverResult`, so `montecarlo.py` can iterate over them uniformly.

Transcribed from `docs/ALGORITHMS.md`, including its three marked deviations
from the authors' source. Each is commented at the line where it applies.

Owner: MD. Shadman Shafie.
"""

from math import hypot, isclose

from ..contracts import Problem, SolverResult, State, Tableau
from ..rootfind import DEFAULT_BRACKET, RootFinder, solve_relaxation_parameter
from ..stepping import PIDController, compute_error_estimate, initial_step_size
from ..tableaus import pid_gains

_DT_MIN = 1e-14
"""The source's step-size floor. Reaching it means the controller is trapped --
usually a relaxation failure that halving cannot fix -- and continuing would
burn the step budget rather than produce an answer."""

_TEND_RTOL = 1.4901161193847656e-8
"""``sqrt(eps(Float64))``, Julia's default relative tolerance for ``≈``.

Used for the one comparison the source makes with ``≈`` rather than ``==``:
whether this step lands on the final time. On the last step the state is still
relaxed but the time is set to `t_end` exactly, so the solution is reported at
the instant the reference is evaluated at."""


def _integrate(problem: Problem,
               tableau: Tableau,
               tol: float,
               *,
               method: str,
               rootfinder: RootFinder | None,
               bracket: tuple[float, float],
               adaptive: bool,
               dt0: float,
               save_history: bool,
               interpolate_fsal: bool,
               max_steps: int) -> SolverResult:
    """The loop all three variants share. See the public wrappers below.

    `method` is one of ``"baseline"``, ``"naive"`` or ``"fsalr"`` and selects
    the three places the variants differ: whether relaxation runs at all, and
    how the FSAL cache is refreshed once it has.
    """
    f = problem.rhs
    entropy = problem.entropy
    entropy_gradient = problem.entropy_gradient
    t_start, t_end = problem.t_span

    A = tableau.A
    b = tableau.b
    b_embedded = tableau.b_embedded
    c = tableau.c
    s = tableau.n_stages
    order = tableau.order

    relaxing = method != "baseline"

    # c0 comes back from the heuristic rather than being remembered here: 2 when
    # it ran, 1 when dt0 was supplied. Gate G3's identity is written in terms of
    # this number, so hardcoding it would turn a stale constant into what looks
    # like a solver bug.
    dt, fsal_cache, nf = initial_step_size(problem, tol, tol, order, dt0)
    controller = PIDController(pid_gains(tableau))

    t = t_start
    u_prev: State = tuple(float(x) for x in problem.u0)
    n_dim = len(u_prev)

    n_accept = 0
    n_reject = 0
    n_root_iterations = 0
    n_relaxation_failures = 0

    t_history: list[float] = []
    entropy_history: list[float] = []
    gamma_history: list[float] = []
    if save_history:
        t_history.append(t)
        entropy_history.append(entropy(u_prev))

    k: list[State] = [fsal_cache] * s
    attempts = 0

    while t < t_end:
        attempts += 1
        if attempts > max_steps:
            raise RuntimeError(
                f"{method}/{tableau.name}: exceeded max_steps={max_steps} at "
                f"t={t!r} with dt={dt!r} (accepted {n_accept}, rejected "
                f"{n_reject})")

        if t + dt > t_end:
            dt = t_end - t

        # --- stages -------------------------------------------------------
        # Stage 1 is served from the cache and never costs a call. That is the
        # FSAL saving, and the whole RHS accounting rests on it.
        k[0] = fsal_cache
        for i in range(1, s):
            row = A[i]
            y = list(u_prev)
            for j in range(i):
                a_ij = row[j]
                if a_ij == 0.0:
                    continue
                weight = a_ij * dt
                k_j = k[j]
                for m in range(n_dim):
                    y[m] += weight * k_j[m]
            k[i] = f(t + c[i] * dt, tuple(y))
            nf += 1

        main = list(u_prev)
        embedded = list(u_prev)
        for i in range(s):
            k_i = k[i]
            b_i = b[i]
            if b_i != 0.0:
                weight = b_i * dt
                for m in range(n_dim):
                    main[m] += weight * k_i[m]
            be_i = b_embedded[i]
            if be_i != 0.0:
                weight = be_i * dt
                for m in range(n_dim):
                    embedded[m] += weight * k_i[m]
        u_new: State = tuple(main)
        u_embedded: State = tuple(embedded)

        # --- error test ---------------------------------------------------
        if adaptive:
            error_estimate = compute_error_estimate(u_new, u_prev, u_embedded,
                                                    tol, tol)
            factor = controller.dt_factor(error_estimate, order)
            accepted = controller.accept(factor)
        else:
            factor = 1.0
            accepted = True

        relaxation_failed = False

        if not accepted:
            controller.on_reject()
            n_reject += 1

        elif not relaxing:
            # DEVIATION 1: the source neither calls `accept_step!` nor
            # increments `naccept` on this path, so its baseline reports zero
            # accepted steps and runs on a PID history that never shifts. Both
            # are done here, which is what makes gate G3 usable for baseline and
            # makes its step sequence comparable with the relaxation variants'.
            controller.on_accept()
            n_accept += 1
            t += dt
            u_prev = u_new
            fsal_cache = k[-1]          # k[-1] IS f(u_new). Free.

        else:
            # Trap 6: the residual is never assembled here. Every variant in the
            # project solves for gamma through this one entry point, configured
            # identically, so a difference between their results can only come
            # from the scheme.
            root = solve_relaxation_parameter(entropy, entropy_gradient,
                                              u_prev, u_new, rootfinder,
                                              bracket)
            n_root_iterations += root.iterations

            if root.converged:
                gamma = root.gamma
                # Only now is the step truly accepted: it passed the error test
                # above and relaxation below, and the history moves once.
                controller.on_accept()
                n_accept += 1

                u_gamma: State = tuple(
                    p + gamma * (n - p) for p, n in zip(u_prev, u_new))
                t_new = t + dt
                # Relaxation moves time as well as state, except on the final
                # step, which lands on t_end exactly so the solution can be
                # compared against the reference at the instant it claims.
                t = t_new if isclose(t_new, t_end, rel_tol=_TEND_RTOL,
                                     abs_tol=0.0) else t + gamma * dt
                u_prev = u_gamma

                if method == "naive":
                    # THE EXTRA RHS CALL. The cached k[-1] is f(u_new), and
                    # integration continues from u_gamma -- a different point.
                    # This one call per accepted step is the cost the two new
                    # schemes remove.
                    fsal_cache = f(t, u_prev)
                    nf += 1
                elif interpolate_fsal:
                    # FSAL-R: interpolate between two stage values already in
                    # hand. Arithmetic, not an evaluation.
                    k_first = k[0]
                    k_last = k[-1]
                    fsal_cache = tuple(
                        a + gamma * (z - a) for a, z in zip(k_first, k_last))
                else:
                    # The authors' "simple" variant: reuse k[-1] unmodified.
                    # Same cost, visibly worse accuracy -- which is what the
                    # interpolation buys.
                    fsal_cache = k[-1]
            else:
                # DEVIATION 2: in the source this branch is unreachable, because
                # the gamma check above it resets the failure flag
                # unconditionally -- which is how the typo `integrator *= 0.5`
                # survives inside it. Implemented properly here: a failed
                # relaxation rejects the step and halves dt. Never proceed with
                # gamma = 1.
                relaxation_failed = True
                controller.on_reject()
                n_reject += 1
                n_relaxation_failures += 1
                dt *= 0.5

        if accepted and not relaxation_failed and save_history:
            t_history.append(t)
            entropy_history.append(entropy(u_prev))
            if relaxing:
                gamma_history.append(gamma)

        if adaptive and not relaxation_failed:
            # Skipped after a relaxation failure: dt was already halved, and
            # scaling it again by a factor from a step that never completed
            # would mix two unrelated decisions.
            dt *= factor

        if dt < _DT_MIN:
            raise RuntimeError(
                f"{method}/{tableau.name}: step size collapsed to {dt!r} at "
                f"t={t!r} (accepted {n_accept}, rejected {n_reject}, "
                f"{n_relaxation_failures} relaxation failures)")

    reference = problem.reference(t)
    error_final = hypot(*(a - r for a, r in zip(u_prev, reference)))

    return SolverResult(
        method=method,
        tableau_name=tableau.name,
        tolerance=tol,
        t_final=t,
        u_final=u_prev,
        error_final=error_final,
        nf=nf,
        n_accept=n_accept,
        n_reject=n_reject,
        n_root_iterations=n_root_iterations,
        n_relaxation_failures=n_relaxation_failures,
        t_history=tuple(t_history),
        entropy_history=tuple(entropy_history),
        gamma_history=tuple(gamma_history),
    )


def solve_baseline(problem: Problem,
                   tableau: Tableau,
                   tol: float,
                   *,
                   rootfinder: RootFinder | None = None,
                   bracket: tuple[float, float] = DEFAULT_BRACKET,
                   adaptive: bool = True,
                   dt0: float = 0.0,
                   save_history: bool = False,
                   max_steps: int = 10_000_000) -> SolverResult:
    """Plain embedded Runge-Kutta with FSAL reuse. No relaxation.

    The reference point for everything else. The invariant drifts here, which
    is the problem relaxation exists to solve, and this method's RHS count is
    the cost the two new schemes must match.

    `rootfinder` is accepted and ignored, so that the signature matches the
    relaxation variants and sweeps need no special case.

    With ``adaptive=False`` and an explicit `dt0`, runs at fixed step size --
    the mode gate G2 uses to check convergence order.

    Returns
    -------
    SolverResult
        With ``n_root_iterations == 0`` and ``n_relaxation_failures == 0``.
    """
    return _integrate(problem, tableau, tol,
                      method="baseline",
                      rootfinder=None,
                      bracket=bracket,
                      adaptive=adaptive,
                      dt0=dt0,
                      save_history=save_history,
                      interpolate_fsal=False,
                      max_steps=max_steps)


def solve_naive(problem: Problem,
                tableau: Tableau,
                tol: float,
                *,
                rootfinder: RootFinder,
                bracket: tuple[float, float] = DEFAULT_BRACKET,
                adaptive: bool = True,
                dt0: float = 0.0,
                save_history: bool = False,
                max_steps: int = 10_000_000) -> SolverResult:
    """Relaxation applied after the step is accepted, the direct way.

    Runs the ordinary step, applies error control, and only then solves for
    gamma and moves to the relaxed point. Because integration continues from
    ``u_gamma`` rather than ``u_np1``, the cached FSAL value is for the wrong
    point and must be recomputed with a fresh right-hand-side evaluation.

    That one extra evaluation per step is exactly what cancels the FSAL saving,
    and demonstrating it is the reason this variant is implemented at all: it
    is the "obvious" approach whose cost the two new schemes improve on.

    Returns
    -------
    SolverResult
        `nf` should exceed `solve_baseline` by the ratio ``s/(s-1)`` -- about
        4/3 for BS3 and 7/6 for DP5. Gate G3 checks this.
    """
    return _integrate(problem, tableau, tol,
                      method="naive",
                      rootfinder=rootfinder,
                      bracket=bracket,
                      adaptive=adaptive,
                      dt0=dt0,
                      save_history=save_history,
                      interpolate_fsal=False,
                      max_steps=max_steps)


def solve_fsalr(problem: Problem,
                tableau: Tableau,
                tol: float,
                *,
                rootfinder: RootFinder,
                bracket: tuple[float, float] = DEFAULT_BRACKET,
                adaptive: bool = True,
                dt0: float = 0.0,
                save_history: bool = False,
                interpolate_fsal: bool = True,
                max_steps: int = 10_000_000) -> SolverResult:
    """FSAL-R: relax after step-size control, then interpolate the FSAL value.

    Identical to `solve_naive` up to the point where the relaxed solution is
    formed. Instead of evaluating the right-hand side afresh at ``u_gamma``,
    it interpolates between two stage values already in hand::

        f(u_gamma) ~= k_first + gamma * (k_last - k_first)

    which costs arithmetic rather than a function evaluation. Theory (Lemma 1
    of the paper) puts the interpolation error at ``O(dt**(p+1))``, small
    enough that the method's order is unchanged.

    `interpolate_fsal=False` disables the interpolation and reuses ``k_last``
    unmodified -- the authors' "simple" variant, kept as a comparison point
    showing what the interpolation actually buys.

    Returns
    -------
    SolverResult
        `nf` must equal `solve_baseline` exactly. Gate G3.
    """
    return _integrate(problem, tableau, tol,
                      method="fsalr",
                      rootfinder=rootfinder,
                      bracket=bracket,
                      adaptive=adaptive,
                      dt0=dt0,
                      save_history=save_history,
                      interpolate_fsal=interpolate_fsal,
                      max_steps=max_steps)
