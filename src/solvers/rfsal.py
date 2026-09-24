"""R-FSAL time-stepping loop.

Structurally distinct from the three variants in `classic.py`, which is why it
has its own module and its own owner rather than being a fourth branch in a
shared loop.

Transcribed from section 4 of `docs/ALGORITHMS.md` and its addenda A-D. Where
the loop departs from the authors' `solve_rfsal!`, a comment says so at the line
where it applies.

Owner: Asikur Rahman (step 7 of docs/PLAN_SEQUENTIAL.md; the step-0 scaffold
named Tousif, but the plan's assignment table gives steps 7-8 to Asikur).
"""

from math import hypot, isclose

from ..contracts import Problem, SolverResult, State, Tableau
from ..rootfind import DEFAULT_BRACKET, RootFinder, solve_relaxation_parameter
from ..stepping import PIDController, compute_error_estimate, initial_step_size
from ..tableaus import pid_gains
# ADDENDUM D: R-FSAL must use *the same* end-of-span tolerance and step-size
# floor as the classic three, or the four methods land at different final times
# and their global errors stop being comparable. Imported rather than copied so
# the two cannot drift apart; if the lead moves them to `contracts.py` or
# `stepping.py`, only this import changes.
from .classic import _DT_MIN, _TEND_RTOL


def extrapolate_last_stage(f_prev: State, f_gamma: State, gamma: float) -> State:
    """R-FSAL's reconstruction of ``f(u_np1)`` from two exact evaluations.

    ::

        f(u_np1) ~= f(u_prev) + (1/gamma) * (f(u_gamma) - f(u_prev))

    The factor is ``1/gamma``, not ``gamma``: ``u_gamma`` sits a fraction
    ``gamma`` of the way from ``u_prev`` to ``u_np1``, so reaching ``u_np1``
    means *stretching past* ``u_gamma`` by ``1/gamma``. FSAL-R's interpolation
    goes the other way (from ``k_last`` back to ``u_gamma``) and uses ``gamma``.

    Exact for a linear right-hand side, because a linear map commutes with the
    affine combination performed here; ``tests/test_rfsal.py`` checks that to
    round-off. Public so the golden-step value in ALGORITHMS.md can be checked
    against this exact function rather than a re-typed copy of it.
    """
    inv_gamma = 1.0 / gamma
    return tuple(a + inv_gamma * (z - a) for a, z in zip(f_prev, f_gamma))



def solve_rfsal(problem: Problem,
                tableau: Tableau,
                tol: float,
                *,
                rootfinder: RootFinder,
                bracket: tuple[float, float] = DEFAULT_BRACKET,
                adaptive: bool = True,
                dt0: float = 0.0,
                save_history: bool = False,
                interpolate_fsal: bool = True,
                relax_embedded: bool = True,
                relax_main: bool = True,
                max_steps: int = 10_000_000) -> SolverResult:
    """R-FSAL: relax before the error estimate, then extrapolate.

    The step is reordered rather than patched at the end:

    1. Compute stages ``1 .. s-1`` only. Stage ``s`` is skipped, which is
       legitimate because ``b[-1] == 0`` for any explicit FSAL method, so that
       stage never contributed to the propagated solution anyway.
    2. Assemble ``u_np1`` from those stages and solve for gamma immediately,
       before any error test.
    3. Make the step's single right-hand-side evaluation at the *relaxed*
       point, giving ``f(u_gamma)``.
    4. Use that one value twice: exactly as the next step's cached first stage,
       and -- via extrapolation with factor ``1/gamma`` -- as the approximation
       of ``f(u_np1)`` that the embedded solution needs::

           f(u_np1) ~= f(u_gamma_prev) + (1/gamma) * (f(u_gamma) - f(u_gamma_prev))

    5. Build the embedded solution from relaxed quantities and apply error
       control.

    So the approximation sits in a different place than in FSAL-R. Here the
    FSAL reuse is *exact* and the error estimate is approximated; in FSAL-R it
    is the other way round. Theory (Lemma 2) bounds the extrapolation error at
    ``O(dt**(p+1))``, preserving the embedded method's order.

    The three boolean switches reproduce the authors' variant study:
    `interpolate_fsal` selects extrapolation versus using ``f(u_gamma)``
    unmodified, `relax_embedded` selects whether the embedded solution uses
    ``gamma*dt`` or plain ``dt``, and `relax_main` selects whether the error
    estimate compares against the relaxed or unrelaxed main solution. The
    defaults are the combination the paper recommends; the others are
    available so the comparison can be reproduced rather than taken on trust.

    Returns
    -------
    SolverResult
        Gate G3: ``nf == c0 + (s-1)*(n_accept + n_reject) - n_relaxation_failures``
        (DEVIATION 3). With no relaxation failures that is the baseline
        identity exactly -- s-2 stage calls plus one at the relaxed point.

    Raises
    ------
    RuntimeError
        On step-size collapse or exceeding `max_steps`, as in `classic.py`.
    ValueError
        If the tableau's last propagated weight is nonzero, which would make
        skipping stage s wrong.
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
    method = "rfsal"

    if b[-1] != 0.0:
        # Skipping stage s is only legitimate because it never enters the
        # propagated solution. A tableau without that property would make this
        # loop silently wrong rather than merely slow.
        raise ValueError(f"R-FSAL needs b[-1] == 0; {tableau.name} has "
                         f"b[-1] = {b[-1]!r}")

    # c0 is reported by the heuristic, exactly as in classic.py.
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

        # --- stages 1 .. s-1 ----------------------------------------------
        # Stage 1 is the cache -- here an *exact* f(u_prev), because the
        # previous step evaluated f at the relaxed point it moved to. Stage s is
        # never computed: s-2 calls, not s-1.
        k[0] = fsal_cache
        for i in range(1, s - 1):
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

        # Main solution from stages 1 .. s-1 only. Nothing is missing: b[s] = 0.
        main = list(u_prev)
        for i in range(s - 1):
            b_i = b[i]
            if b_i != 0.0:
                weight = b_i * dt
                k_i = k[i]
                for m in range(n_dim):
                    main[m] += weight * k_i[m]
        u_new: State = tuple(main)

        # --- relaxation, BEFORE any error test ------------------------------
        # Trap 6 / ADDENDUM F: gamma comes only from the shared entry point, with
        # its default xtol and residual_tol, and the bracket passed straight
        # through -- configured identically to classic.py.
        root = solve_relaxation_parameter(entropy, entropy_gradient,
                                          u_prev, u_new, rootfinder, bracket)
        n_root_iterations += root.iterations

        if not root.converged:
            # DEVIATION 2 (implemented, not dead): reject and halve dt, never
            # proceed with gamma = 1. The step returns before its f(u_gamma)
            # call, so this attempt cost s-2 rather than s-1 -- DEVIATION 3's
            # `- n_relax_fail` term in the G3 identity.
            controller.on_reject()
            n_reject += 1
            n_relaxation_failures += 1
            dt *= 0.5
            if dt < _DT_MIN:
                raise RuntimeError(
                    f"{method}/{tableau.name}: step size collapsed to {dt!r} "
                    f"at t={t!r} (accepted {n_accept}, rejected {n_reject}, "
                    f"{n_relaxation_failures} relaxation failures)")
            continue

        gamma = root.gamma
        u_gamma: State = tuple(p + gamma * (q - p) for p, q in zip(u_prev, u_new))
        t_step = t + dt
        # ADDENDUM B/D: the relaxation_at_last_step = true behaviour, with the
        # tolerance shared with classic.py. The last step's state is relaxed,
        # but its time lands on t_end exactly.
        t_new = t_step if isclose(t_step, t_end, rel_tol=_TEND_RTOL,
                                  abs_tol=0.0) else t + gamma * dt

        # --- the step's one relaxed RHS call ---------------------------------
        f_gamma = f(t_new, u_gamma)
        nf += 1

        # Reconstruct the stage never computed, for the embedded solution only.
        if interpolate_fsal:
            k[-1] = extrapolate_last_stage(fsal_cache, f_gamma, gamma)
        else:
            k[-1] = f_gamma

        # Embedded solution from relaxed quantities.
        scale = gamma * dt if relax_embedded else dt
        embedded = list(u_prev)
        for i in range(s):
            be_i = b_embedded[i]
            if be_i != 0.0:
                weight = be_i * scale
                k_i = k[i]
                for m in range(n_dim):
                    embedded[m] += weight * k_i[m]
        u_embedded: State = tuple(embedded)

        # --- error test --------------------------------------------------------
        if adaptive:
            target = u_gamma if relax_main else u_new
            error_estimate = compute_error_estimate(target, u_prev, u_embedded,
                                                    tol, tol)
            factor = controller.dt_factor(error_estimate, order)
            accepted = controller.accept(factor)
        else:
            factor = 1.0
            accepted = True

        if accepted:
            controller.on_accept()
            n_accept += 1
            t = t_new
            u_prev = u_gamma
            # EXACT, not approximated: the trajectory really does continue from
            # u_gamma, and f_gamma is f evaluated there.
            fsal_cache = f_gamma
            if save_history:
                t_history.append(t)
                entropy_history.append(entropy(u_prev))
                gamma_history.append(gamma)
        else:
            controller.on_reject()
            n_reject += 1

        if adaptive:
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
