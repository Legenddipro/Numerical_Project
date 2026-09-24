"""Error-based step-size control.

Holds the PID controller, the scaled error norm, and the initial step-size
heuristic. Kept separate from the solver loops so that all four method variants
share one controller: if they each had their own, any difference between their
work-precision curves could be blamed on differing step-size logic rather than
on the relaxation scheme, which is the thing actually under study.

Everything here is a port of the authors' Julia (``compute_dt_factor!``,
``accept_step``, ``compute_error_estimate``, ``ode_determine_initdt``), kept
deliberately close to the original so that gate G5 -- our numbers against
theirs, to several digits -- is a check on the solvers rather than on the
controller. Where the port departs from the source, a comment says so.

Owner: MD. Abir Hossain.
"""

from math import atan, hypot, isfinite, isnan, log10, nextafter, ulp

from .contracts import Problem, State

_EPS = 2.220446049250313e-16
"""``eps(Float64)``. Used as the floor on an error estimate, matching the
source's ``eps(typeof(error_estimate))``."""

_SMALL_DT = 1e-6
"""The source's ``smalldt``: ``oneunit * 1//10^6``, its fallback step whenever
the curvature probe cannot say anything useful."""


class PIDController:
    """Proportional-Integral-Derivative step-size controller.

    Chooses the next step size from the last *three* error estimates rather
    than only the current one, which smooths the step-size sequence and avoids
    the oscillation a plain controller shows on oscillatory problems.

    This is the piece that makes the whole study impossible in
    ``scipy.integrate.solve_ivp``, which offers no pluggable controller.

    Following the authors' implementation, with ``err`` the history of
    reciprocal error estimates and ``k`` the order used for control::

        dt_factor = err1**(beta1/k) * err2**(beta2/k) * err3**(beta3/k)

    then passed through a limiter and compared against `accept_safety`.

    The history holds *reciprocal* error estimates, so a number above 1 means
    the step came in under tolerance and the step size may grow. Storing the
    reciprocal is the source's convention and is kept because the gains are
    signed to match it: negating one without the other silently inverts the
    integral term.
    """

    def __init__(self,
                 beta: tuple[float, float, float],
                 accept_safety: float = 0.81) -> None:
        """Initialise with the gains from `tableaus.pid_gains`.

        The error history starts at 1.0 in all three slots, so the first step
        behaves as though the two preceding steps were exactly on tolerance.
        """
        self.beta = (float(beta[0]), float(beta[1]), float(beta[2]))
        self.accept_safety = float(accept_safety)
        self.err = [1.0, 1.0, 1.0]

    @staticmethod
    def limiter(x: float) -> float:
        """The source's ``default_dt_factor_limiter``: ``1 + atan(x - 1)``.

        Fixes 1 and is monotone, so it leaves an on-tolerance step alone while
        squashing extremes: the factor cannot exceed ``1 + pi/2 ~ 2.571`` however
        small the error estimate. Without it a single near-zero estimate would
        inflate the step size without bound and the next step would fail
        catastrophically, which for the relaxation variants can mean a
        relaxation failure rather than merely a rejected step.
        """
        return 1.0 + atan(x - 1.0)

    def dt_factor(self, error_estimate: float, order: int) -> float:
        """Factor by which the step size should be multiplied.

        Records `error_estimate` as the newest entry in the history, but does
        *not* yet shift the history -- that happens in `on_accept`, because a
        step may still be rejected afterwards by the relaxation stage even
        though the error test passed.

        `error_estimate` is clamped away from zero before use; an exactly-zero
        estimate would otherwise produce an infinite factor.

        Returns
        -------
        float
            Limited multiplicative factor, via ``1 + atan(x - 1)``.

        Raises
        ------
        ArithmeticError
            If the factor comes out ``nan``, which the source also treats as
            fatal. It means the error history contains something impossible
            (a negative or non-finite estimate), and continuing would produce a
            step-size sequence nobody could reconstruct afterwards.
        """
        if error_estimate < 0.0 or isnan(error_estimate):
            raise ArithmeticError(
                f"error estimate must be a non-negative number, got "
                f"{error_estimate!r}")

        # Clamp to eps rather than special-casing zero: an exactly-zero estimate
        # and a 1e-300 one should behave the same, and both are then capped by
        # the limiter anyway.
        error_estimate = max(error_estimate, _EPS)

        self.err[0] = 1.0 / error_estimate
        err1, err2, err3 = self.err
        beta1, beta2, beta3 = self.beta
        k = order

        factor = (err1 ** (beta1 / k)
                  * err2 ** (beta2 / k)
                  * err3 ** (beta3 / k))

        if not isfinite(factor):
            raise ArithmeticError(
                f"dt factor is not finite: factor={factor!r}, "
                f"err={tuple(self.err)!r}, beta={self.beta!r}, k={k}")

        return self.limiter(factor)

    def accept(self, dt_factor: float) -> bool:
        """Whether a step with this factor passes the error test.

        True when ``dt_factor >= accept_safety``.

        The test is on the *factor*, not on the error estimate, which is why
        `accept_safety` is 0.81 rather than 1: a step whose error sits slightly
        above tolerance still earns a factor near 1, and rejecting it would cost
        a full attempt to buy a negligible accuracy gain.
        """
        return dt_factor >= self.accept_safety

    def on_accept(self) -> None:
        """Shift the error history after a step is finally accepted.

        Separate from `dt_factor` because the relaxation variants decide
        acceptance in two stages: the error test first, then whether a valid
        relaxation parameter exists. Only when both pass does the history move.
        """
        self.err[2] = self.err[1]
        self.err[1] = self.err[0]

    def on_reject(self) -> None:
        """Called after a rejected step. The history does not shift.

        A no-op, as in the source. It exists so the solver loops read
        symmetrically and so that the accept/reject bookkeeping is impossible to
        omit by accident -- a missing `on_accept` is a bug, and a loop that
        calls neither is visibly unfinished.
        """
        return None


def compute_error_estimate(u: State,
                           uprev: State,
                           uembd: State,
                           abstol: float,
                           reltol: float) -> float:
    """Scaled error norm comparing the main and embedded solutions.

    Each component is scaled by ``abstol + reltol * max(|u_i|, |uprev_i|)``
    before the root-mean-square is taken, so absolute and relative tolerance
    both apply, and components of very different magnitude are weighted fairly.

    Components whose scale comes out non-positive are skipped, as in the source;
    that can only happen when both tolerances are zero, and dividing by it would
    turn a meaningful comparison into ``inf``. If *every* component is skipped
    the result is 0.0, which reads as "meets tolerance" -- the only consistent
    answer when no tolerance was actually requested.

    Returns
    -------
    float
        A value <= 1 means the step meets the requested tolerance.
    """
    total = 0.0
    counted = 0
    for ui, uprevi, uembdi in zip(u, uprev, uembd):
        scale = abstol + reltol * max(abs(ui), abs(uprevi))
        if scale > 0.0:
            total += (abs(ui - uembdi) / scale) ** 2
            counted += 1
    if counted == 0:
        return 0.0
    return (total / counted) ** 0.5


def initial_step_size(problem: Problem,
                      abstol: float,
                      reltol: float,
                      order: int,
                      dt: float = 0.0) -> tuple[float, State, int]:
    """Heuristic first step size, before any error history exists.

    Ports the authors' ``ode_determine_initdt``: it takes a tiny explicit Euler
    probe step, compares the right-hand side before and after to estimate the
    solution's second derivative, and picks a step predicted to land near
    tolerance. One slope says which way the solution is heading; the difference
    between two says how fast that direction is turning, and Runge-Kutta error
    depends on the latter -- hence two evaluations rather than one.

    Passing a nonzero `dt` skips the heuristic and uses that step size, which is
    the fixed-step mode gate G2 needs. Only one evaluation is then required.

    Also returns ``f(t0, u0)``, which the probe computed anyway and which every
    solver needs as its first FSAL cache entry. Returning it avoids a redundant
    evaluation that gate G3 would otherwise see.

    The third return value is the number of right-hand-side evaluations this
    call made: 2 when the heuristic runs, 1 when `dt` was supplied. Solvers
    initialise their own counter from it::

        dt, fsal_cache, nf = initial_step_size(problem, abstol, reltol, order)

    so the startup cost `c0` is reported rather than remembered. A hardcoded
    constant would silently go stale the moment this heuristic changed, and the
    resulting gate G3 failure would look like a bug in the solver rather than
    in the bookkeeping.

    Integration is forward in time only (the source's ``tdir = true``), which is
    all this study needs.

    Returns
    -------
    tuple[float, State, int]
        ``(dt0, f(t0, u0), n_rhs_calls)``.

    Raises
    ------
    ValueError
        If `dt` is negative, or if both tolerances are zero (which leaves the
        scaling vector `sk` with a zero entry and no way to measure anything).
    """
    if dt < 0.0:
        raise ValueError(f"dt must be non-negative for forward integration, "
                         f"got {dt!r}")

    t0, t_end = problem.t_span
    u0 = problem.u0
    f = problem.rhs

    if abstol <= 0.0 and reltol <= 0.0:
        raise ValueError("at least one of abstol and reltol must be positive")

    # Scaling vector: the same abstol/reltol mix `compute_error_estimate` uses,
    # evaluated at u0 alone since no second point exists yet.
    sk = tuple(abstol + abs(x) * reltol for x in u0)
    if any(s <= 0.0 for s in sk):
        raise ValueError(
            f"scaling vector has a non-positive entry {sk!r}; with abstol=0 a "
            f"zero state component cannot be scaled")

    f0 = f(t0, u0)

    if dt > 0.0:
        # Fixed-step mode: the heuristic is skipped, but f(t0, u0) is still
        # needed as the first FSAL cache entry, so exactly one call was made.
        return (float(dt), f0, 1)

    dtmax = t_end - t0
    # The source's ``nextfloat(prob2dtmin(prob))``: a floor of roughly one ulp of
    # the time span. It exists to keep a degenerate probe from returning a step
    # of literally zero; on any problem in this study it never binds.
    dtmin = nextafter(max(ulp(t0), ulp(t_end)), float("inf"))

    d0 = hypot(*(x / s for x, s in zip(u0, sk)))
    d1 = hypot(*(x / s for x, s in zip(f0, sk)))

    if d0 < 1e-5 or d1 < 1e-5:
        # Either the state or its slope is negligible against the tolerance, so
        # the ratio below would say nothing. Start small and let the controller
        # grow the step.
        dt0 = _SMALL_DT
    else:
        dt0 = (d0 / d1) / 100.0
    dt0 = min(dt0, dtmax)

    if dt0 < 10.0 * _EPS:
        # Degenerate: the span itself is near machine precision. One call made.
        #
        # DEVIATION: the source returns a bare scalar here instead of the
        # (dt, f0) tuple it promises everywhere else, so this branch would fail
        # at the call site if it were ever reached. Ours returns the full triple.
        return (_SMALL_DT, f0, 1)

    # Explicit Euler probe, then the second slope. This is the only other RHS
    # call, and the reason the reported startup count is 2.
    u1 = tuple(x + dt0 * fx for x, fx in zip(u0, f0))
    f1 = f(t0 + dt0, u1)

    if tuple(f1) == tuple(f0):
        # No curvature detected at all -- a linear-in-time solution, or a probe
        # too small to register. Nothing here bounds the step, so take the
        # largest value this heuristic ever returns.
        return (max(dtmin, 100.0 * dt0), f0, 2)

    d2 = hypot(*((b - a) / s for a, b, s in zip(f0, f1, sk))) / dt0

    max_d1d2 = max(d1, d2)
    if max_d1d2 <= 1e-15:
        dt1 = max(_SMALL_DT, dt0 * 1e-3)
    else:
        # Invert the local error model: a step of this size is predicted to land
        # near tolerance for a method of this order.
        dt1 = 10.0 ** (-(2.0 + log10(max_d1d2)) / order)

    dt0_final = max(dtmin, min(100.0 * dt0, dt1, dtmax))
    return (dt0_final, f0, 2)
