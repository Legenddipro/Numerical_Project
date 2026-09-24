"""Root-finding for the relaxation parameter gamma.

Every relaxation step solves one scalar equation

    r(gamma) = eta(u_n + gamma*(u_np1 - u_n)) - eta(u_n) = 0

Comparing how different methods solve it is one of this project's three
contributions, so the finders share a single signature and are interchangeable,
and each reports its own iteration count.

Three methods, chosen to span the design space:

    newton      fast, needs the derivative, started from gamma_0 = 1
    bisection   slow, robust, fails loudly when the bracket has no sign change
    toms748     Algorithm 748 of Alefeld-Potra-Shi via scipy.optimize; the
                identical algorithm the base paper uses, included so the
                comparison has the paper's own choice as its baseline

The trivial root
----------------
``gamma = 0`` always satisfies the equation, since it means "do not move at
all". Bracketing methods must exclude it, which is why the default bracket is
centred on 1 rather than starting at 0.

Reporting failure
-----------------
A finder that cannot locate a root returns ``converged=False``. The two
bracketing methods then return ``gamma = nan``, because with no sign change
there is no estimate to report and a plausible-looking number invites a caller
to use it anyway. Newton returns its last iterate instead, which is a genuine
diagnostic -- where it escaped to says whether the residual went flat or the
root simply lies outside the bracket -- but still with ``converged=False``.
Either way the caller must reject the step; never proceed with gamma = 1.

Owner: MD. Shadman Shafie.
"""

from math import inf, isfinite, nan
from typing import Callable

from scipy.optimize import toms748 as _scipy_toms748

from .contracts import EntropyFunction, EntropyGradient, RootResult, State

ScalarFunction = Callable[[float], float]

# Every finder takes the same arguments so solvers can swap them freely.
# Methods ignore what they do not need: bracketing methods ignore `dr`, and
# Newton uses the bracket only to reject a result that escaped it.
RootFinder = Callable[..., RootResult]

DEFAULT_BRACKET: tuple[float, float] = (0.8, 1.2)
"""Matches the authors' choice. Centred on 1 because theory gives
gamma = 1 + O(dt**(p-1)), and deliberately excludes the trivial root at 0."""

DEFAULT_XTOL: float = 1e-14

_FAILED = RootResult(gamma=nan, iterations=0, converged=False, residual=inf)
"""What a bracketing method returns when the bracket holds no sign change.

``nan`` rather than a midpoint or 1.0: there is no root in hand, and any finite
value would be a fabricated one that a caller could use by accident."""


def make_residual(entropy: EntropyFunction,
                  u_old: State,
                  u_new: State) -> ScalarFunction:
    """Build ``r(gamma)`` for one step.

    Closes over the two endpoints so the returned function is cheap to call
    repeatedly: each evaluation builds the candidate point
    ``u_old + gamma*(u_new - u_old)`` and evaluates the invariant there. No
    right-hand-side evaluation is involved, which is why relaxation costs
    almost nothing despite being iterative.

    Returns
    -------
    ScalarFunction
        ``gamma -> eta(candidate) - eta(u_old)``.
    """
    base = tuple(u_old)
    direction = tuple(b - a for a, b in zip(base, u_new))
    # Evaluated once rather than per call: it does not depend on gamma, and
    # bisection asks for forty-odd evaluations of this residual per step.
    eta_old = entropy(base)

    def residual(gamma: float) -> float:
        point = tuple(a + gamma * d for a, d in zip(base, direction))
        return entropy(point) - eta_old

    return residual


def make_residual_derivative(entropy_gradient: EntropyGradient,
                             u_old: State,
                             u_new: State) -> ScalarFunction:
    """Build ``dr/dgamma`` for one step, for Newton's method.

    By the chain rule, with ``d = u_new - u_old``::

        dr/dgamma = grad eta(u_old + gamma*d) . d

    Returns
    -------
    ScalarFunction
    """
    base = tuple(u_old)
    direction = tuple(b - a for a, b in zip(base, u_new))

    def derivative(gamma: float) -> float:
        point = tuple(a + gamma * d for a, d in zip(base, direction))
        gradient = entropy_gradient(point)
        return sum(g * d for g, d in zip(gradient, direction))

    return derivative


def newton(r: ScalarFunction,
           dr: ScalarFunction,
           bracket: tuple[float, float] = DEFAULT_BRACKET,
           *,
           xtol: float = DEFAULT_XTOL,
           max_iter: int = 50) -> RootResult:
    """Newton-Raphson from gamma_0 = 1.

    Theory puts the root at ``1 + O(dt**(p-1))``, so starting at exactly 1 is
    an unusually good initial guess and convergence is typically 3-4
    iterations. Converging to the trivial root at 0 is therefore not a
    practical concern, but a result landing outside `bracket` is reported as
    not converged rather than returned.

    Unlike the two bracketing methods, Newton offers no convergence guarantee:
    a near-zero derivative can throw an iterate far from the bracket. Such a
    result must come back with ``converged=False``, never as a silent
    fallback to gamma = 1.

    Returns
    -------
    RootResult
    """
    lo, hi = bracket
    gamma = 1.0
    iterations = 0
    converged = False

    while iterations < max_iter:
        r_value = r(gamma)
        dr_value = dr(gamma)
        iterations += 1

        if not isfinite(r_value) or not isfinite(dr_value) or dr_value == 0.0:
            # A flat residual gives no direction to move in. Stopping here and
            # reporting the last iterate is honest; dividing by it anyway would
            # put gamma at infinity and lose which step actually went wrong.
            break

        delta = r_value / dr_value
        candidate = gamma - delta
        if not isfinite(candidate):
            break
        gamma = candidate

        if abs(delta) <= xtol:
            converged = True
            break

    if converged and not lo <= gamma <= hi:
        # Converged to something, but not to a usable relaxation parameter. The
        # caller must reject the step; reporting success here would let a root
        # from another branch -- including the trivial one at 0 -- into the
        # solution.
        converged = False

    residual = abs(r(gamma)) if isfinite(gamma) else inf
    return RootResult(gamma=gamma, iterations=iterations,
                      converged=converged, residual=residual)


def bisection(r: ScalarFunction,
              dr: ScalarFunction,
              bracket: tuple[float, float] = DEFAULT_BRACKET,
              *,
              xtol: float = DEFAULT_XTOL,
              max_iter: int = 100) -> RootResult:
    """Classical bisection. `dr` is accepted and ignored.

    Guaranteed to converge given a sign change, but slowly: halving a bracket
    of width 0.4 down to 1e-14 takes about 45 iterations, against 3-4 for
    Newton. That gap is a headline number for the root-finder comparison.

    Returns `converged=False` immediately if the endpoints do not straddle a
    root, rather than returning a fabricated value.

    Returns
    -------
    RootResult
    """
    lo, hi = bracket
    r_lo = r(lo)
    r_hi = r(hi)

    if r_lo == 0.0:
        return RootResult(gamma=lo, iterations=0, converged=True, residual=0.0)
    if r_hi == 0.0:
        return RootResult(gamma=hi, iterations=0, converged=True, residual=0.0)
    if not isfinite(r_lo) or not isfinite(r_hi) or r_lo * r_hi > 0.0:
        return _FAILED

    iterations = 0
    while hi - lo > xtol and iterations < max_iter:
        mid = 0.5 * (lo + hi)
        r_mid = r(mid)
        iterations += 1
        if r_mid == 0.0:
            lo = hi = mid
            break
        if r_lo * r_mid < 0.0:
            hi = mid
            r_hi = r_mid
        else:
            lo = mid
            r_lo = r_mid

    gamma = 0.5 * (lo + hi)
    return RootResult(gamma=gamma, iterations=iterations,
                      converged=hi - lo <= xtol, residual=abs(r(gamma)))


def toms748(r: ScalarFunction,
            dr: ScalarFunction,
            bracket: tuple[float, float] = DEFAULT_BRACKET,
            *,
            xtol: float = DEFAULT_XTOL,
            max_iter: int = 100) -> RootResult:
    """Algorithm 748, via ``scipy.optimize.toms748``. `dr` is ignored.

    The same algorithm as Julia's ``Roots.AlefeldPotraShi()``, which the base
    paper uses, so this is the reference point the other two are measured
    against. Mixes inverse cubic interpolation with Newton-quadratic steps
    while maintaining a bracket, giving near-Newton speed with bisection's
    guarantee.

    Iteration count comes from scipy's ``full_output``.

    Returns
    -------
    RootResult
    """
    lo, hi = bracket
    try:
        gamma, result = _scipy_toms748(r, lo, hi, xtol=xtol, maxiter=max_iter,
                                       full_output=True, disp=False)
    except (ValueError, RuntimeError):
        # scipy raises ValueError when the endpoints share a sign -- the same
        # condition `bisection` checks above, reported identically here so the
        # two remain interchangeable from the caller's side.
        return _FAILED

    gamma = float(gamma)
    return RootResult(gamma=gamma,
                      iterations=int(result.iterations),
                      converged=bool(result.converged),
                      residual=abs(r(gamma)))


ROOT_FINDERS: dict[str, RootFinder] = {
    "newton": newton,
    "bisection": bisection,
    "toms748": toms748,
}
"""Name -> finder, so sweeps can iterate over methods by string.

Expected keys: ``"newton"``, ``"bisection"``, ``"toms748"``."""


def solve_relaxation_parameter(entropy: EntropyFunction,
                               entropy_gradient: EntropyGradient,
                               u_old: State,
                               u_new: State,
                               finder: RootFinder,
                               bracket: tuple[float, float] = DEFAULT_BRACKET,
                               *,
                               xtol: float = DEFAULT_XTOL,
                               residual_tol: float = 1e-12) -> RootResult:
    """Full relaxation solve for one step: the single entry point solvers call.

    Builds the residual and its derivative, runs `finder`, and validates the
    result against `residual_tol` so that a returned gamma which does not
    actually solve the equation becomes ``converged=False`` rather than being
    propagated into the solution.

    Both solver modules call this rather than assembling residuals themselves.
    That is what keeps the comparison honest: this project compares FSAL
    schemes, so everything that is not an FSAL scheme must be held identical
    between them. If each solver configured its own tolerance or bracket, a
    difference in the results could come either from the scheme under study or
    from an incidental difference in how gamma was solved -- and the two could
    not be told apart afterwards.

    The effect is not hypothetical. A gamma differing in its thirteenth digit
    shifts the next error estimate in its thirteenth digit, which can flip a
    borderline accept/reject decision, after which the two runs take entirely
    different step sequences and report different RHS counts. Over a sweep of
    millions of accept/reject decisions, that will happen.

    A returned result with ``converged=False`` means the caller must reject the
    step and retry with a smaller step size -- not proceed with gamma = 1.

    The already-conserved case
    --------------------------
    When the step is small enough that the invariant is conserved to round-off
    across the whole bracket, the residual has no sign change in it and every
    bracketing method reports failure -- yet nothing is wrong, and halving `dt`
    would not help, because a smaller step only flattens the residual further.
    The authors handle this by taking gamma = 1 whenever both bracket endpoints
    sit within a few eps of zero. We do the equivalent, but check it where it
    can be verified: if the finder fails and ``|r(1)| <= residual_tol``, then
    gamma = 1 *is* a root to the accuracy demanded of any other answer, so it is
    returned as converged with its true residual attached.

    This is not the silent fallback the finders forbid. A finder may not invent
    a gamma it did not compute; this returns one that passes the same
    independent residual check every successful result must pass, and reports
    that residual so a caller can see how near zero it actually was.

    Returns
    -------
    RootResult
    """
    r = make_residual(entropy, u_old, u_new)
    dr = make_residual_derivative(entropy_gradient, u_old, u_new)

    result = finder(r, dr, bracket, xtol=xtol)

    gamma = result.gamma
    # Recomputed rather than taken from the finder: the point is to check the
    # answer independently of the method's own verdict, and Newton in
    # particular can report success from somewhere unhelpful.
    residual = abs(r(gamma)) if isfinite(gamma) else inf
    converged = result.converged and isfinite(gamma) and residual <= residual_tol

    if not converged:
        residual_at_one = abs(r(1.0))
        if residual_at_one <= residual_tol:
            return RootResult(gamma=1.0, iterations=result.iterations,
                              converged=True, residual=residual_at_one)

    return RootResult(gamma=gamma, iterations=result.iterations,
                      converged=converged, residual=residual)
