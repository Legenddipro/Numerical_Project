"""Tests for src/rootfind.py. Owner: MD. Shadman Shafie.

Root-finders take plain callables, so these tests need no ODE code and can be
written before any solver exists.
"""

from math import cos, exp, isnan, log, log2, sin, sqrt

import pytest

from src.contracts import RootResult
from src.problems import pendulum_entropy, pendulum_entropy_gradient
from src.rootfind import (DEFAULT_BRACKET, DEFAULT_XTOL, ROOT_FINDERS,
                          bisection, make_residual, make_residual_derivative,
                          newton, solve_relaxation_parameter, toms748)

# The golden step: one BS3 step on the pendulum from u0 = (1.5, 1.0) with
# dt = 0.4. Written out as literals so this file depends on no other module's
# arithmetic -- if the solvers are wrong, these tests still mean something.
GOLDEN_U_PREV = (1.5, 1.0)
GOLDEN_U_NEW = (1.1215518656013794, 1.526171003849539)
GOLDEN_GAMMA = 1.0036353183

FINDERS = tuple(ROOT_FINDERS.values())

# (function, derivative, bracket, known root). Each bracket straddles the root
# and contains gamma_0 = 1 or sits close enough that Newton, which always
# starts there, reaches it.
TOY_FUNCTIONS = (
    (lambda x: x * x - 2.0, lambda x: 2.0 * x, (1.0, 2.0), sqrt(2.0)),
    (lambda x: cos(x) - x, lambda x: -sin(x) - 1.0, (0.5, 1.5), 0.7390851332151607),
    (lambda x: exp(x) - 2.0, exp, (0.5, 1.0), log(2.0)),
    (lambda x: x ** 3 - x - 1.0, lambda x: 3.0 * x * x - 1.0, (1.1, 1.8),
     1.3247179572447458),
)


def golden_residual():
    return make_residual(pendulum_entropy, GOLDEN_U_PREV, GOLDEN_U_NEW)


def golden_derivative():
    return make_residual_derivative(pendulum_entropy_gradient, GOLDEN_U_PREV,
                                    GOLDEN_U_NEW)


def test_all_finders_agree_on_toy_functions():
    """Gate G1.

    Every method must locate the same root of several known functions to
    within tolerance. Disagreement means at least one implementation is wrong,
    and this catches it without any ODE machinery in the picture.
    """
    for f, df, bracket, root in TOY_FUNCTIONS:
        results = [finder(f, df, bracket) for finder in FINDERS]
        for finder, result in zip(FINDERS, results):
            assert result.converged, f"{finder.__name__} failed on {bracket}"
            assert result.gamma == pytest.approx(root, abs=1e-12), (
                f"{finder.__name__} found {result.gamma!r}, expected {root!r}")
            # The residual is the independent check: a method can be confident
            # and wrong, and only this notices.
            assert result.residual < 1e-12


def test_all_finders_agree_on_the_golden_step():
    """Gate G1's other half, on the real residual rather than a toy one.

    gamma = 1.0036353183 for one BS3 step from (1.5, 1.0) with dt = 0.4 --
    computed independently of this code, so it checks the residual's
    construction (which endpoint the invariant is measured against, and in
    which direction) and not merely that three methods agree with each other.
    """
    for name, finder in ROOT_FINDERS.items():
        result = solve_relaxation_parameter(
            pendulum_entropy, pendulum_entropy_gradient,
            GOLDEN_U_PREV, GOLDEN_U_NEW, finder)
        assert result.converged, f"{name} did not converge"
        assert result.gamma == pytest.approx(GOLDEN_GAMMA, abs=5e-11), name
        assert result.residual <= 1e-12, name


def test_newton_converges_in_few_iterations():
    """Starting at gamma_0 = 1 on a realistic residual, Newton should need
    roughly 3-5 iterations."""
    result = newton(golden_residual(), golden_derivative())
    assert result.converged
    assert 3 <= result.iterations <= 5, result.iterations


def test_bisection_iteration_count_matches_theory():
    """Halving a bracket of width w to tolerance x should take about
    log2(w/x) iterations -- roughly 45 for (0.8, 1.2) down to 1e-14."""
    lo, hi = DEFAULT_BRACKET
    predicted = log2((hi - lo) / DEFAULT_XTOL)

    result = bisection(golden_residual(), golden_derivative())
    assert result.converged
    assert result.iterations == pytest.approx(predicted, abs=2.0)
    assert 40 <= result.iterations <= 50, result.iterations


def test_toms748_between_newton_and_bisection_in_cost():
    """Algorithm 748 should need far fewer iterations than bisection while
    keeping the same bracketing guarantee -- the reason the base paper chose
    it, and the baseline the other two are measured against."""
    r, dr = golden_residual(), golden_derivative()
    fast = newton(r, dr).iterations
    middle = toms748(r, dr).iterations
    slow = bisection(r, dr).iterations

    assert middle < slow / 4, (middle, slow)
    # Near Newton, not necessarily below it: both are superlinear here, and the
    # claim being checked is that bracketing costs little, not that 748 wins.
    assert middle <= fast + 3, (middle, fast)


def test_bisection_fails_loudly_without_sign_change():
    """No sign change in the bracket must yield converged=False, not a
    fabricated root."""
    positive = lambda x: 1.0 + x * x                     # noqa: E731
    slope = lambda x: 2.0 * x                            # noqa: E731

    for finder in (bisection, toms748):
        result = finder(positive, slope)
        assert not result.converged, finder.__name__
        assert isnan(result.gamma), (finder.__name__, result.gamma)


def test_newton_reports_failure_rather_than_escaping_bracket():
    """Newton has no bracketing guarantee, so a near-zero derivative can throw
    an iterate far from the bracket. That must come back as converged=False,
    never as a silent fallback to gamma = 1 -- which would look like a
    successful step while quietly abandoning conservation."""
    # Roots at 4 and 6, both far outside [0.8, 1.2]. Newton converges happily
    # to 4.0 and must still report failure.
    escaping = lambda x: (x - 5.0) ** 2 - 1.0            # noqa: E731
    escaping_dr = lambda x: 2.0 * (x - 5.0)              # noqa: E731

    result = newton(escaping, escaping_dr)
    assert not result.converged
    assert result.gamma != 1.0, "fell back to gamma = 1 instead of reporting"

    # Derivative exactly zero at the starting point: no direction to move in.
    flat = lambda x: (x - 1.0) ** 2 + 1.0                # noqa: E731
    flat_dr = lambda x: 2.0 * (x - 1.0)                  # noqa: E731

    result = newton(flat, flat_dr)
    assert not result.converged
    assert result.residual > 0.0


def test_solve_relaxation_parameter_rejects_bad_residual():
    """A finder reporting success with a residual above residual_tol must be
    converted to converged=False.

    The residual is checked independently of the method's own verdict, so a
    method that is confidently wrong is still caught.
    """
    def liar(r, dr, bracket, *, xtol=DEFAULT_XTOL, max_iter=50):
        # Inside the bracket, plausible, and not the root.
        return RootResult(gamma=1.05, iterations=1, converged=True,
                          residual=0.0)

    result = solve_relaxation_parameter(
        pendulum_entropy, pendulum_entropy_gradient,
        GOLDEN_U_PREV, GOLDEN_U_NEW, liar)

    assert not result.converged
    # The residual reported back is the true one, not the finder's claim.
    assert result.residual > 1e-12


def test_already_conserved_step_returns_gamma_one():
    """A step that already conserves the invariant to round-off has no sign
    change in the bracket, and must come back as gamma = 1 rather than as a
    failure that halves the step size forever.

    The direction here is perpendicular to grad eta, so the invariant changes
    only at second order -- about 1e-16 over a 1e-8 step, which is exactly the
    regime a tight tolerance drives the solvers into.
    """
    omega, theta = GOLDEN_U_PREV
    grad = pendulum_entropy_gradient(GOLDEN_U_PREV)
    perpendicular = (-grad[1], grad[0])
    scale = 1e-8
    u_new = (omega + scale * perpendicular[0], theta + scale * perpendicular[1])

    r = make_residual(pendulum_entropy, GOLDEN_U_PREV, u_new)
    assert abs(r(1.0)) < 1e-12, "test setup: step is not already conserved"
    assert r(0.8) * r(1.2) > 0.0, "test setup: bracket does have a sign change"

    for finder in (bisection, toms748):
        # The finder itself must still fail -- that honesty is what the
        # fallback is built on.
        assert not finder(r, None).converged

        result = solve_relaxation_parameter(
            pendulum_entropy, pendulum_entropy_gradient,
            GOLDEN_U_PREV, u_new, finder)
        assert result.converged, finder.__name__
        assert result.gamma == 1.0
        assert result.residual <= 1e-12


def test_trivial_root_at_zero_is_not_returned():
    """gamma = 0 always solves the relaxation equation. No method may return
    it when a nontrivial root exists."""
    r = golden_residual()
    assert r(0.0) == 0.0, "test setup: gamma = 0 should be an exact root"

    for name, finder in ROOT_FINDERS.items():
        result = solve_relaxation_parameter(
            pendulum_entropy, pendulum_entropy_gradient,
            GOLDEN_U_PREV, GOLDEN_U_NEW, finder)
        assert result.gamma == pytest.approx(GOLDEN_GAMMA, abs=5e-11), name


def test_residual_and_derivative_are_consistent():
    """Compare make_residual_derivative against a finite difference of
    make_residual. Catches a sign or chain-rule error in the gradient, which
    would otherwise only show up as Newton converging slowly."""
    r = golden_residual()
    dr = golden_derivative()

    h = 1e-6
    for gamma in (0.85, 1.0, 1.15):
        finite_difference = (r(gamma + h) - r(gamma - h)) / (2.0 * h)
        assert dr(gamma) == pytest.approx(finite_difference, rel=1e-8)


def test_residual_is_zero_at_the_starting_point():
    """r(0) = 0 by construction, and r(1) is the unrelaxed step's drift.

    The second is the number the whole scheme exists to remove: -3.689e-4 for
    the golden step, matching the drift recorded in ALGORITHMS.md.
    """
    r = golden_residual()
    assert r(0.0) == 0.0
    assert r(1.0) == pytest.approx(-3.689e-4, rel=1e-3)
