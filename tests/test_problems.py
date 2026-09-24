"""Tests for src/problems.py. Owner: project lead."""

from math import cos, pi, sin

import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from src.problems import (
    DEFAULT_TSPAN,
    DEFAULT_U0,
    build_reference,
    make_pendulum,
    pendulum_entropy,
    pendulum_entropy_gradient,
    pendulum_rhs,
    regime_of,
    sample_initial_conditions,
    separatrix_energy,
)


def test_rhs_and_entropy_match_golden_step():
    """The reference values every later step checks against."""
    u0 = (1.5, 1.0)
    k1 = pendulum_rhs(0.0, u0)
    assert k1[0] == pytest.approx(-0.8414710, abs=1e-7)
    assert k1[1] == pytest.approx(1.5000000, abs=1e-7)
    assert pendulum_entropy(u0) == pytest.approx(0.5846976941, abs=1e-10)


def test_rhs_component_order_matches_julia():
    """State is (omega, theta): rhs returns (-sin(theta), omega).

    Swapping the components yields a system that still oscillates plausibly, so
    this is asserted explicitly rather than assumed. Gate G5 compares against
    the authors' numbers and would otherwise fail confusingly.
    """
    omega, theta = 0.3, 1.2
    domega, dtheta = pendulum_rhs(0.0, (omega, theta))
    assert domega == pytest.approx(-sin(theta))
    assert dtheta == pytest.approx(omega)


def test_rhs_ignores_time():
    """The pendulum is autonomous; `t` is accepted only for signature uniformity."""
    u = (0.7, -0.4)
    assert pendulum_rhs(0.0, u) == pendulum_rhs(123.456, u)


def test_entropy_gradient_matches_finite_difference():
    """Analytic gradient against a central difference.

    A sign error here would surface only as Newton mysteriously failing to
    converge, with nothing obviously wrong in the root-finder.
    """
    h = 1e-6
    for u in [(1.5, 1.0), (-0.3, 2.4), (0.0, 0.0), (2.0, -1.7)]:
        grad = pendulum_entropy_gradient(u)
        for i in range(2):
            up = list(u); up[i] += h
            um = list(u); um[i] -= h
            fd = (pendulum_entropy(tuple(up)) - pendulum_entropy(tuple(um))) / (2 * h)
            assert grad[i] == pytest.approx(fd, abs=1e-7)


def test_entropy_conserved_along_reference_solution():
    """The reference must hold the invariant far tighter than anything we measure
    against it, or every error figure downstream is contaminated."""
    prob = make_pendulum()
    eta0 = prob.entropy(prob.u0)
    drift = max(abs(prob.entropy(prob.reference(t)) - eta0)
                for t in [i * 0.1 for i in range(101)])
    assert drift < 1e-10, f"reference drifts by {drift:.2e}"


def test_reference_starts_at_initial_condition():
    prob = make_pendulum()
    u = prob.reference(prob.t_span[0])
    assert u[0] == pytest.approx(prob.u0[0], abs=1e-12)
    assert u[1] == pytest.approx(prob.u0[1], abs=1e-12)


@pytest.mark.parametrize("theta0", [0.01, 0.5, 1.0, 2.0])
def test_period_matches_elliptic_integral(theta0):
    """Independent physics check on the whole right-hand side.

    Released from rest at amplitude theta0, the exact period is

        T = 4 * sqrt(L/g) * K(sin^2(theta0 / 2))

    with K the complete elliptic integral of the first kind. This is an
    analytic result about pendulums, derived nowhere in our code, so agreeing
    with it tests the physics rather than our internal consistency. As
    theta0 -> 0 it reduces to 2*pi, the small-angle limit.

    Measured by releasing from rest at theta0 and detecting the next time omega
    returns to zero, which is half a period. The bob starts at rest, so omega
    is already zero at t = 0 and that first crossing is discarded.
    """
    def event(t, y):
        return y[0]          # omega = 0
    event.terminal = False
    event.direction = 0

    sol = solve_ivp(lambda t, y: [-sin(y[1]), y[0]], (0.0, 30.0), [0.0, theta0],
                    method="DOP853", rtol=1e-12, atol=1e-13, events=event)
    turning_points = [float(t) for t in sol.t_events[0] if t > 1e-6]
    assert turning_points, "no turning point found within the integration window"

    measured = 2 * turning_points[0]
    exact = 4 * ellipk(sin(theta0 / 2) ** 2)
    assert measured == pytest.approx(exact, rel=1e-8)


def test_small_angle_period_approaches_two_pi():
    """The textbook limit, stated separately because it is the check a reader
    can sanity-test in their head."""
    exact = 4 * ellipk(sin(0.001 / 2) ** 2)
    assert exact == pytest.approx(2 * pi, rel=1e-6)


def test_separatrix_divides_the_regimes():
    """At the separatrix the bob is balanced motionless at the top."""
    upright_at_rest = (0.0, pi)
    assert pendulum_entropy(upright_at_rest) == pytest.approx(separatrix_energy())
    assert regime_of((0.0, 0.5)) == "libration"      # low energy, swings
    assert regime_of((3.0, 0.0)) == "rotation"       # fast, goes over the top


def test_sampler_respects_requested_regime():
    sep = separatrix_energy()
    for u in sample_initial_conditions(200, seed=0, regime="libration"):
        assert pendulum_entropy(u) < sep
    for u in sample_initial_conditions(200, seed=0, regime="rotation"):
        assert pendulum_entropy(u) > sep


def test_sampler_is_reproducible_from_seed():
    """Every Monte Carlo result must be reproducible from its recorded seed alone."""
    a = sample_initial_conditions(50, seed=12345)
    b = sample_initial_conditions(50, seed=12345)
    c = sample_initial_conditions(50, seed=54321)
    assert a == b
    assert a != c


def test_sampler_returns_exactly_n_states():
    for n in (1, 7, 100):
        assert len(sample_initial_conditions(n, seed=1)) == n
    assert sample_initial_conditions(0, seed=1) == ()


def test_sampler_raises_rather_than_looping_forever():
    """An impossible request must fail loudly, not hang."""
    with pytest.raises(RuntimeError):
        sample_initial_conditions(
            10, seed=0, regime="rotation",
            theta_range=(-0.01, 0.01), omega_range=(-0.01, 0.01))


def test_make_pendulum_defaults_match_the_authors():
    prob = make_pendulum()
    assert prob.u0 == DEFAULT_U0 == (1.5, 1.0)
    assert prob.t_span == DEFAULT_TSPAN == (0.0, 10.0)
    assert prob.rhs(0.0, prob.u0) == pendulum_rhs(0.0, (1.5, 1.0))
    assert prob.entropy(prob.u0) == pendulum_entropy((1.5, 1.0))
    assert prob.entropy_gradient(prob.u0) == pendulum_entropy_gradient((1.5, 1.0))


def test_g_over_l_is_bound_into_the_problem():
    """Solvers see plain (t, u) and (u) signatures; the parameter is closed over."""
    prob = make_pendulum(g_over_l=4.0)
    assert prob.rhs(0.0, (0.0, 1.0))[0] == pytest.approx(-4.0 * sin(1.0))
    assert prob.entropy((0.0, 1.0)) == pytest.approx(-4.0 * cos(1.0))
