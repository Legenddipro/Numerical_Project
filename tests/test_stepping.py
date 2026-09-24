"""Tests for src/stepping.py. Owner: MD. Abir Hossain."""

from math import atan, isfinite, pi

import pytest

from src.problems import make_pendulum, pendulum_rhs
from src.stepping import PIDController, compute_error_estimate, initial_step_size
from src.tableaus import BS3, DP5, pid_gains

# One BS3 step on the pendulum from u0 = (1.5, 1.0) with dt = 0.4 -- the golden
# step every module checks part of. Written out as literals so this file depends
# on no other module's arithmetic.
GOLDEN_U_PREV = (1.5, 1.0)
GOLDEN_U_NEW = (1.1215518656013794, 1.526171003849539)
GOLDEN_U_EMBEDDED = (1.1241401271854055, 1.5257058461672233)
GOLDEN_ERROR_AT_1E3 = 0.7436


def bs3_controller() -> PIDController:
    return PIDController(pid_gains(BS3))


def test_error_estimate_zero_for_identical_solutions():
    """Identical main and embedded solutions give exactly zero error."""
    u = (1.25, -0.5)
    assert compute_error_estimate(u, (1.0, 1.0), u, 1e-6, 1e-6) == 0.0


def test_error_estimate_scales_with_tolerance():
    """Tightening abstol/reltol by ten must raise the reported error by about
    ten for the same solution pair."""
    loose = compute_error_estimate(GOLDEN_U_NEW, GOLDEN_U_PREV,
                                   GOLDEN_U_EMBEDDED, 1e-3, 1e-3)
    tight = compute_error_estimate(GOLDEN_U_NEW, GOLDEN_U_PREV,
                                   GOLDEN_U_EMBEDDED, 1e-4, 1e-4)
    assert tight / loose == pytest.approx(10.0, rel=1e-12)


def test_error_estimate_matches_the_golden_step():
    """The reference value: 0.7436 at tolerance 1e-3, which accepts.

    Computed independently of this code, so it checks the scaling convention
    (``abstol + reltol*max(|u|, |uprev|)``, then RMS) rather than merely the
    arithmetic. Getting the mix wrong -- scaling by |u| alone, or summing
    instead of averaging -- still produces a plausible number that moves in the
    right direction with tolerance, so it would pass every other test here.
    """
    err = compute_error_estimate(GOLDEN_U_NEW, GOLDEN_U_PREV,
                                 GOLDEN_U_EMBEDDED, 1e-3, 1e-3)
    assert err == pytest.approx(GOLDEN_ERROR_AT_1E3, abs=5e-5)

    # Below 1, so the step meets tolerance, and the controller agrees.
    assert err < 1.0
    controller = bs3_controller()
    assert controller.accept(controller.dt_factor(err, BS3.order))


def test_error_estimate_weights_components_by_their_own_scale():
    """A large component's absolute error is judged against its own magnitude.

    Two states differing by the same absolute amount must give different
    estimates when their magnitudes differ, otherwise the relative part of the
    tolerance is not doing anything.
    """
    small = compute_error_estimate((1.0, 1.0), (1.0, 1.0), (1.0 + 1e-6, 1.0),
                                   1e-8, 1e-6)
    large = compute_error_estimate((1000.0, 1.0), (1000.0, 1.0),
                                   (1000.0 + 1e-6, 1.0), 1e-8, 1e-6)
    assert large < small


def test_controller_shrinks_step_on_large_error():
    """Error above tolerance must give a factor below 1, and vice versa."""
    controller = bs3_controller()
    assert controller.dt_factor(50.0, BS3.order) < 1.0
    assert not controller.accept(controller.dt_factor(50.0, BS3.order))

    fresh = bs3_controller()
    assert fresh.dt_factor(0.1, BS3.order) > 1.0
    assert fresh.accept(fresh.dt_factor(0.1, BS3.order))


def test_controller_factor_is_limited():
    """The atan limiter must bound the factor even for an absurdly small error
    estimate. Without it a single near-zero error would blow the step size up
    and the next step would fail catastrophically."""
    ceiling = 1.0 + pi / 2.0
    for estimate in (1e-1, 1e-6, 1e-100, 1e-300, 0.0):
        controller = bs3_controller()
        factor = controller.dt_factor(estimate, BS3.order)
        assert factor < ceiling, f"factor {factor} unbounded at err={estimate}"

    # And it is a real limit, not a coincidence of these inputs: the limiter
    # fixes 1, so an on-tolerance step is left alone.
    assert PIDController.limiter(1.0) == 1.0
    assert PIDController.limiter(3.0) == pytest.approx(1.0 + atan(2.0))


def test_controller_handles_zero_error_estimate():
    """An exactly-zero estimate must not produce inf or nan."""
    controller = bs3_controller()
    factor = controller.dt_factor(0.0, BS3.order)
    assert isfinite(factor)
    assert factor > 1.0
    # Same treatment as a merely tiny estimate: both are clamped to eps, so the
    # boundary between them cannot produce a discontinuity in step size.
    assert factor == pytest.approx(bs3_controller().dt_factor(1e-300, BS3.order))


def test_controller_rejects_a_nonsensical_error_estimate():
    """A negative or nan estimate is a bug upstream, not a small step size.

    Raising means an impossible error estimate surfaces where it was produced.
    Silently clamping it would leave a step-size sequence nobody could
    reconstruct afterwards, which is fatal for a study whose results are
    reproducible-from-seed.
    """
    controller = bs3_controller()
    with pytest.raises(ArithmeticError):
        controller.dt_factor(-1.0, BS3.order)
    with pytest.raises(ArithmeticError):
        controller.dt_factor(float("nan"), BS3.order)


def test_history_shifts_only_on_accept():
    """on_accept shifts the error history; on_reject leaves it unchanged.

    The distinction matters because the relaxation variants accept in two
    stages -- error test, then relaxation -- and a step passing the first can
    still fail the second.
    """
    controller = bs3_controller()
    assert controller.err == [1.0, 1.0, 1.0]

    controller.dt_factor(0.5, BS3.order)
    after_factor = list(controller.err)
    assert after_factor[0] == pytest.approx(2.0)
    assert after_factor[1:] == [1.0, 1.0], "history shifted too early"

    controller.on_reject()
    assert controller.err == after_factor, "on_reject must not shift the history"

    controller.on_accept()
    assert controller.err == [after_factor[0], after_factor[0], 1.0]

    controller.dt_factor(0.25, BS3.order)
    controller.on_accept()
    assert controller.err == pytest.approx([4.0, 4.0, 2.0])


def test_history_affects_the_next_factor():
    """The controller is genuinely multi-step, not just the current error.

    Same current error, different history, different factor. If these agreed,
    the second and third gains would be inert and the controller would be a
    plain proportional one -- which would still work, but would not be what the
    paper's step sequences were produced with, so gate G5 would drift.
    """
    fresh = bs3_controller()
    first = fresh.dt_factor(0.5, BS3.order)

    warmed = bs3_controller()
    warmed.dt_factor(1e-3, BS3.order)
    warmed.on_accept()
    second = warmed.dt_factor(0.5, BS3.order)

    assert first != pytest.approx(second)


@pytest.mark.parametrize("tableau", [BS3, DP5], ids=lambda t: t.name)
@pytest.mark.parametrize("tol", [1e-3, 1e-7, 1e-11])
def test_initial_step_size_is_positive_and_bounded(tableau, tol):
    """The heuristic must return a positive step no larger than the interval,
    and its accompanying f(t0, u0) must equal a direct RHS evaluation."""
    problem = make_pendulum()
    t0, t_end = problem.t_span

    dt0, f0, _calls = initial_step_size(problem, tol, tol, tableau.order)

    assert dt0 > 0.0
    assert dt0 <= t_end - t0
    assert f0 == pytest.approx(pendulum_rhs(t0, problem.u0), abs=0.0)


def test_initial_step_size_tightens_with_tolerance():
    """A tighter tolerance must not ask for a larger first step.

    The heuristic inverts the local error model, so the predicted step shrinks
    as the tolerance does. A version that ignored the tolerance would pass every
    other check in this file.
    """
    problem = make_pendulum()
    steps = [initial_step_size(problem, tol, tol, BS3.order)[0]
             for tol in (1e-3, 1e-5, 1e-7, 1e-9)]
    assert steps == sorted(steps, reverse=True)

    # And a higher-order method tolerates a larger first step at the same
    # tolerance, since its error falls off faster with dt.
    coarse = initial_step_size(problem, 1e-7, 1e-7, BS3.order)[0]
    fine = initial_step_size(problem, 1e-7, 1e-7, DP5.order)[0]
    assert fine > coarse


def test_initial_step_size_reports_its_own_call_count():
    """Third return value is c0: 2 when the heuristic runs, 1 when dt is given.

    Verify by spying -- wrap the problem's rhs in a counter and check the
    reported number equals the calls that actually happened. A hardcoded
    constant would pass a weaker test and then go stale silently.
    """
    base = make_pendulum()
    calls = {"n": 0}

    def spy(t, u):
        calls["n"] += 1
        return base.rhs(t, u)

    problem = base._replace(rhs=spy)

    calls["n"] = 0
    _dt, _f0, reported = initial_step_size(problem, 1e-6, 1e-6, BS3.order)
    assert reported == calls["n"] == 2

    calls["n"] = 0
    dt, _f0, reported = initial_step_size(problem, 1e-6, 1e-6, BS3.order, dt=0.4)
    assert reported == calls["n"] == 1
    assert dt == 0.4


def test_initial_step_size_honours_a_supplied_step():
    """Fixed-step mode, which gate G2's convergence study needs.

    The supplied step is returned untouched -- not clipped, not adjusted by the
    heuristic -- because a convergence study halves dt deliberately and any
    adjustment would flatten the measured order.
    """
    problem = make_pendulum()
    for dt in (0.4, 0.2, 0.1, 0.05):
        got, f0, calls = initial_step_size(problem, 1e-6, 1e-6, BS3.order, dt=dt)
        assert got == dt
        assert calls == 1
        assert f0 == pendulum_rhs(0.0, problem.u0)


def test_initial_step_size_rejects_impossible_arguments():
    """Backwards integration and a zero tolerance pair are configuration errors.

    Both would otherwise produce a step size that looks usable: a negative dt
    integrates the wrong way, and two zero tolerances leave the scaling vector
    at zero and every error estimate infinite.
    """
    problem = make_pendulum()
    with pytest.raises(ValueError):
        initial_step_size(problem, 1e-6, 1e-6, BS3.order, dt=-0.1)
    with pytest.raises(ValueError):
        initial_step_size(problem, 0.0, 0.0, BS3.order)


def test_initial_step_size_survives_a_stationary_state():
    """u0 at rest at the bottom: f(u0) = 0, so the ratio says nothing.

    The pendulum hanging straight down is an equilibrium, so both probe slopes
    vanish and the heuristic must fall back rather than divide by a zero norm.
    This is reachable from the Monte Carlo sampler, which draws omega and theta
    near zero often enough to matter.
    """
    problem = make_pendulum(u0=(0.0, 0.0))
    dt0, f0, calls = initial_step_size(problem, 1e-6, 1e-6, BS3.order)

    assert dt0 > 0.0
    assert isfinite(dt0)
    assert f0 == (0.0, 0.0)
    # The probe still runs -- it is the probe that discovers there is no
    # curvature to measure, since f1 == f0 -- so two calls happen and the
    # reported count says two. Checked with a spy rather than assumed, because
    # this is precisely the sort of path where a hardcoded c0 would go wrong.
    spied = {"n": 0}

    def spy(t, u):
        spied["n"] += 1
        return problem.rhs(t, u)

    _dt, _f, reported = initial_step_size(problem._replace(rhs=spy),
                                          1e-6, 1e-6, BS3.order)
    assert reported == spied["n"] == calls == 2
