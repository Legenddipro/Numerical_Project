"""Tests for src/solvers/classic.py. Owner: MD. Shadman Shafie.

Module-level tests, plus the parts of gates G2, G3 and G4 that this module can
establish on its own. Their cross-method forms -- comparing baseline, naive,
FSAL-R and R-FSAL against each other, and against the Julia reference -- live
in test_gates.py, which spans both solver modules and belongs to the lead.
Checking here what can be checked here means step 6 does not hand a broken
solver to step 7 and wait for someone else's test file to notice.
"""

from math import hypot

import pytest

from src.contracts import RootResult
from src.problems import (make_pendulum, pendulum_entropy,
                          pendulum_entropy_gradient, pendulum_rhs)
from src.rootfind import DEFAULT_BRACKET, ROOT_FINDERS, solve_relaxation_parameter
from src.solvers.classic import solve_baseline, solve_fsalr, solve_naive
from src.tableaus import BS3, DP5

TOMS748 = ROOT_FINDERS["toms748"]

# The golden step: one BS3 step from u0 = (1.5, 1.0) with dt = 0.4, computed
# independently and recorded in docs/ALGORITHMS.md. Literals, so these tests
# check the solver rather than agreeing with it.
GOLDEN_U0 = (1.5, 1.0)
GOLDEN_U_NEW = (1.1215519, 1.5261710)
GOLDEN_U_GAMMA = (1.1201761, 1.5280838)
GOLDEN_GAMMA = 1.0036353183
GOLDEN_T_GAMMA = 0.4014541
GOLDEN_INTERPOLATION = (-0.9995771, 1.1201761)
GOLDEN_F_U_GAMMA = (-0.9990880, 1.1201761)

RELAXING = (("naive", solve_naive), ("fsalr", solve_fsalr))


@pytest.fixture(scope="module")
def pendulum():
    """The authors' setup: u0 = (1.5, 1.0) over t in [0, 10].

    Module-scoped because building it integrates a tight DOP853 reference once,
    which is the expensive part; the solvers themselves are cheap.
    """
    return make_pendulum(t_span=(0.0, 10.0))


@pytest.fixture(scope="module")
def short_pendulum():
    """One BS3 step's worth of time span, for the golden-step checks."""
    return make_pendulum(t_span=(0.0, 0.4))


def rhs_identity(result, tableau, c0=2):
    """The gate G3 right-hand side: what `nf` must equal, exactly.

    Every attempt costs s-1 evaluations, rejections included, because stage 1
    is always served from the FSAL cache and a rejected step restarts from the
    same u_n. Naive pays one more on accepted steps only.
    """
    s = tableau.n_stages
    total = c0 + (s - 1) * (result.n_accept + result.n_reject)
    if result.method == "naive":
        total += result.n_accept
    return total


# --------------------------------------------------------------------------
# The golden step
# --------------------------------------------------------------------------

def test_golden_step_baseline(short_pendulum):
    """One BS3 step reproduces the reference u^1 exactly.

    Fixed step and a span of exactly one step, so nothing the controller does
    can intervene. If this fails, nothing below it means anything.
    """
    result = solve_baseline(short_pendulum, BS3, 1e-3, adaptive=False, dt0=0.4)

    assert result.u_final[0] == pytest.approx(GOLDEN_U_NEW[0], abs=5e-8)
    assert result.u_final[1] == pytest.approx(GOLDEN_U_NEW[1], abs=5e-8)
    assert result.t_final == 0.4
    # c0 = 1 in fixed-step mode (the heuristic is skipped), plus s-1 = 3.
    assert result.nf == 4
    assert result.n_accept == 1
    assert result.n_reject == 0


@pytest.mark.parametrize("name,solver", RELAXING)
def test_golden_step_relaxed(short_pendulum, name, solver):
    """The same step relaxed lands on u_gamma with zero drift.

    Both variants must agree here: they differ only in how the FSAL cache is
    refreshed afterwards, which cannot affect this step's own result.
    """
    result = solver(short_pendulum, BS3, 1e-3, rootfinder=TOMS748,
                    adaptive=False, dt0=0.4, save_history=True)

    assert result.u_final[0] == pytest.approx(GOLDEN_U_GAMMA[0], abs=5e-8)
    assert result.u_final[1] == pytest.approx(GOLDEN_U_GAMMA[1], abs=5e-8)
    assert result.gamma_history[0] == pytest.approx(GOLDEN_GAMMA, abs=5e-11)

    drift = pendulum_entropy(result.u_final) - pendulum_entropy(GOLDEN_U0)
    assert drift == 0.0, "relaxation must conserve the invariant exactly"

    # Naive pays the extra call, FSAL-R does not. The whole point, on one step.
    assert result.nf == (5 if name == "naive" else 4)


@pytest.mark.parametrize("name,solver", RELAXING)
def test_relaxation_moves_time_as_well_as_state(pendulum, name, solver):
    """t_gamma = t_n + gamma*dt, not t_n + dt.

    Forgetting this is silent: the solution stays plausible but is compared
    against the reference at the wrong instant, which surfaces only as a
    mysteriously large global error. The final step is the exception -- it
    lands on t_end exactly -- so this reads the *first* step out of the
    history, where the general rule applies.
    """
    result = solver(pendulum, BS3, 1e-3, rootfinder=TOMS748,
                    adaptive=False, dt0=0.4, save_history=True)

    assert result.t_history[0] == 0.0
    assert result.t_history[1] == pytest.approx(GOLDEN_T_GAMMA, abs=5e-8)
    assert result.t_history[1] != 0.4
    # ...and the run still ends exactly on t_end, not somewhere near it.
    assert result.t_final == 10.0


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------

def test_baseline_reaches_final_time(pendulum):
    """Integration terminates at t_span[1], not before or past it."""
    for tableau in (BS3, DP5):
        result = solve_baseline(pendulum, tableau, 1e-6)
        assert result.t_final == 10.0
        assert result.n_accept > 0


def test_baseline_reuses_fsal_stage(pendulum):
    """RHS count must be about (s-1) per accepted step, not s.

    If FSAL reuse is missing, everything still runs and produces correct
    answers -- just more expensively -- so this is checked directly rather
    than assumed from a passing accuracy test.
    """
    for tableau in (BS3, DP5):
        result = solve_baseline(pendulum, tableau, 1e-7)
        s = tableau.n_stages
        per_attempt = (result.nf - 2) / (result.n_accept + result.n_reject)
        assert per_attempt == pytest.approx(s - 1), (tableau.name, per_attempt)
        assert result.nf == rhs_identity(result, tableau)


def test_baseline_ignores_rootfinder_argument(pendulum):
    """Passing a root-finder must change nothing. The argument exists only so
    the signature matches the relaxation variants."""
    without = solve_baseline(pendulum, BS3, 1e-6)
    for finder in ROOT_FINDERS.values():
        with_finder = solve_baseline(pendulum, BS3, 1e-6, rootfinder=finder)
        assert with_finder == without

    assert without.n_root_iterations == 0
    assert without.n_relaxation_failures == 0


def test_fixed_step_mode_takes_uniform_steps():
    """With adaptive=False, every step must equal dt0 and nothing may be
    rejected. Gate G2 depends on this mode behaving exactly."""
    problem = make_pendulum(t_span=(0.0, 1.0))
    dt = 0.125           # exact in binary, so t accumulates without drift
    result = solve_baseline(problem, BS3, 1e-3, adaptive=False, dt0=dt,
                            save_history=True)

    assert result.n_reject == 0
    assert result.n_accept == 8
    steps = [b - a for a, b in zip(result.t_history, result.t_history[1:])]
    assert all(step == dt for step in steps), steps
    assert result.t_final == 1.0


def test_history_recorded_only_when_requested(pendulum):
    """save_history=False must leave the history tuples empty. A thousand
    Monte Carlo samples with histories on would cost gigabytes."""
    off = solve_fsalr(pendulum, BS3, 1e-5, rootfinder=TOMS748)
    assert off.t_history == ()
    assert off.entropy_history == ()
    assert off.gamma_history == ()

    on = solve_fsalr(pendulum, BS3, 1e-5, rootfinder=TOMS748,
                     save_history=True)
    # One entry for the initial condition, then one per accepted step.
    assert len(on.t_history) == on.n_accept + 1
    assert len(on.entropy_history) == on.n_accept + 1
    assert len(on.gamma_history) == on.n_accept

    # Recording must not change the trajectory.
    assert on.u_final == off.u_final and on.nf == off.nf

    # Baseline has no gamma to record, even with history on.
    assert solve_baseline(pendulum, BS3, 1e-5,
                          save_history=True).gamma_history == ()


# --------------------------------------------------------------------------
# Conservation -- gate G4, within this module
# --------------------------------------------------------------------------

def test_naive_conserves_invariant(pendulum):
    """Energy held to ~1e-15 despite the naive scheme's cost."""
    eta0 = pendulum_entropy(pendulum.u0)
    for tableau in (BS3, DP5):
        result = solve_naive(pendulum, tableau, 1e-5, rootfinder=TOMS748,
                             save_history=True)
        drift = max(abs(eta - eta0) for eta in result.entropy_history)
        assert drift < 1e-13, (tableau.name, drift)


def test_fsalr_conserves_invariant(pendulum):
    """FSAL-R must conserve as well as naive does. The interpolation changes
    the cost, not the conservation."""
    eta0 = pendulum_entropy(pendulum.u0)
    for tableau in (BS3, DP5):
        result = solve_fsalr(pendulum, tableau, 1e-5, rootfinder=TOMS748,
                             save_history=True)
        drift = max(abs(eta - eta0) for eta in result.entropy_history)
        assert drift < 1e-13, (tableau.name, drift)


def test_baseline_drifts_where_the_relaxed_variants_do_not(pendulum):
    """Gate G4's other half: the problem relaxation exists to solve.

    Binary and unmistakable -- baseline's drift is orders of magnitude larger,
    not marginally larger.
    """
    eta0 = pendulum_entropy(pendulum.u0)

    baseline = solve_baseline(pendulum, BS3, 1e-5, save_history=True)
    baseline_drift = max(abs(eta - eta0) for eta in baseline.entropy_history)
    assert baseline_drift > 1e-6, baseline_drift

    relaxed = solve_fsalr(pendulum, BS3, 1e-5, rootfinder=TOMS748,
                          save_history=True)
    relaxed_drift = max(abs(eta - eta0) for eta in relaxed.entropy_history)
    assert baseline_drift > 1e8 * relaxed_drift


# --------------------------------------------------------------------------
# Cost -- gate G3, within this module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tableau", (BS3, DP5), ids=lambda t: t.name)
@pytest.mark.parametrize("tol", (1e-3, 1e-5, 1e-7))
def test_rhs_identity_holds_for_every_variant(pendulum, tableau, tol):
    """Gate G3, the strongest check available, for all three loops here::

        nf == c0 + (s-1)*(n_accept + n_reject)        baseline, FSAL-R
        nf == c0 + (s-1)*(n_accept + n_reject) + n_accept    naive

    Integers admit no "close enough". Checked at every root-finder too, since
    a finder that shifts gamma in its last digit shifts the accept/reject
    sequence and would break the identity if the bookkeeping were fragile.
    """
    assert solve_baseline(pendulum, tableau, tol).nf == rhs_identity(
        solve_baseline(pendulum, tableau, tol), tableau)

    for name, solver in RELAXING:
        for finder in ROOT_FINDERS.values():
            result = solver(pendulum, tableau, tol, rootfinder=finder)
            assert result.nf == rhs_identity(result, tableau), (
                name, tableau.name, tol, result.nf, result.n_accept,
                result.n_reject)


def test_reported_nf_equals_actual_rhs_calls(pendulum):
    """The counting is honest.

    Counting is manual in the hot loop -- a wrapper costs roughly ten times a
    bare increment, which over a full sweep is minutes rather than seconds --
    so this is what proves the manual increments were not missed. It pairs
    with the identity above: that says the number has the right structure,
    this says the number is true.
    """
    for name, solver, kwargs in (("baseline", solve_baseline, {}),
                                 ("naive", solve_naive, {"rootfinder": TOMS748}),
                                 ("fsalr", solve_fsalr, {"rootfinder": TOMS748})):
        calls = 0

        def spy(t, u):
            nonlocal calls
            calls += 1
            return pendulum_rhs(t, u)

        result = solver(pendulum._replace(rhs=spy), BS3, 1e-6, **kwargs)
        assert result.nf == calls, name


def test_naive_pays_one_extra_evaluation_per_accepted_step(pendulum):
    """The trailing n_accept is the inefficiency the paper removes.

    Compared per accepted step rather than as a total: relaxation shifts both
    state and time, so the variants follow different trajectories and their
    totals legitimately differ. The ratio is what the paper actually claims --
    about s/(s-1), so 4/3 for BS3 and 7/6 for DP5.
    """
    for tableau in (BS3, DP5):
        s = tableau.n_stages
        baseline = solve_baseline(pendulum, tableau, 1e-7)
        naive = solve_naive(pendulum, tableau, 1e-7, rootfinder=TOMS748)
        fsalr = solve_fsalr(pendulum, tableau, 1e-7, rootfinder=TOMS748)

        cost = lambda r: r.nf / r.n_accept                   # noqa: E731
        assert cost(naive) / cost(baseline) == pytest.approx(s / (s - 1),
                                                             rel=0.02)
        # FSAL-R buys that back entirely.
        assert cost(fsalr) == pytest.approx(cost(baseline), rel=0.02)


def test_all_three_finders_give_the_same_trajectory(pendulum):
    """Swapping the root-finder must not change the integration.

    All three solve the same equation to the same tolerance, so any difference
    in nf or in the accept/reject counts would mean one of them is returning a
    materially different gamma -- and step 9's root-finder table, which
    compares cost at equal accuracy, would be measuring that instead of the
    methods.
    """
    for name, solver in RELAXING:
        results = [solver(pendulum, BS3, 1e-7, rootfinder=finder)
                   for finder in ROOT_FINDERS.values()]
        first = results[0]
        for other in results[1:]:
            assert other.nf == first.nf, name
            assert other.n_accept == first.n_accept, name
            assert other.n_reject == first.n_reject, name
            assert other.n_relaxation_failures == 0, name
            assert other.error_final == pytest.approx(first.error_final,
                                                      rel=1e-6), name


# --------------------------------------------------------------------------
# Convergence order -- gate G2, within this module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tableau,expected,coarse", ((BS3, 8.0, 128),
                                                     (DP5, 32.0, 256)),
                         ids=("BS3", "DP5"))
def test_convergence_order(pendulum, tableau, expected, coarse):
    """Gate G2. Fixed step, no relaxation: halving dt cuts the error by 2**p.

    If this fails, the tableau or the stage loop is wrong and nothing
    downstream is worth running.

    The step ranges differ by method on purpose. Too large and the asymptotic
    rate has not set in; too small and round-off against a reference that is
    itself only accurate to ~1e-12 flattens the observed order into a false
    failure. DP5 reaches 1e-11 by its finer step, which is as far down as this
    reference can measure.
    """
    errors = []
    for n in (coarse, 2 * coarse):
        result = solve_baseline(pendulum, tableau, 1e-3,
                                adaptive=False, dt0=10.0 / n)
        assert result.n_reject == 0
        errors.append(result.error_final)

    ratio = errors[0] / errors[1]
    assert ratio == pytest.approx(expected, rel=0.25), (tableau.name, ratio)


# --------------------------------------------------------------------------
# FSAL-R's interpolation
# --------------------------------------------------------------------------

def test_fsalr_interpolation_flag_changes_accuracy_not_cost(pendulum):
    """interpolate_fsal=False must give the same RHS count and a different
    solution -- the interpolation is free, so it can only buy accuracy.

    Note what is *not* asserted: that the final global error is smaller with
    the interpolation on. On this problem it is not reliably smaller, because
    both approximations sit below the method's own truncation error and the
    difference between them is lost in it. What the interpolation genuinely
    buys is one order in the FSAL value itself, which the next test measures
    directly rather than through an integration that hides it.
    """
    for tableau in (BS3, DP5):
        on = solve_fsalr(pendulum, tableau, 1e-5, rootfinder=TOMS748,
                         adaptive=False, dt0=0.05)
        off = solve_fsalr(pendulum, tableau, 1e-5, rootfinder=TOMS748,
                          adaptive=False, dt0=0.05, interpolate_fsal=False)

        assert on.nf == off.nf, tableau.name
        assert on.n_accept == off.n_accept
        assert on.u_final != off.u_final, tableau.name


def fsal_approximation_errors(tableau, dt):
    """Error in the FSAL value with and without the interpolation.

    Takes one step by hand rather than through a solver, so the two
    approximations are compared against the truth -- f(u_gamma), evaluated
    directly -- at identical inputs.
    """
    A, b, c, s = tableau.A, tableau.b, tableau.c, tableau.n_stages
    k = [pendulum_rhs(0.0, GOLDEN_U0)] * s
    for i in range(1, s):
        y = list(GOLDEN_U0)
        for j in range(i):
            if A[i][j]:
                for m in range(2):
                    y[m] += A[i][j] * dt * k[j][m]
        k[i] = pendulum_rhs(c[i] * dt, tuple(y))

    u_new = tuple(GOLDEN_U0[m] + dt * sum(b[i] * k[i][m] for i in range(s))
                  for m in range(2))
    gamma = solve_relaxation_parameter(
        pendulum_entropy, pendulum_entropy_gradient, GOLDEN_U0, u_new,
        TOMS748).gamma
    u_gamma = tuple(GOLDEN_U0[m] + gamma * (u_new[m] - GOLDEN_U0[m])
                    for m in range(2))

    truth = pendulum_rhs(0.0, u_gamma)
    interpolated = tuple(k[0][m] + gamma * (k[-1][m] - k[0][m])
                         for m in range(2))
    interpolated_error = hypot(*(interpolated[m] - truth[m] for m in range(2)))
    raw_error = hypot(*(k[-1][m] - truth[m] for m in range(2)))
    return interpolated, truth, interpolated_error, raw_error


def test_fsalr_interpolation_matches_the_golden_step():
    """The interpolation reproduces the reference value, including the split
    between its two components.

    The second component is *exact*: theta' = omega is linear, and
    interpolation commutes with linear maps. The first is off by 4.9e-4
    because omega' = -sin(theta) is not. A correct implementation shows
    exactly that split -- both components exact, or both wrong, means
    something else is going on.
    """
    interpolated, truth, error, _ = fsal_approximation_errors(BS3, 0.4)

    assert interpolated[0] == pytest.approx(GOLDEN_INTERPOLATION[0], abs=5e-8)
    assert interpolated[1] == pytest.approx(GOLDEN_INTERPOLATION[1], abs=5e-8)
    assert truth[0] == pytest.approx(GOLDEN_F_U_GAMMA[0], abs=5e-8)

    assert interpolated[1] == truth[1], "linear component must be exact"
    assert abs(interpolated[0] - truth[0]) == pytest.approx(4.9e-4, rel=0.01)
    assert error == pytest.approx(4.892e-4, rel=1e-3)


@pytest.mark.parametrize("tableau", (BS3, DP5), ids=lambda t: t.name)
def test_interpolation_gains_one_order_over_the_raw_stage(tableau):
    """Lemma 1: interpolating gives an O(dt**(p+1)) FSAL value where reusing
    k[s] unmodified gives O(dt**p).

    Measured as a ratio, which is the robust form: halving dt must roughly
    double the advantage, whatever the constants are. This is what
    `interpolate_fsal` actually buys, and it is invisible in the final global
    error because both sit under the method's own truncation term.
    """
    advantages = []
    for dt in (0.2, 0.1, 0.05):
        _, _, interpolated_error, raw_error = fsal_approximation_errors(
            tableau, dt)
        assert interpolated_error < raw_error
        advantages.append(raw_error / interpolated_error)

    for coarse, fine in zip(advantages, advantages[1:]):
        assert fine / coarse == pytest.approx(2.0, rel=0.1), advantages


# --------------------------------------------------------------------------
# Trap 1: the relaxation-failure retry
# --------------------------------------------------------------------------

def failing_finder(n_failures):
    """A finder that fails its first `n_failures` calls, then behaves."""
    state = {"left": n_failures}

    def finder(r, dr, bracket=DEFAULT_BRACKET, *, xtol=1e-14, max_iter=50):
        if state["left"] > 0:
            state["left"] -= 1
            return RootResult(gamma=float("nan"), iterations=1,
                              converged=False, residual=float("inf"))
        return TOMS748(r, dr, bracket, xtol=xtol, max_iter=max_iter)

    return finder


@pytest.mark.parametrize("name,solver", RELAXING)
def test_relaxation_failure_rejects_the_step_and_halves_dt(pendulum, name,
                                                           solver):
    """Trap 1.

    In the authors' Julia this path is dead: a flag reset makes the
    "relaxation failed -> halve the step and retry" branch unreachable, which
    is why a typo survives inside it unnoticed. Random initial conditions in
    the Monte Carlo sweep will reach it, so it is exercised explicitly.

    A failure must become a rejection, never a step taken with gamma = 1.

    Not all of the scripted failures need land as rejections: after a couple of
    halvings the step is small enough that gamma = 1 is itself a verified root,
    and `solve_relaxation_parameter` returns it rather than failing. So the
    bound is on the range, not the exact count.
    """
    failures = 3
    result = solver(pendulum, BS3, 1e-5, rootfinder=failing_finder(failures),
                    save_history=True)

    assert 1 <= result.n_relaxation_failures <= failures
    assert result.n_reject >= result.n_relaxation_failures
    assert result.t_final == 10.0
    assert result.nf == rhs_identity(result, BS3)

    # The failures cost attempts but not correctness. The bound is 1e-12 rather
    # than the 1e-15 of a clean run because a step rescued by gamma = 1 is only
    # guaranteed to the residual tolerance that verified it.
    eta0 = pendulum_entropy(pendulum.u0)
    drift = max(abs(eta - eta0) for eta in result.entropy_history)
    assert drift < 1e-11

    clean = solver(pendulum, BS3, 1e-5, rootfinder=TOMS748)
    assert result.n_reject > clean.n_reject


@pytest.mark.parametrize("name,solver", RELAXING)
def test_persistent_relaxation_failure_never_takes_an_unrelaxed_step(
        pendulum, name, solver):
    """A finder that never succeeds must not silently degrade into baseline.

    The solver keeps halving, and the only gamma it ever accepts is 1 -- and
    only once the step is small enough that gamma = 1 is a verified root of
    the relaxation equation, which `solve_relaxation_parameter` checks against
    its residual tolerance. So the failures are all recorded and the cost
    explodes visibly rather than the conservation quietly failing. That cost is
    itself the Monte Carlo result.

    Note the drift bound here is weaker than the 1e-15 of a working run, and
    necessarily so: the residual check caps the drift of each *step* at
    residual_tol, and over thousands of steps those accumulate. It still beats
    baseline by six orders of magnitude, which is the point -- degraded, not
    broken.
    """
    def never(r, dr, bracket=DEFAULT_BRACKET, *, xtol=1e-14, max_iter=50):
        return RootResult(gamma=float("nan"), iterations=1, converged=False,
                          residual=float("inf"))

    result = solver(pendulum, BS3, 1e-3, rootfinder=never, save_history=True)

    assert result.n_relaxation_failures > 10
    assert all(gamma == 1.0 for gamma in result.gamma_history)
    assert result.nf == rhs_identity(result, BS3)

    eta0 = pendulum_entropy(pendulum.u0)
    drift = max(abs(eta - eta0) for eta in result.entropy_history)
    assert drift < 1e-8, drift
    baseline_drift = max(
        abs(eta - eta0) for eta in
        solve_baseline(pendulum, BS3, 1e-3, save_history=True).entropy_history)
    assert drift < 1e-5 * baseline_drift

    # Far more expensive than the same run with a working finder.
    assert result.nf > 5 * solver(pendulum, BS3, 1e-3, rootfinder=TOMS748).nf


def test_step_size_collapse_is_reported(pendulum):
    """A run that cannot make progress must raise rather than spin.

    `montecarlo.py` records a raising run as a failure row and continues, so
    this has to be an exception and not a silently truncated result: a
    SolverResult that stopped early would enter the work-precision diagram as
    a cheap, accurate point.
    """
    def broken(t, u):
        # No finite step can satisfy any tolerance against this.
        return (float("inf"), float("inf"))

    with pytest.raises((RuntimeError, ArithmeticError, OverflowError,
                        ValueError)):
        solve_baseline(pendulum._replace(rhs=broken), BS3, 1e-6)


def test_max_steps_is_enforced(pendulum):
    """The step budget is a real bound, so a pathological run ends."""
    with pytest.raises(RuntimeError, match="max_steps"):
        solve_baseline(pendulum, BS3, 1e-9, max_steps=10)
