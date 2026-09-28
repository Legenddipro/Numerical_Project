"""Tests for src/solvers/rfsal.py. Owner: Asikur Rahman.

Module-level tests, plus the parts of gates G3 and G4 that R-FSAL can establish
on its own. Gate G5 (``nf`` against the Julia reference) needs
``tests/data/julia_reference.csv`` from step 4, which was never produced, so it
is skipped here for that reason and no other -- see the step-7 addendum in
docs/PLAN_SEQUENTIAL.md.
"""

from functools import partial
from itertools import product
from math import cos, sin
from pathlib import Path

import pytest

from src.contracts import Problem
from src.problems import make_pendulum, pendulum_rhs
from src.rootfind import ROOT_FINDERS
from src.solvers.classic import solve_baseline, solve_fsalr
from src.solvers.rfsal import extrapolate_last_stage, solve_rfsal
from src.tableaus import BS3, DP5

TOMS748 = ROOT_FINDERS["toms748"]

# The golden step (docs/ALGORITHMS.md). Literals, so these tests check the
# solver rather than agreeing with it.
GOLDEN_U0 = (1.5, 1.0)
GOLDEN_K1 = (-0.8414710, 1.5000000)
GOLDEN_U_GAMMA = (1.1201761, 1.5280838)
GOLDEN_GAMMA = 1.0036353183
GOLDEN_F_U_GAMMA = (-0.9990880, 1.1201761)
GOLDEN_EXTRAPOLATION = (-0.9985170, 1.1215519)
GOLDEN_WRONG_FACTOR = (-0.9996609, 1.1187953)   # what gamma instead of 1/gamma gives

JULIA_TABLE = Path(__file__).parent / "data" / "julia_reference.csv"


@pytest.fixture(scope="module")
def pendulum():
    return make_pendulum(t_span=(0.0, 10.0))


@pytest.fixture(scope="module")
def short_pendulum():
    return make_pendulum(t_span=(0.0, 0.4))


def harmonic_oscillator(t_span=(0.0, 10.0)) -> Problem:
    """omega' = -theta, theta' = omega: the linearised pendulum, exact solution
    known in closed form. Component order (omega, theta) as everywhere else."""
    w0, th0 = GOLDEN_U0

    def rhs(t, u):
        return (-u[1], u[0])

    def reference(t):
        return (w0 * cos(t) - th0 * sin(t), th0 * cos(t) + w0 * sin(t))

    return Problem(
        name="harmonic_oscillator",
        rhs=rhs,
        entropy=lambda u: 0.5 * (u[0] * u[0] + u[1] * u[1]),
        entropy_gradient=lambda u: (u[0], u[1]),
        u0=GOLDEN_U0,
        t_span=t_span,
        reference=reference,
    )


class CallSpy:
    """Wraps an rhs, recording every point it is evaluated at."""

    def __init__(self, rhs):
        self.rhs = rhs
        self.points = []

    def __call__(self, t, u):
        self.points.append((t, tuple(u)))
        return self.rhs(t, u)


def with_spy(problem):
    spy = CallSpy(problem.rhs)
    return problem._replace(rhs=spy), spy


def rfsal_identity(result, tableau, c0=2):
    """Gate G3 for R-FSAL, including DEVIATION 3's correction term."""
    s = tableau.n_stages
    return (c0 + (s - 1) * (result.n_accept + result.n_reject)
            - result.n_relaxation_failures)


# --------------------------------------------------------------------------
# The golden step
# --------------------------------------------------------------------------

def test_extrapolation_matches_golden_value():
    """The sharpest check in the golden table: the 1/gamma extrapolation.

    The wrong factor gives a value different in the third decimal, so this
    discriminates the gamma / 1/gamma swap outright.
    """
    got = extrapolate_last_stage(GOLDEN_K1, GOLDEN_F_U_GAMMA, GOLDEN_GAMMA)
    assert got == pytest.approx(GOLDEN_EXTRAPOLATION, abs=5e-7)
    assert got[0] != pytest.approx(GOLDEN_WRONG_FACTOR[0], abs=1e-5)


def test_golden_step(short_pendulum):
    """One fixed BS3 step lands on the golden u_gamma and uses 4 RHS calls:
    1 at startup (dt supplied), 2 stages, 1 at the relaxed point."""
    problem, spy = with_spy(short_pendulum)
    result = solve_rfsal(problem, BS3, 1e-3, rootfinder=TOMS748,
                         adaptive=False, dt0=0.4, save_history=True)

    assert result.u_final == pytest.approx(GOLDEN_U_GAMMA, abs=5e-8)
    assert result.gamma_history[0] == pytest.approx(GOLDEN_GAMMA, abs=1e-10)
    assert result.nf == 4 == len(spy.points)
    # Final step lands on t_end exactly (relaxation_at_last_step = true).
    assert result.t_final == 0.4


def test_relaxation_precedes_error_estimate(short_pendulum):
    """Verified by construction: the only post-stage RHS call is at the
    *relaxed* point, and f is never evaluated at the unrelaxed u_np1.

    If relaxation ran after the error test, the loop would need f(u_np1) for
    the embedded solution before gamma existed -- a call at u_np1 would appear.
    """
    problem, spy = with_spy(short_pendulum)
    solve_rfsal(problem, BS3, 1e-3, rootfinder=TOMS748, adaptive=False, dt0=0.4)

    last_point = spy.points[-1][1]
    assert last_point == pytest.approx(GOLDEN_U_GAMMA, abs=5e-8)
    u_new = (1.1215519, 1.5261710)
    for _, point in spy.points:
        assert abs(point[0] - u_new[0]) > 1e-6 or abs(point[1] - u_new[1]) > 1e-6


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tableau", [BS3, DP5], ids=lambda tb: tb.name)
def test_final_stage_is_never_computed_directly(short_pendulum, tableau):
    """(s-2) stage evaluations plus one at the relaxed point, per step."""
    problem, spy = with_spy(short_pendulum)
    result = solve_rfsal(problem, tableau, 1e-3, rootfinder=TOMS748,
                         adaptive=False, dt0=0.4)
    assert result.n_accept == 1
    assert result.nf == len(spy.points) == 1 + (tableau.n_stages - 2) + 1


def test_fsal_cache_is_exact_not_approximated():
    """The value carried into the next step is f at the relaxed point.

    Two fixed steps: step 2 makes exactly s-2 stage calls, so its first stage
    came from the cache -- and the cache was filled by a genuine call at the
    point step 2 starts from. That is R-FSAL's structural difference from
    FSAL-R, whose cache is interpolated instead.
    """
    problem, spy = with_spy(make_pendulum(t_span=(0.0, 10.0)))
    result = solve_rfsal(problem, BS3, 1e-3, rootfinder=TOMS748,
                         adaptive=False, dt0=0.4, save_history=True)
    # startup, then per step: 2 stages + 1 relaxed call
    assert len(spy.points) == 1 + 3 * result.n_accept

    first_relaxed_t, first_relaxed_u = spy.points[3]
    assert first_relaxed_u == pytest.approx(GOLDEN_U_GAMMA, abs=5e-8)
    # Step 2 starts where step 1's relaxed call was made -- time included.
    assert result.t_history[1] == first_relaxed_t
    # And its second stage is built from exactly that cache value.
    k1 = pendulum_rhs(0.0, first_relaxed_u)
    c2, a21 = BS3.c[1], BS3.A[1][0]
    dt = 0.4
    expected_y = tuple(u + dt * a21 * k for u, k in zip(first_relaxed_u, k1))
    assert spy.points[4][1] == pytest.approx(expected_y, abs=1e-15)
    assert spy.points[4][0] == pytest.approx(first_relaxed_t + c2 * dt, abs=1e-15)


def test_extrapolation_is_exact_for_linear_rhs():
    """On a linear f, the 1/gamma extrapolation reproduces f(u_np1) exactly.

    Uses a real BS3 step on the harmonic oscillator so gamma is the genuine
    relaxation parameter rather than a made-up number.
    """
    problem = harmonic_oscillator()
    f, u0, dt = problem.rhs, problem.u0, 0.4

    k = [f(0.0, u0)]
    for i in range(1, BS3.n_stages - 1):
        y = tuple(u0[m] + dt * sum(BS3.A[i][j] * k[j][m] for j in range(i))
                  for m in range(2))
        k.append(f(BS3.c[i] * dt, y))
    u_new = tuple(u0[m] + dt * sum(BS3.b[i] * k[i][m] for i in range(3))
                  for m in range(2))

    from src.rootfind import solve_relaxation_parameter
    gamma = solve_relaxation_parameter(problem.entropy, problem.entropy_gradient,
                                       u0, u_new, TOMS748).gamma
    assert gamma != 1.0          # otherwise the check could not tell gamma from 1/gamma
    u_gamma = tuple(a + gamma * (z - a) for a, z in zip(u0, u_new))

    extrapolated = extrapolate_last_stage(k[0], f(0.0, u_gamma), gamma)
    exact = f(0.0, u_new)
    assert extrapolated == pytest.approx(exact, rel=0, abs=1e-15)

    wrong = tuple(a + gamma * (z - a) for a, z in zip(k[0], f(0.0, u_gamma)))
    assert abs(wrong[0] - exact[0]) > 1e-4


def test_harmonic_oscillator_run_matches_closed_form():
    """End-to-end on a problem with a closed-form solution, independent of the
    DOP853 reference."""
    result = solve_rfsal(harmonic_oscillator(), BS3, 1e-8, rootfinder=TOMS748)
    assert result.error_final < 1e-6
    assert result.t_final == 10.0


@pytest.mark.parametrize("flag", ["interpolate_fsal", "relax_embedded", "relax_main"])
def test_each_variant_switch_changes_results(pendulum, flag):
    """Each switch changes results on its own."""
    default = solve_rfsal(pendulum, BS3, 1e-5, rootfinder=TOMS748)
    flipped = solve_rfsal(pendulum, BS3, 1e-5, rootfinder=TOMS748, **{flag: False})
    assert (flipped.error_final, flipped.nf) != (default.error_final, default.nf)


def test_all_eight_variant_combinations_are_distinct(pendulum):
    """All 2**3 combinations are reachable and give eight different runs, so
    the authors' variant study can be reproduced."""
    outcomes = set()
    for interp, emb, main in product([True, False], repeat=3):
        r = solve_rfsal(pendulum, BS3, 1e-5, rootfinder=TOMS748,
                        interpolate_fsal=interp, relax_embedded=emb,
                        relax_main=main)
        assert r.nf == rfsal_identity(r, BS3)
        outcomes.add((r.nf, r.error_final))
    assert len(outcomes) == 8


# --------------------------------------------------------------------------
# Gates G3 and G4, R-FSAL's share
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tableau", [BS3, DP5], ids=lambda tb: tb.name)
@pytest.mark.parametrize("tol", [1e-3, 1e-5, 1e-7, 1e-9])
@pytest.mark.parametrize("finder", sorted(ROOT_FINDERS))
def test_rhs_identity(pendulum, tableau, tol, finder):
    """Gate G3, exact. ADDENDUM C: on the authors' initial condition relaxation
    never fails, so the correction term must be zero and the identity collapses
    to the baseline form."""
    r = solve_rfsal(pendulum, tableau, tol, rootfinder=ROOT_FINDERS[finder])
    assert r.n_relaxation_failures == 0
    assert r.nf == rfsal_identity(r, tableau)


def test_reported_nf_equals_actual_rhs_calls(pendulum):
    problem, spy = with_spy(pendulum)
    for tableau in (BS3, DP5):
        spy.points.clear()
        r = solve_rfsal(problem, tableau, 1e-6, rootfinder=TOMS748)
        assert r.nf == len(spy.points)


def test_relaxation_failure_rejects_and_halves_step(pendulum):
    """DEVIATION 2/3, with a finder that always fails.

    Every attempt whose r(1) is not already ~0 must be rejected and dt halved;
    it must never proceed with gamma = 1 on the finder's say-so. Halving
    continues until the step is small enough that gamma = 1 genuinely passes
    the residual check (ADDENDUM A) -- so the run completes, every accepted
    gamma is exactly 1, and each failed attempt cost s-2 calls rather than s-1.
    """
    from src.contracts import RootResult

    def failing(r, dr, bracket, *, xtol):
        return RootResult(gamma=float("nan"), iterations=1, converged=False,
                          residual=float("inf"))

    problem, spy = with_spy(pendulum)
    r = solve_rfsal(problem, BS3, 1e-3, rootfinder=failing, save_history=True)

    assert r.n_relaxation_failures > 0
    assert set(r.gamma_history) == {1.0}
    assert r.nf == rfsal_identity(r, BS3) == len(spy.points)
    # Each rescued step is only guaranteed to residual_tol = 1e-12.
    eta0 = r.entropy_history[0]
    drift = max(abs(e - eta0) for e in r.entropy_history)
    assert drift <= 1e-12 * r.n_accept


def test_step_size_collapse_raises():
    """A relaxation that can never succeed ends in RuntimeError, not a hang
    and not a silent gamma = 1. The 'invariant' is a step function that jumps
    by 1 as soon as omega moves at all, so r(1) = 1 however small dt gets and
    ADDENDUM A can never rescue the step."""
    from src.contracts import RootResult

    def failing(r, dr, bracket, *, xtol):
        return RootResult(gamma=float("nan"), iterations=1, converged=False,
                          residual=float("inf"))

    problem = harmonic_oscillator()._replace(
        rhs=lambda t, u: (1.0, 1.0),
        entropy=lambda u: 1.0 if u[0] > 1.5 else 0.0,
        entropy_gradient=lambda u: (0.0, 0.0))
    with pytest.raises(RuntimeError, match="collapsed"):
        solve_rfsal(problem, BS3, 1e-3, rootfinder=failing, max_steps=1000)


@pytest.mark.parametrize("tableau", [BS3, DP5], ids=lambda tb: tb.name)
def test_rfsal_conserves_invariant(pendulum, tableau):
    """Gate G4: energy held to ~1e-15 over the whole run."""
    r = solve_rfsal(pendulum, tableau, 1e-6, rootfinder=TOMS748, save_history=True)
    eta0 = r.entropy_history[0]
    assert max(abs(e - eta0) for e in r.entropy_history) < 1e-14


@pytest.mark.parametrize("tableau", [BS3, DP5], ids=lambda tb: tb.name)
@pytest.mark.parametrize("tol", [1e-3, 1e-6, 1e-9])
def test_same_cost_as_baseline_and_fsalr_class(pendulum, tableau, tol):
    """The paper's claim for R-FSAL: baseline cost, relaxed accuracy.

    Asserted as a per-attempt cost (s-1 RHS calls), not as nf equality with
    baseline -- the two take different step sequences, so their totals may
    legitimately differ by a few attempts.
    """
    r = solve_rfsal(pendulum, tableau, tol, rootfinder=TOMS748)
    b = solve_baseline(pendulum, tableau, tol)
    f = solve_fsalr(pendulum, tableau, tol, rootfinder=TOMS748)
    s = tableau.n_stages
    assert (r.nf - 2) / (r.n_accept + r.n_reject) == s - 1
    assert r.nf <= 1.1 * b.nf and r.nf <= 1.1 * f.nf


def test_extrapolation_order_at_fixed_step(pendulum):
    """ADDENDUM F: measure what the 1/gamma extrapolation buys directly.

    Against f(u_np1) computed exactly, the extrapolation error must shrink
    faster than the O(dt**p) gap left by using f(u_gamma) unmodified.
    """
    f = pendulum.rhs
    u0 = pendulum.u0
    from src.rootfind import solve_relaxation_parameter

    def errors(dt):
        k = [f(0.0, u0)]
        for i in range(1, BS3.n_stages - 1):
            y = tuple(u0[m] + dt * sum(BS3.A[i][j] * k[j][m] for j in range(i))
                      for m in range(2))
            k.append(f(BS3.c[i] * dt, y))
        u_new = tuple(u0[m] + dt * sum(BS3.b[i] * k[i][m] for i in range(3))
                      for m in range(2))
        gamma = solve_relaxation_parameter(pendulum.entropy,
                                           pendulum.entropy_gradient,
                                           u0, u_new, TOMS748).gamma
        u_gamma = tuple(a + gamma * (z - a) for a, z in zip(u0, u_new))
        f_gamma = f(0.0, u_gamma)
        exact = f(0.0, u_new)
        extr = extrapolate_last_stage(k[0], f_gamma, gamma)
        return (max(abs(a - b) for a, b in zip(extr, exact)),
                max(abs(a - b) for a, b in zip(f_gamma, exact)))

    e1, s1 = errors(0.2)
    e2, s2 = errors(0.1)
    assert e1 / e2 > 2 ** 3.5          # extrapolation: order p+1 = 4 in dt
    assert s1 / s2 < e1 / e2           # beats reusing f(u_gamma) unmodified


@pytest.mark.skipif(not JULIA_TABLE.exists(),
                    reason="gate G5: tests/data/julia_reference.csv (step 4) "
                           "was never produced")
def test_nf_matches_julia_reference(pendulum):
    import csv
    with JULIA_TABLE.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("method") == "rfsal"]
    assert rows, "no rfsal rows in julia_reference.csv"
    tableaus = {"BS3": BS3, "DP5": DP5}
    for row in rows:
        r = solve_rfsal(pendulum, tableaus[row["tableau"]], float(row["tol"]),
                        rootfinder=TOMS748)
        # Step 0: compare nf - c0 on both sides.
        assert r.nf - 2 == int(row["nf"]) - int(row.get("c0", 0))
