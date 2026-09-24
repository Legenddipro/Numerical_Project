"""Tests for src/tableaus.py. Owner: MD. Abir Hossain.

The cheapest place in the whole project to catch a mistyped coefficient. A
single wrong digit here otherwise surfaces much later as a failed
convergence-order check, where the cause is far harder to locate.

Every order-condition check runs in exact rational arithmetic. These are
identities, not approximations: ``sum(b*c**2) == 1/3`` is either true or false,
and checking it against a float tolerance would turn a structural fact into a
judgement call. It matters most for the conditions that must fail *exactly* --
the embedded weights miss the highest order on purpose.
"""

from fractions import Fraction

import pytest

from src.tableaus import (
    BS3,
    BS3_EXACT,
    DP5,
    DP5_EXACT,
    EXACT,
    ORDER_CONDITION_TARGETS,
    TABLEAUS,
    order_conditions,
    pid_gains,
    validate_tableau,
)


def test_bs3_satisfies_tableau_identities():
    """sum(b)==1, c[i]==sum(A[i]), A[-1]==b, c[-1]==1, b[-1]==0."""
    validate_tableau(BS3)


def test_dp5_satisfies_tableau_identities():
    """Same identities as BS3."""
    validate_tableau(DP5)


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_embedded_weights_sum_to_one(name):
    """Consistency of the embedded method: sum(b_embedded) == 1.

    Exact, not approximate: the rationals are the source of truth and they sum
    to one over the integers.
    """
    exact = EXACT[name]
    assert sum(exact.b_embedded) == Fraction(1)
    assert sum(TABLEAUS[name].b_embedded) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_embedded_uses_final_stage(name):
    """b_embedded[-1] != 0 even though b[-1] == 0.

    This asymmetry is the whole reason the last stage is computed at all, and
    the reason R-FSAL can skip it for the main solution but still needs a
    substitute for the embedded one.
    """
    tableau = TABLEAUS[name]
    assert tableau.b[-1] == 0.0
    assert tableau.b_embedded[-1] != 0.0

    exact = EXACT[name]
    assert exact.b[-1] == 0
    assert exact.b_embedded[-1] != 0


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_fsal_structure(name):
    """A[-1] == b and c[-1] == 1, exactly, in both tableaus.

    These two are what make the final stage equal ``f`` at the new solution
    point, which is the entire basis of FSAL reuse. Asserted on the rationals
    because ``b`` is *defined* as the last row of ``A``, so any difference would
    mean the construction itself was edited wrongly.
    """
    exact = EXACT[name]
    assert exact.A[-1] == exact.b
    assert exact.c[-1] == 1

    tableau = TABLEAUS[name]
    assert tableau.A[-1] == tableau.b
    assert tableau.c[-1] == 1.0


def test_bs3_main_weights_are_order_three_exactly():
    """All four order conditions hold for ``b``, and order 4 fails.

    Both halves matter: the first says BS3 is order 3, the second says it is not
    accidentally order 4, which would mean the coefficients belong to some other
    method than the one the paper used.
    """
    got = order_conditions(BS3_EXACT.A, BS3_EXACT.b, BS3_EXACT.c)

    assert got["sum_b"] == Fraction(1)
    assert got["sum_b_c"] == Fraction(1, 2)
    assert got["sum_b_c2"] == Fraction(1, 3)
    assert got["sum_b_A_c"] == Fraction(1, 6)

    failing_orders = {ORDER_CONDITION_TARGETS[k][0]
                      for k, v in got.items() if v != ORDER_CONDITION_TARGETS[k][1]}
    assert min(failing_orders) == 4, (
        f"BS3 should first fail at order 4, failed at {sorted(failing_orders)}")


def test_bs3_embedded_weights_fail_the_order_three_conditions():
    """The two failures that make the embedded method order 2.

    ``sum(b*c**2)`` comes out 3/8 rather than 1/3 and ``sum(b*A*c)`` 3/16 rather
    than 1/6. These are the sharpest checks in this file. They are not bugs: if
    the embedded weights satisfied the order-3 conditions too, the two solutions
    would agree to order 3 and their difference would estimate nothing, leaving
    the adaptive controller driving on noise.
    """
    got = order_conditions(BS3_EXACT.A, BS3_EXACT.b_embedded, BS3_EXACT.c)

    # Order 1 and 2 still hold -- the embedded method is order 2, not order 0.
    assert got["sum_b"] == Fraction(1)
    assert got["sum_b_c"] == Fraction(1, 2)

    # And the order-3 pair must fail, at exactly these values.
    assert got["sum_b_c2"] == Fraction(3, 8) != Fraction(1, 3)
    assert got["sum_b_A_c"] == Fraction(3, 16) != Fraction(1, 6)


def test_dp5_main_weights_satisfy_conditions_through_order_four():
    """DP5's propagated weights satisfy every condition on record.

    Complete through order 4 plus one order-5 condition -- see
    `ORDER_CONDITION_TARGETS` on why the order-5 family is only sampled. A
    mistyped digit anywhere in the 7x7 ``A`` breaks at least one of these eight
    identities, which is what this test is really for.
    """
    got = order_conditions(DP5_EXACT.A, DP5_EXACT.b, DP5_EXACT.c)
    for key, (_order, required) in ORDER_CONDITION_TARGETS.items():
        assert got[key] == required, f"DP5 b fails {key}: {got[key]} != {required}"


def test_dp5_embedded_weights_are_order_four_not_five():
    """DP5's embedded pair is order 4, so it misses only at order 5.

    Unlike BS3, the four conditions in the plan's table do not separate DP5's
    two weight vectors at all -- both satisfy them. The discriminating condition
    is ``sum(b*c**4)``, which the embedded weights miss.
    """
    got = order_conditions(DP5_EXACT.A, DP5_EXACT.b_embedded, DP5_EXACT.c)

    for key, (order, required) in ORDER_CONDITION_TARGETS.items():
        if order <= 4:
            assert got[key] == required, (
                f"DP5 b_embedded should be order 4 but fails {key}")

    assert got["sum_b_c4"] != Fraction(1, 5)


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_float_tableau_matches_its_exact_source(name):
    """The floats solvers use are the rationals, converted -- nothing retyped."""
    exact = EXACT[name]
    tableau = TABLEAUS[name]

    assert tableau.name == exact.name
    assert tableau.order == exact.order
    assert tableau.n_stages == exact.n_stages
    assert tableau.b == tuple(float(x) for x in exact.b)
    assert tableau.b_embedded == tuple(float(x) for x in exact.b_embedded)
    assert tableau.c == tuple(float(x) for x in exact.c)
    assert tableau.A == tuple(tuple(float(a) for a in row) for row in exact.A)


def test_validate_tableau_rejects_corrupted_coefficients():
    """Perturb one coefficient; validate_tableau must raise."""
    # Consistency broken: sum(b) != 1.
    broken_b = BS3._replace(b=(BS3.b[0] + 1e-6,) + BS3.b[1:])
    with pytest.raises(AssertionError, match="sum"):
        validate_tableau(broken_b)

    # Row-sum condition broken: A's last row no longer sums to c[-1] == 1.
    bad_row = (BS3.A[-1][0] + 1e-6,) + BS3.A[-1][1:]
    broken_A = BS3._replace(A=BS3.A[:-1] + (bad_row,))
    with pytest.raises(AssertionError):
        validate_tableau(broken_A)

    # FSAL broken: b no longer equals A's last row, so the last stage is not f
    # at the new solution point and reusing it as the next first stage is wrong.
    # The two entries are swapped rather than perturbed, so sum(b) is still 1 --
    # otherwise the consistency check would fire first and this identity would
    # never be reached.
    swapped_b = (BS3.b[1], BS3.b[0]) + BS3.b[2:]
    with pytest.raises(AssertionError, match="FSAL"):
        validate_tableau(BS3._replace(b=swapped_b))

    # c[-1] != 1 puts the last stage somewhere other than the step end. Caught
    # by the row-sum condition before the FSAL one, since the two disagree now.
    broken_c = BS3._replace(c=BS3.c[:-1] + (0.9,))
    with pytest.raises(AssertionError):
        validate_tableau(broken_c)

    # Explicitness broken: a nonzero entry on the diagonal. Solver stage loops
    # would silently drop it rather than report an unsupported method.
    implicit_row = (BS3.A[1][0], 0.25, 0.0, 0.0)
    broken_explicit = BS3._replace(A=BS3.A[:1] + (implicit_row,) + BS3.A[2:])
    with pytest.raises(AssertionError, match="not explicit"):
        validate_tableau(broken_explicit)

    # The embedded method must actually use the final stage.
    broken_embedded = BS3._replace(
        b_embedded=BS3.b_embedded[:-1] + (0.0,))
    with pytest.raises(AssertionError):
        validate_tableau(broken_embedded)


def test_pid_gains_match_reference_implementation():
    """BS3 -> (0.6, -0.2, 0.0); DP5 -> (0.7, -0.4, 0.0)."""
    assert pid_gains(BS3) == (0.6, -0.2, 0.0)
    assert pid_gains(DP5) == (0.7, -0.4, 0.0)


def test_pid_gains_refuses_an_unknown_tableau():
    """No silent default: a substituted gain would break the guarantee that all
    four methods share identical step-size logic, and would do it invisibly."""
    with pytest.raises(KeyError):
        pid_gains(BS3._replace(name="RK4"))


def test_weights_reproduce_the_golden_step():
    """Gate-adjacent check: BS3's weights must reproduce the reference step.

    From u0 = (1.5, 1.0) with dt = 0.4 and g/L = 1, the stage recurrence built
    from ``A`` and ``c`` must give the tabulated k values, ``b`` must give u1 and
    ``b_embedded`` must give the embedded solution. This exercises the
    coefficients against numbers computed independently of this code, which the
    order conditions -- being internal consistency checks -- cannot do.
    """
    from src.problems import pendulum_rhs

    u0 = (1.5, 1.0)
    dt = 0.4
    s = BS3.n_stages

    k = [pendulum_rhs(0.0, u0)]
    for i in range(1, s):
        y = tuple(u0[m] + dt * sum(BS3.A[i][j] * k[j][m] for j in range(i))
                  for m in range(len(u0)))
        k.append(pendulum_rhs(0.0 + BS3.c[i] * dt, y))

    expected_k = [(-0.8414710, 1.5000000),
                  (-0.9635582, 1.3317058),
                  (-0.9853666, 1.2109325),
                  (-0.9990045, 1.1215519)]
    for i, (got, want) in enumerate(zip(k, expected_k)):
        assert got[0] == pytest.approx(want[0], abs=1e-7), f"k[{i}] omega"
        assert got[1] == pytest.approx(want[1], abs=1e-7), f"k[{i}] theta"

    u1 = tuple(u0[m] + dt * sum(BS3.b[i] * k[i][m] for i in range(s))
               for m in range(len(u0)))
    assert u1[0] == pytest.approx(1.1215519, abs=1e-7)
    assert u1[1] == pytest.approx(1.5261710, abs=1e-7)

    u_emb = tuple(u0[m] + dt * sum(BS3.b_embedded[i] * k[i][m] for i in range(s))
                  for m in range(len(u0)))
    assert u_emb[0] == pytest.approx(1.1241401, abs=1e-7)
    assert u_emb[1] == pytest.approx(1.5257058, abs=1e-7)

    # The last stage is f at the new solution point -- the FSAL property, here
    # as a number rather than as an identity on the coefficients.
    assert k[-1] == pytest.approx(pendulum_rhs(dt, u1), abs=1e-14)
