"""Butcher tableaus and matching PID controller gains.

Two embedded Runge-Kutta pairs, both with FSAL structure:

    BS3  Bogacki-Shampine 3(2), 4 stages, order 3
    DP5  Dormand-Prince 5(4),   7 stages, order 5

FSAL means the last row of ``A`` equals ``b`` and ``c[-1] == 1``, so the final
stage evaluates the right-hand side exactly at the new solution point and can
be reused as the next step's first stage. A consequence worth remembering:
``b[-1] == 0`` for any explicit FSAL method, so the last stage contributes
nothing to the propagated solution -- only to the embedded one. R-FSAL exploits
precisely this by skipping that stage entirely.

Coefficients are transcribed from the authors' ``ButcherTableau`` constructors.

Exact rationals are the single source of truth
----------------------------------------------
Every coefficient is written once, as a ``fractions.Fraction``, in `EXACT`. The
float tableaus solvers actually use are derived from those rationals by
`_to_float`. Nothing is typed twice, so the exact order-condition checks in
``tests/test_tableaus.py`` and the floats used in the hot loop cannot disagree:
a mistyped digit fails the exact check instead of quietly weakening the method.

Rationals are also what the order conditions require. ``sum(b*c**2) == 1/3`` is
either true or false; asking it of floats turns a structural identity into a
tolerance question. That matters most for the case that must *fail* exactly --
the embedded weights miss the two order-3 conditions, which is precisely why
the embedded solution is order 2 and its difference from the main solution
estimates anything at all.

Owner: MD. Abir Hossain.
"""

from fractions import Fraction
from typing import NamedTuple

from .contracts import Tableau

Rational = Fraction


class ExactTableau(NamedTuple):
    """A tableau in exact arithmetic, for verification only.

    Deliberately not a `contracts.Tableau`: that type is what solvers consume
    and its fields are floats. Keeping the two apart means no solver can end up
    doing `Fraction` arithmetic in its inner loop, which would be correct and
    unusably slow.
    """

    name: str
    A: tuple[tuple[Rational, ...], ...]
    b: tuple[Rational, ...]
    b_embedded: tuple[Rational, ...]
    c: tuple[Rational, ...]
    order: int
    n_stages: int


_F = Fraction

# Bogacki-Shampine 3(2), from the authors' ``ButcherTableau(alg::BS3)``:
#   A = [0 0 0 0; 1/2 0 0 0; 0 3/4 0 0; 2/9 1/3 4/9 0]
#   b = A[end, :];  bembd = [7/24, 1/4, 1/3, 1/8];  c = [0, 1/2, 3/4, 1]
_BS3_A: tuple[tuple[Rational, ...], ...] = (
    (_F(0), _F(0), _F(0), _F(0)),
    (_F(1, 2), _F(0), _F(0), _F(0)),
    (_F(0), _F(3, 4), _F(0), _F(0)),
    (_F(2, 9), _F(1, 3), _F(4, 9), _F(0)),
)

BS3_EXACT = ExactTableau(
    name="BS3",
    A=_BS3_A,
    # FSAL, so b is the last row of A. Taken from it rather than retyped: if the
    # two were written separately they could disagree, and that is exactly the
    # identity the FSAL reuse depends on.
    b=_BS3_A[-1],
    b_embedded=(_F(7, 24), _F(1, 4), _F(1, 3), _F(1, 8)),
    c=(_F(0), _F(1, 2), _F(3, 4), _F(1)),
    order=3,
    n_stages=4,
)

# Dormand-Prince 5(4), from the authors' ``ButcherTableau(alg::DP5)``.
_DP5_A: tuple[tuple[Rational, ...], ...] = (
    (_F(0), _F(0), _F(0), _F(0), _F(0), _F(0), _F(0)),
    (_F(1, 5), _F(0), _F(0), _F(0), _F(0), _F(0), _F(0)),
    (_F(3, 40), _F(9, 40), _F(0), _F(0), _F(0), _F(0), _F(0)),
    (_F(44, 45), _F(-56, 15), _F(32, 9), _F(0), _F(0), _F(0), _F(0)),
    (_F(19372, 6561), _F(-25360, 2187), _F(64448, 6561), _F(-212, 729),
     _F(0), _F(0), _F(0)),
    (_F(9017, 3168), _F(-355, 33), _F(46732, 5247), _F(49, 176),
     _F(-5103, 18656), _F(0), _F(0)),
    (_F(35, 384), _F(0), _F(500, 1113), _F(125, 192), _F(-2187, 6784),
     _F(11, 84), _F(0)),
)

DP5_EXACT = ExactTableau(
    name="DP5",
    A=_DP5_A,
    b=_DP5_A[-1],
    b_embedded=(_F(5179, 57600), _F(0), _F(7571, 16695), _F(393, 640),
                _F(-92097, 339200), _F(187, 2100), _F(1, 40)),
    c=(_F(0), _F(1, 5), _F(3, 10), _F(4, 5), _F(8, 9), _F(1), _F(1)),
    order=5,
    n_stages=7,
)

EXACT: dict[str, ExactTableau] = {"BS3": BS3_EXACT, "DP5": DP5_EXACT}
"""Exact-arithmetic tableaus, keyed by name. Used by the tests, not by solvers."""

# PID gains from the authors' ``default_controller``. A zero third coefficient
# makes the controller effectively PI, which is what both methods use.
_PID_GAINS: dict[str, tuple[float, float, float]] = {
    "BS3": (0.6, -0.2, 0.0),
    "DP5": (0.7, -0.4, 0.0),
}


def _to_float(exact: ExactTableau) -> Tableau:
    """Convert an exact tableau into the float form solvers consume."""
    return Tableau(
        name=exact.name,
        A=tuple(tuple(float(a) for a in row) for row in exact.A),
        b=tuple(float(x) for x in exact.b),
        b_embedded=tuple(float(x) for x in exact.b_embedded),
        c=tuple(float(x) for x in exact.c),
        order=exact.order,
        n_stages=exact.n_stages,
    )


BS3: Tableau = _to_float(BS3_EXACT)
DP5: Tableau = _to_float(DP5_EXACT)

TABLEAUS: dict[str, Tableau] = {"BS3": BS3, "DP5": DP5}
"""Name -> tableau, so sweeps can iterate over integrators by string."""


def pid_gains(tableau: Tableau) -> tuple[float, float, float]:
    """PID controller coefficients (beta1, beta2, beta3) for this tableau.

    Taken from the authors' ``default_controller``: BS3 uses (0.6, -0.2, 0.0)
    and DP5 uses (0.7, -0.4, 0.0). A zero third coefficient makes the
    controller effectively PI, which is what these two methods use.

    Returns
    -------
    tuple[float, float, float]

    Raises
    ------
    KeyError
        For a tableau with no gains on record. Falling back to a default would
        be worse than failing loudly: the controller is shared across all four
        methods precisely so step-size logic cannot differ between them, and a
        silently substituted gain would break that guarantee invisibly.
    """
    try:
        return _PID_GAINS[tableau.name]
    except KeyError:
        raise KeyError(
            f"no PID gains recorded for tableau {tableau.name!r}; "
            f"known tableaus are {sorted(_PID_GAINS)}") from None


ORDER_CONDITION_TARGETS: dict[str, tuple[int, Rational]] = {
    "sum_b": (1, _F(1)),
    "sum_b_c": (2, _F(1, 2)),
    "sum_b_c2": (3, _F(1, 3)),
    "sum_b_A_c": (3, _F(1, 6)),
    "sum_b_c3": (4, _F(1, 4)),
    "sum_b_c_A_c": (4, _F(1, 8)),
    "sum_b_A_c2": (4, _F(1, 12)),
    "sum_b_A_A_c": (4, _F(1, 24)),
    "sum_b_c4": (5, _F(1, 5)),
}
"""Order condition -> (the order it belongs to, its exact required value).

Complete through order 4. Order 5 has seventeen conditions and only one of them
is listed, which is enough for the job here: distinguishing order-5 propagated
weights from order-4 embedded ones. Satisfying every condition through order 4
plus this one is strong evidence of order 5, not a proof of it -- the proof is
the convergence-order measurement at gate G2, which tests the assembled method
rather than its coefficients.
"""


def order_conditions(A: tuple[tuple[Rational, ...], ...],
                     b: tuple[Rational, ...],
                     c: tuple[Rational, ...]) -> dict[str, Rational]:
    """Runge-Kutta order conditions in exact arithmetic, keyed as in
    `ORDER_CONDITION_TARGETS`.

    Returns the left-hand sides; the caller compares each against the required
    value. For a method of order p, every condition of order <= p must hold
    exactly and at least one of order p+1 must fail.

    Applied to `b`, all conditions up to the tableau's order hold. Applied to
    `b_embedded`, the conditions one order higher must **fail**:

    - BS3's embedded weights are order 2, so ``sum_b_c2`` comes out 3/8 instead
      of 1/3 and ``sum_b_A_c`` 3/16 instead of 1/6.
    - DP5's embedded weights are order 4, so they satisfy everything through
      order 4 and miss ``sum_b_c4``.

    Those failures are the point, not a defect. If the embedded weights matched
    the main ones at full order, the two solutions would agree too closely and
    their difference would estimate nothing.

    Takes loose coefficient arrays rather than a whole tableau so `b_embedded`
    can be passed in place of `b`: here the failing case is as important as the
    passing one.
    """
    n = len(b)
    zero = Fraction(0)
    rng = range(n)

    # A*c, the vector with entries sum_j A[i][j]*c[j]. Appears in three of the
    # conditions below, so it is formed once.
    Ac = tuple(sum((A[i][j] * c[j] for j in rng), zero) for i in rng)
    Ac2 = tuple(sum((A[i][j] * c[j] ** 2 for j in rng), zero) for i in rng)
    AAc = tuple(sum((A[i][j] * Ac[j] for j in rng), zero) for i in rng)

    return {
        "sum_b": sum(b, zero),
        "sum_b_c": sum((b[i] * c[i] for i in rng), zero),
        "sum_b_c2": sum((b[i] * c[i] ** 2 for i in rng), zero),
        "sum_b_A_c": sum((b[i] * Ac[i] for i in rng), zero),
        "sum_b_c3": sum((b[i] * c[i] ** 3 for i in rng), zero),
        "sum_b_c_A_c": sum((b[i] * c[i] * Ac[i] for i in rng), zero),
        "sum_b_A_c2": sum((b[i] * Ac2[i] for i in rng), zero),
        "sum_b_A_A_c": sum((b[i] * AAc[i] for i in rng), zero),
        "sum_b_c4": sum((b[i] * c[i] ** 4 for i in rng), zero),
    }


def validate_tableau(tableau: Tableau) -> None:
    """Check the structural identities every tableau here must satisfy.

    Verifies consistency (``sum(b) == 1``), the row-sum condition
    (``c[i] == sum(A[i])``), and the three FSAL conditions (``A[-1] == b``,
    ``c[-1] == 1``, ``b[-1] == 0``), all to floating-point tolerance.

    Called from the test suite rather than at import time. Catching a mistyped
    coefficient here is the cheapest possible place to catch it -- a single
    wrong digit otherwise surfaces much later as a failed convergence-order
    check at gate G2, where the cause is far harder to locate.

    Raises
    ------
    AssertionError
        Naming which identity failed and by how much.
    """
    name = tableau.name
    s = tableau.n_stages
    tol = 1e-14

    assert len(tableau.A) == s, f"{name}: len(A)={len(tableau.A)}, expected {s}"
    assert len(tableau.b) == s, f"{name}: len(b)={len(tableau.b)}, expected {s}"
    assert len(tableau.b_embedded) == s, (
        f"{name}: len(b_embedded)={len(tableau.b_embedded)}, expected {s}")
    assert len(tableau.c) == s, f"{name}: len(c)={len(tableau.c)}, expected {s}"
    for i, row in enumerate(tableau.A):
        assert len(row) == s, f"{name}: len(A[{i}])={len(row)}, expected {s}"

    # Strictly lower triangular, i.e. explicit. Every solver's stage loop reads
    # only k[j] for j < i, so a nonzero entry on or above the diagonal would be
    # dropped silently rather than reported as an unsupported method.
    for i, row in enumerate(tableau.A):
        for j in range(i, s):
            assert row[j] == 0.0, (
                f"{name}: A[{i}][{j}]={row[j]!r} is nonzero, so the method is "
                f"not explicit")

    residual = sum(tableau.b) - 1.0
    assert abs(residual) <= tol, f"{name}: sum(b)=1 fails by {residual:.3e}"

    residual = sum(tableau.b_embedded) - 1.0
    assert abs(residual) <= tol, (
        f"{name}: sum(b_embedded)=1 fails by {residual:.3e}")

    for i, row in enumerate(tableau.A):
        residual = sum(row) - tableau.c[i]
        assert abs(residual) <= tol, (
            f"{name}: row-sum condition c[{i}]=sum(A[{i}]) fails by "
            f"{residual:.3e}")

    for i, (a, w) in enumerate(zip(tableau.A[-1], tableau.b)):
        assert abs(a - w) <= tol, (
            f"{name}: FSAL condition A[-1]==b fails in component {i} by "
            f"{a - w:.3e}")

    residual = tableau.c[-1] - 1.0
    assert abs(residual) <= tol, (
        f"{name}: FSAL condition c[-1]==1 fails by {residual:.3e}")

    assert abs(tableau.b[-1]) <= tol, (
        f"{name}: FSAL condition b[-1]==0 fails by {tableau.b[-1]:.3e}")

    # Not a tolerance check but an exact one: were this weight zero, the
    # embedded solution would not need the final stage at all, and R-FSAL's
    # extrapolation -- which exists only to fill that slot -- would be solving a
    # problem that does not arise.
    assert tableau.b_embedded[-1] != 0.0, (
        f"{name}: b_embedded[-1] is zero, so the final stage never enters the "
        f"embedded solution and there is no error estimate to form")

    assert tableau.order >= 1, f"{name}: order={tableau.order}"
