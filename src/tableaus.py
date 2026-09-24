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

Owner: MD. Abir Hossain.
"""

from .contracts import Tableau

# Populated with the literal coefficients. Declared here rather than built by a
# function so that they are importable constants and trivially inspectable.
BS3: Tableau
DP5: Tableau

# Name -> tableau, so sweeps can iterate over integrators by string.
TABLEAUS: dict[str, Tableau]


def pid_gains(tableau: Tableau) -> tuple[float, float, float]:
    """PID controller coefficients (beta1, beta2, beta3) for this tableau.

    Taken from the authors' ``default_controller``: BS3 uses (0.6, -0.2, 0.0)
    and DP5 uses (0.7, -0.4, 0.0). A zero third coefficient makes the
    controller effectively PI, which is what these two methods use.

    Returns
    -------
    tuple[float, float, float]
    """
    raise NotImplementedError


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
    raise NotImplementedError
