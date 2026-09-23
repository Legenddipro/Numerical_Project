"""Butcher tableaus and matching step-size controller gains.

Coefficients for the two embedded Runge-Kutta pairs used in the base paper:

    BS3  Bogacki-Shampine 3(2), 4 stages, FSAL
    DP5  Dormand-Prince 5(4),   7 stages, FSAL

Both satisfy the FSAL condition: the last row of A equals b, and c[-1] == 1, so
the final stage evaluates the right-hand side exactly at the new solution point.
A consequence worth remembering: b[-1] == 0 for any explicit FSAL method, so the
last stage never contributes to the propagated solution -- only to the embedded
one.

Interfaces to be fixed in the Phase 0 contract session.
"""
