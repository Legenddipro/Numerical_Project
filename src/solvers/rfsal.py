"""R-FSAL time-stepping loop.

Structurally distinct from the three variants in `classic.py`, which is why it
has its own module and its own owner rather than being a fourth branch in a
shared loop.

Owner: Tousif Fahmeed Quadir.
"""

from ..contracts import Problem, SolverResult, Tableau
from ..rootfind import DEFAULT_BRACKET, RootFinder


def solve_rfsal(problem: Problem,
                tableau: Tableau,
                tol: float,
                *,
                rootfinder: RootFinder,
                bracket: tuple[float, float] = DEFAULT_BRACKET,
                adaptive: bool = True,
                dt0: float = 0.0,
                save_history: bool = False,
                interpolate_fsal: bool = True,
                relax_embedded: bool = True,
                relax_main: bool = True,
                max_steps: int = 10_000_000) -> SolverResult:
    """R-FSAL: relax before the error estimate, then extrapolate.

    The step is reordered rather than patched at the end:

    1. Compute stages ``1 .. s-1`` only. Stage ``s`` is skipped, which is
       legitimate because ``b[-1] == 0`` for any explicit FSAL method, so that
       stage never contributed to the propagated solution anyway.
    2. Assemble ``u_np1`` from those stages and solve for gamma immediately,
       before any error test.
    3. Make the step's single right-hand-side evaluation at the *relaxed*
       point, giving ``f(u_gamma)``.
    4. Use that one value twice: exactly as the next step's cached first stage,
       and -- via extrapolation with factor ``1/gamma`` -- as the approximation
       of ``f(u_np1)`` that the embedded solution needs::

           f(u_np1) ~= f(u_gamma_prev) + (1/gamma) * (f(u_gamma) - f(u_gamma_prev))

    5. Build the embedded solution from relaxed quantities and apply error
       control.

    So the approximation sits in a different place than in FSAL-R. Here the
    FSAL reuse is *exact* and the error estimate is approximated; in FSAL-R it
    is the other way round. Theory (Lemma 2) bounds the extrapolation error at
    ``O(dt**(p+1))``, preserving the embedded method's order.

    The three boolean switches reproduce the authors' variant study:
    `interpolate_fsal` selects extrapolation versus using ``f(u_gamma)``
    unmodified, `relax_embedded` selects whether the embedded solution uses
    ``gamma*dt`` or plain ``dt``, and `relax_main` selects whether the error
    estimate compares against the relaxed or unrelaxed main solution. The
    defaults are the combination the paper recommends; the others are
    available so the comparison can be reproduced rather than taken on trust.

    Returns
    -------
    SolverResult
        `nf` must equal `solve_baseline` exactly. Gate G3.
    """
    raise NotImplementedError
