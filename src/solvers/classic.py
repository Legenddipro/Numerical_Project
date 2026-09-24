"""Baseline, naive relaxation, and FSAL-R time-stepping loops.

These three share the great majority of their structure -- FSAL-R differs from
naive by only a handful of lines at the end of the step -- so they live in one
module under one owner. Splitting them across files would duplicate the shared
loop and invite the two copies to drift apart.

All four solver entry points in the project take the same arguments and return
`SolverResult`, so `montecarlo.py` can iterate over them uniformly.

Owner: MD. Shadman Shafie.
"""

from ..contracts import Problem, SolverResult, Tableau
from ..rootfind import DEFAULT_BRACKET, RootFinder


def solve_baseline(problem: Problem,
                   tableau: Tableau,
                   tol: float,
                   *,
                   rootfinder: RootFinder | None = None,
                   bracket: tuple[float, float] = DEFAULT_BRACKET,
                   adaptive: bool = True,
                   dt0: float = 0.0,
                   save_history: bool = False,
                   max_steps: int = 10_000_000) -> SolverResult:
    """Plain embedded Runge-Kutta with FSAL reuse. No relaxation.

    The reference point for everything else. The invariant drifts here, which
    is the problem relaxation exists to solve, and this method's RHS count is
    the cost the two new schemes must match.

    `rootfinder` is accepted and ignored, so that the signature matches the
    relaxation variants and sweeps need no special case.

    With ``adaptive=False`` and an explicit `dt0`, runs at fixed step size --
    the mode gate G2 uses to check convergence order.

    Returns
    -------
    SolverResult
        With ``n_root_iterations == 0`` and ``n_relaxation_failures == 0``.
    """
    raise NotImplementedError


def solve_naive(problem: Problem,
                tableau: Tableau,
                tol: float,
                *,
                rootfinder: RootFinder,
                bracket: tuple[float, float] = DEFAULT_BRACKET,
                adaptive: bool = True,
                dt0: float = 0.0,
                save_history: bool = False,
                max_steps: int = 10_000_000) -> SolverResult:
    """Relaxation applied after the step is accepted, the direct way.

    Runs the ordinary step, applies error control, and only then solves for
    gamma and moves to the relaxed point. Because integration continues from
    ``u_gamma`` rather than ``u_np1``, the cached FSAL value is for the wrong
    point and must be recomputed with a fresh right-hand-side evaluation.

    That one extra evaluation per step is exactly what cancels the FSAL saving,
    and demonstrating it is the reason this variant is implemented at all: it
    is the "obvious" approach whose cost the two new schemes improve on.

    Returns
    -------
    SolverResult
        `nf` should exceed `solve_baseline` by the ratio ``s/(s-1)`` -- about
        4/3 for BS3 and 7/6 for DP5. Gate G3 checks this.
    """
    raise NotImplementedError


def solve_fsalr(problem: Problem,
                tableau: Tableau,
                tol: float,
                *,
                rootfinder: RootFinder,
                bracket: tuple[float, float] = DEFAULT_BRACKET,
                adaptive: bool = True,
                dt0: float = 0.0,
                save_history: bool = False,
                interpolate_fsal: bool = True,
                max_steps: int = 10_000_000) -> SolverResult:
    """FSAL-R: relax after step-size control, then interpolate the FSAL value.

    Identical to `solve_naive` up to the point where the relaxed solution is
    formed. Instead of evaluating the right-hand side afresh at ``u_gamma``,
    it interpolates between two stage values already in hand::

        f(u_gamma) ~= k_first + gamma * (k_last - k_first)

    which costs arithmetic rather than a function evaluation. Theory (Lemma 1
    of the paper) puts the interpolation error at ``O(dt**(p+1))``, small
    enough that the method's order is unchanged.

    `interpolate_fsal=False` disables the interpolation and reuses ``k_last``
    unmodified -- the authors' "simple" variant, kept as a comparison point
    showing what the interpolation actually buys.

    Returns
    -------
    SolverResult
        `nf` must equal `solve_baseline` exactly. Gate G3.
    """
    raise NotImplementedError
