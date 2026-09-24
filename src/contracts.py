"""Shared type definitions used across every module.

This module exists so that no module owner has to import from another owner's
file merely to obtain a type. Both solver modules need `SolverResult`; if it
lived in `solvers/classic.py`, then `solvers/rfsal.py` would depend on a file
owned by someone else. Putting shared types here keeps the dependency graph
one-directional: everything imports from `contracts`, and `contracts` imports
from nothing.

Owner: project lead. Changes here affect every module, so they are discussed
before being made.

State representation
--------------------
A state is a tuple of plain Python floats, not a NumPy array. For this
problem the state has two components, and at that size NumPy's per-call
dispatch overhead (~1 microsecond) dwarfs the arithmetic itself (~50
nanoseconds) -- NumPy would be roughly 20x slower here.

Tuples are used at module boundaries because they are immutable and cannot be
aliased by accident. Solver loops are free to use mutable lists internally for
stage storage; only what crosses a module boundary must be a tuple.
"""

from typing import Callable, NamedTuple

# A point in phase space. For the pendulum: (omega, theta).
State = tuple[float, ...]

# Right-hand side of the ODE, u' = f(t, u).
# Takes time first to match the usual convention, even though the pendulum is
# autonomous and ignores it.
RHSFunction = Callable[[float, State], State]

# The conserved quantity eta(u). For the pendulum: 0.5*omega**2 - (g/L)*cos(theta).
EntropyFunction = Callable[[State], float]

# Gradient of the invariant, grad eta(u). Required by Newton's method, which
# needs d(eta)/d(gamma) = grad eta . d along the relaxation direction d.
EntropyGradient = Callable[[State], State]


class Problem(NamedTuple):
    """Everything defining one test problem.

    Bundled into a single object so that solvers take one argument instead of
    six, and so that adding a second test problem (e.g. the harmonic
    oscillator, useful for cross-validating against the authors' Julia code)
    requires no solver changes.
    """

    name: str
    rhs: RHSFunction
    entropy: EntropyFunction
    entropy_gradient: EntropyGradient
    u0: State
    t_span: tuple[float, float]
    reference: Callable[[float], State]
    """High-accuracy solution used to measure true global error."""


class Tableau(NamedTuple):
    """An embedded explicit Runge-Kutta pair with FSAL structure.

    Invariants every tableau here must satisfy, worth asserting in tests:
      - ``A[-1] == b``      the last stage evaluates f at the new solution point
      - ``c[-1] == 1.0``    that stage sits at the end of the step
      - ``b[-1] == 0.0``    follows from the two above for any explicit method,
                            so the last stage never contributes to the
                            propagated solution -- only to the embedded one
      - ``sum(b) == 1.0``   consistency
    """

    name: str
    A: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    b_embedded: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    """Order of the propagated solution: 3 for BS3, 5 for DP5."""
    n_stages: int


class RootResult(NamedTuple):
    """Outcome of one relaxation-parameter solve.

    Carries more than just gamma because comparing root-finders is one of this
    project's three contributions: `iterations` is the cost metric being
    measured, and `residual` lets the caller verify that the returned value
    actually solves the equation rather than trusting the method to say so.
    """

    gamma: float
    iterations: int
    converged: bool
    residual: float
    """|r(gamma)| at the returned gamma. Should be near zero; a large value
    means the method returned a non-root and the step must be rejected.

    Checked independently of `converged` because a method can report success
    while having converged somewhere unhelpful -- Newton, which has no
    bracketing guarantee, is the case to watch."""


class SolverResult(NamedTuple):
    """Outcome of integrating one problem with one method to the final time.

    Consumed by both `montecarlo.py` and `analysis.py`, so its shape is fixed
    before either is written.
    """

    method: str
    tableau_name: str
    tolerance: float

    t_final: float
    """Time actually reached. Relaxation advances to t_n + gamma*dt rather than
    t_n + dt, so this need not equal t_span[1] unless the run was configured to
    land exactly on it."""

    u_final: State
    error_final: float
    """Norm of (u_final - reference(t_final)). The y-axis of work-precision
    diagrams."""

    nf: int
    """Right-hand-side evaluations. The cost metric, and the x-axis of
    work-precision diagrams. Gate G3 compares this across methods as an exact
    integer."""

    n_accept: int
    n_reject: int

    n_root_iterations: int
    """Total root-finder iterations over the whole run. Zero for baseline."""

    n_relaxation_failures: int
    """Steps where no valid gamma could be found and the step size had to be
    reduced. Expected to be zero on well-behaved initial conditions; a nonzero
    count is itself a Monte Carlo finding."""

    t_history: tuple[float, ...]
    entropy_history: tuple[float, ...]
    gamma_history: tuple[float, ...]
    """Empty tuples unless the run requested history recording. Storing full
    histories for a thousand Monte Carlo samples costs gigabytes, so sweeps
    leave them off and single diagnostic runs turn them on."""
