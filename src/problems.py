"""Test problem: the nonlinear pendulum.

    theta'' + (g/L) sin(theta) = 0

written as a first-order system in the state u = (omega, theta):

    omega' = -(g/L) sin(theta)
    theta' = omega

with conserved invariant

    eta(u) = 0.5 * omega^2 - (g/L) * cos(theta)

Component order is (omega, theta), matching the authors' Julia code where
``du[1] = -sin(u[2])`` and ``du[2] = u[1]``. Keeping the same ordering matters
for gate G5, which compares our numbers against theirs directly.

State is a tuple of plain floats. At two components NumPy's per-call dispatch
overhead dominates the arithmetic, so `math.sin` on scalars is the fast path
here, not `numpy.sin` on arrays.

Owner: project lead.
"""

from functools import partial
from math import cos, sin
from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import Problem, State

Regime = Literal["libration", "rotation", "both"]

DEFAULT_U0: State = (1.5, 1.0)
"""The authors' initial condition: omega = 1.5, theta = 1.0.

Their ``du[2] = u[1]`` makes ``u[2]`` the angle, so their ``[1.5, 1.0]`` is
(omega, theta) in that order -- confirmed by their invariant
``0.5*u[1]^2 - cos(u[2])``. Reversing it gives a system that still oscillates
plausibly and silently fails gate G5.
"""

DEFAULT_TSPAN: tuple[float, float] = (0.0, 10.0)


def pendulum_rhs(t: float, u: State, g_over_l: float = 1.0) -> State:
    """Right-hand side of the pendulum system.

    `t` is accepted but unused -- the pendulum is autonomous. It is present so
    the signature matches the general RHS convention and stage evaluations at
    ``t + c[i]*dt`` need no special case.

    Returns
    -------
    State
        ``(omega', theta')``.
    """
    omega, theta = u
    return (-g_over_l * sin(theta), omega)


def pendulum_entropy(u: State, g_over_l: float = 1.0) -> float:
    """Total energy per unit mass -- the conserved invariant eta.

    Returns
    -------
    float
        ``0.5*omega**2 - (g/L)*cos(theta)``.
    """
    omega, theta = u
    return 0.5 * omega * omega - g_over_l * cos(theta)


def pendulum_entropy_gradient(u: State, g_over_l: float = 1.0) -> State:
    """Gradient of the invariant, ``grad eta = (omega, (g/L) sin(theta))``.

    Needed only by Newton's method, which requires

        d/dgamma eta(u_n + gamma*d) = grad eta(u_n + gamma*d) . d

    Analytic and cheap, so no automatic differentiation is required.

    Returns
    -------
    State
        The gradient, in the same component order as the state.
    """
    omega, theta = u
    return (omega, g_over_l * sin(theta))


def separatrix_energy(g_over_l: float = 1.0) -> float:
    """Invariant value dividing the pendulum's two regimes.

    Equals the energy of the bob balanced motionless at the top
    (theta = pi, omega = 0), which is ``+g/L``. Below it the pendulum swings
    back and forth; above it, it carries over the top and rotates forever.
    """
    return g_over_l


def regime_of(u: State, g_over_l: float = 1.0) -> str:
    """Classify a state as ``"libration"`` or ``"rotation"``.

    The two behave differently enough that Monte Carlo results must be reported
    separately for each rather than pooled.
    """
    return ("rotation" if pendulum_entropy(u, g_over_l) > separatrix_energy(g_over_l)
            else "libration")


def build_reference(u0: State,
                    t_span: tuple[float, float],
                    g_over_l: float = 1.0) -> Callable[[float], State]:
    """High-accuracy reference solution, for measuring true global error.

    Integrates once with DOP853 (8th order) at very tight tolerance and returns
    a dense interpolant. The authors use Julia's ``Vern9()`` at 1e-14 for the
    same purpose; DOP853 is the closest readily available equivalent and lands
    near 1e-12 here, leaving about three orders of margin over the loosest
    tolerance we test.

    Returns
    -------
    Callable[[float], State]
        Maps time to state. Calling it outside `t_span` extrapolates and should
        not be relied on.
    """
    def fun(t, y):
        return [-g_over_l * sin(y[1]), y[0]]

    sol = solve_ivp(fun, t_span, list(u0), method="DOP853",
                    rtol=1e-13, atol=1e-14, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"reference integration failed: {sol.message}")

    def reference(t: float) -> State:
        y = sol.sol(t)
        return (float(y[0]), float(y[1]))

    return reference


def make_pendulum(u0: State = DEFAULT_U0,
                  t_span: tuple[float, float] = DEFAULT_TSPAN,
                  g_over_l: float = 1.0) -> Problem:
    """Assemble a fully specified pendulum Problem.

    The defaults reproduce the authors' setup exactly, which is what gate G5
    compares against. `g_over_l` is bound into the callables here, so the
    Problem's `rhs`, `entropy` and `entropy_gradient` match the general
    signatures in `contracts` and solvers need not know the parameter exists.

    Building the reference is the expensive part -- one tight integration -- so
    construct a Problem once per initial condition and reuse it across
    tolerances and methods.
    """
    u0 = (float(u0[0]), float(u0[1]))
    return Problem(
        name=f"nonlinear_pendulum(g/L={g_over_l:g}, u0={u0})",
        rhs=partial(pendulum_rhs, g_over_l=g_over_l),
        entropy=partial(pendulum_entropy, g_over_l=g_over_l),
        entropy_gradient=partial(pendulum_entropy_gradient, g_over_l=g_over_l),
        u0=u0,
        t_span=t_span,
        reference=build_reference(u0, t_span, g_over_l),
    )


def sample_initial_conditions(n: int,
                              seed: int,
                              regime: Regime = "libration",
                              theta_range: tuple[float, float] = (-3.0, 3.0),
                              omega_range: tuple[float, float] = (-2.0, 2.0),
                              g_over_l: float = 1.0,
                              max_attempts: int | None = None) -> tuple[State, ...]:
    """Draw random initial conditions for the Monte Carlo study.

    Rejection sampling: draw uniformly from the rectangle, keep what falls in
    the requested regime, repeat. With the default ranges and ``g/L = 1`` the
    rectangle straddles the separatrix, so both regimes are reachable.

    `seed` is mandatory rather than optional: every Monte Carlo result must be
    reproducible from its recorded seed alone.

    Raises
    ------
    RuntimeError
        If `max_attempts` is exhausted -- which means the rectangle and the
        requested regime barely intersect, and the ranges need widening rather
        than the sampler needing more tries.

    Returns
    -------
    tuple[State, ...]
        Exactly `n` states, each ``(omega, theta)``.
    """
    if n <= 0:
        return ()
    if max_attempts is None:
        max_attempts = 1000 * n

    rng = np.random.default_rng(seed)
    out: list[State] = []
    attempts = 0
    while len(out) < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"sampled {attempts} points but found only {len(out)}/{n} in regime "
                f"{regime!r}; widen theta_range/omega_range")
        attempts += 1
        theta = float(rng.uniform(*theta_range))
        omega = float(rng.uniform(*omega_range))
        u = (omega, theta)
        if regime == "both" or regime_of(u, g_over_l) == regime:
            out.append(u)
    return tuple(out)
