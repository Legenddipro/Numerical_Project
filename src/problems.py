"""Test problem: the nonlinear pendulum.

Provides the right-hand side, the conserved invariant and its gradient, a
high-accuracy reference solution for error measurement, and initial-condition
sampling for the Monte Carlo study.

    theta'' + (g/L) sin(theta) = 0,    eta(u) = 0.5 * omega^2 - (g/L) * cos(theta)

State is carried as two plain floats (omega, theta), not a NumPy array: at this
size NumPy's per-call dispatch overhead dominates the arithmetic.

Interfaces to be fixed in the Phase 0 contract session.
"""
