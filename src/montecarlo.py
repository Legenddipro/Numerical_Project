"""Monte Carlo sweep harness.

Samples random initial conditions (theta_0, omega_0), runs every combination of
method variant, integrator, tolerance, and root-finder, and writes raw results
to results/ as CSV.

Sampling must distinguish the pendulum's two regimes: with E = 0.5*omega^2 -
cos(theta), states with E < 1 librate (swing back and forth) while E > 1 rotate
(swing over the top). These behave qualitatively differently and should not be
pooled without saying so.

The sweep is embarrassingly parallel over initial conditions.

Interfaces to be fixed in the Phase 0 contract session.
"""
