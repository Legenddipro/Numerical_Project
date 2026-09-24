"""Tests for src/montecarlo.py. Owner: Asikur Rahman."""

import pytest


@pytest.mark.skip(reason="not yet implemented")
def test_sweep_covers_full_cross_product():
    """Row count must equal the product of the configuration lists, minus the
    root-finder dimension for baseline, which uses none."""


@pytest.mark.skip(reason="not yet implemented")
def test_baseline_not_duplicated_across_rootfinders():
    """baseline runs once per (tableau, tolerance, initial condition).

    Running it once per root-finder would produce identical duplicate rows and
    inflate its apparent share of the compute in any timing comparison.
    """


@pytest.mark.skip(reason="not yet implemented")
def test_solver_failure_is_recorded_not_raised():
    """A configuration that raises must appear as a failure row and the sweep
    must continue. With random initial conditions some failures are expected,
    and how often they occur is itself a result."""


@pytest.mark.skip(reason="not yet implemented")
def test_sweep_is_reproducible_from_seed():
    """Same seed, same output rows."""


@pytest.mark.skip(reason="not yet implemented")
def test_parallel_matches_serial():
    """n_workers=4 must give the same rows as n_workers=1.

    Each worker derives its seed from the master seed, so results must not
    depend on worker count -- a classic source of irreproducible Monte Carlo.
    """


@pytest.mark.skip(reason="not yet implemented")
def test_histories_excluded_from_csv():
    """Output columns must omit the history tuples. Including them would make
    the file unusable at a thousand samples."""
