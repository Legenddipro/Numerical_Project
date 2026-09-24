"""Tests for src/solvers/classic.py. Owner: MD. Shadman Shafie.

Module-level tests only. Cross-method comparisons (gates G2-G6) live in
test_gates.py, since they span both solver modules.
"""

import pytest


@pytest.mark.skip(reason="not yet implemented")
def test_baseline_reaches_final_time():
    """Integration terminates at t_span[1], not before or past it."""


@pytest.mark.skip(reason="not yet implemented")
def test_baseline_reuses_fsal_stage():
    """RHS count must be about (s-1) per accepted step, not s.

    If FSAL reuse is missing, everything still runs and produces correct
    answers -- just more expensively -- so this is checked directly rather
    than assumed from a passing accuracy test.
    """


@pytest.mark.skip(reason="not yet implemented")
def test_baseline_ignores_rootfinder_argument():
    """Passing a root-finder must change nothing. The argument exists only so
    the signature matches the relaxation variants."""


@pytest.mark.skip(reason="not yet implemented")
def test_naive_conserves_invariant():
    """Energy held to ~1e-15 despite the naive scheme's cost."""


@pytest.mark.skip(reason="not yet implemented")
def test_fsalr_conserves_invariant():
    """FSAL-R must conserve as well as naive does. The interpolation changes
    the cost, not the conservation."""


@pytest.mark.skip(reason="not yet implemented")
def test_fsalr_interpolation_flag_changes_accuracy_not_cost():
    """interpolate_fsal=False must give the same RHS count but measurably
    worse accuracy -- this is what the interpolation buys."""


@pytest.mark.skip(reason="not yet implemented")
def test_fixed_step_mode_takes_uniform_steps():
    """With adaptive=False, every step must equal dt0 and nothing may be
    rejected. Gate G2 depends on this mode behaving exactly."""


@pytest.mark.skip(reason="not yet implemented")
def test_history_recorded_only_when_requested():
    """save_history=False must leave the history tuples empty. A thousand
    Monte Carlo samples with histories on would cost gigabytes."""
