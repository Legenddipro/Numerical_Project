"""Tests for src/tableaus.py. Owner: MD. Abir Hossain.

The cheapest place in the whole project to catch a mistyped coefficient. A
single wrong digit here otherwise surfaces much later as a failed
convergence-order check, where the cause is far harder to locate.
"""

import pytest


@pytest.mark.skip(reason="not yet implemented")
def test_bs3_satisfies_tableau_identities():
    """sum(b)==1, c[i]==sum(A[i]), A[-1]==b, c[-1]==1, b[-1]==0."""


@pytest.mark.skip(reason="not yet implemented")
def test_dp5_satisfies_tableau_identities():
    """Same identities as BS3."""


@pytest.mark.skip(reason="not yet implemented")
def test_embedded_weights_sum_to_one():
    """Consistency of the embedded method: sum(b_embedded) == 1."""


@pytest.mark.skip(reason="not yet implemented")
def test_embedded_uses_final_stage():
    """b_embedded[-1] != 0 even though b[-1] == 0.

    This asymmetry is the whole reason the last stage is computed at all, and
    the reason R-FSAL can skip it for the main solution but still needs a
    substitute for the embedded one.
    """


@pytest.mark.skip(reason="not yet implemented")
def test_validate_tableau_rejects_corrupted_coefficients():
    """Perturb one coefficient; validate_tableau must raise."""


@pytest.mark.skip(reason="not yet implemented")
def test_pid_gains_match_reference_implementation():
    """BS3 -> (0.6, -0.2, 0.0); DP5 -> (0.7, -0.4, 0.0)."""
