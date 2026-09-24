"""Tests for src/solvers/rfsal.py. Owner: Tousif Fahmeed Quadir."""

import pytest


@pytest.mark.skip(reason="not yet implemented")
def test_rfsal_conserves_invariant():
    """Energy held to ~1e-15, matching the other relaxation variants."""


@pytest.mark.skip(reason="not yet implemented")
def test_final_stage_is_never_computed_directly():
    """R-FSAL skips stage s. The RHS count must reflect that: (s-2) stage
    evaluations plus one at the relaxed point, not (s-1) plus one."""


@pytest.mark.skip(reason="not yet implemented")
def test_extrapolation_is_exact_for_linear_rhs():
    """On a linear problem such as the harmonic oscillator, the 1/gamma
    extrapolation reproduces f(u_np1) exactly, to round-off.

    Not a coincidence: applying a linear map commutes with the affine
    combination the formula performs. A sharp test of the extrapolation with
    no tolerance fudging, and it fails loudly if the factor is gamma rather
    than 1/gamma.
    """


@pytest.mark.skip(reason="not yet implemented")
def test_fsal_cache_is_exact_not_approximated():
    """The value carried into the next step is a real evaluation at the
    relaxed point, so it must equal a direct RHS call there.

    This is R-FSAL's structural difference from FSAL-R: here the cache is
    exact and the error estimate is approximated, the other way round.
    """


@pytest.mark.skip(reason="not yet implemented")
def test_variant_switches_are_independent():
    """interpolate_fsal, relax_embedded, and relax_main must each change
    results on their own, confirming all eight combinations are reachable and
    the authors' variant study can be reproduced."""


@pytest.mark.skip(reason="not yet implemented")
def test_relaxation_precedes_error_estimate():
    """The embedded solution must be built from relaxed quantities.

    Verified by construction rather than by output: if relaxation ran after
    the error test, this would be R-FSAL in name only, and the accuracy
    difference from FSAL-R would be too small to catch by comparing results.
    """
