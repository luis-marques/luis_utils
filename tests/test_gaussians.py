"""Unit tests for Gaussian-related functions in luis_utils.gaussians."""

import pytest
import torch

from luis_utils.gaussians import (
    confidence_hyperellipsoid_volume,
    mahalanobis_distance_squared,
)


@pytest.mark.parametrize(
    "batch_diff, batch_cov",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("use_cholesky", [True, False])
def test_mahalanobis_hardcoded_all_shapes(batch_diff, batch_cov, use_cholesky):
    """Test mahalanobis_distance_squared returns correct value (4.0) across all shape combinations using hardcoded values."""

    # D = 2
    N = 3  # batch size

    # Choose a known inverse covariance so that diffᵀ Σ⁻¹ diff = 4.0
    inv_cov = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity → Mahalanobis^2 = ||diff||^2

    diff = torch.tensor([2.0, 0.0])  # Squared norm = 4.0
    cov = torch.linalg.inv(inv_cov)
    L = torch.linalg.cholesky(cov)

    expected_val = 4.0

    # Expand for batch cases
    diff_batched = diff.expand(N, -1) if batch_diff else diff
    cov_batched = cov.expand(N, -1, -1) if batch_cov else cov
    L_batched = L.expand(N, -1, -1) if batch_cov else L

    kwargs = {"cholesky_factor": L_batched} if use_cholesky else {"covariance": cov_batched}
    result = mahalanobis_distance_squared(diff_batched, **kwargs)

    if batch_diff or batch_cov:
        assert result.shape == (N,)
        torch.testing.assert_close(result, torch.full((N,), expected_val, dtype=result.dtype), atol=1e-12, rtol=1e-12)
    else:
        assert result.shape == (1,)
        torch.testing.assert_close(result, torch.tensor([expected_val], dtype=result.dtype), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("batch_input", [False, True])
def test_confidence_ellipsoid_shape_covariance(batch_input):
    """Tests the output shape of confidence_hyperellipsoid_volume for batched and unbatched Cholesky factors."""
    torch.manual_seed(0)
    D = 3
    N = 4
    failure_rate = 0.05

    # Create a positive definite matrix
    A = torch.randn(D, D)
    cov = A @ A.T + 1e-2 * torch.eye(D)
    L = torch.linalg.cholesky(cov)  # (D, D)

    if batch_input:
        L = L.expand(N, D, D)  # (N, D, D)

    vol = confidence_hyperellipsoid_volume(L, failure_rate)

    if batch_input:
        assert vol.shape == (N,)
    else:
        assert vol.shape == (1,)
