"""Gaussian-related utilities."""

import math
from typing import Optional, Tuple

import torch
from scipy.special import gamma
from scipy.stats import chi2
from torch.distributions.multivariate_normal import _batch_mahalanobis


def ball_volume(radius: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the volume of a ball of radius r in dim dimensions."""
    return torch.pi ** (dim / 2) * radius**dim / gamma(dim / 2 + 1)


def mahalanobis_distance_squared(
    difference: torch.Tensor,
    *,  # Arguments after this require keyword
    cholesky_factor: Optional[torch.Tensor] = None,
    covariance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute squared Mahalanobis distance: (x - μ)^T Σ⁻¹ (x - μ)

    Args:
        difference (Tensor): shape (..., D)
        cholesky_factor (Tensor, optional): shape (..., D, D)
        covariance (Tensor, optional): shape (..., D, D)

    Returns:
        Tensor: Mahalanobis distance squared, shape (...,)
    """
    if (cholesky_factor is None) == (covariance is None):
        raise ValueError("Provide exactly one of `cholesky_factor` or `covariance`, not both or neither.")

    L = cholesky_factor if cholesky_factor is not None else torch.linalg.cholesky(covariance)

    assert L.shape[-2] == L.shape[-1], "Cholesky factor must be square"
    assert L.shape[-2] == difference.shape[-1], "Cholesky factor and difference must have same last dimension"

    # Normalize input shapes
    if difference.ndim == 1:
        difference = difference.unsqueeze(0)  # (1, D)
    if L.ndim == 2:
        L = L.unsqueeze(0)  # (1, D, D)

    # If only one of the inputs is batched, expand the other to match
    if difference.shape[0] == 1 and L.shape[0] > 1:
        difference = difference.expand(L.shape[0], -1)
    elif L.shape[0] == 1 and difference.shape[0] > 1:
        L = L.expand(difference.shape[0], -1, -1)

    return _batch_mahalanobis(L, difference)  # type: ignore


def mahalanobis_distance(
    difference: torch.Tensor,
    *,
    cholesky_factor: Optional[torch.Tensor] = None,
    covariance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Mahalanobis distance: sqrt((x - μ)^T Σ⁻¹ (x - μ))

    Args:
        difference (Tensor): shape (..., D), difference vector(s)
        cholesky_factor (Tensor, optional): shape (..., D, D), lower-triangular Cholesky factor L (Σ = L @ L.T)
        covariance (Tensor, optional): shape (..., D, D), covariance matrix Σ (L will be computed)

    Returns:
        Tensor: Mahalanobis distance, shape (...)
    """
    return mahalanobis_distance_squared(difference, cholesky_factor=cholesky_factor, covariance=covariance).sqrt()


def confidence_hyperellipsoid_volume(cholesky_factor: torch.Tensor, failure_rate: float) -> torch.Tensor:
    """
    Computes the volume of the (1 - failure_rate) confidence hyperellipsoid
    defined by the Cholesky factor of the covariance matrix.

    Based on:
    - Theorem 3.10 (p. 100) of David Olive, "Robust Statistics"
      http://parker.ad.siu.edu/Olive/runrob.pdf
    - Theorem 3.9 (p. 99) for the chi-squared quantile (h²)

    Args:
        cholesky_factor (Tensor): Cholesky factor of the covariance matrix,
            shape (..., D, D) or (D, D)
        failure_rate (float): Desired failure rate (e.g. 0.05 for 95% confidence)

    Returns:
        Tensor: (...,) volumes of the confidence hyperellipsoids
    """
    if not 0 < failure_rate < 1:
        raise ValueError("failure_rate must be between 0 and 1")

    L = cholesky_factor
    if L.ndim == 2:
        L = L.unsqueeze(0)  # Make it batched: (1, D, D)

    D = L.shape[-1]

    det_term = torch.linalg.det(L)  # shape (...,)

    h2 = critical_mahalanobis_distance(failure_rate=failure_rate, D=D) ** 2

    numerator = 2 * (torch.pi ** (D / 2)) * (h2 ** (D / 2)) * det_term
    volume = numerator / (D * gamma(D / 2))
    return volume if volume.ndim > 0 else volume.unsqueeze(0)


def points_in_confidence_hyperellipsoid_from_mahalanobis(
    maha_squared: torch.Tensor,
    *,
    failure_rate: float,
    D: int,
) -> Tuple[torch.Tensor, float]:
    """
    Check if a Mahalanobis-squared value is within the (1 - failure_rate) confidence region.

    Args:
        maha_squared (Tensor): shape (...,), precomputed Mahalanobis distance squared
        failure_rate (float): e.g. 0.05 for 95% confidence
        D (int, optional): Dimensionality of the data. Required if chi-squared threshold can't be inferred.

    Returns:
        Tensor: Boolean tensor of shape (...) indicating whether each point is inside the ellipsoid
    """
    if not 0 < failure_rate < 1:
        raise ValueError("failure_rate must be between 0 and 1")

    mask = maha_squared.sqrt() <= critical_mahalanobis_distance(failure_rate=failure_rate, D=D)
    fraction = mask.float().mean().item()
    return mask, fraction


def critical_mahalanobis_distance(failure_rate: float, D: int) -> float:
    """Compute the critical Mahalanobis distance for a given failure rate and dimensionality."""
    return math.sqrt(chi2.ppf(1 - failure_rate, D))


def points_in_confidence_hyperellipsoid(
    difference: torch.Tensor,
    *,
    failure_rate: float,
    cholesky_factor: Optional[torch.Tensor] = None,
    covariance: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Check if difference vectors are inside the (1 - failure_rate) confidence hyperellipsoid.

    Args:
        difference (Tensor): shape (..., D), difference from the mean (x - μ)
        failure_rate (float): e.g. 0.05 for 95% confidence
        cholesky_factor (Tensor, optional): shape (..., D, D), lower-triangular matrix L such that Σ = L @ L.T
        covariance (Tensor, optional): shape (..., D, D)

    Returns:
        Tensor: Boolean tensor of shape (...) indicating whether each point is inside the ellipsoid
    """
    if (cholesky_factor is None) == (covariance is None):
        raise ValueError("Provide exactly one of `cholesky_factor` or `covariance`, not both or neither.")
    if not 0 < failure_rate < 1:
        raise ValueError("failure_rate must be between 0 and 1")

    D = difference.shape[-1]

    md_squared = mahalanobis_distance_squared(
        difference,
        cholesky_factor=cholesky_factor,
        covariance=covariance,
    )

    return points_in_confidence_hyperellipsoid_from_mahalanobis(
        maha_squared=md_squared,
        failure_rate=failure_rate,
        D=D,
    )


def sample_unit_circle_points(n_points: int) -> torch.Tensor:
    """Generates uniformly distributed points on the unit circle.

    Args:
        n_points (int): Number of points to generate.

    Returns:
        Tensor of shape (n_points, 2) with unit vectors.
    """
    # Generate angles from 0 to 2*pi
    angles = torch.linspace(0, 2 * torch.pi, n_points, dtype=torch.float32)

    # Calculate x and y coordinates
    x = torch.cos(angles)
    y = torch.sin(angles)

    return torch.stack([x, y], dim=1)


def sample_hypersphere_surface(n_points: int, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Generates uniformly distributed points on the surface of a unit hypersphere.

    Args:
        n_points (int): Number of points to generate.
        dim (int): Dimension of the sphere (e.g., 3 for 3D).
        eps (float): Minimum norm threshold to avoid division by near-zero.

    Returns:
        Tensor of shape (n_points, dim) with unit-norm vectors.
    """
    assert n_points > 0, "Number of points must be positive"
    assert dim > 0, "Dimension must be positive"
    points = torch.randn(n_points, dim)  # each row is a Gaussian vector

    norms = torch.norm(points, dim=1, keepdim=True)
    # If any norms are too small (rare), resample those rows
    too_small = norms.squeeze(-1) < eps
    while too_small.any():
        n_resample = too_small.sum()
        points[too_small] = torch.randn(n_resample, dim)
        norms = torch.norm(points, dim=1, keepdim=True)
        too_small = norms.squeeze(-1) < eps

    return points / norms


def fibonacci_3Dsphere_grid(n_points: int) -> torch.Tensor:
    """Generates approximately uniformly distributed points on the unit sphere using Fibonacci lattice.

    Args:
        n_points (int): Number of points to generate.

    Returns:
        Tensor of shape (n_points, 3) with unit vectors.
    """
    indices = torch.arange(0, n_points)
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor([5.0])).item())  # golden angle ~2.39996

    z = 1 - 2 * indices / (n_points - 1)
    theta = phi * indices

    x = torch.sqrt(1 - z**2) * torch.cos(theta)
    y = torch.sqrt(1 - z**2) * torch.sin(theta)

    return torch.stack([x, y, z], dim=1)


def sample_confidence_ellipsoid_boundary(
    n_points: int,
    *,
    mean: torch.Tensor,  # (D,)
    covariance: torch.Tensor,  # (D, D)
    confidence_level: float,  # 1 - alpha
) -> torch.Tensor:
    """Samples points on the boundary of the confidence ellipsoid defined by the mean and covariance.

    Returns:
        Tensor of shape (n_points, D)
    """
    D = mean.shape[0]
    assert D in [2, 3], "Only 2D and 3D are supported for now"
    L = torch.linalg.cholesky(covariance)  # (D, D)

    # Chi-squared quantile for (1 - alpha) confidence
    h = critical_mahalanobis_distance(failure_rate=1 - confidence_level, D=D)

    # Unit vectors on surface of unit sphere
    # unit_sphere_points = sample_hypersphere_surface(n_points, D)  # shape: (n_points, D)
    if D == 2:
        unit_sphere_points = sample_unit_circle_points(n_points)
    elif D == 3:
        unit_sphere_points = fibonacci_3Dsphere_grid(n_points)
    else:
        raise ValueError(f"Only 2D and 3D are supported for now, got {D}")

    # Scale and transform to ellipsoid boundary
    ellipsoid_points = mean + (h * (unit_sphere_points @ L.T))  # shape: (n_points, D)
    return ellipsoid_points
