"""Conformal prediction utilities."""

import torch


def covariateShiftCP(scores: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """Covariate shift conformal prediction."""
    assert scores.dim() == 1 and weights.dim() == 2 and scores.shape[0] == weights.shape[1]
    assert torch.all(weights >= 0) and 0 < alpha < 1

    _scores = scores.unsqueeze(0).expand(weights.shape[0], -1)  # Expand to (M, N)
    _scores = torch.cat((_scores, torch.full((weights.shape[0], 1), torch.inf)), dim=1)  # Add infinity to residuals (M, N+1) for test point
    _weights = torch.cat((weights, torch.ones(weights.shape[0], 1)), dim=1)  # Add 1.0 to weights (M, N+1) for test point

    _normed_weights = _weights / _weights.sum(dim=1, keepdim=True)  # Normalize weights

    sorted_scores, sorted_indices = torch.sort(_scores, dim=1)  # Sort residuals

    sorted_weights = torch.gather(input=_normed_weights, dim=1, index=sorted_indices)  # Sort weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=1)  # Cumulative sum of weights     (M, N+1)

    quantile_indices = torch.searchsorted(
        cumulative_weights, (1 - alpha) * torch.ones(weights.shape[0], 1), right=False
    )  # Find the fist index where the cumulative sum exceeds 1-alpha

    return torch.gather(input=sorted_scores, dim=1, index=quantile_indices).squeeze(1)


def splitCP(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Split/Inductive Conformal Prediction."""
    return covariateShiftCP(scores=scores, weights=torch.ones((1, len(scores))), alpha=alpha)
