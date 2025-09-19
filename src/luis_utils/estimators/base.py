"""Base classes for state estimators."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseEstimator(ABC):
    """Abstract base class for state estimators."""

    def __init__(self, system: Any) -> None:
        """Initialize the estimator with a system."""
        self.system = system

    @abstractmethod
    def predict_step(self, state_estimate: dict, u: torch.Tensor) -> dict:
        """Perform one prediction step."""
