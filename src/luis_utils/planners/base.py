"""Abstract base class for planners."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BasePlanner(ABC):
    """Abstract base class for planners."""

    def __init__(self, system, horizon: int):
        """Abstract planner base class.

        Args:
            system: The internal planning model (may differ from simulator).
            horizon: Planning horizon H.
        """
        self.system = system
        self.horizon: int = horizon
        self.u_dim: int = system.u_dim

        self._planning_index: Optional[int] = None  # Index of the current planning step
        self.planned_controls: Optional[torch.Tensor] = None  # (H, u_dim)

    def reset(self) -> None:
        """Start a brand-new episode: zero your time cursor."""
        self._planning_index = 0
        self.planned_controls = None

    @abstractmethod
    def plan(self, obs: dict, ep_index: int) -> None:
        """Fill self.planned_controls (H×u_dim) based on (q0,dq0).

        Fixed planners can make this a no-op after the very first call. MPC planners will recompute here each time it’s called.
        """

    def __call__(self, obs: dict, ep_index: int) -> torch.Tensor:
        """Returns the control at time t (after planning)."""
        assert self._planning_index is not None, "Must call `reset()` before using the planner."
        self.plan(obs, ep_index)  # Fixed planners can make this a no-op after the very first call.
        assert self.planned_controls is not None, "Must call `plan()` before using the planner."

        index = min(self._planning_index, self.horizon - 1)
        action = self.planned_controls[index]  # was [index:index+1]

        self._planning_index += 1
        return action
