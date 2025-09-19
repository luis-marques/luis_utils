"""Environment classes for second-order dynamical systems."""

from typing import Callable, Optional, Tuple

import torch

from luis_utils.algorithms import (
    ROBOT_TO_KEY,
    Q_Space,
)
from luis_utils.load import get_callable
from luis_utils.vecops import convert_trajectory


class SecondOrderEnv:
    """Environment for second-order dynamical systems."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        system,
        physics_dt: float,
        max_steps_per_episode: int,
        integrator_name: str,
        decimation: int,
    ) -> None:
        """Initialize the environment.

        Args:
            system: The dynamical system
            physics_dt: Physics timestep
            max_steps_per_episode: Maximum steps per episode
            integrator_name: Name of the integrator
            decimation: Decimation factor
        """
        assert physics_dt > 0, "physics_dt must be > 0"
        assert max_steps_per_episode > 0, "max_steps_per_episode must be > 0"
        assert decimation >= 1, "decimation must be >= 1"
        self.system = system
        self.physics_dt = physics_dt
        self.decimation: int = decimation
        self.max_steps_per_episode = max_steps_per_episode
        if integrator_name != "isaac-sim":
            self.integrator: Optional[Callable] = get_callable("luis_utils.integrators", integrator_name)
        else:
            self.integrator = None
        self.system_key = ROBOT_TO_KEY[system.__class__]

        # Runtime State (populated in reset())
        self.q: Optional[torch.Tensor] = None  # Current configuration (N, q_dim)
        self.dq: Optional[torch.Tensor] = None  # Current velocity (N, q_dim)
        self.step_idx: Optional[int] = None  # Current time step
        self.N: Optional[int] = None  # Batch size (number of robots in parallel)

    def reset(self, s0: dict) -> Tuple[dict, dict]:
        """Reset the environment with initial state."""
        assert s0["q"].shape[0] == s0["dq"].shape[0], "q0 and dq0 must have same batch size"
        assert s0["dq"].shape[1] == self.system.DOF, f"dq0 must have shape (N, {self.system.DOF})"
        self.q, self.dq = s0["q"], s0["dq"]
        self.N = s0["q"].shape[0]
        self.step_idx = 0  # Reset time step

        self._post_reset()

        return self.get_observation(), {}  # (obs, info)

    def _post_reset(self) -> None:
        """Post-reset hook for subclasses."""

    def get_observation(self) -> dict:
        """Get current observation."""
        return {"q": self.q, "dq": self.dq}

    def dynamics_step(self, q: torch.Tensor, dq: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one dynamics step - generic 2nd-order integration wrapper: state = [q, dq]."""
        assert q.shape[0] == dq.shape[0], "q and dq must have same batch size"
        assert dq.shape[1] == self.system.DOF, f"dq must have shape (N, {self.system.DOF})"
        # assert u.shape == (self.N, self.system.u_dim), f"Action must be of shape (N, {self.system.u_dim}), got {u.shape}"

        q_next = q
        dq_next = dq

        for _ in range(self.decimation):
            if self.integrator is not None:
                q_next, dq_next = self.integrator(
                    self.system,
                    q_next,
                    dq_next,
                    u,
                    self.physics_dt,
                )
            else:
                raise RuntimeError("Integrator not set for non-Isaac-sim environment")

        return q_next, dq_next

    def step(self, action: torch.Tensor) -> Tuple[dict, float, bool, bool, dict]:
        """Returns observation, reward, terminated, truncated, info."""
        if self.q is None or self.dq is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        q_next = self.q
        dq_next = self.dq

        self.q, self.dq = self.dynamics_step(q_next, dq_next, u=action)

        if self.step_idx is None:
            self.step_idx = 0
        self.step_idx += 1
        return self.get_observation(), 0.0, False, (self.step_idx >= self.max_steps_per_episode), {}

    def rollout_episode(  # pylint: disable=too-many-positional-arguments
        self, algorithm, q0, dq0, output_space: Q_Space, ep_index: int, noise: Optional[torch.Tensor] = None
    ) -> dict:
        """Run a full episode rollout.

        Args:
            algorithm: The algorithm to use
            q0: Initial configuration
            dq0: Initial velocity
            output_space: Output space
            ep_index: Episode index
            noise: Optional noise tensor

        Returns:
            Dictionary containing trajectory data
        """
        # Reset before start of every episode
        obs, _ = self.reset(s0={"q": q0, "dq": dq0})
        algorithm.planner.reset()

        trace = {}
        # for each obs key -> list of length max_steps_per_episode + 1
        for key, value in obs.items():
            if self.N is None:
                raise RuntimeError("N not set")
            trace[f"{key}_traj"] = torch.empty((self.N, self.max_steps_per_episode + 1, *value.shape[1:]))
            trace[f"{key}_traj"][:, 0] = value

        if self.N is None:
            raise RuntimeError("N not set")
        trace["u_traj"] = torch.empty((self.N, self.max_steps_per_episode, self.system.u_dim))

        if noise is not None:
            assert noise.ndim == 3, "Noise must be of shape (1, num_steps, dq_dim)"
            assert noise.shape[1] == self.max_steps_per_episode, "Noise must have same length as num_steps"
            assert noise.shape[2] == self.system.u_dim, "Noise must have same shape as action space"
            assert noise.shape[0] == self.N

        for t in range(self.max_steps_per_episode):
            action = algorithm.planner(obs, ep_index=ep_index).T
            if self.step_idx is not None:
                trace["u_traj"][:, self.step_idx] = action

            if noise is None:
                u_eff = action
            else:
                u_eff = action + noise[:, t]

            obs, _, terminated, truncated, _ = self.step(u_eff)

            for key, value in obs.items():
                if self.step_idx is not None:
                    trace[f"{key}_traj"][:, self.step_idx] = value

            if terminated or truncated:
                break

        if self.step_idx is None:
            raise RuntimeError("step_idx not set")
        q_mapped, dq_mapped = convert_trajectory(
            trace["q_traj"], trace["dq_traj"], self.step_idx + 1, self.system_key, (self.system_key[0], output_space)
        )

        trace["q_traj"] = q_mapped
        trace["dq_traj"] = dq_mapped
        if self.step_idx is not None:
            for key, value in trace.items():  # Clean unused steps
                if key != "u_traj":
                    trace[key] = value[:, : self.step_idx + 1]
                else:
                    trace[key] = value[:, : self.step_idx]

        out = {
            **trace,
            "noise": noise,
            "params": {
                "DOF": self.system.DOF,
                "u_dim": self.system.u_dim,
                "max_steps_per_episode": self.max_steps_per_episode,
                "inertia_matrix": self.system.inertia_matrix,
                "physics_dt": self.physics_dt,
                "N": self.N,
                "decimation": self.decimation,
                "system": self.system,
                "system_key": self.system_key,
                "integrator_name": self.integrator,
            },
        }

        return out


def rollout_analytical_trajectory(reference_env, real_system, q0, dq0, u, analytical_func: Callable):  # pylint: disable=too-many-positional-arguments
    """Roll out analytical trajectory for comparison."""
    times = torch.linspace(
        0,
        reference_env["params"]["physics_dt"] * reference_env["params"]["decimation"] * reference_env["params"]["max_steps_per_episode"],
        reference_env["params"]["max_steps_per_episode"] + 1,
        device=reference_env["q_traj"].device,
    )

    q, dq = analytical_func(real_system, q0, dq0, times, u)  # (T+1,)
    assert q.shape == (reference_env["params"]["N"], reference_env["params"]["max_steps_per_episode"] + 1, reference_env["params"]["DOF"])
    assert dq.shape == (reference_env["params"]["N"], reference_env["params"]["max_steps_per_episode"] + 1, reference_env["params"]["DOF"])

    out = {
        "q_traj": q,
        "dq_traj": dq,
        "u_traj": torch.zeros((reference_env["params"]["N"], reference_env["params"]["max_steps_per_episode"], reference_env["params"]["u_dim"])),
        "params": reference_env["params"],
    }
    return out
