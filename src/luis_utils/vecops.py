"""Vector operations and trajectory conversion utilities."""

from typing import Tuple

import torch

from luis_utils.algorithms import ConfigMapper, Q_Space, RobotType, VelocityMapper


def wrap_angle(theta: torch.Tensor, low: float = -torch.pi, high: float = torch.pi) -> torch.Tensor:
    """Wrap angles to a given interval [low, high) using floor instead of modulo.

    Args:
        theta: input tensor of angles (radians).
        low: lower bound (inclusive).
        high: upper bound (exclusive).

    Returns:
        Wrapped tensor with all angles inside [low, high).
    """
    assert low < high, "Low must be less than high"
    width = high - low
    return theta - width * torch.floor((theta - low) / width)


def convert_trajectory(
    q_traj: torch.Tensor, dq_traj: torch.Tensor, n_steps: int, input_key: Tuple[RobotType, Q_Space], output_key: Tuple[RobotType, Q_Space]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q_traj and dq_traj are (batch_size, n_steps, q_shape) tensors."""

    assert dq_traj.ndim == 3 and q_traj.shape[0:2] == dq_traj.shape[0:2]
    assert 0 < n_steps <= q_traj.shape[1]

    # Remove unused steps
    q = q_traj[:, :n_steps]
    dq = dq_traj[:, :n_steps]

    B, T, _ = dq.shape

    # Flatten batch and time dimensions (done like this for matrix lie groups)
    flat_q = q.reshape(B * T, *q.shape[2:])
    flat_dq = dq.reshape(B * T, *dq.shape[2:])

    q_mapped = ConfigMapper.map(from_key=input_key, to_key=output_key, q=flat_q)
    q_mapped_unflat = q_mapped.view(B, T, *q_mapped.shape[1:])

    if output_key[1] == Q_Space.LIE and input_key[1] == Q_Space.STATE_SPACE:
        dq_mapped = VelocityMapper.map(from_key=input_key, to_key=output_key, q=flat_q, dq_or_v=flat_dq)
    else:
        dq_mapped = VelocityMapper.map(from_key=input_key, to_key=output_key, q=q_mapped, dq_or_v=flat_dq)
    dq_mapped_unflat = dq_mapped.view(B, T, *dq_mapped.shape[1:])

    return q_mapped_unflat, dq_mapped_unflat
