"""2nd Order Unicycle Dynamics (2D)"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from pymatlie.base_group import NonholonomicGroup
from pymatlie.se2 import SE2
from scipy.special import fresnel  # pylint: disable=no-name-in-module

from luis_utils.systems.base import SecondOrderNonholonomicSystem


@dataclass(frozen=True)
class Unicycle2ndOrder(SecondOrderNonholonomicSystem):
    """2nd-order unicycle dynamics in 2D with mass and inertia."""

    inertia_matrix: torch.Tensor  # Inertia matrix (3x3)

    q_dim: int = field(default=3, init=False)  # (x, y, theta)
    u_dim: int = field(default=2, init=False)  # (F, tau)

    q_names: Tuple[str, ...] = field(default=(r"$x \ (m)$", r"$y \ (m)$", r"$\theta \ (rad)$"), init=False)
    dq_names: Tuple[str, ...] = field(default=(r"$\dot x \ (m/s)$", r"$\dot y \ (m/s)$", r"$\dot \theta \ (rad/s)$"), init=False)

    def force_map(self, q: torch.Tensor) -> torch.Tensor:
        theta = q[:, 2:3]  # (N, 1)
        N = q.shape[0]
        B = torch.zeros((N, 3, 2), dtype=q.dtype, device=q.device)
        B[:, 0, 0] = torch.cos(theta[:, 0])  # F_x in world
        B[:, 1, 0] = torch.sin(theta[:, 0])  # F_y in world
        B[:, 2, 1] = 1.0  # tau_z is direct torque
        return B

    def get_Pfaffian_A(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """State-dependent nonholonomic constraint matrix."""
        _, _, theta = torch.chunk(q, chunks=3, dim=1)
        A = torch.zeros((q.shape[0], 1, self.q_dim), device=q.device, dtype=q.dtype)
        A[:, 0, 0] = torch.sin(theta[:, 0])
        A[:, 0, 1] = -torch.cos(theta[:, 0])
        return A

    def get_Pfaffian_A_dot(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """State-dependent derivative of A(q):

        A(q) = [ sinθ, -cosθ, 0 ], A_dot = [ cosθ,  sinθ, 0 ] * θ̇
        """
        N = q.shape[0]
        A_dot = torch.zeros((N, 1, self.q_dim), device=q.device, dtype=q.dtype)

        # extract θ and θ̇
        theta = q[:, 2]  # (N,)
        theta_dot = dq[:, 2]  # (N,)

        # fill in, scaling by θ̇
        A_dot[:, 0, 0] = torch.cos(theta) * theta_dot
        A_dot[:, 0, 1] = torch.sin(theta) * theta_dot
        # A_dot[:,0,2] stays zero
        return A_dot

    def total_energy(self, _: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Total kinetic energy of unicycle system (translational + rotational)"""
        assert dq.ndim == 2 and dq.shape[1] == self.q_dim, f"dq must be of shape (N, {self.q_dim})"
        dx, dy, dtheta = torch.chunk(dq, chunks=3, dim=1)  # Each (N, 1)

        m = self.inertia_matrix[0, 0]
        I = self.inertia_matrix[2, 2]

        transl = 0.5 * m * (dx**2 + dy**2)
        rot = 0.5 * I * (dtheta**2)

        return (transl + rot).reshape(-1)

    def potential_forces(self, q):
        return torch.zeros_like(q)

    def dissipative_forces(self, q, dq):
        # Damping forces (linear + angular)
        return torch.zeros_like(q)

    def get_linearized_dynamics_matrices(
        self, q: torch.Tensor, dq: torch.Tensor, u: torch.Tensor, error_form: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q.ndim == dq.ndim == u.ndim == 2
        assert q.shape[0] == dq.shape[0] == u.shape[0]
        assert q.shape[1] == dq.shape[1] == self.q_dim
        assert u.shape[1] == self.u_dim

        N = q.shape[0]
        m = self.inertia_matrix[0, 0]
        I = self.inertia_matrix[2, 2]

        theta = q[:, 2]
        vx = dq[:, 0]  # world-frame
        vy = dq[:, 1]  # world-frame
        wz = dq[:, 2]  # ω = θ̇

        A = torch.zeros((N, 2 * self.DOF, 2 * self.DOF), device=q.device, dtype=q.dtype)

        # q̇ depends on dq
        A[:, :3, 3:6] = torch.eye(3, device=q.device, dtype=q.dtype)

        # q̈ partials wrt q (from force direction)
        F = u[:, 0]
        A[:, 3, 2] = -(F / m) * torch.sin(theta)  # ∂ẍ/∂θ
        A[:, 4, 2] = (F / m) * torch.cos(theta)  # ∂ÿ/∂θ
        # A[:, 5, 2] = 0

        # *** q̈ partials wrt dq (convective terms!) ***
        # wrt v_x, v_y, ω  -> fill the lower-right 3x3 block
        A[:, 3, 3] = 0.0
        A[:, 3, 4] = -wz  # ∂ẍ/∂v_y = -ω
        A[:, 3, 5] = -vy  # ∂ẍ/∂ω  = -v_y

        A[:, 4, 3] = wz  # ∂ÿ/∂v_x = +ω
        A[:, 4, 4] = 0.0
        A[:, 4, 5] = vx  # ∂ÿ/∂ω  = +v_x

        # A[:, 5, 3:6] already zeros (θ̈ doesn’t depend on velocities)

        # Process noise mapping (force, torque) -> accelerations (world)
        G = torch.zeros((N, 2 * self.DOF, 2), device=q.device, dtype=q.dtype)
        G[:, 3, 0] = torch.cos(theta) / m
        G[:, 4, 0] = torch.sin(theta) / m
        G[:, 5, 1] = 1 / I

        return A, G


class SE2Unicycle2ndOrder(NonholonomicGroup, SE2):  # type: ignore[misc]
    """Lie Group based 2nd-order unicycle dynamics in 2D with mass and inertia."""

    def get_Pfaffian_A(self, g: torch.Tensor, xi: torch.Tensor):
        """Applies nonholonomic constraints for unicycle model."""
        N = xi.shape[0]
        assert xi.ndim == 2 and xi.shape[-1] == self.g_dim, f"get_Pfaffian_A requires shape (N, {self.g_dim}), got {xi.shape}"
        A = torch.zeros((N, 1, 3))
        A[:, 0, 1] = 1.0
        return A

    def get_linearized_dynamics_matrices(
        self, g: torch.Tensor, xi: torch.Tensor, u: torch.Tensor, error_form: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get linearized dynamics matrices for the SE2 unicycle system."""
        assert g.shape[0] == xi.shape[0] == u.shape[0]

        m = self.inertia_matrix[0, 0]

        N = g.shape[0]

        if error_form == "LI":
            A = torch.zeros((N, 2 * self.DOF, 2 * self.DOF), device=g.device, dtype=g.dtype)
            A[:, :3, :3] = -self.ad_operator(xi)
            A[:, :3, 3:] = torch.eye(3, device=g.device, dtype=g.dtype)

        elif error_form == "RI":
            A = torch.zeros((N, 2 * self.DOF, 2 * self.DOF), device=g.device, dtype=g.dtype)
            A[:, :3, :3] = self.ad_operator(xi)
            A[:, :3, 3:] = -torch.eye(3, device=g.device, dtype=g.dtype)

        else:
            raise ValueError(f"Invalid error_form: {error_form}")

        # Build J
        vx, vy, wz = xi[:, 0], xi[:, 1], xi[:, 2]
        J = torch.zeros((N, 3, 3), device=g.device, dtype=g.dtype)
        J[:, 0, 1] = m * wz
        J[:, 0, 2] = m * vy
        J[:, 1, 0] = -m * wz
        J[:, 1, 2] = -m * vx

        # Fill A22
        A[:, 3:, 3:] = self.inertia_matrix_inv @ self.constraint_projection_matrix_wrench @ J

        force_map = torch.tensor([[1, 0], [0, 0], [0, 1]], device=g.device, dtype=g.dtype)

        B = torch.zeros((2 * self.DOF, self.u_dim))  # Same for all (reduces computation)
        B[3:, :] = self.inertia_matrix_inv @ self.constraint_projection_matrix_wrench @ force_map

        return A, B


def analytical_constant_acceleration_unicycle(
    _: torch.Tensor, q0: torch.Tensor, dq0: torch.Tensor, times: torch.Tensor, u: torch.Tensor  # (N,3)  # (N,3)  # (T+1,)  # (T,2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic unicycle with constant accelerations a, b given by u[0].

    Includes shape and sanity checks after each major step.
    """
    # Unpack
    N = q0.shape[0]
    T1 = times.shape[0]
    assert u.shape[1] == 2, "u must have shape (T,2)"
    assert dq0.shape == q0.shape == (N, 3)
    assert times.dim() == 1

    a0, b0 = u[0, 0].item(), u[0, 1].item()

    # Initial state unpack
    x0, y0, th0 = q0[:, 0:1], q0[:, 1:2], q0[:, 2:3]
    dx0, dy0, dth0 = dq0[:, 0:1], dq0[:, 1:2], dq0[:, 2:3]
    cos0, sin0 = torch.cos(th0), torch.sin(th0)
    v0 = cos0 * dx0 + sin0 * dy0
    w0 = dth0
    # Checks
    assert v0.shape == (N, 1)
    assert w0.shape == (N, 1)

    # Time expansion
    t = times.view(1, T1, 1).expand(N, T1, 1)  # (N,T1,1)
    assert t.shape == (N, T1, 1)

    th = th0.view(N, 1, 1) + w0.view(N, 1, 1) * t + 0.5 * b0 * t * t
    v = v0.view(N, 1, 1) + a0 * t
    w = w0.view(N, 1, 1) + b0 * t
    # Sanity at t=0
    assert torch.allclose(th[:, 0, :], th0, atol=1e-12)
    assert torch.allclose(v[:, 0, :], v0, atol=1e-12)
    assert torch.allclose(w[:, 0, :], w0, atol=1e-12)

    eps_b = torch.finfo(q0.dtype).eps

    assert b0 >= 0, "b0 must be non-negative"
    if abs(b0) > eps_b:
        # Fresnel branch
        sqrt_pi_b = torch.sqrt(torch.tensor(b0 * math.pi, dtype=q0.dtype))
        arg = (w0 + b0 * t) / sqrt_pi_b
        arg_flat = arg.detach().cpu().numpy().reshape(-1)
        S_flat, C_flat = fresnel(arg_flat)
        S = torch.from_numpy(S_flat).view(N, T1, 1).to(q0)
        C = torch.from_numpy(C_flat).view(N, T1, 1).to(q0)
        assert S.shape == C.shape == (N, T1, 1)

        gamma = torch.cos(0.5 * w0**2 / b0 - th0)
        sigma = torch.sin(0.5 * w0**2 / b0 - th0)

        coeff = ((b0 * v0 - a0 * w0) * torch.sqrt(torch.tensor(torch.pi, device=q0.device)) / (b0**1.5)).view(N, 1, 1)
        phase = 0.5 * b0 * t**2 + w0 * t + th0

        Fx = coeff * (sigma * S + gamma * C) + (a0 / b0) * torch.sin(phase)
        Fy = coeff * (gamma * S - sigma * C) - (a0 / b0) * torch.cos(phase)
        # Baseline at t=0
        Fx0 = Fx[:, :1, :]
        Fy0 = Fy[:, :1, :]
        assert torch.allclose(Fx0, Fx0)  # trivial check

        dX = Fx - Fx0
        dY = Fy - Fy0
        # Sanity: at t=0, dX,dY = 0
        assert torch.allclose(dX[:, 0, :], torch.zeros(N, 1, device=q0.device), atol=1e-12)
        assert torch.allclose(dY[:, 0, :], torch.zeros(N, 1, device=q0.device), atol=1e-12)

    else:
        # b0 == 0 branch
        eps_w = torch.finfo(q0.dtype).eps
        w0_v = w0.view(N, 1, 1)
        v0_v = v0.view(N, 1, 1)
        th0_v = th0.view(N, 1, 1)
        dX = torch.zeros(N, T1, 1, device=q0.device, dtype=q0.dtype)
        dY = torch.zeros(N, T1, 1, device=q0.device, dtype=q0.dtype)

        mask = (w0.abs() > eps_w).squeeze(-1)  # shape (N,)
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            ω = w0_v[idx]
            v_sel = v0_v[idx]
            th_sel = th0_v[idx]
            t_sel = t[idx]
            phase = ω * t_sel + th_sel
            Fx = (a0 / ω**2) * torch.cos(phase) + (a0 * t_sel + v_sel) / ω * torch.sin(phase)
            Fy = (a0 / ω**2) * torch.sin(phase) - (a0 * t_sel + v_sel) / ω * torch.cos(phase)
            Fx0 = (a0 / ω**2) * torch.cos(th_sel) + v_sel / ω * torch.sin(th_sel)
            Fy0 = (a0 / ω**2) * torch.sin(th_sel) - v_sel / ω * torch.cos(th_sel)
            dX[idx] = Fx - Fx0
            dY[idx] = Fy - Fy0

        if (~mask).any():
            idx0 = (~mask).nonzero(as_tuple=True)[0]
            v0l = v0_v[idx0]
            th0l = th0_v[idx0]
            t0l = t[idx0]
            dX[idx0] = (v0l * t0l + 0.5 * a0 * t0l**2) * torch.cos(th0l)
            dY[idx0] = (v0l * t0l + 0.5 * a0 * t0l**2) * torch.sin(th0l)

    # Assemble q and dq
    x = x0.view(N, 1, 1) + dX
    y = y0.view(N, 1, 1) + dY
    q = torch.cat([x, y, th], dim=-1)
    bvel = torch.cat([v, torch.zeros_like(v), w], dim=-1)
    dq_flat = SE2.map_velocity_to_dq(q.reshape(-1, 3), bvel.reshape(-1, 3))
    dq = dq_flat.view(N, T1, 3)

    # Final sanity
    assert q.shape == (N, T1, 3)
    assert dq.shape == (N, T1, 3)
    return q, dq
