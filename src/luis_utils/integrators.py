"""Numerical integrators for dynamical systems."""

import time
from typing import Tuple

import numpy as np
import torch
from scipy.stats import linregress

from luis_utils.algorithms import ROBOT_TO_KEY, ConfigMapper, VelocityMapper
from luis_utils.env import SecondOrderEnv, rollout_analytical_trajectory
from luis_utils.systems.base import SecondOrderSystem


def fe(
    system: SecondOrderSystem,
    q: torch.Tensor,
    dq: torch.Tensor,
    u: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward Euler integration for dynamical systems."""
    d_conf, d_vel = system.f(q, dq, u)
    q_next = system.update_configuration(q, d_conf, dt)
    dq_next = system.update_velocity(dq, d_vel, dt)
    return q_next, dq_next


def estimate_order(errors_by_dt):
    """Estimate the order of an integrator."""
    filtered = []
    for dt, err in errors_by_dt:
        if err < 10e-14:
            continue
        filtered.append((dt, err))
    if not filtered:
        return r"$\approx \varepsilon$"

    log_dts = np.log([dt for dt, err in filtered])
    log_errs = np.log([err for dt, err in filtered])
    slope, _, _, _, _ = linregress(log_dts, log_errs)
    return slope


def get_integrators_orders(
    integrator_list, analytical_system_key, real_system, algo, q0, dq0, q_reference_space, analytical_func, u
):  # pylint: disable=too-many-positional-arguments
    """Get the orders of a list of integrators."""
    dt_values = np.logspace(np.log10(0.001), np.log10(0.1), 5)

    accuracy_metrics = {}
    num_steps_for_dt = {dt: int(1.0 / dt) for dt in dt_values}  # e.g., 1s total simulation

    real_system_key = ROBOT_TO_KEY[real_system.__class__]

    for integrator_name in integrator_list:

        metrics = {"q_errors": {}, "dt_values": dt_values, "total_time": {}, "time_per_step": {}, "orders": {}}
        errors_by_dt = []
        for dt in dt_values:
            num_steps = num_steps_for_dt[dt]
            runtimes_total = []
            runtimes_per_step = []
            real_q0 = ConfigMapper.map(from_key=(analytical_system_key[0], q_reference_space), to_key=real_system_key, q=q0)
            real_dq0 = VelocityMapper.map(from_key=(analytical_system_key[0], q_reference_space), to_key=real_system_key, q=q0, dq_or_v=dq0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            env = SecondOrderEnv(real_system, dt, max_steps_per_episode=num_steps, integrator_name=integrator_name, decimation=1)
            env_dict = env.rollout_episode(
                algorithm=algo,
                q0=real_q0,
                dq0=real_dq0,
                output_space=analytical_system_key[1],
                ep_index=0,
                noise=None,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            runtimes_total.append(t1 - t0)
            runtimes_per_step.append(runtimes_total[-1] / num_steps)

            q0_analytical = ConfigMapper.map(from_key=(analytical_system_key[0], q_reference_space), to_key=analytical_system_key, q=q0)
            dq0_analytical = VelocityMapper.map(
                from_key=(analytical_system_key[0], q_reference_space), to_key=analytical_system_key, q=q0, dq_or_v=dq0
            )

            analytical_traj = rollout_analytical_trajectory(env_dict, real_system, q0_analytical, dq0_analytical, u, analytical_func)

            error = ((env_dict["q_traj"] - analytical_traj["q_traj"]) ** 2).mean().sqrt().item()
            errors_by_dt.append((dt, error))
            metrics["q_errors"][dt] = error
            metrics["total_time"][dt] = runtimes_total[-1]
            metrics["time_per_step"][dt] = runtimes_per_step[-1]

        metrics["orders"][integrator_name] = estimate_order(errors_by_dt)
        accuracy_metrics[integrator_name] = metrics
    return accuracy_metrics
