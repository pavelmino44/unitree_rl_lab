"""Small, deployment-neutral rewards for Go1 T3 flat training.

These do not use camera, terrain scans, or privileged observations at inference.
They operate only inside training's reward calculation.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def signed_forward_velocity(env, max_speed: float = 0.6) -> torch.Tensor:
    """Reward SIGNED forward body velocity, unlike an instantaneous Gaussian alone.

    Summing this reward over time makes forward/backward oscillations cancel
    rather than letting brief positive-velocity peaks dominate velocity tracking.
    """
    robot = env.scene["robot"]
    return robot.data.root_lin_vel_b[:, 0].clamp(min=-max_speed, max=max_speed)


def forward_gated_velocity_tracking(
    env,
    command_name: str,
    std: float,
    min_command_speed: float = 0.1,
) -> torch.Tensor:
    """Velocity-tracking reward that cannot be collected by standing still.

    For a positive forward command, multiply Gaussian tracking by the fraction
    of the requested forward speed that has actually been achieved. This
    fraction is clamped in [0, 1], so standing / moving backwards earns ZERO
    linear tracking reward, while overspeed is still penalized by the Gaussian.

    For near-zero forward commands, fall back to standard Gaussian tracking
    (useful if stationary commands are enabled in future training stages).
    The velocity and command are both expressed in the robot body frame.
    """
    robot = env.scene["robot"]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    actual_xy = robot.data.root_lin_vel_b[:, :2]
    error_sq = torch.sum(torch.square(command_xy - actual_xy), dim=-1)
    tracking = torch.exp(-error_sq / (std * std))

    forward_command = command_xy[:, 0] > min_command_speed
    fraction = (actual_xy[:, 0] / command_xy[:, 0].clamp_min(min_command_speed)).clamp(0.0, 1.0)
    return torch.where(forward_command, tracking * fraction, tracking)


def normalized_forward_progress(
    env,
    command_name: str,
    min_command_speed: float = 0.1,
) -> torch.Tensor:
    """Signed progress: -1 at reverse commanded speed, 0 standing, +1 at target.

    Unlike a raw forward-speed bonus, it saturates once commanded speed is
    reached; this avoids making running faster than requested more profitable.
    """
    command_x = env.command_manager.get_command(command_name)[:, 0]
    actual_x = env.scene["robot"].data.root_lin_vel_b[:, 0]
    fraction = (actual_x / command_x.clamp_min(min_command_speed)).clamp(-1.0, 1.0)
    return torch.where(command_x > min_command_speed, fraction, torch.zeros_like(fraction))


def rear_feet_narrow_penalty(
    env,
    min_width: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize rear feet closer in the lateral body axis than ``min_width``.

    ``asset_cfg.body_ids`` must resolve exactly [RR_foot, RL_foot], in that order.
    The result is dimensionless: 0 at >= min_width, 1 at zero separation.
    """
    robot = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("rear_feet_narrow_penalty expects exactly RR_foot, RL_foot")
    rr_id, rl_id = asset_cfg.body_ids
    # Relative vector is immune to world-space translation and follows body yaw.
    rr_to_rl_w = robot.data.body_pos_w[:, rl_id] - robot.data.body_pos_w[:, rr_id]
    rr_to_rl_b = quat_apply_inverse(robot.data.root_quat_w, rr_to_rl_w)
    lateral_separation = rr_to_rl_b[:, 1]
    return ((min_width - lateral_separation) / min_width).clamp(min=0.0, max=2.0)



def diagonal_trot_contact_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    contact_threshold: float = 2.0,
    min_command_speed: float = 0.1,
    full_reward_speed: float = 0.2,
) -> torch.Tensor:
    """Small optional reward for diagonal support, not a clock-driven gait.

    Foot order must be [FR, FL, RR, RL]. The two acceptable support states
    are FR+RL (swing FL+RR) or FL+RR (swing FR+RL). Both-front/rear pairs,
    four-foot stance and flight receive zero. At 0.3 m/s the term is fully
    weighted; it is scaled down if the robot does not actually move forward.

    Training-only contact sensor signal; no new actor observations.
    """
    sensor = env.scene[sensor_cfg.name]
    if len(sensor_cfg.body_ids) != 4:
        raise ValueError("diagonal_trot_contact_reward needs exactly 4 ordered foot sensors")
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = (torch.linalg.vector_norm(forces, dim=-1) > contact_threshold).to(forces.dtype)
    fr, fl, rr, rl = contact.unbind(dim=1)
    diagonal_1 = fr * rl * (1.0 - fl) * (1.0 - rr)
    diagonal_2 = fl * rr * (1.0 - fr) * (1.0 - rl)

    command_x = env.command_manager.get_command(command_name)[:, 0]
    commanded = (command_x > min_command_speed).to(forces.dtype)
    forward_speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    progress_gate = torch.clamp(forward_speed / full_reward_speed, min=0.0, max=1.0)
    return (diagonal_1 + diagonal_2) * commanded * progress_gate
