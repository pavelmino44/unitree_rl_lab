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
