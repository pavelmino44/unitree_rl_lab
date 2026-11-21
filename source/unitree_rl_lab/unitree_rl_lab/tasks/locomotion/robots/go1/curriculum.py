"""Go1 command curriculum evaluated over batches of completed episodes."""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase


class Go1VelocityCurriculum(ManagerTermBase):
    """Keep Go2's reward-based range expansion without synchronized resets.

    Isaac Lab calls this before clearing the resetting environments' reward
    sums. Aggregate at least num_envs episodes and wait at least one episode
    horizon between decisions. A single lucky reset cannot advance all robots.
    Short/fallen episodes contribute their reward divided by the FULL horizon,
    so a brief burst of tracking followed by a fall cannot pass the threshold.
    The inherited reset is deliberately a no-op: these are cross-episode stats.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._reward_sum = torch.zeros((), device=env.device)
        self._episode_count = 0
        self._last_check_step = env.common_step_counter
        self._tracking_score = 0.0

    def __call__(
        self,
        env,
        env_ids,
        reward_term_name: str = "track_lin_vel_xy",
        threshold: float = 0.8,
        increment: float = 0.05,
    ) -> dict[str, float]:
        command = env.command_manager.get_term("base_velocity")
        ranges = command.cfg.ranges
        limits = command.cfg.limit_ranges
        weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        if weight <= 0 or not (0 < threshold <= 1) or increment <= 0:
            raise ValueError(
                "Velocity curriculum needs positive weight/increment and threshold in (0, 1]"
            )

        # Exclude initialization resets, which contain no episode data.
        valid = env.episode_length_buf[env_ids] > 0
        scores = env.reward_manager._episode_sums[reward_term_name][env_ids]
        self._reward_sum += scores[valid].sum()
        self._episode_count += int(valid.sum().item())
        elapsed = env.common_step_counter - self._last_check_step
        if self._episode_count >= env.num_envs and elapsed >= env.max_episode_length:
            self._tracking_score = (
                self._reward_sum
                / (self._episode_count * env.max_episode_length_s * weight)
            ).item()
            if self._tracking_score >= threshold:
                ranges.lin_vel_x = (
                    max(limits.lin_vel_x[0], ranges.lin_vel_x[0] - increment),
                    min(limits.lin_vel_x[1], ranges.lin_vel_x[1] + increment),
                )
            # Start a fresh evidence window after every decision, successful or not.
            self._reward_sum.zero_()
            self._episode_count = 0
            self._last_check_step = env.common_step_counter

        return {
            "min_speed": float(ranges.lin_vel_x[0]),
            "max_speed": float(ranges.lin_vel_x[1]),
            "tracking_score": self._tracking_score,
        }
