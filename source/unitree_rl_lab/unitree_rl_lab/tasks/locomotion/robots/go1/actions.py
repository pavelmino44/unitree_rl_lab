"""Position action with a lightweight approximation of T2 safety filtering.

Unlike the real T2 Safety Block, this term does NOT perform collision checking.
It enforces joint limits and caps target-to-target movement to under five degrees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class RateLimitedJointPositionAction(JointPositionAction):
    """Apply a rate limit to joint-position targets between policy ticks."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not isinstance(self._offset, torch.Tensor):
            raise ValueError(
                "RateLimitedJointPositionAction requires use_default_offset=True"
            )
        if (
            len(cfg.min_target) != self.action_dim
            or len(cfg.max_target) != self.action_dim
        ):
            raise ValueError(
                "min_target/max_target must match the resolved joint count"
            )
        self._q_min = torch.tensor(
            cfg.min_target, dtype=self._offset.dtype, device=self.device
        )[None, :]
        self._q_max = torch.tensor(
            cfg.max_target, dtype=self._offset.dtype, device=self.device
        )[None, :]
        if torch.any(self._q_min >= self._q_max):
            raise ValueError("min_target must be strictly below max_target")
        if not (0.0 < cfg.max_delta_rad < math.radians(5.0)):
            raise ValueError(
                "max_delta_rad must be positive and strictly below 5 degrees"
            )
        self._last_target = torch.minimum(
            torch.maximum(self._offset.clone(), self._q_min), self._q_max
        )
        self._requested_target = self._last_target.clone()

    @property
    def requested_target(self) -> torch.Tensor:
        return self._requested_target

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(
            torch.clamp(actions, -self.cfg.raw_action_clip, self.cfg.raw_action_clip)
        )
        self._requested_target = self._processed_actions.clone()
        requested = torch.minimum(
            torch.maximum(self._processed_actions, self._q_min), self._q_max
        )
        delta = torch.clamp(
            requested - self._last_target,
            -self.cfg.max_delta_rad,
            self.cfg.max_delta_rad,
        )
        self._processed_actions = self._last_target + delta
        self._last_target.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._last_target[:] = torch.minimum(
                torch.maximum(self._offset, self._q_min), self._q_max
            )
        else:
            self._last_target[env_ids] = torch.minimum(
                torch.maximum(self._offset[env_ids], self._q_min), self._q_max
            )
        ids = slice(None) if env_ids is None else env_ids
        self._requested_target[ids] = self._last_target[ids]
        self._processed_actions[ids] = self._last_target[ids]


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RateLimitedJointPositionAction
    max_delta_rad: float = math.radians(4.5)
    raw_action_clip: float = 1.0
    min_target: tuple[float, ...] = ()
    max_target: tuple[float, ...] = ()
