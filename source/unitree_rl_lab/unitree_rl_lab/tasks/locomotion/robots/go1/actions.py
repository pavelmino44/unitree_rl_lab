"""Go1 joint-command filtering, with optional exact CPU T2 collision checks.

Default PPO uses batched joint/step/torque checks. It does not run the MuJoCo
collision sweep; select collision_check='mujoco' for full target validation.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .safety import FLOAT_GUARD_RAD, TORQUE_LIMITS, accept_targets, candidate_targets, torque_acceptance
from .safety_reference.joint_limits import Q_MAX, Q_MIN, STEP_LIMIT_RAD


class RateLimitedJointPositionAction(JointPositionAction):
    """Keep the preceding accepted whole command when validation fails."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.action_dim != 12 or not isinstance(self._offset, torch.Tensor):
            raise ValueError("Go1 safety requires 12 joints and use_default_offset=True")
        from unitree_rl_lab.assets.robots.go1 import GO1_SDK_JOINTS
        if list(self._joint_names) != list(GO1_SDK_JOINTS):
            raise ValueError("Go1 safety requires FR/FL/RR/RL hip/thigh/calf order")
        if list(cfg.min_target) != Q_MIN or list(cfg.max_target) != Q_MAX:
            raise ValueError("Go1 safety bounds must match the T2 joint contract")
        if not (FLOAT_GUARD_RAD < cfg.max_delta_rad <= STEP_LIMIT_RAD):
            raise ValueError("Target delta must be positive and at most 5 degrees")
        if cfg.clip is not None or math.isfinite(cfg.raw_action_clip):
            raise ValueError("Use only T2 target limits; raw action clipping must be disabled")
        if cfg.collision_check not in ("none", "mujoco"):
            raise ValueError("collision_check must be 'none' or 'mujoco'")
        self._q_min = torch.tensor(Q_MIN, device=self.device)[None, :] + FLOAT_GUARD_RAD
        self._q_max = torch.tensor(Q_MAX, device=self.device)[None, :] - FLOAT_GUARD_RAD
        self._torque_limits = torch.tensor(TORQUE_LIMITS, device=self.device)[None, :]
        self._kp = self._asset.data.default_joint_stiffness[:, self._joint_ids]
        self._kd = self._asset.data.default_joint_damping[:, self._joint_ids]
        self._last_target = self._offset.clone().clamp(self._q_min, self._q_max)
        self._requested_target = self._last_target.clone()
        self.safety_metrics = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("bound_clip", "step_clip", "nonfinite_reject", "torque_reject",
                         "collision_reject", "accepted", "exact_collision_check")
        }
        self._collision_checker = None
        if cfg.collision_check == "mujoco":
            from pathlib import Path
            from unitree_rl_lab.assets.robots.go1 import GO1_ASSET_DIR
            from .safety_reference.collision_checker import CollisionChecker
            reference = Path(GO1_ASSET_DIR) / "reference"
            pairs = reference / "t2_collision_pairs.csv"
            expected = {
                "go1.xml": "d5a7466784c8e72fd174cc7fa5a92754a1a0a6d458a1dec79e7ca0abc9525c4a",
                "scene.xml": "eceb4231bf2a3b0a190296963b38b904ed77dc14f5770bbcdd67fb0be8c49374",
                "t2_collision_pairs.csv": "5ef3f72b624ad08c1495407f958617c1dd73f7872227997cfaf46c37ed5ea987",
            }
            for filename, digest in expected.items():
                if hashlib.sha256((reference / filename).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"T2 collision reference differs: {filename}")
            self._collision_checker = CollisionChecker(str(reference / "scene.xml"), str(pairs))
            if self._collision_checker.joint_names != list(GO1_SDK_JOINTS):
                raise ValueError("MuJoCo checker joint order differs from action order")
            if len(self._collision_checker.pairs) != 655:
                raise ValueError("Expected 655 checked T2 collision pairs")
            self.safety_metrics["exact_collision_check"].fill_(1.0)
        print(f"[GO1 ACTION SAFETY] raw_clip=disabled; target_step={math.degrees(cfg.max_delta_rad):g}deg max; "
              f"bounds=T2; torque_reject={cfg.check_torques}; collision_check={cfg.collision_check}")

    @property
    def requested_target(self) -> torch.Tensor:
        return self._requested_target

    def export_safety_config(self):
        """Preserve exact contractual limits after the generic 3-digit export."""
        return {
            "min_target": list(Q_MIN), "max_target": list(Q_MAX),
            "max_delta_rad": self.cfg.max_delta_rad,
            "raw_action_clip": math.inf,
            "float_guard_rad": FLOAT_GUARD_RAD,
            "torque_limits_nm": list(TORQUE_LIMITS),
            "check_torques": self.cfg.check_torques,
            "collision_check": self.cfg.collision_check,
        }

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._requested_target = self._processed_actions.clone()
        candidate, finite, bound_clip, step_clip = candidate_targets(
            self._requested_target, self._last_target,
            self._q_min, self._q_max, self.cfg.max_delta_rad,
        )
        collision_ok = torch.ones_like(finite)
        if self._collision_checker is not None:
            # Exact reference implementation: CPU sweep at 0.1 degree, 655 pairs.
            # This explicit mode can be expensive with thousands of PPO envs.
            previous_cpu = self._last_target.detach().cpu().numpy()
            candidate_cpu = candidate.detach().cpu().numpy()
            for index in finite.nonzero(as_tuple=False).flatten().tolist():
                collision_ok[index] = self._collision_checker.check_transition(
                    previous_cpu[index], candidate_cpu[index]
                )
        torque_ok = torch.ones_like(finite)
        if self.cfg.check_torques:
            torque_ok, _ = torque_acceptance(
                candidate, self._asset.data.joint_pos[:, self._joint_ids],
                self._asset.data.joint_vel[:, self._joint_ids],
                self._kp, self._kd, self._torque_limits,
            )
        self._processed_actions, accepted = accept_targets(
            candidate, self._last_target, finite, torque_ok, collision_ok
        )
        self._last_target.copy_(self._processed_actions)
        values = {
            "bound_clip": bound_clip, "step_clip": step_clip,
            "nonfinite_reject": ~finite,
            "collision_reject": finite & ~collision_ok,
            "torque_reject": finite & collision_ok & ~torque_ok,
            "accepted": accepted,
        }
        for name, value in values.items():
            self.safety_metrics[name].copy_(value)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # SafetyNode seeds last_safe_q from the first measured LowState.
        measured = self._asset.data.joint_pos[:, self._joint_ids]
        self._last_target[ids] = measured[ids].clamp(self._q_min, self._q_max)
        self._requested_target[ids] = self._last_target[ids]
        self._processed_actions[ids] = self._last_target[ids]
        for name, value in self.safety_metrics.items():
            if name != "exact_collision_check":
                value[ids] = 0


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RateLimitedJointPositionAction
    max_delta_rad: float = STEP_LIMIT_RAD
    # Compatibility with older diagnostic scripts; inf explicitly disables it.
    raw_action_clip: float = math.inf
    min_target: tuple[float, ...] = tuple(Q_MIN)
    max_target: tuple[float, ...] = tuple(Q_MAX)
    check_torques: bool = True
    collision_check: str = "none"
