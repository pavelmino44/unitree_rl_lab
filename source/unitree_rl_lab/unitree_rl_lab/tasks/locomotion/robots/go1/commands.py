"""Uniform Go2-style commands with episode-mean Go1 diagnostics."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from unitree_rl_lab.tasks.locomotion.mdp.commands import UniformLevelVelocityCommandCfg


class Go1VelocityCommand(UniformVelocityCommand):
    """Average actual samples, independently of command resampling intervals.

    Upstream velocity errors accumulate over an episode but are divided by
    max command duration. With 20 s episodes and 10 s commands this doubles
    the reported full-episode error. These metrics use running sample means.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._go1_env = env
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        for name in (
            "velocity_x",
            "command_velocity_x",
            "angular_velocity_z",
            "command_angular_velocity_z",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for name in ("bound_clip", "step_clip", "nonfinite_reject", "torque_reject",
                     "collision_reject", "accepted", "exact_collision_check"):
            self.metrics["safety_" + name] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        self._metric_steps += 1
        velocity = self.robot.data.root_lin_vel_b
        samples = {
            "error_vel_xy": torch.linalg.vector_norm(
                self.command[:, :2] - velocity[:, :2], dim=-1
            ),
            "error_vel_yaw": torch.abs(
                self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
            ),
            "velocity_x": velocity[:, 0],
            "command_velocity_x": self.command[:, 0],
            "angular_velocity_z": self.robot.data.root_ang_vel_b[:, 2],
            "command_angular_velocity_z": self.command[:, 2],
        }
        manager = getattr(self._go1_env, "action_manager", None)
        if manager is not None:
            term = manager.get_term("JointPositionAction")
            for name, value in term.safety_metrics.items():
                samples["safety_" + name] = value
        for name, sample in samples.items():
            self.metrics[name] += (sample - self.metrics[name]) / self._metric_steps

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        self._metric_steps[slice(None) if env_ids is None else env_ids] = 0
        return result

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Keep the exact Go2 target/actual XY arrows, and add axial yaw arrows.
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and not hasattr(self, "goal_yaw_visualizer"):
            self.goal_yaw_visualizer = VisualizationMarkers(
                self.cfg.goal_yaw_visualizer_cfg
            )
            self.current_yaw_visualizer = VisualizationMarkers(
                self.cfg.current_yaw_visualizer_cfg
            )
        if hasattr(self, "goal_yaw_visualizer"):
            self.goal_yaw_visualizer.set_visibility(debug_vis)
            self.current_yaw_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized:
            return
        for visualizer, velocity, offset in (
            (self.goal_yaw_visualizer, self.command[:, 2], -0.12),
            (self.current_yaw_visualizer, self.robot.data.root_ang_vel_b[:, 2], 0.12),
        ):
            position = self.robot.data.root_pos_w.clone()
            position[:, 0] += offset
            position[:, 2] += 0.8
            scale, quat = self._resolve_yaw_velocity_to_arrow(velocity)
            visualizer.visualize(position, quat, scale)

    def _resolve_yaw_velocity_to_arrow(self, velocity: torch.Tensor):
        """Axial vector: +wz points up, -wz down, length tracks |wz|.

        Zero speed has zero longitudinal scale, so it cannot suggest rotation.
        The marker is a yaw component indicator, independent of body tilt.
        """
        scale = torch.empty((velocity.shape[0], 3), device=velocity.device)
        scale[:, 0] = velocity.abs() * 0.3
        scale[:, 1:] = 0.08
        zero = torch.zeros_like(velocity)
        pitch = torch.where(velocity >= 0, -torch.pi / 2, torch.pi / 2)
        return scale, quat_from_euler_xyz(zero, pitch, zero)


@configclass
class Go1VelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = Go1VelocityCommand
    goal_yaw_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/go1_yaw_goal"
    )
    current_yaw_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/go1_yaw_current")
    )
