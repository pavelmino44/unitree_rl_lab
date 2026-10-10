"""Fixed evaluation scenarios for the Go2 velocity policy.

A scenario = terrain + constant velocity command + spawn point + goal line.
Everything else (no randomization, no noise, fixed seed, 1 env) lives in
RobotEvalEnvCfg (velocity_env_cfg.py).

Coordinates spawn_x / goal_x / midpoint_x are in metres, along +x, relative to the CENTRE of
the terrain tile (the terrain origin). The robot spawns facing heading (default +x) and a
heading-hold event overrides the yaw-rate command every step, so the robot keeps walking
perpendicular to the obstacle even after stumbling.

NOTE: import this module only after the Isaac Sim app has been launched (it imports isaaclab.*).

To add a scenario, add one entry to EVAL_SCENARIOS at the bottom of this file.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Distance between the robot and the nearest edge of the obstacle at spawn.
SPAWN_CLEARANCE = 0.7

# Heading hold: wz = clip(HEADING_STIFFNESS * (heading - yaw), +-MAX_YAW_RATE). The yaw-rate limit
# should not exceed the range the policy was trained on (ang_vel_z in limit_ranges).
HEADING_STIFFNESS = 0.5
MAX_YAW_RATE = 1.0


##
# Reset event
##


def reset_root_state_fixed(
    env,
    env_ids: torch.Tensor,
    x: float,
    y: float = 0.0,
    yaw: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot to a fixed pose given relative to the terrain tile centre.

    x/y are measured from the tile centre. Height is the robot's default height above the WORLD
    ground (z = 0), not above the terrain origin: the origin of pyramid-like tiles lies on top of the
    pyramid, so using it would drop the robot from that height when it spawns on the flat border.
    """
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    pos = torch.stack([origins[:, 0] + x, origins[:, 1] + y, default[:, 2]], dim=-1)
    yaw_t = torch.full((len(env_ids),), yaw, device=env.device)
    zero = torch.zeros_like(yaw_t)
    quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zero, zero, yaw_t), default[:, 3:7])

    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def hold_heading(
    env,
    env_ids: torch.Tensor,
    heading: float,
    stiffness: float = HEADING_STIFFNESS,
    max_rate: float = MAX_YAW_RATE,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Keep the base yaw at heading (world frame) by overriding the yaw-rate command.

    Registered as an "interval" event that fires every env step. Interval events run after the command
    manager has updated the command and before observations are computed, so the policy sees the
    corrected command on the same step.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name).command  # (num_envs, 3): vx, vy, wz in body frame
    error = math_utils.wrap_to_pi(heading - asset.data.heading_w[env_ids])
    command[env_ids, 2] = torch.clip(stiffness * error, -max_rate, max_rate)


def goal_reached(
    env,
    goal_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination: the base crossed the goal line (x relative to the terrain tile centre)."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]) >= goal_x


##
# Scenario definition
##


@dataclass(frozen=True)
class EvalScenario:
    description: str
    sub_terrain: Any  # an isaaclab.terrains sub-terrain cfg (proportion must be 1.0)
    size: float  # side of the square terrain tile, m
    vx: float  # constant velocity command, m/s
    spawn_x: float  # spawn position relative to the tile centre, m
    goal_x: float  # success line relative to the tile centre, m
    vy: float = 0.0
    wz: float = 0.0  # only used when hold_heading is False
    heading: float = 0.0  # spawn yaw and target yaw, rad (0 = facing +x)
    hold_heading: bool = True
    midpoint_x: float | None = None  # optional milestone, e.g. the top of a pyramid
    timeout_margin_s: float = 5.0

    @property
    def episode_length_s(self) -> float:
        """2x the ideal time to reach the goal, plus a margin."""
        ideal = abs(self.goal_x - self.spawn_x) / max(abs(self.vx), 1e-3)
        return 2.0 * ideal + self.timeout_margin_s


def apply_eval_scenario(env_cfg, name: str) -> EvalScenario:
    """Apply scenario name to a RobotEvalEnvCfg instance (in place)."""
    if name not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown eval scenario '{name}'. Available: {', '.join(EVAL_SCENARIOS)}")
    sc = EVAL_SCENARIOS[name]

    # terrain: a single fixed tile surrounded by the generator's flat border
    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(sc.size, sc.size),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 0.0),
        use_cache=False,
        curriculum=False,
        sub_terrains={name: copy.deepcopy(sc.sub_terrain)},
    )
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # constant velocity command
    cmd = env_cfg.commands.base_velocity
    for r in (cmd.ranges, cmd.limit_ranges):
        r.lin_vel_x = (sc.vx, sc.vx)
        r.lin_vel_y = (sc.vy, sc.vy)
        r.ang_vel_z = (sc.wz, sc.wz)
    cmd.resampling_time_range = (1.0e6, 1.0e6)

    env_cfg.episode_length_s = sc.episode_length_s

    # fixed spawn pose
    env_cfg.events.reset_base = EventTerm(
        func=reset_root_state_fixed,
        mode="reset",
        params={"x": sc.spawn_x, "y": 0.0, "yaw": sc.heading},
    )

    # end the episode as soon as the goal line is crossed (not a time-out: counted as a termination)
    env_cfg.terminations.goal_reached = DoneTerm(func=goal_reached, params={"goal_x": sc.goal_x}, time_out=False)

    # keep walking in the spawn direction (closed loop on yaw)
    if sc.hold_heading:
        step_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.events.hold_heading = EventTerm(
            func=hold_heading,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            params={"heading": sc.heading},
        )
    return sc


##
# Scenario builders
##

_STAIRS_SIZE = 6.6  # with step_width=0.3 and platform_width=3.0 this gives 6 stair layers


def _flat() -> EvalScenario:
    return EvalScenario(
        description="Flat ground, 6 m straight walk",
        sub_terrain=terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        size=8.0,
        vx=0.5,
        spawn_x=-3.0,
        goal_x=3.0,
    )


def _stairs_up(step_h: float) -> EvalScenario:
    return EvalScenario(
        description=f"Pyramid staircase, {step_h * 100:.0f} cm steps, approached from the front",
        sub_terrain=terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(step_h, step_h),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.0,
            holes=False,
        ),
        size=_STAIRS_SIZE,
        vx=0.5,
        spawn_x=-(_STAIRS_SIZE / 2 + SPAWN_CLEARANCE),
        goal_x=_STAIRS_SIZE / 2 + SPAWN_CLEARANCE,  # far side of the pyramid: up AND down
        midpoint_x=0.0,  # centre of the top platform
    )


def _slope_up(slope: float) -> EvalScenario:
    size = 8.0
    return EvalScenario(
        description=f"Pyramid slope (rise/run = {slope}), approached from the front",
        sub_terrain=terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(slope, slope), platform_width=2.0, border_width=0.25
        ),
        size=size,
        vx=0.5,
        spawn_x=-(size / 2 + SPAWN_CLEARANCE),
        goal_x=size / 2 + SPAWN_CLEARANCE,
        midpoint_x=0.0,
    )


##
# Registry. Add new scenarios here.
##

EVAL_SCENARIOS: dict[str, EvalScenario] = {
    "flat": _flat(),
    **{f"stairs_up_{round(h * 100)}cm": _stairs_up(h) for h in (0.05, 0.10, 0.15, 0.20)},
    "slope_up_0p2": _slope_up(0.2),
    "slope_up_0p4": _slope_up(0.4),
}