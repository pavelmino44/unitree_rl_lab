"""Fixed evaluation scenarios for the Go2 velocity policy.

A scenario = terrain + constant velocity command + spawn point + goal line.
Everything else (no randomization, no noise, fixed seed, 1 env) lives in
``RobotEvalEnvCfg`` (velocity_env_cfg.py).

Coordinates ``spawn_x`` / ``goal_x`` / ``midpoint_x`` are in metres, along +x, relative to the CENTRE of
the terrain tile (the terrain origin). The robot spawns on the flat ground in front of the obstacle,
facing ``heading`` (default +x), and a heading-hold event overrides the yaw-rate command every step, so
the robot keeps walking perpendicular to the obstacle even after stumbling. The episode ends as soon as
the goal line is crossed. Pyramids are crossed completely (up AND down, or down AND up).

Two groups of scenarios:
  * hand-made sweeps:  ``flat``, ``stairs_up_<N>cm`` (N = 5, 10, 11, ..., 20), ``slope_up_<s>``
  * ``rough_*``: the exact sub-terrains of ``ROUGH_TERRAINS_CFG`` (the terrain set used for training),
    each at fixed difficulty levels. They are read from the same config object the training uses, so
    they follow any change you make to it.

NOTE: import this module only after the Isaac Sim app has been launched (it imports isaaclab.*).

To add a scenario, add one entry to ``EVAL_SCENARIOS`` at the bottom of this file.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch

import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # terrain set used for training

# Distance between the robot and the nearest edge of the obstacle at spawn.
SPAWN_CLEARANCE = 0.7

# Heading hold: wz = clip(HEADING_STIFFNESS * (heading - yaw), +-MAX_YAW_RATE). The yaw-rate limit
# should not exceed the range the policy was trained on (ang_vel_z in limit_ranges).
HEADING_STIFFNESS = 0.5
MAX_YAW_RATE = 1.0


##
# Events / terminations
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
    ground (z = 0), not above the terrain origin: the origin of pyramid-like tiles lies on top of (or
    inside) the obstacle, so using it would drop the robot from that height when it spawns on the flat
    border. The flat ground around every obstacle of the training terrain set is at z = 0.
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
    """Keep the base yaw at ``heading`` (world frame) by overriding the yaw-rate command.

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
    sub_terrain: Any  # an isaaclab.terrains sub-terrain cfg (proportion is forced to 1.0)
    size: float  # side of the square terrain tile, m
    vx: float  # constant velocity command, m/s
    spawn_x: float  # spawn position relative to the tile centre, m
    goal_x: float  # success line relative to the tile centre, m
    vy: float = 0.0
    wz: float = 0.0  # only used when hold_heading is False
    heading: float = 0.0  # spawn yaw and target yaw, rad (0 = facing +x)
    hold_heading: bool = True
    difficulty: float = 0.0  # terrain difficulty in [0, 1] (interpolates the *_range fields of the sub-terrain)
    midpoint_x: float | None = None  # optional milestone, e.g. the top of a pyramid
    midpoint_label: str = "top"  # what the midpoint is: top / bottom / center
    timeout_margin_s: float = 5.0

    @property
    def episode_length_s(self) -> float:
        """2x the ideal time to reach the goal, plus a margin."""
        ideal = abs(self.goal_x - self.spawn_x) / max(abs(self.vx), 1e-3)
        return 2.0 * ideal + self.timeout_margin_s


def apply_eval_scenario(env_cfg, name: str) -> EvalScenario:
    """Apply scenario ``name`` to a ``RobotEvalEnvCfg`` instance (in place).

    ``env_cfg.seed`` must be set BEFORE calling this: it also seeds the terrain generator, which
    matters for the randomized terrains (boxes, random_rough).
    """
    if name not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown eval scenario '{name}'. Available: {', '.join(EVAL_SCENARIOS)}")
    sc = EVAL_SCENARIOS[name]

    # terrain: a single fixed tile. Start from the training generator so that scales, border and slope
    # threshold are identical to training.
    gen = copy.deepcopy(ROUGH_TERRAINS_CFG)
    gen.size = (sc.size, sc.size)
    gen.num_rows = 1
    gen.num_cols = 1
    gen.curriculum = False
    gen.difficulty_range = (sc.difficulty, sc.difficulty)
    gen.use_cache = False
    gen.seed = env_cfg.seed
    sub = copy.deepcopy(sc.sub_terrain)
    sub.proportion = 1.0
    gen.sub_terrains = {name: sub}
    env_cfg.scene.terrain.terrain_generator = gen
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
# Hand-made scenario builders
##

_STAIRS_SIZE = 6.6  # with step_width=0.3 and platform_width=3.0 this gives 6 stair layers

# Step heights of the hand-made staircase sweep, in cm: 5 and 10 as before, then every centimetre up to 20.
STAIRS_STEP_HEIGHTS_CM: tuple[int, ...] = (5, *range(10, 21))


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
# Scenarios derived from ROUGH_TERRAINS_CFG (the terrain set used for training)
##

# ROUGH sub-terrain key -> difficulty levels to test. None = the generator ignores difficulty for it.
# Stairs only differ by a few millimetres across difficulty in ROUGH, so only the hardest level is tested.
_ROUGH_LEVELS: dict[str, tuple[float | None, ...]] = {
    "pyramid_stairs": (1.0,),
    "pyramid_stairs_inv": (1.0,),
    "boxes": (0.5, 1.0),
    "random_rough": (None,),
    "hf_pyramid_slope": (0.5, 1.0),
    "hf_pyramid_slope_inv": (0.5, 1.0),
}


def _lerp(rng: tuple[float, float], d: float) -> float:
    return rng[0] + d * (rng[1] - rng[0])


def _obstacle_half_width(sub, size: float) -> float:
    """Half of the side of the part of the tile that actually contains the obstacle."""
    if hasattr(sub, "grid_width"):  # MeshRandomGridTerrain: whole cells + a thin flat border
        return int(size / sub.grid_width) * sub.grid_width / 2.0
    return size / 2.0 - getattr(sub, "border_width", 0.0)


def _describe_obstacle(sub, d: float) -> str:
    if hasattr(sub, "step_height_range"):
        return f"steps {_lerp(sub.step_height_range, d) * 100:.1f} cm high, {sub.step_width * 100:.0f} cm wide"
    if hasattr(sub, "slope_range"):
        s = _lerp(sub.slope_range, d)
        return f"slope {s:.2f} rise/run ({math.degrees(math.atan(s)):.0f} deg)"
    if hasattr(sub, "grid_height_range"):
        return f"random blocks {sub.grid_width * 100:.0f} cm wide, height +-{_lerp(sub.grid_height_range, d) * 100:.1f} cm"
    if hasattr(sub, "noise_range"):
        return f"uniform height noise {sub.noise_range[0] * 100:.0f}..{sub.noise_range[1] * 100:.0f} cm"
    return type(sub).__name__


def _rough_scenario(key: str, difficulty: float | None) -> tuple[str, EvalScenario]:
    sub = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains[key])
    size = float(ROUGH_TERRAINS_CFG.size[0])
    half = _obstacle_half_width(sub, size)
    d = 1.0 if difficulty is None else difficulty

    inverted = bool(getattr(sub, "inverted", False)) or "Inverted" in type(sub).__name__
    if hasattr(sub, "noise_range"):  # rough ground: no obvious milestone
        midpoint_x, label = None, "top"
    elif hasattr(sub, "grid_height_range"):
        midpoint_x, label = 0.0, "center"
    else:
        midpoint_x, label = 0.0, ("bottom" if inverted else "top")

    name = f"rough_{key}" + ("" if difficulty is None else f"_d{round(difficulty * 100)}")
    level = "" if difficulty is None else f" @ difficulty {d:.2f}"
    return name, EvalScenario(
        description=f"ROUGH '{key}'{level}: {_describe_obstacle(sub, d)}; crossed completely",
        sub_terrain=sub,
        size=size,
        vx=0.5,
        spawn_x=-(half + SPAWN_CLEARANCE),
        goal_x=half + SPAWN_CLEARANCE,
        difficulty=d,
        midpoint_x=midpoint_x,
        midpoint_label=label,
    )


##
# Registry. Add new scenarios here.
##

EVAL_SCENARIOS: dict[str, EvalScenario] = {
    "flat": _flat(),
    **{f"stairs_up_{cm}cm": _stairs_up(cm / 100.0) for cm in STAIRS_STEP_HEIGHTS_CM},
    "slope_up_0p2": _slope_up(0.2),
    "slope_up_0p4": _slope_up(0.4),
}

for _key in ROUGH_TERRAINS_CFG.sub_terrains:
    if _key not in _ROUGH_LEVELS:
        warnings.warn(f"eval_scenarios: ROUGH sub-terrain '{_key}' has no evaluation scenario (add it to _ROUGH_LEVELS).")
for _key, _levels in _ROUGH_LEVELS.items():
    if _key not in ROUGH_TERRAINS_CFG.sub_terrains:
        warnings.warn(f"eval_scenarios: '{_key}' is not in ROUGH_TERRAINS_CFG, scenario skipped.")
        continue
    for _level in _levels:
        _name, _scenario = _rough_scenario(_key, _level)
        EVAL_SCENARIOS[_name] = _scenario