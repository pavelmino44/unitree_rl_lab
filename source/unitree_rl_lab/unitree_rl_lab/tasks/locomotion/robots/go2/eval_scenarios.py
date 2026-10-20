"""Fixed evaluation scenarios for the Go2 velocity policy.

A scenario = terrain + velocity command (constant or scheduled in time) + spawn point + end condition.
Everything else (no randomization, no noise, fixed seed, 1 env) lives in RobotEvalEnvCfg
(velocity_env_cfg.py).

Coordinates spawn_x / goal_x / midpoint_x are in metres along the course direction
course_dir (default +x; sideways and backward walks use another world direction), relative to the CENTRE
of the terrain tile. The robot spawns facing heading (default +x), and a heading-hold event overrides
the yaw-rate command every step, so the robot keeps its heading even after stumbling. Pyramids are crossed
completely (up AND down, or down AND up).

Kinds of motion (EvalScenario.motion):
  * walk          cross the goal line (distance along course_dir); the episode ends there
  * turn          rotate on the spot until the total rotation reaches goal_x degrees
  * sprint_brake  accelerate straight to the maximum speed of the training range, then command 0 at
                      brake_time_s and watch the stop; succeeds when the robot survives and stops

Scenario families:
  * flat, flat_backward, flat_strafe_right, flat_strafe_left, flat_turn_360,
    flat_max_speed
  * rough_*: the exact sub-terrains of ROUGH_TERRAINS_CFG (the terrain set used for training), read
    from the same config object the training uses, so they follow any change made to it:
      - rough_<terrain>_d<NN>                 fixed difficulty level NN % (interpolates the terrain ranges)
      - rough_pyramid_stairs_<N>cm            the ROUGH stairs with a fixed step height of N cm
      - rough_pyramid_stairs_inv_<N>cm        the same for the inverted pyramid (a pit)
      - rough_hf_pyramid_slope_<A>deg         the ROUGH slope with a fixed angle of A degrees
  * slope_up_<s>: hand-made slopes (rise/run = s)

The commanded speed of the walk scenarios can be overridden with apply_eval_scenario(..., speed=...)
(--speed in evaluate.py / play.py). It is ignored by flat_turn_360 and flat_max_speed.

NOTE: import this module only after the Isaac Sim app has been launched (it imports isaaclab.*).

To add a scenario, add one entry to EVAL_SCENARIOS at the bottom of this file.
"""

from __future__ import annotations

import copy
import dataclasses
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

# Defaults of the flat-ground motions (inside the training ranges lin_vel_x/lin_vel_y/ang_vel_z).
BACKWARD_SPEED = 0.5  # m/s
SIDEWAYS_SPEED = 0.3  # m/s
TURN_RATE = 0.8  # rad/s
SPRINT_BRAKE_TIME_S = 6.0  # the stop command is given after this long
SPRINT_EPISODE_S = 10.0


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
    """Keep the base yaw at heading (world frame) by overriding the yaw-rate command.

    Registered as an "interval" event that fires every env step. Interval events run after the command
    manager has updated the command and before observations are computed, so the policy sees the
    corrected command on the same step.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name).command  # (num_envs, 3): vx, vy, wz in body frame
    error = math_utils.wrap_to_pi(heading - asset.data.heading_w[env_ids])
    command[env_ids, 2] = torch.clip(stiffness * error, -max_rate, max_rate)


def command_schedule(
    env,
    env_ids: torch.Tensor,
    schedule: tuple,
    command_name: str = "base_velocity",
):
    """Piecewise-constant command in episode time: schedule = ((t_start_s, vx, vy, wz), ...) sorted by time.

    Registered as an "interval" event that fires every env step (before hold_heading, which then overrides
    the yaw rate). The command that the policy sees right after a reset is the first segment (set through the
    command ranges).
    """
    command = env.command_manager.get_term(command_name).command  # (num_envs, 3): vx, vy, wz in body frame
    time_s = env.episode_length_buf[env_ids].float() * env.step_dt  # time since the reset, at the end of the step
    new = torch.zeros(len(env_ids), 3, device=env.device)
    for start_s, vx, vy, wz in schedule:
        segment = torch.tensor([vx, vy, wz], device=env.device).expand(len(env_ids), 3)
        new = torch.where((time_s >= start_s - 1e-6).unsqueeze(-1), segment, new)
    command[env_ids, :] = new


def goal_reached(
    env,
    goal_x: float,
    dir_x: float = 1.0,
    dir_y: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination: the base crossed the goal line (coordinate along (dir_x, dir_y), tile centre = 0)."""
    asset = env.scene[asset_cfg.name]
    offset = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (offset[:, 0] * dir_x + offset[:, 1] * dir_y) >= goal_x


def rotation_reached(
    env,
    goal_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination: the base has turned by goal_deg degrees in total since the reset (either direction).

    The yaw rate is integrated over the steps, in a buffer that lives on the env and restarts with every episode.
    """
    asset = env.scene[asset_cfg.name]
    total = getattr(env, "_eval_yaw_total", None)
    if total is None or total.shape[0] != env.num_envs:
        total = torch.zeros(env.num_envs, device=env.device)
    total = torch.where(env.episode_length_buf <= 1, torch.zeros_like(total), total)  # a new episode has started
    total = total + asset.data.root_ang_vel_w[:, 2] * env.step_dt
    env._eval_yaw_total = total
    return torch.rad2deg(total.abs()) >= goal_deg


##
# Scenario definition
##


@dataclass(frozen=True)
class EvalScenario:
    description: str
    sub_terrain: Any  # an isaaclab.terrains sub-terrain cfg (proportion is forced to 1.0)
    size: float  # side of the square terrain tile, m
    vx: float  # commanded velocity (first segment of the schedule), body frame: forward m/s
    spawn_x: float  # spawn coordinate along course_dir relative to the tile centre (m)
    goal_x: float | None  # end line along course_dir (m; degrees for motion="turn"); None = no goal line
    vy: float = 0.0  # sideways m/s (left is positive)
    wz: float = 0.0  # yaw rate rad/s (counter-clockwise is positive); only used when hold_heading is False
    heading: float = 0.0  # spawn yaw and target yaw, rad (0 = facing +x)
    hold_heading: bool = True
    difficulty: float = 0.0  # terrain difficulty in [0, 1] (interpolates the *_range fields of the sub-terrain)
    midpoint_x: float | None = None  # optional milestone, e.g. the top of a pyramid
    midpoint_label: str = "top"  # what the midpoint is: top / bottom / center
    obstacle_half_width: float | None = None  # the obstacle spans [-w, +w] along course_dir (for plots)
    timeout_margin_s: float = 5.0
    # --- kind of motion ---
    motion: str = "walk"  # "walk" | "turn" | "sprint_brake"
    course_dir: tuple[float, float] = (1.0, 0.0)  # world direction of travel (unit vector)
    speed_label: str = "forward speed"  # name of the main speed in the plots
    command_schedule: tuple | None = None  # ((t_start_s, vx, vy, wz), ...); None = constant command
    fixed_episode_length_s: float | None = None
    brake_time_s: float | None = None  # sprint_brake: when the stop command is given
    success_on_timeout: bool = False  # success = survive until the end of the episode (and stop)
    generator_overrides: dict | None = None  # attributes of the terrain generator to change, e.g. slope_threshold

    @property
    def primary_channel(self) -> str:
        """Velocity channel that carries the main command: "vx", "vy" or "wz"."""
        if self.motion == "turn":
            return "wz"
        return "vy" if abs(self.vy) > abs(self.vx) else "vx"

    @property
    def primary_sign(self) -> float:
        return -1.0 if getattr(self, self.primary_channel) < 0.0 else 1.0

    @property
    def primary_command(self) -> float:
        return abs(getattr(self, self.primary_channel))

    @property
    def primary_unit(self) -> str:
        return "rad/s" if self.motion == "turn" else "m/s"

    @property
    def progress_unit(self) -> str:
        return "deg" if self.motion == "turn" else "m"

    @property
    def episode_length_s(self) -> float:
        """2x the ideal time to reach the goal, plus a margin (or the fixed length)."""
        if self.fixed_episode_length_s is not None:
            return self.fixed_episode_length_s
        distance = abs(self.goal_x - self.spawn_x)
        if self.motion == "turn":
            ideal = math.radians(distance) / max(abs(self.wz), 1e-3)
        else:
            ideal = distance / max(math.hypot(self.vx, self.vy), 1e-3)
        return 2.0 * ideal + self.timeout_margin_s


def find_trunk_contact_terms(env_cfg) -> list[dict]:
    """Termination terms that end the episode on contact of a robot body with the surroundings.

    Detected by their sensor_cfg parameter (e.g. illegal_contact on the "base" body). Returns
    dicts with the term name, the contact sensor name, the body name(s) and the force threshold, i.e. the
    exact definition of "trunk contact" used by the training.
    """
    terms = []
    for term_name, term in vars(env_cfg.terminations).items():
        params = getattr(term, "params", None) or {}
        sensor_cfg = params.get("sensor_cfg")
        if sensor_cfg is not None and getattr(sensor_cfg, "name", None):
            terms.append(
                {
                    "term": term_name,
                    "sensor": sensor_cfg.name,
                    "body_names": sensor_cfg.body_names,
                    "threshold": float(params.get("threshold", 1.0)),
                }
            )
    return terms


def scenario_meta(sc: EvalScenario) -> dict:
    """Scenario data that the report viewer needs for its plots."""
    return {
        "description": sc.description,
        "motion": sc.motion,
        "course_dir": list(sc.course_dir),
        "command_speed_mps": None if sc.motion == "turn" else sc.primary_command,
        "primary_channel": sc.primary_channel,
        "primary_sign": sc.primary_sign,
        "primary_label": sc.speed_label,
        "primary_unit": sc.primary_unit,
        "primary_command": sc.primary_command,
        "progress_unit": sc.progress_unit,
        "brake_time_s": sc.brake_time_s,
        "command_schedule": [list(segment) for segment in sc.command_schedule] if sc.command_schedule else None,
        "spawn_x": sc.spawn_x,
        "goal_x": sc.goal_x,
        "midpoint_x": sc.midpoint_x,
        "midpoint_label": sc.midpoint_label,
        "obstacle_half_width": sc.obstacle_half_width,
    }


def _resolve_command(sc: EvalScenario, name: str, speed: float | None, limits) -> EvalScenario:
    """Apply --speed and fill in what depends on the training ranges (maximum speed of the sprint)."""
    if speed is not None and speed <= 0.0:
        raise ValueError(f"The commanded speed must be positive, got {speed}.")
    lim_x, lim_y, lim_w = tuple(limits.lin_vel_x), tuple(limits.lin_vel_y), tuple(limits.ang_vel_z)

    def check(label: str, value: float, lim: tuple, unit: str):
        if not lim[0] <= value <= lim[1]:
            print(f"[WARN] '{name}': {label} {value} {unit} is outside the training range [{lim[0]}, {lim[1]}] {unit}.", flush=True)

    if sc.motion == "sprint_brake":
        vmax = float(lim_x[1])
        if speed is not None:
            print(f"[INFO] --speed is ignored by '{name}': it always runs up to {vmax} m/s, the maximum of the training range.", flush=True)
        schedule = ((0.0, vmax, 0.0, 0.0), (float(sc.brake_time_s), 0.0, 0.0, 0.0))
        return dataclasses.replace(sc, vx=vmax, command_schedule=schedule)
    if sc.motion == "turn":
        if speed is not None:
            print(f"[INFO] --speed is ignored by '{name}': it turns at {sc.wz} rad/s.", flush=True)
        check("turn rate", sc.wz, lim_w, "rad/s")
        return sc
    if speed is not None:
        sc = dataclasses.replace(sc, **{sc.primary_channel: sc.primary_sign * float(speed)})
    check("speed", sc.vx, lim_x, "m/s")
    check("sideways speed", sc.vy, lim_y, "m/s")
    return sc


def apply_eval_scenario(
    env_cfg, name: str, speed: float | None = None, contact_termination: bool = True
) -> EvalScenario:
    """Apply scenario name to a RobotEvalEnvCfg instance (in place) and return the effective scenario.

    env_cfg.seed must be set BEFORE calling this: it also seeds the terrain generator, which
    matters for the randomized terrains (boxes, random_rough).

    speed: commanded speed in m/s for the walking scenarios; None keeps the scenario's own. Ignored by
        flat_turn_360 and flat_max_speed.
    contact_termination: False removes the terminations on trunk contact ("base_contact"), so that the
        episode goes on after the trunk touches the ground and every contact can be recorded.
    """
    if name not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown eval scenario '{name}'. Available: {', '.join(EVAL_SCENARIOS)}")
    cmd = env_cfg.commands.base_velocity
    sc = _resolve_command(EVAL_SCENARIOS[name], name, speed, cmd.limit_ranges)  # reads the training ranges first
    if not contact_termination:
        for term in find_trunk_contact_terms(env_cfg):
            setattr(env_cfg.terminations, term["term"], None)

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
    for key, value in (sc.generator_overrides or {}).items():
        setattr(gen, key, value)
    sub = copy.deepcopy(sc.sub_terrain)
    sub.proportion = 1.0
    gen.sub_terrains = {name: sub}
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # velocity command: the first segment is what the policy sees after a reset
    for r in (cmd.ranges, cmd.limit_ranges):
        r.lin_vel_x = (sc.vx, sc.vx)
        r.lin_vel_y = (sc.vy, sc.vy)
        r.ang_vel_z = (sc.wz, sc.wz)
    cmd.resampling_time_range = (1.0e6, 1.0e6)

    env_cfg.episode_length_s = sc.episode_length_s
    dir_x, dir_y = sc.course_dir

    # fixed spawn pose
    env_cfg.events.reset_base = EventTerm(
        func=reset_root_state_fixed,
        mode="reset",
        params={"x": sc.spawn_x * dir_x, "y": sc.spawn_x * dir_y, "yaw": sc.heading},
    )

    # end of the episode (not a time-out: counted as a termination)
    if sc.motion == "walk":
        env_cfg.terminations.goal_reached = DoneTerm(
            func=goal_reached, params={"goal_x": sc.goal_x, "dir_x": dir_x, "dir_y": dir_y}, time_out=False
        )
    elif sc.motion == "turn":
        env_cfg.terminations.goal_reached = DoneTerm(func=rotation_reached, params={"goal_deg": sc.goal_x}, time_out=False)

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    if sc.command_schedule:  # must be registered before hold_heading, which overrides the yaw rate afterwards
        env_cfg.events.command_schedule = EventTerm(
            func=command_schedule,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            params={"schedule": sc.command_schedule},
        )
    # keep the heading (closed loop on yaw)
    if sc.hold_heading:
        env_cfg.events.hold_heading = EventTerm(
            func=hold_heading,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            params={"heading": sc.heading},
        )
    return sc


##
# Flat-ground scenarios
##


def _flat(description: str, **kwargs) -> EvalScenario:
    base = dict(
        sub_terrain=terrain_gen.MeshPlaneTerrainCfg(proportion=1.0), size=8.0, vx=0.5, spawn_x=-3.0, goal_x=3.0
    )
    base.update(kwargs)
    return EvalScenario(description=description, **base)


def _flat_scenarios() -> dict[str, EvalScenario]:
    return {
        "flat": _flat("Flat ground, 6 m straight walk"),
        "flat_backward": _flat(
            "Flat ground, 6 m walking backwards",
            vx=-BACKWARD_SPEED,
            course_dir=(-1.0, 0.0),
            speed_label="backward speed",
        ),
        "flat_strafe_right": _flat(
            "Flat ground, 6 m sideways to the right (right side leading), facing forward",
            vx=0.0,
            vy=-SIDEWAYS_SPEED,
            course_dir=(0.0, -1.0),
            speed_label="sideways speed (right)",
        ),
        "flat_strafe_left": _flat(
            "Flat ground, 6 m sideways to the left (left side leading), facing forward",
            vx=0.0,
            vy=SIDEWAYS_SPEED,
            course_dir=(0.0, 1.0),
            speed_label="sideways speed (left)",
        ),
        "flat_turn_360": _flat(
            "Flat ground, one full turn (360 deg) on the spot, counter-clockwise",
            vx=0.0,
            wz=TURN_RATE,
            spawn_x=0.0,
            goal_x=360.0,
            motion="turn",
            hold_heading=False,
            speed_label="turn rate (counter-clockwise)",
        ),
        "flat_max_speed": _flat(
            "Flat ground: accelerate straight to the maximum speed of the training range, then command 0 at once",
            vx=1.0,  # replaced by the maximum of the training range when the scenario is applied
            spawn_x=-4.0,
            goal_x=None,
            motion="sprint_brake",
            brake_time_s=SPRINT_BRAKE_TIME_S,
            fixed_episode_length_s=SPRINT_EPISODE_S,
            success_on_timeout=True,
            speed_label="forward speed",
        ),
    }


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
        obstacle_half_width=size / 2 - 0.25,  # the 0.25 m border of the sloped terrain is flat
    )


##
# Scenarios derived from ROUGH_TERRAINS_CFG (the terrain set used for training)
##

# ROUGH sub-terrain key -> difficulty levels to test. None = the generator ignores difficulty for it.
# The step height of the stairs and the angle of the slopes are swept separately below (cm / deg variants).
_ROUGH_LEVELS: dict[str, tuple[float | None, ...]] = {
    "pyramid_stairs": (1.0,),
    "pyramid_stairs_inv": (1.0,),
    "boxes": (0.25, 0.5, 0.75, 1.0),
    "random_rough": (None,),
    "hf_pyramid_slope": (0.5, 1.0),
    "hf_pyramid_slope_inv": (0.5, 1.0),
}

# Step heights (cm) of rough_pyramid_stairs_<N>cm and rough_pyramid_stairs_inv_<N>cm: 5, 10, then every cm up to 20.
STAIRS_STEP_HEIGHTS_CM: tuple[int, ...] = (5, *range(10, 21))
# Angles (deg) of rough_hf_pyramid_slope_<A>deg.
SLOPE_ANGLES_DEG: tuple[int, ...] = tuple(range(10, 46, 5))


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
        s = abs(_lerp(sub.slope_range, d))
        return f"slope {s:.2f} rise/run ({math.degrees(math.atan(s)):.0f} deg)"
    if hasattr(sub, "grid_height_range"):
        return f"random blocks {sub.grid_width * 100:.0f} cm wide, height +-{_lerp(sub.grid_height_range, d) * 100:.1f} cm"
    if hasattr(sub, "noise_range"):
        return f"uniform height noise {sub.noise_range[0] * 100:.0f}..{sub.noise_range[1] * 100:.0f} cm"
    return type(sub).__name__


def _rough_scenario(key: str, difficulty: float | None, name: str | None = None, modify=None, **extra) -> tuple[str, EvalScenario]:
    """Scenario for the ROUGH sub-terrain key. modify(sub) may change the copied sub-terrain config."""
    sub = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains[key])
    if modify is not None:
        modify(sub)
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

    if name is None:
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
        obstacle_half_width=half,
        **extra,
    )


def _set(attr: str, value):
    return lambda sub: setattr(sub, attr, value)


def _rough_family(key: str) -> dict[str, EvalScenario]:
    """All scenarios of one ROUGH sub-terrain: its difficulty levels plus the cm / deg sweeps."""
    out: dict[str, EvalScenario] = {}
    for level in _ROUGH_LEVELS[key]:
        name, sc = _rough_scenario(key, level)
        out[name] = sc
    if key in ("pyramid_stairs", "pyramid_stairs_inv"):
        for cm in STAIRS_STEP_HEIGHTS_CM:
            h = cm / 100.0
            name, sc = _rough_scenario(key, None, name=f"rough_{key}_{cm}cm", modify=_set("step_height_range", (h, h)))
            out[name] = sc
    if key == "hf_pyramid_slope":
        for deg in SLOPE_ANGLES_DEG:
            t = math.tan(math.radians(deg))
            # no vertical-wall correction: above rise/run = 0.75 (37 deg) the generator would turn the ramp into terraces
            name, sc = _rough_scenario(
                key, None, name=f"rough_{key}_{deg}deg", modify=_set("slope_range", (t, t)),
                generator_overrides={"slope_threshold": None},
            )  # fmt: skip
            out[name] = sc
    return out


##
# Registry. Add new scenarios here.
##

EVAL_SCENARIOS: dict[str, EvalScenario] = {**_flat_scenarios()}

for _key in ROUGH_TERRAINS_CFG.sub_terrains:
    if _key not in _ROUGH_LEVELS:
        warnings.warn(f"eval_scenarios: ROUGH sub-terrain '{_key}' has no evaluation scenario (add it to _ROUGH_LEVELS).")
for _key in _ROUGH_LEVELS:
    if _key not in ROUGH_TERRAINS_CFG.sub_terrains:
        warnings.warn(f"eval_scenarios: '{_key}' is not in ROUGH_TERRAINS_CFG, scenario skipped.")
        continue
    EVAL_SCENARIOS.update(_rough_family(_key))

EVAL_SCENARIOS["slope_up_0p2"] = _slope_up(0.2)
EVAL_SCENARIOS["slope_up_0p4"] = _slope_up(0.4)