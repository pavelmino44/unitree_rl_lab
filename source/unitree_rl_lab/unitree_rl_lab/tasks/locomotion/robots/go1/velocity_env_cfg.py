"""Flat-only Go1 PPO task for Hackathon T3 (Isaac Lab 2.3 / Isaac Sim 5.x).
Default PPO batches T2 joint/step/torque checks on GPU; optional MuJoCo mode also
checks collision sweeps. The Go1 USD matches the MJCF mass/inertias/geometry;
solver behavior still requires MuJoCo sim-to-sim validation before submission.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import (
    ISAAC_NUCLEUS_DIR,
    ISAACLAB_NUCLEUS_DIR,
)

from unitree_rl_lab.assets.robots.go1 import GO1_SDK_JOINTS, UNITREE_GO1_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CurriculumCfg as Go2CurriculumCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    EventCfg as Go2EventsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ObservationsCfg as Go2ObservationsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RewardsCfg as Go2RewardsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    TerminationsCfg as Go2TerminationsCfg,
)

from .actions import RateLimitedJointPositionActionCfg
from .commands import Go1VelocityCommandCfg
from .physics import validate_go1_mjcf_physics
from .safety_reference.joint_limits import Q_MAX, Q_MIN, STEP_LIMIT_RAD
from .stability import base_too_low
from .terrain import Go1TerrainImporter

# SDK/T2 joint order: FR, FL, RR, RL; hip, thigh, calf per leg.
assert len(GO1_SDK_JOINTS) == 12
# Preserve the v3/Go2 curriculum start, increment and success conditions.
# Extend only the longitudinal cap for training headroom; official T3 is <=0.6.
GO1_JUDGE_MAX_SPEED = 0.6
GO1_MAX_SPEED = 0.7
GO1_LINEAR_SPEED_SCALE = GO1_JUDGE_MAX_SPEED / 1.0
JOINT_MIN = tuple(Q_MIN)
JOINT_MAX = tuple(Q_MAX)


@configclass
class Go1FlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        class_type=Go1TerrainImporter,
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Only the contact sensor is used: no ray caster, camera or rough terrain.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class Go1CommandsCfg:
    base_velocity = Go1VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=Go1VelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1 * GO1_LINEAR_SPEED_SCALE, 0.1 * GO1_LINEAR_SPEED_SCALE),
            lin_vel_y=(-0.1 * GO1_LINEAR_SPEED_SCALE, 0.1 * GO1_LINEAR_SPEED_SCALE),
            ang_vel_z=(-1.0, 1.0),
        ),
        limit_ranges=Go1VelocityCommandCfg.Ranges(
            lin_vel_x=(-GO1_MAX_SPEED, GO1_MAX_SPEED),
            lin_vel_y=(-0.4 * GO1_LINEAR_SPEED_SCALE, 0.4 * GO1_LINEAR_SPEED_SCALE),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class Go1ActionsCfg:
    # The term name stays compatible with the project's deploy.yaml exporter.
    JointPositionAction = RateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=GO1_SDK_JOINTS,
        preserve_order=True,
        scale={
            ".*_hip_joint": 0.15,
            ".*_thigh_joint": 0.35,
            ".*_calf_joint": 0.35,
        },
        use_default_offset=True,
        clip=None,
        raw_action_clip=math.inf,  # No extra +/-1 envelope before affine scaling.
        max_delta_rad=STEP_LIMIT_RAD,
        min_target=JOINT_MIN,
        max_target=JOINT_MAX,
        check_torques=True,
        collision_check="none",  # Full CPU reference sweep is enabled by evaluation.
    )


@configclass
class Go1ObservationsCfg(Go2ObservationsCfg):
    # Policy is exactly the Go2 proprioception-only policy, with five steps of
    # observation history. Do not add base_lin_vel or ground-truth state here.
    @configclass
    class CriticCfg(Go2ObservationsCfg.CriticCfg):
        # The Go2 critic's height scan would be invalid without a ray caster.
        height_scanner: ObsTerm | None = None

    critic: CriticCfg = CriticCfg()


@configclass
class Go1EventsCfg(Go2EventsCfg):
    # Go1 USD physical root is trunk; no auxiliary base/sensor rigid bodies.
    # Stage 1: a repeatable stand/walk task. Restore mild domain randomization
    # and pushes only after upright velocity tracking is learned.
    # Ground offsets are authored by Go1TerrainImporter before simulation.
    mjcf_physics = EventTerm(func=validate_go1_mjcf_physics, mode="startup")
    physics_material = None
    add_base_mass = None
    base_external_force_torque = None
    push_robot = None
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class Go1RewardsCfg(Go2RewardsCfg):
    """Inherit every Go2 reward unchanged except Go1 contact body names."""

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["trunk", ".*_hip", ".*_thigh", ".*_calf"]
            ),
        },
    )


@configclass
class Go1CurriculumCfg(Go2CurriculumCfg):
    """Keep Go2 conditions and the v3 nominal 0.06 m/s increment."""

    terrain_levels: CurrTerm | None = None
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"increment": 0.1 * GO1_LINEAR_SPEED_SCALE},
    )


@configclass
class Go1TerminationsCfg(Go2TerminationsCfg):
    # These named terms are logged by Isaac Lab as Episode_Termination/*.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # T3 Judge tolerates up to 50 degrees; training uses a tighter upright margin.
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.55})
    # Contact on trunk is NOT sufficient: the URDF visual head and the collision
    # geometry are not identical. Height ensures nose-down exploits reset.
    base_too_low = DoneTerm(func=base_too_low, params={"minimum_height": 0.14})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 1.0,
        },
    )


@configclass
class Go1FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go1FlatSceneCfg = Go1FlatSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: Go1ObservationsCfg = Go1ObservationsCfg()
    actions: Go1ActionsCfg = Go1ActionsCfg()
    commands: Go1CommandsCfg = Go1CommandsCfg()
    rewards: Go1RewardsCfg = Go1RewardsCfg()
    terminations: Go1TerminationsCfg = Go1TerminationsCfg()
    events: Go1EventsCfg = Go1EventsCfg()
    curriculum: Go1CurriculumCfg = (
        Go1CurriculumCfg()
    )  # Train: velocity-command curriculum only.

    def __post_init__(self):
        self.decimation = 10  # MJCF 0.002s * 10 = 20 ms, preserve v3 policy tick
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go1FlatPlayEnvCfg(Go1FlatEnvCfg):
    """Repeatable 0.30 m/s visualization; no training randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None  # Play: freeze the commanded speed at 0.30 m/s.
        self.scene.num_envs = 1
        self.episode_length_s = 95.0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        for ranges in (
            self.commands.base_velocity.ranges,
            self.commands.base_velocity.limit_ranges,
        ):
            ranges.lin_vel_x = (0.30, 0.30)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
