"""Flat-only Go1 PPO task for Hackathon T3 (Isaac Lab 2.3 / Isaac Sim 5.x).

This environment models a simplified T2 joint target clamp/rate limit, not the
collision-aware Safety Block. The URDF physics also differ from the judge's
MuJoCo Menagerie model; MuJoCo sim-to-sim remains necessary before submission.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.go1 import GO1_SDK_JOINTS, UNITREE_GO1_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    EventCfg as Go2EventsCfg,
    ObservationsCfg as Go2ObservationsCfg,
    RewardsCfg as Go2RewardsCfg,
    TerminationsCfg as Go2TerminationsCfg,
)

from .actions import RateLimitedJointPositionActionCfg
from .rewards import safety_target_correction_l2
from .stability import base_too_low, low_base_height_penalty

# SDK/T2 joint order: FR, FL, RR, RL; hip, thigh, calf per leg.
assert len(GO1_SDK_JOINTS) == 12
# Use stricter margins than the T2 contractual limits. This is not a
# guarantee of collision freedom: only the full Safety Block can give that.
JOINT_MIN = (-0.70, 0.05, -2.60) * 4
JOINT_MAX = (+0.70, 2.25, -1.00) * 4


@configclass
class Go1FlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
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
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.40), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.40), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class Go1ActionsCfg:
    # The term name stays compatible with the project's deploy.yaml exporter.
    JointPositionAction = RateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=GO1_SDK_JOINTS,
        preserve_order=True,
        scale=0.45,
        use_default_offset=True,
        clip=None,  # Raw actions are clipped before affine scaling by our action term.
        raw_action_clip=1.0,
        max_delta_rad=math.radians(4.5),
        min_target=JOINT_MIN,
        max_target=JOINT_MAX,
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
    # Go1 URDF physical base is trunk; 'base' is a massless fixed parent link.
    # Stage 1: a repeatable stand/walk task. Restore mild domain randomization
    # and pushes only after upright velocity tracking is learned.
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
    # A narrower velocity-tracking kernel distinguishes real forward walking
    # from stationary nose-down poses (which score well with a wide kernel).
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.06)},
    )
    # No turning is requested; don't make the constant no-yaw reward dominate.
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.4,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Orientation and body height explicitly discourage the front-body-down pose.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)
    low_base_height = RewTerm(
        func=low_base_height_penalty,
        weight=-5.0,
        params={"target_height": 0.30},
    )
    # Low-speed task: reduce penalty for deviating from the rest pose while walking.
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 3.0,
            "velocity_threshold": 0.1,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.2,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["trunk", ".*_hip", ".*_thigh", ".*_calf"]
            ),
        },
    )
    safety_target_correction = RewTerm(func=safety_target_correction_l2, weight=-2.0)


@configclass
class Go1TerminationsCfg(Go2TerminationsCfg):
    # T3 Judge tolerates up to 50 degrees; training uses a tighter upright margin.
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.55})
    # Contact on trunk is NOT sufficient: the URDF visual head and the collision
    # geometry are not identical. Height ensures nose-down exploits reset.
    base_too_low = DoneTerm(func=base_too_low, params={"minimum_height": 0.21})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
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
    curriculum = None  # Deliberately NO rough terrain or speed curriculum.

    def __post_init__(self):
        self.decimation = 4  # 0.005s * 4 = 20 ms, 50 Hz policy tick
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go1FlatPlayEnvCfg(Go1FlatEnvCfg):
    """Repeatable 0.30 m/s visualization; no training randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 95.0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        for ranges in (self.commands.base_velocity.ranges, self.commands.base_velocity.limit_ranges):
            ranges.lin_vel_x = (0.30, 0.30)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
