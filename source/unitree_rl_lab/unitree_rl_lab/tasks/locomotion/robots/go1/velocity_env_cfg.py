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
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import (
    ISAAC_NUCLEUS_DIR,
    ISAACLAB_NUCLEUS_DIR,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from unitree_rl_lab.assets.robots.go1 import GO1_SDK_JOINTS, UNITREE_GO1_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
EventCfg as Go2EventsCfg,
ObservationsCfg as Go2ObservationsCfg,
RewardsCfg as Go2RewardsCfg,
TerminationsCfg as Go2TerminationsCfg,
CurriculumCfg as Go2CurriculumCfg,
)
from .actions import RateLimitedJointPositionActionCfg
from .rewards import safety_target_correction_l2
from .gait import (
diagonal_trot_contact_reward,
front_rear_pair_swing_penalty,
forward_gated_velocity_tracking,
normalized_forward_progress,
rear_feet_narrow_penalty,
)
from .stability import base_too_low, low_base_height_penalty
# SDK/T2 joint order: FR, FL, RR, RL; hip, thigh, calf per leg.
assert len(GO1_SDK_JOINTS) == 12
# Use stricter margins than the T2 contractual limits. This is not a
# guarantee of collision freedom: only the full Safety Block can give that.
# Stage 1 (straight line only): keep right hips slightly abducted (negative)
# and left hips slightly abducted (positive). This prevents splaying/crossing
# exploits while self-collision detection is off for this URDF import.
JOINT_MIN = (-0.32, 0.05, -2.60, +0.03, 0.05, -2.60) * 2
JOINT_MAX = (-0.03, 2.25, -1.00, +0.32, 2.25, -1.00) * 2

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)


@configclass
class Go1FlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
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
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.40), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.60), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
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
    """Reward terms for the MDP."""
# -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
# -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
# -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
# stuck_penalty = RewTerm(
#     func=mdp.stuck_penalty,
#     weight=-2.0,
#     params={
#         "command_name": "base_velocity",
#         "vel_threshold": 0.1,
#     },
# )
# -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
# feet_contact_forces = RewTerm(
#     func=mdp.contact_forces,
#     weight=-0.02,
#     params={
#         "threshold": 100.0,
#         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
#     },
# )
# -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

@configclass
class Go1CurriculumCfg(Go2CurriculumCfg):
    """Curriculum terms for the MDP."""
    terrain_levels: CurrTerm | None = None  # Disable inherited rough-terrain curriculum on plane.
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels, params={"reward_term_name": "track_lin_vel_xy"})

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
    curriculum: Go1CurriculumCfg = Go1CurriculumCfg()  # Train: velocity-command curriculum only.
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
        self.curriculum = None  # Play: freeze the commanded speed at 0.30 m/s.
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
