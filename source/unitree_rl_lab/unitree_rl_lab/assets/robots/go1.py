"""Go1 physics built from the SHA-pinned hackathon MuJoCo 3.3.7 model.

This Go1-only USD bypasses the unitree_ros URDF and its implicit frame masses.
The distributed USD includes exact authored inertias and MJCF collision shapes.
PhysX contacts remain an approximation of MuJoCo contacts; see physics.json.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUsdFileCfg

GO1_ASSET_DIR = Path(__file__).with_name("go1_mjcf")
GO1_USD = GO1_ASSET_DIR / "go1.usda"
GO1_PHYSICS_MANIFEST = GO1_ASSET_DIR / "physics.json"
GO1_SDK_JOINTS = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]

UNITREE_GO1_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=str(GO1_USD),
        # Filters in USD preserve MuJoCo welded-body/parent-child exclusions.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # Preserve v3's policy offsets/action contract. The MJCF home keyframe is
    # not a required physical property and has hips outside v3 target margins.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*R_hip_joint": -0.10,
            ".*L_hip_joint": 0.10,
            ".*_thigh_joint": 0.85,
            ".*_calf_joint": -1.70,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO1": IdealPDActuatorCfg(
            joint_names_expr=GO1_SDK_JOINTS,
            stiffness=100.0,  # Position servo in the supplied, unmodified MJCF.
            damping=0.0,  # MJCF servo kv=0; passive damping is separate below.
            effort_limit={
                ".*_hip_joint": 23.7,
                ".*_thigh_joint": 23.7,
                ".*_calf_joint": 35.55,
            },
            # MuJoCo clamps the servo BEFORE adding passive joint resistance.
            # Avoid an extra solver cap on the sum of servo/passive efforts.
            effort_limit_sim=1.0e9,
            # MJCF has no hard angular-velocity limit; do not inherit URDF's.
            velocity_limit=1.0e6,
            velocity_limit_sim=1.0e6,
            armature=0.01,
            # Isaac Sim >=5.0: effort [N*m], not legacy dimensionless friction.
            friction=0.2,
            dynamic_friction=0.2,
            viscous_friction={
                ".*_hip_joint": 1.0,
                ".*_thigh_joint": 2.0,
                ".*_calf_joint": 2.0,
            },
        )
    },
    joint_sdk_names=GO1_SDK_JOINTS.copy(),
)
