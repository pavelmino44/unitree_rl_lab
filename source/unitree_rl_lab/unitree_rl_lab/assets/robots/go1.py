"""Unitree Go1 articulation used for the Hackathon T3 flat task.

The description is taken from the existing unitree_ros submodule.  The judge
uses MuJoCo Menagerie, so matching this model to its inertias, collision shapes
and motor dynamics remains an explicit sim-to-sim validation task.
"""

import os
import tempfile

import isaaclab.sim as sim_utils
from pathlib import Path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg

# Works in a checkout of unitree-go2-rl with sibling submodules, or with the
# UNITREE_ROS_DIR environment variable pointing at an independent checkout.
_DEFAULT_UNITREE_ROS_DIR = Path(__file__).resolve().parents[6] / "unitree_ros"
UNITREE_ROS_DIR = Path(os.environ.get("UNITREE_ROS_DIR", str(_DEFAULT_UNITREE_ROS_DIR))).expanduser()
GO1_URDF = UNITREE_ROS_DIR / "robots/go1_description/urdf/go1.urdf"


def _resolve_go1_mesh_uris() -> str:
    """Create a temporary URDF with absolute mesh paths (no source mutations).

    The unitree_ros Go1 URDF uses package://go1_description/meshes/*.dae.
    The Isaac Sim importer does not need a ROS package index when those paths
    are absolute. Keep every joint, mass and collision geometry unchanged.
    """
    if not GO1_URDF.is_file():
        return str(GO1_URDF)  # The importer will report the missing URDF.
    content = GO1_URDF.read_text(encoding="utf-8")
    mesh_dir = (GO1_URDF.parent.parent / "meshes").resolve()
    prefix = "package://go1_description/meshes/"
    if prefix not in content:
        return str(GO1_URDF)
    resolved = content.replace(prefix, mesh_dir.as_posix() + "/")
    out_dir = Path(tempfile.gettempdir()) / "unitree_rl_lab_go1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "go1_resolved.urdf"
    if not out_path.exists() or out_path.read_text(encoding="utf-8") != resolved:
        out_path.write_text(resolved, encoding="utf-8")
    return str(out_path)


GO1_RESOLVED_URDF = _resolve_go1_mesh_uris()

GO1_SDK_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

UNITREE_GO1_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=GO1_RESOLVED_URDF,
        # Preserve foot bodies as separate links for contact-based rewards.
        merge_fixed_joints=False,
        # The source URDF has overlapping collision meshes among fixed links
        # (e.g. trunk and hip rotor links). Prevent those internal self-contacts
        # from triggering illegal_contact(trunk) in the first policy step.
        # This does NOT replace the Menagerie/T2 safety collision checks.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
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
            effort_limit=23.7,
            velocity_limit=30.1,
            stiffness=35.0,
            damping=0.8,
        )
    },
    joint_sdk_names=GO1_SDK_JOINTS.copy(),
)
