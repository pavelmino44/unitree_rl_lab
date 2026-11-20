"""No-Isaac smoke check: Go1 URDF exists and has the T2 joint contract.

Run from unitree_rl_lab with `python scripts/project/check_go1_urdf.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]
EXPECTED_BOUNDS = {
    "hip": (-0.863, 0.863),
    "thigh": (-0.686, 4.501),
    "calf": (-2.818, -0.888),
}


def check(urdf_path: Path) -> None:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Go1 URDF missing: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    joints = {node.attrib["name"]: node for node in root.findall("joint")}
    print(f"Checking {urdf_path}")
    for name in JOINT_NAMES:
        node = joints[name]
        assert node.attrib["type"] == "revolute", name
        joint_type = name.split("_")[1]
        expected_lo, expected_hi = EXPECTED_BOUNDS[joint_type]
        limit = node.find("limit")
        assert limit is not None, name
        lo, hi = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        assert abs(lo - expected_lo) <= 1e-3, (name, lo, expected_lo)
        assert abs(hi - expected_hi) <= 1e-3, (name, hi, expected_hi)
        print(f"{name:19s}  [{lo:+.3f}, {hi:+.3f}] rad")
    print("OK: 12 named Go1 motor joints and limits match the T2 specification.")
    print("Note: this does NOT test Isaac Sim URDF import or judge Menagerie collisions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parents[3] / "unitree_ros"
    parser.add_argument("--unitree-ros-dir", default=os.environ.get("UNITREE_ROS_DIR", str(default_dir)))
    args = parser.parse_args()
    try:
        check(Path(args.unitree_ros_dir) / "robots/go1_description/urdf/go1.urdf")
    except (FileNotFoundError, KeyError, AssertionError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
