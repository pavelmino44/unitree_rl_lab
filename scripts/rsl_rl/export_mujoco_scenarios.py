"""Export the evaluation scenarios (eval_scenarios.py) and their exact terrains for the MuJoCo evaluation.

The MuJoCo evaluation (ros2_ws/src/go2_eval + unitree_mujoco/simulate_python/eval_sim.py) runs the SAME
scenarios as evaluate.py, against go2_policy_runner. Everything it needs is read from the training code
here, so that a change of ROUGH_TERRAINS_CFG, eval_scenarios.py, the commands or the terminations is picked
up by re-running this script:

  * for every scenario: the EvalScenario fields (spawn, goal, command, motion, episode length, ...);
  * for every (scenario, seed): the terrain tile exactly as IsaacLab generates it for evaluate.py. The
    TerrainGenerator is built from the same config (apply_eval_scenario) after the same seeding as the
    evaluation environment, and the raw meshes returned by the sub-terrain function are stored (vertices /
    faces in world coordinates). unitree_mujoco/simulate_python/eval_scene.py turns them into MuJoCo geoms;
  * the task data that the evaluation needs: control step, command ranges, termination thresholds, the
    bodies of the trunk contact, the robot's initial pose and joint positions.

Run from the unitree_rl_lab root, in the IsaacLab conda env:

    python scripts/rsl_rl/export_mujoco_scenarios.py --headless
    python scripts/rsl_rl/export_mujoco_scenarios.py --headless --eval_seed 42,43,44
    python scripts/rsl_rl/export_mujoco_scenarios.py --headless --scenarios "rough_boxes*,flat" --output /tmp/cat

Output (default: ../unitree_mujoco/eval_scenarios, i.e. next to unitree_rl_lab in unitree-go2-rl):
    catalog.json             scenarios, task data, terrain file per (scenario, seed)
    terrains/<hash>.npz      one file per distinct terrain (identical terrains of different seeds are shared)

The seed only changes the randomized terrains (boxes, random_rough). They are reproduced exactly when this
script runs on the same machine as evaluate.py (the box heights are drawn on the GPU when one is present).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fnmatch
import functools
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime

import numpy as np

from isaaclab.app import AppLauncher

CATALOG_VERSION = 1

# Parts of the robot body whose contacts are recorded (the same patterns as CONTACT_PARTS in evaluate.py).
CONTACT_PARTS = {
    "head": ["Head_.*"],
    "legs": [".*_hip", ".*_thigh", ".*_calf"],
}

parser = argparse.ArgumentParser(description="Export the evaluation scenarios and terrains for MuJoCo.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity", help="Name of the task.")
parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated names / glob patterns / 'rough' / 'all'.")
parser.add_argument("--eval_seed", type=str, default="42", help="Comma-separated terrain seeds to export.")
parser.add_argument(
    "--output", type=str, default=None, help="Output folder (default: ../unitree_mujoco/eval_scenarios)."
)
parser.add_argument(
    "--append",
    action="store_true",
    default=False,
    help="Keep the scenarios / seeds already in the catalog that are not exported now.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.terrains import TerrainGenerator  # noqa: E402
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.locomotion.robots.go2.eval_scenarios import (  # noqa: E402
    EVAL_SCENARIOS,
    apply_eval_scenario,
    find_trunk_contact_terms,
    scenario_meta,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _expand(text: str, registry: list[str]) -> list[str]:
    names: list[str] = []
    for token in [t.strip() for t in text.split(",") if t.strip()]:
        pattern = "*" if token == "all" else ("rough_*" if token == "rough" else token)
        matches = [n for n in registry if fnmatch.fnmatchcase(n, pattern)]
        if not matches:
            raise SystemExit(f"[ERROR] '{token}' matches no scenario. Available: {registry}")
        names += [m for m in matches if m not in names]
    return names


def _plain(value):
    """JSON-friendly copy of a config value (tuples -> lists, callables / objects -> their name)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    if hasattr(value, "to_dict"):
        try:
            return _plain(value.to_dict())
        except Exception:
            pass
    return str(value)


def _scenario_record(name: str, sc) -> dict:
    """The EvalScenario fields (without the terrain config object) plus the meta of the report viewer."""
    fields = {}
    for f in dataclasses.fields(sc):
        if f.name == "sub_terrain":
            continue
        fields[f.name] = _plain(getattr(sc, f.name))
    sub = sc.sub_terrain
    return {
        "description": sc.description,
        "fields": fields,
        "sub_terrain": {"type": type(sub).__name__, "cfg": _plain(sub)},
        "episode_length_s": sc.episode_length_s,
        "meta": scenario_meta(sc),
        "terrains": {},
    }


def _func_name(term) -> str:
    func = getattr(term, "func", None)
    return getattr(func, "__name__", "") if func is not None else ""


def _task_record(env_cfg, contact_terms: list[dict]) -> dict:
    """Task data used by the MuJoCo evaluation (from the evaluation config, before a scenario is applied)."""
    cmd = env_cfg.commands.base_velocity
    lim = cmd.limit_ranges
    bad_orientation = None
    for term_name, term in vars(env_cfg.terminations).items():
        if term is not None and _func_name(term) == "bad_orientation":
            bad_orientation = {"term": term_name, "limit_angle": float(term.params.get("limit_angle", 0.8))}
    robot = env_cfg.scene.robot
    return {
        "task": args_cli.task,
        "physics_dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "command_limit_ranges": {
            "lin_vel_x": list(lim.lin_vel_x),
            "lin_vel_y": list(lim.lin_vel_y),
            "ang_vel_z": list(lim.ang_vel_z),
        },
        "terminations": {
            "bad_orientation": bad_orientation,
            "trunk_contact": contact_terms,  # [{"term", "sensor", "body_names", "threshold"}]
        },
        "contact_parts": CONTACT_PARTS,
        "robot": {
            "init_pos": [float(v) for v in robot.init_state.pos],
            "init_rot": [float(v) for v in robot.init_state.rot],
            "init_joint_pos": {k: float(v) for k, v in robot.init_state.joint_pos.items()},  # regex -> value
            "joint_sdk_names": list(getattr(robot, "joint_sdk_names", []) or []),
        },
    }


def _capturing(function, calls: list):
    """Wrap the function of a sub-terrain config: it keeps the meshes it returns (before concatenation).

    A plain function (functools.wraps) survives the copies of the config and keeps the name that the
    generator uses in its hash, so the generation is not changed in any way.
    """

    @functools.wraps(function)
    def wrapper(difficulty, cfg):
        meshes, origin = function(difficulty, cfg)
        calls.append(([m.copy() for m in meshes], np.array(origin, dtype=np.float64), float(difficulty)))
        return meshes, origin

    return wrapper


def _terrain(gen_cfg, seed: int, device: str) -> dict:
    """Generate the single evaluation tile exactly like the evaluation environment and return its raw data."""
    gen_cfg = copy.deepcopy(gen_cfg)
    (key, sub), = gen_cfg.sub_terrains.items()
    calls: list = []
    sub.function = _capturing(sub.function, calls)

    # the same seeding as evaluate.py (before gym.make) and as the environment (ManagerBasedEnv.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ManagerBasedEnv.seed(seed)
    generator = TerrainGenerator(cfg=gen_cfg, device=device)

    if len(calls) != 1:
        raise RuntimeError(f"expected one sub-terrain, got {len(calls)}")
    meshes, _origin_local, difficulty = calls[0]
    sx, sy = gen_cfg.size
    if gen_cfg.num_rows != 1 or gen_cfg.num_cols != 1:
        raise RuntimeError("the evaluation terrain must be a single tile")
    # local tile frame [0, size] -> world: the generator centers the tile (-size/2) and the 1x1 grid at 0
    shift = np.array([-sx / 2.0, -sy / 2.0, 0.0])
    verts, faces, offset = [], [], 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float64) + shift
        f = np.asarray(m.faces, dtype=np.int64) + offset
        verts.append(v)
        faces.append(f)
        offset += len(v)
    vertices = np.concatenate(verts)
    faces = np.concatenate(faces).astype(np.int32)
    origin = np.asarray(generator.terrain_origins[0, 0], dtype=np.float64)

    # sanity check against the final terrain mesh of the generator (tile + border)
    full = generator.terrain_mesh
    inner = np.all(np.abs(full.vertices[:, :2]) <= max(sx, sy) / 2.0 + 1e-6, axis=1)
    if inner.any():
        lo, hi = full.vertices[inner].min(axis=0), full.vertices[inner].max(axis=0)
        if not (np.allclose(lo[:2], vertices.min(axis=0)[:2], atol=1e-4) and np.allclose(hi[2], vertices.max(axis=0)[2], atol=1e-4)):
            print(f"[WARN] terrain '{key}': captured tile does not match the generator mesh bounds", flush=True)
    return {
        "vertices": vertices,
        "faces": faces,
        "origin": origin,
        "tile_size": np.array([sx, sy], dtype=np.float64),
        "border_width": float(gen_cfg.border_width),
        "border_height": float(getattr(gen_cfg, "border_height", 1.0)),
        "sub_terrain": type(sub).__name__,
        "difficulty": float(difficulty),
    }


def _write_terrain(out_dir: str, data: dict) -> str:
    h = hashlib.sha1()
    for key in ("vertices", "faces", "origin", "tile_size"):
        h.update(np.ascontiguousarray(data[key]).tobytes())
    h.update(f"{data['border_width']}/{data['border_height']}/{data['sub_terrain']}/{data['difficulty']:.6f}".encode())
    name = f"terrains/{h.hexdigest()[:16]}.npz"
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            vertices=data["vertices"],
            faces=data["faces"],
            origin=data["origin"],
            tile_size=data["tile_size"],
            border_width=np.float64(data["border_width"]),
            border_height=np.float64(data["border_height"]),
            sub_terrain=np.array(data["sub_terrain"]),
            difficulty=np.float64(data["difficulty"]),
        )
    return name


def main():
    out_dir = os.path.abspath(args_cli.output or os.path.join("..", "unitree_mujoco", "eval_scenarios"))
    seeds = [int(s) for s in args_cli.eval_seed.split(",") if s.strip()]
    names = _expand(args_cli.scenarios, list(EVAL_SCENARIOS))
    os.makedirs(out_dir, exist_ok=True)
    catalog_path = os.path.join(out_dir, "catalog.json")

    base_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, entry_point_key="eval_env_cfg_entry_point")
    task = _task_record(base_cfg, find_trunk_contact_terms(base_cfg))
    device = base_cfg.sim.device

    catalog = {"scenarios": {}}
    if args_cli.append and os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)
    catalog.update(
        {
            "version": CATALOG_VERSION,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": {
                "task": args_cli.task,
                "rough_terrains_cfg": _plain(ROUGH_TERRAINS_CFG),
                "terrain_device": device,
            },
            "task": task,
        }
    )
    scenarios = catalog.setdefault("scenarios", {})
    # keep the registry order (the order of evaluate.py)
    ordered = {n: scenarios.get(n) for n in EVAL_SCENARIOS if n in scenarios or n in names}

    total = len(names) * len(seeds)
    done = 0
    for name in names:
        record = None
        for seed in seeds:
            done += 1
            env_cfg = parse_env_cfg(
                args_cli.task, device=args_cli.device, num_envs=1, entry_point_key="eval_env_cfg_entry_point"
            )
            env_cfg.seed = seed
            sc = apply_eval_scenario(env_cfg, name, speed=None, contact_termination=True)
            if record is None:
                old = ordered.get(name) if args_cli.append else None
                record = _scenario_record(name, EVAL_SCENARIOS[name])
                record["resolved"] = _scenario_record(name, sc)["fields"]  # with the training-range maximum etc.
                if old:
                    record["terrains"].update(old.get("terrains", {}))
            data = _terrain(env_cfg.scene.terrain.terrain_generator, seed, device)
            record["terrains"][str(seed)] = _write_terrain(out_dir, data)
            print(
                f"[INFO] ({done}/{total}) {name} seed {seed}: {data['sub_terrain']}, difficulty "
                f"{data['difficulty']:.3f}, {len(data['faces'])} triangles -> {record['terrains'][str(seed)]}",
                flush=True,
            )
        ordered[name] = record
    catalog["scenarios"] = ordered

    with open(catalog_path, "w", encoding="utf-8") as f:
        text = json.dumps(catalog, indent=1, ensure_ascii=False)
        text = re.sub(r"\[\s+([^\[\]{}\"]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)
        f.write(text)
    print(f"[INFO] {len(names)} scenario(s) x {len(seeds)} seed(s) exported to {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
