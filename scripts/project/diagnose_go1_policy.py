"""Deterministic Isaac Lab probe of a Go1 checkpoint; no training or MDP edits.

Run after stopping training:
    python scripts/project/diagnose_go1_policy.py --checkpoint /path/model_400.pt --headless

Prints one JSON record per command. Action statistics are after a one-second
warmup. The original action filter stays active throughout the probe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from xml.etree import ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint", help="Required for policy/joint probes; optional for model_check"
)
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument(
    "--mode",
    choices=(
        "all",
        "stand",
        "forward",
        "turn_left",
        "turn_right",
        "joint_probe",
        "model_check",
    ),
    default="all",
    help="Probe all commands or just one command (default: all)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.duration) or args.duration <= 1.0:
    parser.error("--duration must be finite and greater than 1 second")
checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
if args.mode != "model_check" and checkpoint is None:
    parser.error("--checkpoint is required except with --mode model_check")
if checkpoint is not None and not checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {checkpoint}")
app = AppLauncher(args).app

import gymnasium as gym
import torch
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from unitree_rl_lab.tasks.locomotion.robots.go1.agents import Go1FlatPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go1.velocity_env_cfg import (
    Go1FlatPlayEnvCfg,
)


@torch.inference_mode()
def main():
    cfg = Go1FlatPlayEnvCfg()
    cfg.seed = 42
    cfg.episode_length_s = args.duration + 2.0
    cfg.sim.device = args.device
    cfg.commands.base_velocity.debug_vis = not args.headless
    raw_env = gym.make("Unitree-Go1-T3-Flat", cfg=cfg)
    agent_cfg = Go1FlatPPORunnerCfg()
    agent_cfg.device = raw_env.unwrapped.device
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        term = unwrapped.action_manager.get_term("JointPositionAction")
        command = unwrapped.command_manager.get_term("base_velocity")
        joint_ids = term._joint_ids
        print("Action joint order:", term._joint_names)
        print("Action scale:", term.cfg.scale)
        print("Fixed base:", robot.is_fixed_base)
        print(
            "Joint stiffness:",
            robot.data.default_joint_stiffness[0, joint_ids].tolist(),
        )
        print("Joint damping:", robot.data.default_joint_damping[0, joint_ids].tolist())
        print(
            "Command observation:",
            unwrapped.cfg.observations.policy.velocity_commands.params,
        )
        masses = robot.root_physx_view.get_masses()[0].detach().cpu()
        mass_report = {
            "physics_total_mass_kg": float(masses.sum()),
            "body_count": len(robot.body_names),
            "body_masses_kg": dict(zip(robot.body_names, masses.tolist(), strict=True)),
        }
        source_robot = getattr(getattr(cfg, "scene", None), "robot", None)
        source_path = getattr(getattr(source_robot, "spawn", None), "asset_path", None)
        if (
            source_path
            and Path(source_path).suffix.lower() == ".urdf"
            and Path(source_path).is_file()
        ):
            root = ET.parse(source_path).getroot()
            source_masses = []
            missing = []
            for link in root.findall("link"):
                mass = link.find("inertial/mass")
                if mass is None:
                    missing.append(link.get("name"))
                else:
                    source_masses.append(float(mass.get("value")))
            mass_report["urdf_specified_total_mass_kg"] = sum(source_masses)
            mass_report["urdf_links_without_mass"] = missing
            mass_report["mass_difference_kg"] = float(masses.sum()) - sum(source_masses)
        print("[MODEL MASS] " + json.dumps(mass_report, ensure_ascii=False), flush=True)
        if args.mode == "model_check":
            print("[MODEL CHECK COMPLETE] no policy rollout", flush=True)
            return mass_report
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        dt = float(unwrapped.step_dt)
        warmup_steps = math.ceil(1.0 / dt)
        total_steps = math.ceil(args.duration / dt)
        cases = [
            ("stand", 0.0, 0.0),
            ("forward", 0.3, 0.0),
            ("turn_left", 0.0, 0.6),
            ("turn_right", 0.0, -0.6),
        ]
        probe_index = None
        if args.mode == "joint_probe":
            cases = [("joint_probe", 0.0, 0.0)]
            probe_index = term._joint_names.index("FR_calf_joint")
            print(
                "Joint probe: FR_calf_joint, raw amplitude 0.5, frequency 0.5 Hz",
                flush=True,
            )
        elif args.mode != "all":
            cases = [case for case in cases if case[0] == args.mode]
        records = []
        for case_index, (name, vx, wz) in enumerate(cases, start=1):
            print(
                f"[PROBE {case_index}/{len(cases)}] {name}: setting command and resetting",
                flush=True,
            )
            for ranges in (command.cfg.ranges, command.cfg.limit_ranges):
                ranges.lin_vel_x = (vx, vx)
                ranges.lin_vel_y = (0.0, 0.0)
                ranges.ang_vel_z = (wz, wz)
            obs, _ = env.reset()
            print(f"[PROBE {case_index}/{len(cases)}] {name}: rollout", flush=True)
            samples = []
            start_xy = None
            previous_yaw = None
            yaw_travel = 0.0
            terminated = False
            for step in range(total_steps):
                if probe_index is None:
                    action = policy(obs)
                else:
                    # Hold the nominal stance during warmup, then request two
                    # slow calf oscillations over the four-second measurement.
                    action = torch.zeros_like(term.processed_actions)
                    probe_time = max(0, step - warmup_steps + 1) * dt
                    action[:, probe_index] = 0.5 * math.sin(math.pi * probe_time)
                obs, _, done, _ = env.step(action)
                if bool(done[0]):
                    terminated = True
                    break  # state after done has already been reset
                if step < warmup_steps:
                    continue
                position = robot.data.root_pos_w[0, :2]
                yaw = float(euler_xyz_from_quat(robot.data.root_quat_w[:1])[2][0])
                if start_xy is None:
                    start_xy = position.clone()
                    previous_yaw = yaw
                delta_yaw = yaw - previous_yaw
                yaw_travel += math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
                previous_yaw = yaw
                bounded = term.requested_target.clamp(min=term._q_min, max=term._q_max)
                samples.append(
                    {
                        "policy": action[0].detach().cpu().clone(),
                        "target": term.processed_actions[0].detach().cpu().clone(),
                        "joint": robot.data.joint_pos[0, joint_ids]
                        .detach()
                        .cpu()
                        .clone(),
                        "correction": (term.requested_target - term.processed_actions)[
                            0
                        ]
                        .detach()
                        .cpu()
                        .clone(),
                        "joint_error": (
                            robot.data.joint_pos[:, joint_ids] - term.processed_actions
                        )[0]
                        .detach()
                        .cpu()
                        .clone(),
                        "raw_clip": float(
                            (action.abs() > term.cfg.raw_action_clip).float().mean()
                        ),
                        "bound_clip": float(
                            (term.requested_target != bounded).float().mean()
                        ),
                        "rate_clip": float(
                            ((bounded - term.processed_actions).abs() > 1e-6)
                            .float()
                            .mean()
                        ),
                        "velocity_x": float(robot.data.root_lin_vel_b[0, 0]),
                        "angular_velocity_z": float(robot.data.root_ang_vel_b[0, 2]),
                        "command_x": float(command.command[0, 0]),
                        "command_yaw": float(command.command[0, 2]),
                        "height": float(robot.data.root_link_pos_w[0, 2]),
                        "joint_velocity": robot.data.joint_vel[0, joint_ids]
                        .detach()
                        .cpu()
                        .clone(),
                        "torque": robot.data.applied_torque[0, joint_ids]
                        .detach()
                        .cpu()
                        .clone(),
                    }
                )
            result = {
                "mode": name,
                "samples": len(samples),
                "terminated": terminated,
                "action_source": "actor_mean"
                if probe_index is None
                else "joint_probe_signal",
            }
            if samples:
                stacked = {
                    key: torch.stack([row[key] for row in samples])
                    for key in (
                        "policy",
                        "target",
                        "joint",
                        "correction",
                        "joint_error",
                        "joint_velocity",
                        "torque",
                    )
                }
                for key in (
                    "velocity_x",
                    "angular_velocity_z",
                    "command_x",
                    "command_yaw",
                    "raw_clip",
                    "bound_clip",
                    "rate_clip",
                    "height",
                ):
                    result[key] = sum(row[key] for row in samples) / len(samples)
                result["policy_rms"] = float(stacked["policy"].square().mean().sqrt())
                for key in ("policy", "target", "joint"):
                    result[key + "_temporal_std"] = float(
                        stacked[key].std(dim=0, correction=0).mean()
                    )
                    result[key + "_mean"] = [
                        round(v, 5) for v in stacked[key].mean(dim=0).tolist()
                    ]
                result["correction_rms_rad"] = float(
                    stacked["correction"].square().mean().sqrt()
                )
                result["joint_error_rms_rad"] = float(
                    stacked["joint_error"].square().mean().sqrt()
                )
                if probe_index is not None:
                    result["probe_joint"] = term._joint_names[probe_index]
                    result["probe_target_std_rad"] = float(
                        stacked["target"][:, probe_index].std(correction=0)
                    )
                    result["probe_actual_std_rad"] = float(
                        stacked["joint"][:, probe_index].std(correction=0)
                    )
                    result["probe_joint_error_rms_rad"] = float(
                        stacked["joint_error"][:, probe_index].square().mean().sqrt()
                    )
                    result["probe_torque_rms_nm"] = float(
                        stacked["torque"][:, probe_index].square().mean().sqrt()
                    )
                    result["probe_joint_velocity_rms_rad_s"] = float(
                        stacked["joint_velocity"][:, probe_index].square().mean().sqrt()
                    )
                    for key in list(result):
                        if key.startswith("policy_"):
                            result[key.replace("policy_", "action_", 1)] = result.pop(
                                key
                            )
                if not terminated:
                    result["distance_xy_m"] = float(
                        (robot.data.root_pos_w[0, :2] - start_xy).norm()
                    )
                    result["yaw_change_rad"] = yaw_travel
            records.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        print(f"[PROBE COMPLETE] {len(records)}/{len(cases)} commands", flush=True)
        return records
    except BaseException:
        # Print before env.close()/app.close(), which may terminate Kit before
        # Python gets a chance to print an unhandled exception.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()
