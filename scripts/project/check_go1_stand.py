"""Zero-action standing sanity test for the Go1 T3 task.

Run before PPO training: if Go1 collapses even with all-zero policy actions,
fix its spawn pose, actuator gains or URDF physics before reward tuning.

Example (from unitree_rl_lab, with env_isaaclab activated):
    python scripts/project/check_go1_stand.py --headless --duration 5
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go1 zero-action pose and contact check")
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from unitree_rl_lab.tasks.locomotion.robots.go1.velocity_env_cfg import (
    Go1FlatPlayEnvCfg,
)


def main() -> None:
    cfg = Go1FlatPlayEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.commands.base_velocity.debug_vis = not args.headless
    cfg.scene.num_envs = 1
    if not args.headless:
        cfg.viewer.eye = (1.2, -1.2, 0.8)
        cfg.viewer.lookat = (0.0, 0.0, 0.25)
    cfg.episode_length_s = max(20.0, args.duration + 1.0)
    print("[GO1 STAND] creating environment", flush=True)
    env = gym.make("Unitree-Go1-T3-Flat", cfg=cfg)
    try:
        print("[GO1 STAND] resetting environment", flush=True)
        _, reset_info = env.reset()
        unwrapped = env.unwrapped
        expected_logs = {
            f"Episode_Termination/{name}"
            for name in ("time_out", "base_contact", "bad_orientation", "base_too_low")
        }
        missing_logs = expected_logs.difference(reset_info.get("log", {}))
        if missing_logs:
            raise RuntimeError(f"Missing Isaac Lab termination logs: {sorted(missing_logs)}")
        print("Termination log keys:", sorted(expected_logs))
        robot = unwrapped.scene["robot"]
        contacts = unwrapped.scene["contact_forces"]
        print("Sensor bodies:", contacts.body_names)
        foot_ids, foot_names = robot.find_bodies(
            ["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True
        )
        print("Standing foot bodies:", foot_names)
        actions = torch.zeros(
            (1, unwrapped.action_manager.total_action_dim), device=unwrapped.device
        )
        dt = float(unwrapped.step_dt)
        n_steps = math.ceil(args.duration / dt)
        stride = max(1, round(0.5 / dt))
        for step in range(n_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            if step % stride == 0 or bool(terminated[0]) or bool(truncated[0]):
                z = float(robot.data.root_link_pos_w[0, 2].item())
                feet_w = robot.data.body_pos_w[:, foot_ids]
                foot_surface_z = feet_w[0, :, 2] - 0.023  # MJCF sphere radius.
                front_delta_w = feet_w[:, 1] - feet_w[:, 0]
                rear_delta_w = feet_w[:, 3] - feet_w[:, 2]
                front_width = float(quat_apply_inverse(
                    robot.data.root_quat_w, front_delta_w
                )[0, 1].item())
                rear_width = float(quat_apply_inverse(
                    robot.data.root_quat_w, rear_delta_w
                )[0, 1].item())
                q = robot.data.root_link_quat_w[:1]
                roll, pitch, _ = euler_xyz_from_quat(q)
                norms = contacts.data.net_forces_w[0].norm(dim=-1)
                indices = torch.argsort(norms, descending=True)[:5].tolist()
                forces = ", ".join(
                    f"{contacts.body_names[i]}:{float(norms[i]):.1f}N" for i in indices
                )
                print(
                    f"t={(step + 1)*dt:.2f}s | z={z:.3f}m | "
                    f"roll={math.degrees(float(roll[0])):+.1f}deg | "
                    f"pitch={math.degrees(float(pitch[0])):+.1f}deg | "
                    f"front_width={front_width:.3f}m | rear_width={rear_width:.3f}m | "
                    f"foot_surface_z=[{float(foot_surface_z.min()):+.4f}, "
                    f"{float(foot_surface_z.max()):+.4f}]m | "
                    f"contacts=[{forces}] | done={bool(terminated[0]) or bool(truncated[0])}"
                )
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("FAILED: reset/termination during zero-action stance")
        else:
            print("PASSED: no early termination with zero policy actions")
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        print("[GO1 STAND FAILED]", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
    raise SystemExit(exit_code)
