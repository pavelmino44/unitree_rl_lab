"""Zero-action standing sanity test for the Go1 T3 task.

Run before PPO training: if Go1 collapses even with all-zero policy actions,
fix its spawn pose, actuator gains or URDF physics before reward tuning.

Example (from unitree_rl_lab, with env_isaaclab activated):
    python scripts/project/check_go1_stand.py --headless --duration 5
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go1 zero-action pose and contact check")
parser.add_argument("--duration", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import euler_xyz_from_quat

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1FlatPlayEnvCfg


def main() -> None:
    cfg = Go1FlatPlayEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(20.0, args.duration + 1.0)
    env = gym.make("Unitree-Go1-T3-Flat", cfg=cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        contacts = unwrapped.scene["contact_forces"]
        print("Sensor bodies:", contacts.body_names)
        actions = torch.zeros(
            (1, unwrapped.action_manager.total_action_dim), device=unwrapped.device
        )
        dt = float(unwrapped.step_dt)
        n_steps = math.ceil(args.duration / dt)
        stride = max(1, int(round(0.5 / dt)))
        for step in range(n_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            if step % stride == 0 or bool(terminated[0]) or bool(truncated[0]):
                z = float(robot.data.root_link_pos_w[0, 2].item())
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
                    f"contacts=[{forces}] | done={bool(terminated[0]) or bool(truncated[0])}"
                )
            if bool(terminated[0]) or bool(truncated[0]):
                print("FAILED: reset/termination during zero-action stance")
                break
        else:
            print("PASSED: no early termination with zero policy actions")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
