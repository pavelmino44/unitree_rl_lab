"""Local, approximate T3 flat-rollout test using Isaac Lab ground truth.

IMPORTANT: This is NOT the official Judge and cannot substitute for MuJoCo
sim-to-sim with the T2 collision-aware Safety Block. Each episode performs an
Isaac Lab reset, not a full Docker/physics engine restart.

Example:
    python scripts/project/evaluate_go1_t3.py \
      --checkpoint logs/rsl_rl/unitree_go1_t3_flat/RUN/model_5000.pt \
      --episodes 10 --speed 0.3 --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go1 T3 flat local evaluation (not official Judge)")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--output", type=str, default="logs/go1_t3_eval.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1:
    parser.error("--episodes must be positive")
if not (0.1 <= args.speed <= 0.6):
    parser.error("--speed must be between 0.1 and 0.6 m/s")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.go1.agents import Go1FlatPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1FlatPlayEnvCfg


def main():
    cfg = Go1FlatPlayEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 94.0
    cfg.terminations.bad_orientation = None
    cfg.terminations.base_contact = None
    cfg.terminations.base_too_low = None
    # T3: constant forward Twist, no lateral command and no turning.
    for ranges in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        ranges.lin_vel_x = (args.speed, args.speed)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)

    raw_env = gym.make("Unitree-Go1-T3-Flat", cfg=cfg)
    agent_cfg = Go1FlatPPORunnerCfg()
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(Path(args.checkpoint).expanduser().resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    step_dt = float(env.unwrapped.step_dt)
    warmup_steps = math.ceil(1.5 / step_dt)
    movement_steps = math.ceil(90.0 / step_dt)
    persistence_steps = math.floor(0.5 / step_dt) + 1  # strictly > 0.5 s
    records = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        start_xy = None
        # The judge measures forward progress relative to the initial heading.
        _, _, initial_yaw = euler_xyz_from_quat(robot.data.root_quat_w[:1])
        y0 = initial_yaw[0].item()
        heading = torch.tensor([math.cos(y0), math.sin(y0)], device=robot.device)
        fall_steps = 0
        fell = False
        forward = 0.0
        lateral = 0.0
        scored_s = 0.0
        with torch.inference_mode():
            for step in range(warmup_steps + movement_steps):
                obs, _, done, _ = env.step(policy(obs))
                pos = robot.data.root_pos_w[0]
                quat = robot.data.root_quat_w[:1]
                roll, pitch, yaw = euler_xyz_from_quat(quat)
                if step == warmup_steps - 1:
                    start_xy = pos[:2].clone()
                    fall_steps = 0
                    continue
                if step < warmup_steps:
                    continue
                scored_s = (step - warmup_steps + 1) * step_dt
                dxy = pos[:2] - start_xy
                forward = float(torch.dot(dxy, heading).item())
                lateral = float(heading[0] * dxy[1] - heading[1] * dxy[0])
                bad = (
                    (pos[2] < 0.16).item()
                    or (abs(roll[0].item()) > math.radians(50.0))
                    or (abs(pitch[0].item()) > math.radians(50.0))
                )
                fall_steps = fall_steps + 1 if bad else 0
                if fall_steps >= persistence_steps or bool(done[0].item()):
                    fell = True
                    break
                if forward >= 3.0:
                    break
        success = (forward >= 3.0) and not fell
        record = {
            "episode": ep + 1,
            "success": int(success),
            "forward_m": round(forward, 3),
            "lateral_m": round(lateral, 3),
            "scored_time_s": round(scored_s, 3),
            "fell": int(fell),
            "speed_cmd_mps": args.speed,
        }
        print(json.dumps(record))
        records.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    success_count = sum(record["success"] for record in records)
    print(f"Local Isaac Lab success: {success_count}/{len(records)}. CSV: {output}")
    print("Not equivalent to the T3 MuJoCo Judge, including collision-aware Safety filtering.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
