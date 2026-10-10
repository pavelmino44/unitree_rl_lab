"""Batch evaluation of an RSL-RL checkpoint on fixed scenarios (see eval_scenarios.py).

Example:
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless \
        --scenarios stairs_up_5cm,stairs_up_10cm,stairs_up_15cm --episodes 5
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless --scenarios all \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_5000.pt

Metrics per scenario:
    Metrics per scenario:
    success_rate        fraction of episodes in which the base crossed the goal line
                        (for pyramids: the far side, i.e. the climb AND the descent)
    midpoint_rate       fraction of episodes that reached the midpoint (pyramids: the top platform);
                        success_rate << midpoint_rate means the descent is the problem
    fall_rate           fraction of episodes terminated (fall / bad orientation) BEFORE reaching the goal
    mean_time_to_goal_s mean time to the goal line over successful episodes
    mean_progress_m     mean best forward progress from the spawn point
    max_yaw_err_deg     worst deviation of the heading from the target heading (mean/max over episodes)
    max_lateral_m       worst sideways deviation from the start line (mean/max over episodes)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import statistics
from datetime import datetime
from importlib.metadata import version
 
from isaaclab.app import AppLauncher
 
# local imports
import cli_args  # isort: skip
 
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint on fixed scenarios.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel copies of the same scenario.")
parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated scenario names, or 'all'.")
parser.add_argument("--episodes", type=int, default=5, help="Episodes per scenario.")
parser.add_argument("--eval_seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--output", type=str, default=None, help="Path of the JSON report.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
 
"""Rest everything follows."""
 
import gymnasium as gym
import torch
 
from rsl_rl.runners import OnPolicyRunner
 
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import wrap_to_pi
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
 
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.go2.eval_scenarios import EVAL_SCENARIOS, apply_eval_scenario
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
 
_UNSET = -1.0e9
 
 
def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    ok = [e for e in episodes if e["success"]]
    return {
        "episodes": n,
        "success_rate": len(ok) / n,
        "fall_rate": sum(e["fell_before_goal"] for e in episodes) / n,
        "mean_time_to_goal_s": statistics.mean(e["time_to_goal_s"] for e in ok) if ok else None,
        "midpoint_rate": sum(e["reached_midpoint"] for e in episodes) / n,
        "mean_progress_m": statistics.mean(e["progress_m"] for e in episodes),
        "mean_max_yaw_err_deg": statistics.mean(e["max_yaw_err_deg"] for e in episodes),
        "worst_max_yaw_err_deg": max(e["max_yaw_err_deg"] for e in episodes),
        "mean_max_lateral_m": statistics.mean(e["max_lateral_m"] for e in episodes),
        "worst_max_lateral_m": max(e["max_lateral_m"] for e in episodes),
    }
 
 
def run_scenario(name: str, agent_cfg, resume_path: str) -> dict:
    # --- environment ---------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="eval_env_cfg_entry_point",
    )
    sc = apply_eval_scenario(env_cfg, name)
    env_cfg.seed = args_cli.eval_seed
 
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
 
    # --- policy ----------------------------------------------------------------
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
 
    # --- bookkeeping -----------------------------------------------------------
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    origin_x = unwrapped.scene.env_origins[:, 0].clone()
    origin_y = unwrapped.scene.env_origins[:, 1].clone()
    n, dev, dt = unwrapped.num_envs, unwrapped.device, unwrapped.step_dt
 
    steps = torch.zeros(n, device=dev)
    max_x = torch.full((n,), _UNSET, device=dev)
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    t_goal = torch.full((n,), float("nan"), device=dev)
    reached_mid = torch.zeros(n, dtype=torch.bool, device=dev)
    max_yaw_err = torch.zeros(n, device=dev)
    max_lat = torch.zeros(n, device=dev)
    episodes: list[dict] = []
 
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
 
    # sanity check of the spawn pose (x relative to tile centre, z in world frame)
    x0 = (robot.data.root_pos_w[0, 0] - origin_x[0]).item()
    z0 = robot.data.root_pos_w[0, 2].item()
    print(f"[INFO] [{name}] {sc.description}")
    print(f"[INFO] [{name}] spawn x={x0:.2f} m (rel. to tile centre), z={z0:.2f} m (world); goal x={sc.goal_x} m")
 
    # --- rollout ---------------------------------------------------------------
    while len(episodes) < args_cli.episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
 
        x = (robot.data.root_pos_w[:, 0] - origin_x).clone()
        y = (robot.data.root_pos_w[:, 1] - origin_y).clone()
        yaw_err = wrap_to_pi(robot.data.heading_w - sc.heading).abs().clone()
        done = dones.bool()
        timeout = extras["time_outs"].bool()
        alive = ~done
 
        # after a reset the root pose already belongs to the next episode, so only use alive envs
        max_x = torch.where(alive, torch.maximum(max_x, x), max_x)
        steps += 1
        hit = alive & ~reached & (max_x >= sc.goal_x)
        t_goal[hit] = steps[hit] * dt
        reached |= hit
        if sc.midpoint_x is not None:
            reached_mid |= alive & (max_x >= sc.midpoint_x)
        max_yaw_err = torch.where(alive, torch.maximum(max_yaw_err, yaw_err), max_yaw_err)
        max_lat = torch.where(alive, torch.maximum(max_lat, y.abs()), max_lat)
 
        for i in done.nonzero().flatten().tolist():
            best_x = max_x[i].item() if max_x[i].item() > _UNSET / 2 else sc.spawn_x
            episodes.append(
                {
                    "success": bool(reached[i]),
                    "fell_before_goal": bool((not timeout[i]) and (not reached[i])),
                    "reached_midpoint": bool(reached_mid[i]) if sc.midpoint_x is not None else bool(reached[i]),
                    "time_to_goal_s": float(t_goal[i]) if reached[i] else None,
                    "duration_s": float(steps[i] * dt),
                    "progress_m": best_x - sc.spawn_x,
                    "max_yaw_err_deg": float(torch.rad2deg(max_yaw_err[i])),
                    "max_lateral_m": float(max_lat[i]),
                }
            )
 
        steps[done] = 0
        max_x[done] = _UNSET
        reached[done] = False
        t_goal[done] = float("nan")
        reached_mid[done] = False
        max_yaw_err[done] = 0.0
        max_lat[done] = 0.0
 
    episodes = episodes[: args_cli.episodes]
    env.close()
    return {"description": sc.description, "summary": summarize(episodes), "episodes": episodes}
 
 
def print_table(results: dict):
    print("\n" + "=" * 110)
    print(
        f"{'scenario':<18}{'episodes':>9}{'success':>9}{'midpoint':>10}{'falls':>7}{'t_goal, s':>11}"
        f"{'progress, m':>13}{'yaw err, deg':>14}{'lateral, m':>12}"
    )
    print("-" * 110)
    for name, r in results.items():
        s = r["summary"]
        tg = f"{s['mean_time_to_goal_s']:.1f}" if s["mean_time_to_goal_s"] is not None else "-"
        print(
            f"{name:<18}{s['episodes']:>9}{s['success_rate']:>9.0%}{s['midpoint_rate']:>10.0%}"
            f"{s['fall_rate']:>7.0%}{tg:>11}{s['mean_progress_m']:>13.2f}"
            f"{s['worst_max_yaw_err_deg']:>14.1f}{s['worst_max_lateral_m']:>12.2f}"
        )
    print("=" * 110)
    print("yaw err / lateral: worst case over episodes")
 
 
def main():
    names = list(EVAL_SCENARIOS) if args_cli.scenarios == "all" else [s.strip() for s in args_cli.scenarios.split(",")]
    unknown = [s for s in names if s not in EVAL_SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Available: {list(EVAL_SCENARIOS)}")
 
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}")
 
    results = {name: run_scenario(name, agent_cfg, resume_path) for name in names}
    print_table(results)
 
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(resume_path))[0]
    out = args_cli.output or os.path.join(os.path.dirname(resume_path), "eval", f"eval_{model_name}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "seed": args_cli.eval_seed,
        "num_envs": args_cli.num_envs,
        "date": stamp,
        "scenarios": results,
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved to: {out}")
 
 
if __name__ == "__main__":
    main()
    simulation_app.close()
 