"""Batch evaluation of an RSL-RL checkpoint on fixed scenarios (see eval_scenarios.py).

Every scenario runs in its OWN process (a fresh Isaac Sim instance). Re-creating an environment inside
one process is unreliable, so with several scenarios this script does not launch Isaac Sim itself: it
starts one child process per scenario and merges their reports.

Examples (run from the unitree_rl_lab root, in the conda env):
    # watch the scene like in play.py, with metrics printed per episode and as a table at the end
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --scenarios stairs_up_10cm --real-time

    # fast, no window
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless --scenarios all --episodes 5

Metrics per scenario:
    success_rate        fraction of episodes in which the base crossed the goal line
                        (for pyramids: the far side, i.e. the climb AND the descent)
    midpoint_rate       fraction of episodes that reached the midpoint (pyramids: the top platform);
                        success_rate << midpoint_rate means the descent is the problem
    fall_rate           fraction of episodes terminated (fall / bad orientation) BEFORE reaching the goal
    mean_time_to_goal_s mean time to the goal line over successful episodes
    mean_progress_m     mean best forward progress from the spawn point
    max_yaw_err_deg     worst deviation of the heading from the target heading
    max_lateral_m       worst sideways deviation from the start line
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

##
# Pure-python helpers (usable before Isaac Sim is launched)
##


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
    print("yaw err / lateral: worst case over episodes", flush=True)


def _drop_args(argv: list[str], flags: set[str]) -> list[str]:
    """Remove ``--flag value`` and ``--flag=value`` pairs from an argv list."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in flags:
            skip = True
        elif any(a.startswith(f + "=") for f in flags):
            pass
        else:
            out.append(a)
    return out


def _default_report_path(checkpoint: str, stamp: str) -> str:
    model_name = os.path.splitext(os.path.basename(checkpoint))[0]
    return os.path.join(os.path.dirname(checkpoint), "eval", f"eval_{model_name}_{stamp}.json")


##
# Arguments
##

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint on fixed scenarios.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel copies of the same scenario.")
parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated scenario names, or 'all'.")
parser.add_argument("--episodes", type=int, default=5, help="Episodes per scenario.")
parser.add_argument("--eval_seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--output", type=str, default=None, help="Path of the JSON report.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time (useful with the GUI).")
parser.add_argument("--list_scenarios_to", type=str, default=None, help=argparse.SUPPRESS)  # internal
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

##
# Parent mode: several scenarios -> one child process per scenario, Isaac Sim is NOT launched here
##


def _child_cmd(extra: list[str]) -> list[str]:
    base = _drop_args(sys.argv[1:], {"--scenarios", "--output", "--list_scenarios_to"})
    return [sys.executable, os.path.abspath(__file__), *base, *extra]


def _run_children(args) -> int:
    tmp = tempfile.mkdtemp(prefix="go2_eval_")

    if args.scenarios == "all":
        names_file = os.path.join(tmp, "names.json")
        print("[INFO] Resolving the list of scenarios (short headless start) ...", flush=True)
        subprocess.run(_child_cmd(["--headless", "--list_scenarios_to", names_file]))
        if not os.path.exists(names_file):
            print("[ERROR] Could not get the list of scenarios.")
            return 1
        with open(names_file) as f:
            names = json.load(f)
    else:
        names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    results, first = {}, None
    for i, name in enumerate(names, 1):
        print(f"\n[INFO] ===== scenario {i}/{len(names)}: {name} =====", flush=True)
        out = os.path.join(tmp, f"{name}.json")
        subprocess.run(_child_cmd(["--scenarios", name, "--output", out]))
        if not os.path.exists(out):
            print(f"[ERROR] Scenario '{name}' produced no report (see the log above). Continuing.", flush=True)
            continue
        with open(out) as f:
            report = json.load(f)
        first = first or report
        if report["checkpoint"] != first["checkpoint"]:
            print(f"[WARN] '{name}' used a different checkpoint: {report['checkpoint']}")
        results.update(report["scenarios"])

    if not results:
        print("[ERROR] No scenario finished.")
        return 1

    print_table(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or _default_report_path(first["checkpoint"], stamp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({**{k: v for k, v in first.items() if k != "scenarios"}, "date": stamp, "scenarios": results}, f, indent=2)
    print(f"[INFO] Report saved to: {out}")
    return 0 if len(results) == len(names) else 1


_is_single = args_cli.scenarios != "all" and "," not in args_cli.scenarios
if not _is_single and not args_cli.list_scenarios_to:
    sys.exit(_run_children(args_cli))

##
# Child mode: one scenario, one Isaac Sim instance
##

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
    print(f"[INFO] [{name}] running {args_cli.episodes} episodes, up to {sc.episode_length_s:.0f} s each", flush=True)

    # --- rollout ---------------------------------------------------------------
    while len(episodes) < args_cli.episodes and simulation_app.is_running():
        t_start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        x = (robot.data.root_pos_w[:, 0] - origin_x).clone()
        y = (robot.data.root_pos_w[:, 1] - origin_y).clone()
        yaw_err = wrap_to_pi(robot.data.heading_w - sc.heading).abs().clone()
        done = dones.bool()
        timeout = extras["time_outs"].bool()
        # flag of the goal_reached termination of THIS step (it survives the automatic reset)
        goal_hit = unwrapped.termination_manager.get_term("goal_reached").bool()
        alive = ~done

        # after a reset the root pose already belongs to the next episode, so only use alive envs
        max_x = torch.where(alive, torch.maximum(max_x, x), max_x)
        steps += 1
        if sc.midpoint_x is not None:
            reached_mid |= alive & (max_x >= sc.midpoint_x)
        max_yaw_err = torch.where(alive, torch.maximum(max_yaw_err, yaw_err), max_yaw_err)
        max_lat = torch.where(alive, torch.maximum(max_lat, y.abs()), max_lat)

        for i in done.nonzero().flatten().tolist():
            success = bool(goal_hit[i])
            best_x = max_x[i].item() if max_x[i].item() > _UNSET / 2 else sc.spawn_x
            if success:  # the pose of the terminating step is already reset, but the goal line was crossed
                best_x = max(best_x, sc.goal_x)
            ep = {
                "success": success,
                "fell_before_goal": bool((not timeout[i]) and (not success)),
                "reached_midpoint": success or (bool(reached_mid[i]) if sc.midpoint_x is not None else False),
                "time_to_goal_s": float(steps[i] * dt) if success else None,
                "duration_s": float(steps[i] * dt),
                "progress_m": best_x - sc.spawn_x,
                "max_yaw_err_deg": float(torch.rad2deg(max_yaw_err[i])),
                "max_lateral_m": float(max_lat[i]),
            }
            episodes.append(ep)
            verdict = "SUCCESS" if ep["success"] else ("FELL" if ep["fell_before_goal"] else "TIMEOUT")
            print(
                f"[{name}] episode {len(episodes)}/{args_cli.episodes}: {verdict:<8} "
                f"time={ep['duration_s']:.1f}s progress={ep['progress_m']:.2f}m "
                f"top={'yes' if ep['reached_midpoint'] else 'no'} "
                f"yaw_err={ep['max_yaw_err_deg']:.1f}deg lateral={ep['max_lateral_m']:.2f}m",
                flush=True,
            )

        steps[done] = 0
        max_x[done] = _UNSET
        reached_mid[done] = False
        max_yaw_err[done] = 0.0
        max_lat[done] = 0.0

        if args_cli.real_time:
            sleep_time = dt - (time.time() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    episodes = episodes[: args_cli.episodes]
    env.close()
    if not episodes:
        raise RuntimeError(f"Scenario '{name}': no episode finished (was the simulator window closed?).")
    return {"description": sc.description, "summary": summarize(episodes), "episodes": episodes}


def main():
    if args_cli.list_scenarios_to:
        with open(args_cli.list_scenarios_to, "w") as f:
            json.dump(list(EVAL_SCENARIOS), f)
        return

    name = args_cli.scenarios.strip()
    if name not in EVAL_SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(EVAL_SCENARIOS)}")

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}", flush=True)

    results = {name: run_scenario(name, agent_cfg, resume_path)}
    print_table(results)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args_cli.output or _default_report_path(resume_path, stamp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
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
    print(f"[INFO] Report saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()