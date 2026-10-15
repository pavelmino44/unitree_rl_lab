"""Batch evaluation of an RSL-RL checkpoint on fixed scenarios (see eval_scenarios.py).

Every (scenario, seed) pair runs in its OWN process (a fresh Isaac Sim instance). Re-creating an
environment inside one process is unreliable, so with several scenarios/seeds this script does not
launch Isaac Sim itself: it starts one child process per pair and merges their reports.

Examples (run from the unitree_rl_lab root, in the conda env):
    # watch the scene like in play.py, metrics are printed per episode and as tables at the end
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --scenarios stairs_up_10cm --real-time

    # all obstacles of ROUGH_TERRAINS_CFG, 3 terrain realizations, no window
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless \
        --scenarios rough --eval_seed 42,43,44 --episodes 3

    # patterns work too
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless --scenarios "rough_boxes*,flat"

--scenarios accepts: exact names, glob patterns (stairs_up_*), rough (= rough_*) and all.
--eval_seed accepts a comma-separated list. The seed controls the terrain realization of the randomized
obstacles (boxes, random_rough); episodes of all seeds of a scenario are pooled in the summary.

Metrics (per scenario, over all pooled episodes)
  Task:        success_rate (goal line crossed; for pyramids: the far side = climb AND descent),
               midpoint_rate (top / bottom / center reached), fall_rate (terminated before the goal),
               timeout_rate, mean_time_to_goal_s, mean_progress_m
  Path:        max_yaw_err_deg (heading vs. target), max_lateral_m (sideways deviation from the start line)
  Tracking:    vel_err_xy_mps, vel_err_yaw_radps (commanded vs. measured base velocity, body frame),
               forward_speed_mps
  Efficiency:  mean_power_w (sum |torque * joint_vel|), cost_of_transport (E / (m g d))
  Smoothness:  action_rate (|a_t - a_t-1|), joint_acc (|qdd|)
  Stability:   tilt_deg (angle between base z-axis and gravity: mean and max)
  Training-like: total_reward per episode, reward_terms_per_s (each reward term averaged per second, the
               same quantity as Episode_Reward/<term> in the training logs for a full-length episode),
               end_reasons (which termination terms ended the episode)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import fnmatch
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

##
# Pure-python helpers (usable before Isaac Sim is launched)
##


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _f(value, spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    ok = [e for e in episodes if e["success"]]
    fell = sum(e["fell_before_goal"] for e in episodes)

    def mean_of(key):
        return _mean(e.get(key) for e in episodes)

    def worst_of(key):
        vals = [e[key] for e in episodes if e.get(key) is not None]
        return max(vals) if vals else None

    term_keys = sorted({k for e in episodes for k in e.get("reward_terms_per_s", {})})
    return {
        "episodes": n,
        # task
        "success_rate": len(ok) / n,
        "midpoint_rate": sum(e["reached_midpoint"] for e in episodes) / n,
        "fall_rate": fell / n,
        "timeout_rate": (n - len(ok) - fell) / n,
        "mean_time_to_goal_s": _mean(e["time_to_goal_s"] for e in ok),
        "mean_progress_m": mean_of("progress_m"),
        # path
        "mean_max_yaw_err_deg": mean_of("max_yaw_err_deg"),
        "worst_max_yaw_err_deg": worst_of("max_yaw_err_deg"),
        "mean_max_lateral_m": mean_of("max_lateral_m"),
        "worst_max_lateral_m": worst_of("max_lateral_m"),
        # tracking
        "mean_vel_err_xy_mps": mean_of("vel_err_xy_mps"),
        "mean_vel_err_yaw_radps": mean_of("vel_err_yaw_radps"),
        "mean_forward_speed_mps": mean_of("forward_speed_mps"),
        # efficiency
        "mean_power_w": mean_of("power_w"),
        "mean_cost_of_transport": mean_of("cost_of_transport"),
        # smoothness
        "mean_action_rate": mean_of("action_rate"),
        "mean_joint_acc": mean_of("joint_acc"),
        # stability
        "mean_tilt_deg": mean_of("mean_tilt_deg"),
        "worst_max_tilt_deg": worst_of("max_tilt_deg"),
        # training-like
        "mean_total_reward": mean_of("total_reward"),
        "reward_terms_per_s": {k: _mean(e.get("reward_terms_per_s", {}).get(k) for e in episodes) for k in term_keys},
        "end_reasons": dict(Counter(e.get("end_reason", "unknown") for e in episodes)),
    }


def print_tables(results: dict):
    w = max(len(k) for k in results) + 2
    line = "=" * (w + 100)

    # --- table 1: task result and path --------------------------------------------------------
    print("\n" + line)
    print(
        f"{'scenario':<{w}}{'episodes':>9}{'success':>9}{'midpoint':>10}{'falls':>7}{'timeouts':>10}"
        f"{'t_goal, s':>11}{'progress, m':>13}{'yaw err, deg':>14}{'lateral, m':>12}"
    )
    print("-" * (w + 100))
    for name, r in results.items():
        s = r["summary"]
        print(
            f"{name:<{w}}{s['episodes']:>9}{s['success_rate']:>9.0%}{s['midpoint_rate']:>10.0%}"
            f"{s['fall_rate']:>7.0%}{s['timeout_rate']:>10.0%}{_f(s['mean_time_to_goal_s'], '.1f'):>11}"
            f"{_f(s['mean_progress_m'], '.2f'):>13}{_f(s['worst_max_yaw_err_deg'], '.1f'):>14}"
            f"{_f(s['worst_max_lateral_m'], '.2f'):>12}"
        )
    print(line)
    print("yaw err / lateral: worst case over episodes")

    # --- table 2: motion quality ---------------------------------------------------------------
    print("\n" + line)
    print(
        f"{'scenario':<{w}}{'vel err xy':>11}{'vel err yaw':>12}{'speed':>8}{'power, W':>10}{'CoT':>7}"
        f"{'act. rate':>11}{'joint acc':>11}{'tilt max':>10}{'reward':>9}"
    )
    print("-" * (w + 100))
    for name, r in results.items():
        s = r["summary"]
        print(
            f"{name:<{w}}{_f(s['mean_vel_err_xy_mps'], '.3f'):>11}{_f(s['mean_vel_err_yaw_radps'], '.3f'):>12}"
            f"{_f(s['mean_forward_speed_mps'], '.2f'):>8}{_f(s['mean_power_w'], '.0f'):>10}"
            f"{_f(s['mean_cost_of_transport'], '.2f'):>7}{_f(s['mean_action_rate'], '.2f'):>11}"
            f"{_f(s['mean_joint_acc'], '.0f'):>11}{_f(s['worst_max_tilt_deg'], '.0f'):>10}"
            f"{_f(s['mean_total_reward'], '.1f'):>9}"
        )
    print(line)
    print("vel err: m/s and rad/s | speed: m/s along +x | tilt max: deg, worst case | reward: mean sum per episode")

    # --- end reasons -----------------------------------------------------------------------------
    print("\nEpisode end reasons:")
    for name, r in results.items():
        reasons = ", ".join(f"{k} x{v}" for k, v in sorted(r["summary"]["end_reasons"].items()))
        print(f"  {name:<{w}}{reasons}")
    print(flush=True)


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


def _tokens(scenarios: str) -> list[str]:
    return [t.strip() for t in scenarios.split(",") if t.strip()]


def _is_pattern(token: str) -> bool:
    return token in ("all", "rough") or any(c in token for c in "*?[")


def _expand(tokens: list[str], registry: list[str]) -> list[str]:
    names: list[str] = []
    for t in tokens:
        pattern = "*" if t == "all" else ("rough_*" if t == "rough" else t)
        matches = [n for n in registry if fnmatch.fnmatchcase(n, pattern)]
        if not matches:
            raise ValueError(f"'{t}' matches no scenario. Available: {registry}")
        names += [m for m in matches if m not in names]
    return names


##
# Arguments
##

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint on fixed scenarios.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel copies of the same scenario.")
parser.add_argument(
    "--scenarios", type=str, default="all", help="Comma-separated names / glob patterns / 'rough' / 'all'."
)
parser.add_argument("--episodes", type=int, default=5, help="Episodes per scenario and seed.")
parser.add_argument(
    "--eval_seed", type=str, default="42", help="Environment/terrain seed; a comma-separated list pools several."
)
parser.add_argument("--output", type=str, default=None, help="Path of the JSON report.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time (useful with the GUI).")
parser.add_argument("--list_scenarios_to", type=str, default=None, help=argparse.SUPPRESS)  # internal
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

##
# Parent mode: several scenarios / seeds -> one child process per pair, Isaac Sim is NOT launched here
##


def _child_cmd(extra: list[str]) -> list[str]:
    base = _drop_args(sys.argv[1:], {"--scenarios", "--output", "--list_scenarios_to", "--eval_seed"})
    return [sys.executable, os.path.abspath(__file__), *base, *extra]


def _run_children(args) -> int:
    tmp = tempfile.mkdtemp(prefix="go2_eval_")
    tokens = _tokens(args.scenarios)
    seeds = [s.strip() for s in args.eval_seed.split(",") if s.strip()]

    if any(_is_pattern(t) for t in tokens):
        names_file = os.path.join(tmp, "names.json")
        print("[INFO] Resolving the list of scenarios (short headless start) ...", flush=True)
        subprocess.run(_child_cmd(["--headless", "--list_scenarios_to", names_file]))
        if not os.path.exists(names_file):
            print("[ERROR] Could not get the list of scenarios.")
            return 1
        with open(names_file) as f:
            registry = json.load(f)
        try:
            names = _expand(tokens, registry)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return 1
    else:
        names = tokens

    total = len(names) * len(seeds)
    print(f"[INFO] {len(names)} scenario(s) x {len(seeds)} seed(s) = {total} run(s): {names}", flush=True)

    pooled: dict[str, dict] = {}
    first, done_runs, failed = None, 0, 0
    for name in names:
        for seed in seeds:
            done_runs += 1
            print(f"\n[INFO] ===== run {done_runs}/{total}: {name}, seed {seed} =====", flush=True)
            out = os.path.join(tmp, f"{name}_{seed}.json")
            subprocess.run(_child_cmd(["--scenarios", name, "--eval_seed", seed, "--output", out]))
            if not os.path.exists(out):
                print(f"[ERROR] '{name}' (seed {seed}) produced no report (see the log above). Continuing.", flush=True)
                failed += 1
                continue
            with open(out) as f:
                report = json.load(f)
            first = first or report
            if report["checkpoint"] != first["checkpoint"]:
                print(f"[WARN] '{name}' used a different checkpoint: {report['checkpoint']}")
            entry = report["scenarios"][name]
            slot = pooled.setdefault(name, {"description": entry["description"], "seeds": [], "episodes": []})
            slot["seeds"].append(int(seed))
            slot["episodes"] += entry["episodes"]

    if not pooled:
        print("[ERROR] No run finished.")
        return 1

    results = {}
    for name, slot in pooled.items():
        slot["summary"] = summarize(slot["episodes"])
        results[name] = slot

    print_tables(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or _default_report_path(first["checkpoint"], stamp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    meta = {k: v for k, v in first.items() if k not in ("scenarios", "seed")}
    with open(out, "w") as f:
        json.dump({**meta, "seeds": [int(s) for s in seeds], "date": stamp, "scenarios": results}, f, indent=2)
    print(f"[INFO] Report saved to: {out}")
    if failed:
        print(f"[WARN] {failed} run(s) failed, their episodes are missing from the report.")
    return 0 if not failed else 1


_tok = _tokens(args_cli.scenarios)
_single_seed = len([s for s in args_cli.eval_seed.split(",") if s.strip()]) == 1
_is_single = len(_tok) == 1 and not _is_pattern(_tok[0]) and _single_seed
if not _is_single and not args_cli.list_scenarios_to:
    sys.exit(_run_children(args_cli))

##
# Child mode: one scenario, one seed, one Isaac Sim instance
##

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import random
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
_GRAVITY = 9.81


def run_scenario(name: str, seed: int, agent_cfg, resume_path: str) -> dict:
    # --- environment ---------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="eval_env_cfg_entry_point",
    )
    env_cfg.seed = seed  # must be set before apply_eval_scenario (it seeds the terrain generator)
    sc = apply_eval_scenario(env_cfg, name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- policy ----------------------------------------------------------------
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- handles ---------------------------------------------------------------
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    cmd_mgr = unwrapped.command_manager
    rew_mgr = unwrapped.reward_manager
    term_mgr = unwrapped.termination_manager
    origin_x = unwrapped.scene.env_origins[:, 0].clone()
    origin_y = unwrapped.scene.env_origins[:, 1].clone()
    n, dev, dt = unwrapped.num_envs, unwrapped.device, unwrapped.step_dt

    reward_names = list(rew_mgr.active_terms)
    done_names = list(term_mgr.active_terms)
    try:  # robot mass for the cost of transport (not available in every version)
        mass = robot.data.default_mass.sum(dim=-1).to(dev)
    except Exception:
        mass = None

    def step_reward_terms() -> torch.Tensor:
        """Weighted reward rate of every reward term for the step that was just executed, (n, num_terms)."""
        buf = getattr(rew_mgr, "_step_reward", None)
        if buf is not None:
            return buf
        return torch.tensor(
            [[v[0] for _, v in rew_mgr.get_active_iterable_terms(i)] for i in range(n)], device=dev
        )

    # --- per-episode accumulators ----------------------------------------------
    def zeros():
        return torch.zeros(n, device=dev)

    steps = zeros()  # all steps of the episode, including the terminating one
    alive_steps = zeros()  # steps whose state is valid (not the terminating one)
    max_x = torch.full((n,), _UNSET, device=dev)
    reached_mid = torch.zeros(n, dtype=torch.bool, device=dev)
    max_yaw_err, max_lat, max_tilt = zeros(), zeros(), zeros()
    sums = {k: zeros() for k in ("vel_err_xy", "vel_err_yaw", "fwd_speed", "power", "action_rate", "joint_acc", "tilt")}
    total_reward = zeros()
    term_sums = torch.zeros(n, max(len(reward_names), 1), device=dev)
    prev_actions = None
    episodes: list[dict] = []

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # sanity check of the spawn pose (x relative to tile centre, z in world frame)
    x0 = (robot.data.root_pos_w[0, 0] - origin_x[0]).item()
    z0 = robot.data.root_pos_w[0, 2].item()
    print(f"[INFO] [{name}] seed {seed}: {sc.description}")
    print(f"[INFO] [{name}] spawn x={x0:.2f} m (rel. to tile centre), z={z0:.2f} m (world); goal x={sc.goal_x:.2f} m")
    print(f"[INFO] [{name}] running {args_cli.episodes} episodes, up to {sc.episode_length_s:.0f} s each", flush=True)

    # --- rollout ---------------------------------------------------------------
    while len(episodes) < args_cli.episodes and simulation_app.is_running():
        t_start = time.time()
        # command that the policy is about to act on (body frame: vx, vy, wz)
        cmd = cmd_mgr.get_command("base_velocity").clone()

        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)

        if prev_actions is None:
            prev_actions = torch.zeros_like(actions)

        # --- state after the step ---
        x = (robot.data.root_pos_w[:, 0] - origin_x).clone()
        y = (robot.data.root_pos_w[:, 1] - origin_y).clone()
        yaw_err = wrap_to_pi(robot.data.heading_w - sc.heading).abs().clone()
        lin_b, ang_b = robot.data.root_lin_vel_b, robot.data.root_ang_vel_b
        vel_err_xy = torch.norm(cmd[:, :2] - lin_b[:, :2], dim=-1)
        vel_err_yaw = (cmd[:, 2] - ang_b[:, 2]).abs()
        fwd_speed = robot.data.root_lin_vel_w[:, 0]
        power = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(dim=-1)
        action_rate = torch.norm(actions - prev_actions, dim=-1)
        joint_acc = torch.norm(robot.data.joint_acc, dim=-1)
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))

        done = dones.bool()
        timeout = extras["time_outs"].bool()
        # flag of the goal_reached termination of THIS step (it survives the automatic reset)
        goal_hit = term_mgr.get_term("goal_reached").bool()
        alive = ~done
        zero = torch.zeros_like(x)

        # rewards of the terminating step belong to the ending episode (they are computed before the reset)
        total_reward = total_reward + rew
        if reward_names:
            term_sums = term_sums + step_reward_terms() * dt

        # kinematic quantities after a reset belong to the next episode, so only use alive envs
        max_x = torch.where(alive, torch.maximum(max_x, x), max_x)
        max_yaw_err = torch.where(alive, torch.maximum(max_yaw_err, yaw_err), max_yaw_err)
        max_lat = torch.where(alive, torch.maximum(max_lat, y.abs()), max_lat)
        max_tilt = torch.where(alive, torch.maximum(max_tilt, tilt), max_tilt)
        for key, value in (
            ("vel_err_xy", vel_err_xy),
            ("vel_err_yaw", vel_err_yaw),
            ("fwd_speed", fwd_speed),
            ("power", power),
            ("action_rate", action_rate),
            ("joint_acc", joint_acc),
            ("tilt", tilt),
        ):
            sums[key] = sums[key] + torch.where(alive, value, zero)
        alive_steps = alive_steps + alive.float()
        steps = steps + 1
        if sc.midpoint_x is not None:
            reached_mid = reached_mid | (alive & (max_x >= sc.midpoint_x))

        if done.any():
            flags = torch.stack([term_mgr.get_term(nm).bool() for nm in done_names], dim=1)
            for i in done.nonzero().flatten().tolist():
                success = bool(goal_hit[i])
                best_x = max_x[i].item() if max_x[i].item() > _UNSET / 2 else sc.spawn_x
                if success:  # the pose of the terminating step is already reset, but the goal line was crossed
                    best_x = max(best_x, sc.goal_x)
                progress = best_x - sc.spawn_x
                duration = float(steps[i] * dt)
                k = max(float(alive_steps[i]), 1.0)
                mean_power = float(sums["power"][i]) / k
                energy = mean_power * k * dt
                cot = None
                if mass is not None and progress > 0.5:
                    cot = energy / (float(mass[i]) * _GRAVITY * progress)
                reasons = [nm for j, nm in enumerate(done_names) if bool(flags[i, j])]
                ep = {
                    "seed": seed,
                    "success": success,
                    "fell_before_goal": bool((not timeout[i]) and (not success)),
                    "reached_midpoint": success or (bool(reached_mid[i]) if sc.midpoint_x is not None else False),
                    "time_to_goal_s": duration if success else None,
                    "duration_s": duration,
                    "progress_m": progress,
                    "max_yaw_err_deg": float(torch.rad2deg(max_yaw_err[i])),
                    "max_lateral_m": float(max_lat[i]),
                    "vel_err_xy_mps": float(sums["vel_err_xy"][i]) / k,
                    "vel_err_yaw_radps": float(sums["vel_err_yaw"][i]) / k,
                    "forward_speed_mps": float(sums["fwd_speed"][i]) / k,
                    "power_w": mean_power,
                    "cost_of_transport": cot,
                    "action_rate": float(sums["action_rate"][i]) / k,
                    "joint_acc": float(sums["joint_acc"][i]) / k,
                    "mean_tilt_deg": float(torch.rad2deg(sums["tilt"][i] / k)),
                    "max_tilt_deg": float(torch.rad2deg(max_tilt[i])),
                    "total_reward": float(total_reward[i]),
                    "reward_terms_per_s": {nm: float(term_sums[i, j]) / duration for j, nm in enumerate(reward_names)},
                    "end_reason": "+".join(sorted(reasons)) if reasons else ("time_out" if timeout[i] else "unknown"),
                }
                episodes.append(ep)
                verdict = "SUCCESS" if success else ("FELL" if ep["fell_before_goal"] else "TIMEOUT")
                mid = f" {sc.midpoint_label}={'yes' if ep['reached_midpoint'] else 'no'}" if sc.midpoint_x is not None else ""
                print(
                    f"[{name}] episode {len(episodes)}/{args_cli.episodes}: {verdict:<8} "
                    f"time={duration:.1f}s progress={progress:.2f}m{mid} yaw_err={ep['max_yaw_err_deg']:.1f}deg "
                    f"lateral={ep['max_lateral_m']:.2f}m vel_err={ep['vel_err_xy_mps']:.2f}m/s "
                    f"reward={ep['total_reward']:.1f} end={ep['end_reason']}",
                    flush=True,
                )

        # --- reset the accumulators of the finished episodes (out-of-place for inference-mode tensors) ---
        prev_actions = torch.where(done.unsqueeze(-1), torch.zeros_like(actions), actions)
        steps[done] = 0
        alive_steps[done] = 0
        max_x[done] = _UNSET
        reached_mid[done] = False
        max_yaw_err[done] = 0.0
        max_lat[done] = 0.0
        max_tilt[done] = 0.0
        total_reward[done] = 0.0
        term_sums[done] = 0.0
        for key in sums:
            sums[key][done] = 0.0

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
    seed = int(args_cli.eval_seed)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}", flush=True)

    results = {name: run_scenario(name, seed, agent_cfg, resume_path)}
    results[name]["seeds"] = [seed]
    print_tables(results)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args_cli.output or _default_report_path(resume_path, stamp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    report = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "seed": seed,
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