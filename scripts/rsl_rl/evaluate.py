"""Batch evaluation of an RSL-RL checkpoint on fixed scenarios (see eval_scenarios.py).

Every (scenario, seed) pair runs in its OWN process (a fresh Isaac Sim instance). Re-creating an
environment inside one process is unreliable, so with several scenarios/seeds this script does not
launch Isaac Sim itself: it starts one child process per pair and merges their reports.

Two ways to start it (run from the unitree_rl_lab root, in the conda env):
  * with a window to set everything up (task, scenarios, episodes, seeds, checkpoint / load run, report
    path, headless / real-time, device; configurations can be saved and loaded):
        python scripts/rsl_rl/evaluate.py --gui
  * the classic way, everything on the command line (examples below).

Examples:
    # watch the scene like in play.py, metrics are printed per episode and as tables at the end
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --scenarios rough_pyramid_stairs_10cm --real-time

    # all obstacles of ROUGH_TERRAINS_CFG, 3 terrain realizations, no window
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless \
        --scenarios rough --eval_seed 42,43,44 --episodes 3

    # patterns work too
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless --scenarios "rough_boxes*,flat"

    # another commanded speed, and let the episode go on after the trunk touches the ground
    python scripts/rsl_rl/evaluate.py --task Unitree-Go2-Velocity --headless --scenarios "rough_pyramid_stairs_*cm" \
        --speed 0.3 --no_contact_termination

--scenarios accepts: exact names, glob patterns (rough_pyramid_stairs_*cm, flat*), rough (= rough_*) and all.
--eval_seed accepts a comma-separated list. The seed controls the terrain realization of the randomized
obstacles (boxes, random_rough); episodes of all seeds of a scenario are pooled in the summary.
--speed sets the commanded forward speed in m/s (default: 0.5, the value built into the scenarios).
--no_contact_termination removes the training termination on trunk contact, so that the whole episode is
observed and every trunk contact can be shown on the timeline (by default one contact ends the episode,
exactly as in training).

Scenario kinds (see eval_scenarios.py): walking along a course (forward, backward, sideways, obstacles),
turning on the spot (progress in degrees) and the sprint with a sudden stop (flat_max_speed: it ignores --speed
and uses the maximum speed of the training range; success = the robot survives and comes to a stop).

When the runs are finished the report viewer (eval_report.py) opens: one window per episode (speed vs. time
with the commanded speed, trunk-contact timeline, all numbers), one window per scenario (averages over its
episodes) and one window for the whole run. It can also be started on its own on a saved report:
    python scripts/rsl_rl/eval_report.py path/to/report.json
Use --no_report to skip it, --report_save DIR to also write the figures as PNG files.

Metrics (per scenario, over all pooled episodes)
  Task:        success_rate (goal line crossed; for pyramids: the far side = climb AND descent),
               midpoint_rate (top / bottom / center reached), fall_rate (terminated before the goal),
               timeout_rate, mean_time_to_goal_s, mean_progress_m
  Path:        max_yaw_err_deg (heading vs. target), max_lateral_m (sideways deviation from the start line;
               for the turn on the spot: drift from the start point)
  Tracking:    vel_err_xy_mps, vel_err_yaw_radps (commanded vs. measured base velocity, body frame),
               forward_speed_mps
  Efficiency:  mean_power_w (sum |torque * joint_vel|), cost_of_transport (E / (m g d))
  Smoothness:  action_rate (|a_t - a_t-1|), joint_acc (|qdd|)
  Stability:   tilt_deg (angle between base z-axis and gravity: mean and max)
  Trunk contact: events (contacts closer than 0.1 s count as one), contact time, first contact, max force;
               "trunk" = the bodies and force threshold of the training termination (e.g. base_contact)
  Sprint/brake (flat_max_speed): peak_speed, time to 90 % of the command, speed at the stop command,
               stop time (speed < 0.1 m/s for 0.5 s), stop distance and the largest roll-back after it
  Time series: every episode stores a down-sampled (--record_dt, default 0.1 s) time series of the speed,
               command, position, heading error, tilt, power and trunk contact for the report viewer
  Training-like: total_reward per episode, reward_terms_per_s (each reward term averaged per second, the
               same quantity as Episode_Reward/<term> in the training logs for a full-length episode),
               end_reasons (which termination terms ended the episode)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import fnmatch
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime
from importlib.metadata import version

import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

##
# Pure-python helpers (usable before Isaac Sim is launched)
##


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


# Channels of the recorded time series (columns of the per-step rows).
SERIES_CHANNELS = (
    "vx", "vy", "wz",  # measured base velocity, body frame (m/s, m/s, rad/s)
    "cmd_vx", "cmd_vy", "cmd_wz",  # commanded velocity
    "x", "y",  # position relative to the terrain tile centre (m)
    "yaw_err_deg", "tilt_deg", "power_w",
    "contact_force_n",  # largest trunk contact force during the step (N)
    "contact",  # 1 while the trunk is in contact
)  # fmt: skip
CONTACT_COL, FORCE_COL = SERIES_CHANNELS.index("contact"), SERIES_CHANNELS.index("contact_force_n")
X_COL = SERIES_CHANNELS.index("x")
CONTACT_MERGE_GAP_S = 0.1  # trunk contacts closer than this count as one event
STOP_SPEED_MPS, STOP_HOLD_S = 0.1, 0.5  # "stopped" = |speed| below this for this long
SPRINT_KEYS = (
    "peak_speed_mps", "time_to_90pct_s", "speed_at_brake_mps", "stop_time_s", "stop_distance_m",
    "min_speed_after_brake_mps",
)  # fmt: skip
CONTACT_KEYS = (
    "trunk_contact_events", "trunk_contact_time_s", "first_trunk_contact_s",
    "max_trunk_contact_force_n", "trunk_contact_intervals_s",
)  # fmt: skip


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the runs of True in a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def contact_intervals(mask: np.ndarray, dt: float, merge_gap_s: float = CONTACT_MERGE_GAP_S) -> list[list[float]]:
    """Time intervals (s) in which mask (one value per control step) is True; close ones are merged."""
    gap = int(round(merge_gap_s / dt))
    merged: list[list[int]] = []
    for start, end in _runs(mask):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [[start * dt, end * dt] for start, end in merged]


def sprint_metrics(arr: np.ndarray, dt: float, brake_t: float, v_cmd: float) -> dict:
    """Metrics of "accelerate to v_cmd, command 0 at brake_t" from the full-rate rows (see SERIES_CHANNELS).

    peak speed and time to 90 % of v_cmd (up to the stop command), the speed at the stop command and, after it,
    the stop time (|speed| < STOP_SPEED_MPS for STOP_HOLD_S), the distance travelled until then and the lowest
    speed afterwards (negative = the robot rolled back). Keys are None when the episode ended too early.
    """
    out = dict.fromkeys(SPRINT_KEYS)
    vx, x = arr[:, 0], arr[:, X_COL]
    n = len(arr)
    t = (np.arange(n) + 1) * dt
    run = np.isfinite(vx) & (t <= brake_t + 1e-9)
    if run.any():
        out["peak_speed_mps"] = float(vx[run].max())
        hit = np.flatnonzero(run & (vx >= 0.9 * v_cmd))
        out["time_to_90pct_s"] = float(t[hit[0]]) if hit.size else None
    kb = int(round(brake_t / dt))  # index of the first step that is driven by the stop command
    if kb < 1 or n <= kb:
        return out
    out["speed_at_brake_mps"] = float(vx[kb - 1]) if np.isfinite(vx[kb - 1]) else None
    after = vx[kb:]
    if np.isfinite(after).any():
        out["min_speed_after_brake_mps"] = float(np.nanmin(after))
    hold = max(1, int(round(STOP_HOLD_S / dt)))
    slow = np.isfinite(after) & (np.abs(after) < STOP_SPEED_MPS)
    if slow.size >= hold:
        csum = np.concatenate(([0], np.cumsum(slow)))
        starts = np.flatnonzero(csum[hold:] - csum[:-hold] == hold)  # first window of `hold` slow steps
        if starts.size:
            j = int(starts[0])
            out["stop_time_s"] = j * dt
            if np.isfinite(x[kb + j]) and np.isfinite(x[kb - 1]):
                out["stop_distance_m"] = float(x[kb + j] - x[kb - 1])
    return out


def _jsonable(column: np.ndarray, digits: int = 4) -> list:
    return [round(float(v), digits) if np.isfinite(v) else None for v in column]


def finalize_episode(
    rows: list, dt: float, record_dt: float, brake_time_s: float | None = None, v_cmd: float | None = None
) -> tuple[dict, dict]:
    """Turn the per-step rows of one finished episode into (time series for the report, metrics).

    The metrics are the trunk-contact metrics and, if brake_time_s is given (sprint with a sudden stop),
    the sprint/brake metrics.

    The series is down-sampled to record_dt: block means, except the contact force and the contact flag,
    which use the block maximum so that a short contact is never lost. The contact metrics are computed
    from the full-rate data.
    """
    arr = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(SERIES_CHANNELS))
    steps = len(arr)
    contact = arr[:, CONTACT_COL] > 0.5
    intervals = contact_intervals(contact, dt)
    force = arr[contact & np.isfinite(arr[:, FORCE_COL]), FORCE_COL]
    metrics = {
        "trunk_contact_events": len(intervals),
        "trunk_contact_time_s": float(contact.sum() * dt),
        "first_trunk_contact_s": round(intervals[0][0], 3) if intervals else None,
        "max_trunk_contact_force_n": float(force.max()) if force.size else None,
        "trunk_contact_intervals_s": [[round(a, 3), round(b, 3)] for a, b in intervals],
    }
    if brake_time_s is not None and v_cmd:
        metrics.update(sprint_metrics(arr, dt, brake_time_s, v_cmd))

    k = max(1, int(round(record_dt / dt)))
    blocks = -(-steps // k)
    if blocks * k > steps:
        arr = np.vstack([arr, np.full((blocks * k - steps, arr.shape[1]), np.nan)])
    grouped = arr.reshape(blocks, k, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # blocks without valid data
        reduced = np.nanmean(grouped, axis=1)
        biggest = np.nanmax(grouped, axis=1)
    reduced[:, [FORCE_COL, CONTACT_COL]] = biggest[:, [FORCE_COL, CONTACT_COL]]
    times = np.minimum((np.arange(blocks) + 1) * k, steps) * dt
    series = {"dt": round(k * dt, 4), "t": _jsonable(times, 3)}
    for name, column in zip(SERIES_CHANNELS, reduced.T):
        series[name] = _jsonable(column)
    return series, metrics


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
        # trunk contacts (None if the contact sensor was not found)
        "mean_trunk_contacts_per_episode": mean_of("trunk_contact_events"),
        "trunk_contact_episode_rate": (
            sum(e["trunk_contact_events"] > 0 for e in episodes if e.get("trunk_contact_events") is not None)
            / max(1, sum(e.get("trunk_contact_events") is not None for e in episodes))
            if any(e.get("trunk_contact_events") is not None for e in episodes) else None
        ),
        "mean_trunk_contact_time_s": mean_of("trunk_contact_time_s"),
        "worst_trunk_contact_force_n": worst_of("max_trunk_contact_force_n"),
        # sprint with a sudden stop (None for the other scenarios)
        "mean_peak_speed_mps": mean_of("peak_speed_mps"),
        "mean_time_to_90pct_s": mean_of("time_to_90pct_s"),
        "mean_speed_at_brake_mps": mean_of("speed_at_brake_mps"),
        "mean_stop_time_s": mean_of("stop_time_s"),
        "mean_stop_distance_m": mean_of("stop_distance_m"),
        "lowest_speed_after_brake_mps": (
            min(e["min_speed_after_brake_mps"] for e in episodes if e.get("min_speed_after_brake_mps") is not None)
            if any(e.get("min_speed_after_brake_mps") is not None for e in episodes) else None
        ),
        "stopped_rate": (
            sum(e.get("stop_time_s") is not None for e in episodes) / n
            if any("peak_speed_mps" in e for e in episodes) else None
        ),
        # training-like
        "mean_total_reward": mean_of("total_reward"),
        "reward_terms_per_s": {k: _mean(e.get("reward_terms_per_s", {}).get(k) for e in episodes) for k in term_keys},
        "end_reasons": dict(Counter(e.get("end_reason", "unknown") for e in episodes)),
    }


def _fp(value, meta: dict) -> str:
    """Progress with its unit: metres along the course, or degrees for a turn on the spot."""
    if value is None:
        return "-"
    return f"{value:.0f} deg" if (meta or {}).get("progress_unit") == "deg" else f"{value:.2f} m"


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
        has_milestone = (r.get("meta") or {}).get("midpoint_x", 0.0) is not None  # flat ground / sprint: no milestone
        mid_txt = f"{s['midpoint_rate']:.0%}" if has_milestone else "-"
        print(
            f"{name:<{w}}{s['episodes']:>9}{s['success_rate']:>9.0%}{mid_txt:>10}"
            f"{s['fall_rate']:>7.0%}{s['timeout_rate']:>10.0%}{_f(s['mean_time_to_goal_s'], '.1f'):>11}"
            f"{_fp(s['mean_progress_m'], r.get('meta')):>13}{_f(s['worst_max_yaw_err_deg'], '.1f'):>14}"
            f"{_f(s['worst_max_lateral_m'], '.2f'):>12}"
        )
    print(line)
    print("yaw err / lateral: worst case over episodes (turn on the spot: lateral = drift from the start point)")

    # --- table 2: motion quality ---------------------------------------------------------------
    line = "=" * (w + 123)
    print("\n" + line)
    print(
        f"{'scenario':<{w}}{'vel err xy':>11}{'vel err yaw':>12}{'speed':>8}{'power, W':>10}{'CoT':>7}"
        f"{'act. rate':>11}{'joint acc':>11}{'tilt max':>10}{'reward':>9}{'contacts/ep':>13}{'w/ contact':>11}"
    )
    print("-" * (w + 123))
    for name, r in results.items():
        s = r["summary"]
        print(
            f"{name:<{w}}{_f(s['mean_vel_err_xy_mps'], '.3f'):>11}{_f(s['mean_vel_err_yaw_radps'], '.3f'):>12}"
            f"{_f(s['mean_forward_speed_mps'], '.2f'):>8}{_f(s['mean_power_w'], '.0f'):>10}"
            f"{_f(s['mean_cost_of_transport'], '.2f'):>7}{_f(s['mean_action_rate'], '.2f'):>11}"
            f"{_f(s['mean_joint_acc'], '.0f'):>11}{_f(s['worst_max_tilt_deg'], '.0f'):>10}"
            f"{_f(s['mean_total_reward'], '.1f'):>9}{_f(s['mean_trunk_contacts_per_episode'], '.2f'):>13}"
            f"{_f(s['trunk_contact_episode_rate'], '.0%'):>11}"
        )
    print(line)
    print("vel err: m/s and rad/s | speed: along the commanded direction (m/s) | tilt max: deg, worst case | reward: mean sum per episode")
    print("contacts/ep: trunk contact events per episode | w/ contact: share of episodes with a trunk contact")

    # --- sprint with a sudden stop ----------------------------------------------------------------
    sprint = {k: r for k, r in results.items() if r["summary"].get("mean_peak_speed_mps") is not None}
    if sprint:
        line = "=" * (w + 87)
        print("\n" + line)
        print(
            f"{'scenario (sprint + stop)':<{w}}{'peak, m/s':>11}{'t to 90%, s':>13}{'at brake':>10}{'stop time':>11}"
            f"{'stop dist':>11}{'lowest':>9}{'stopped':>9}"
        )
        print("-" * (w + 87))
        for name, r in sprint.items():
            s = r["summary"]
            print(
                f"{name:<{w}}{_f(s['mean_peak_speed_mps'], '.2f'):>11}{_f(s['mean_time_to_90pct_s'], '.1f'):>13}"
                f"{_f(s['mean_speed_at_brake_mps'], '.2f'):>10}{_f(s['mean_stop_time_s'], '.2f'):>11}"
                f"{_f(s['mean_stop_distance_m'], '.2f'):>11}{_f(s['lowest_speed_after_brake_mps'], '.2f'):>9}"
                f"{_f(s['stopped_rate'], '.0%'):>9}"
            )
        print(line)
        print("speeds in m/s | stop time: s from the stop command until |v| < 0.1 m/s for 0.5 s | stop dist: m | lowest: min speed after the command (negative = rolled back)")

    # --- end reasons -----------------------------------------------------------------------------
    print("\nEpisode end reasons:")
    for name, r in results.items():
        reasons = ", ".join(f"{k} x{v}" for k, v in sorted(r["summary"]["end_reasons"].items()))
        print(f"  {name:<{w}}{reasons}")
    print(flush=True)


def _drop_args(argv: list[str], flags: set[str]) -> list[str]:
    """Remove --flag value and --flag=value pairs from an argv list."""
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


def _dump_json(obj, path: str):
    """Write indented JSON, but keep every list of plain numbers on one line (the time series are long)."""
    text = json.dumps(obj, indent=2)
    text = re.sub(r'\[\s+([^\[\]{}"]*?)\s+\]', lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)
    with open(path, "w") as f:
        f.write(text)


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
parser.add_argument(
    "--speed",
    type=float,
    default=None,
    help="Commanded forward speed in m/s (default: 0.5, the value built into the scenarios).",
)
parser.add_argument(
    "--no_contact_termination",
    action="store_true",
    default=False,
    help="Do not end the episode on trunk contact, so that all trunk contacts are recorded.",
)
parser.add_argument(
    "--record_dt", type=float, default=0.1, help="Time step (s) of the time series stored in the report."
)
parser.add_argument(
    "--no_report", action="store_true", default=False, help="Do not open the report viewer when the runs are done."
)
parser.add_argument(
    "--report_save", type=str, default=None, help="Also save the report figures as PNG files into this folder."
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time (useful with the GUI).")
parser.add_argument(
    "--gui",
    action="store_true",
    default=False,
    help="Open a window to configure the run. All other arguments are then set in the window.",
)
parser.add_argument("--list_scenarios_to", type=str, default=None, help=argparse.SUPPRESS)  # internal
parser.add_argument("--internal_child", action="store_true", default=False, help=argparse.SUPPRESS)  # internal
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _check_args(args):
    if args.speed is not None and args.speed <= 0.0:
        parser.error("--speed must be a positive number (m/s).")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if args.record_dt <= 0.0:
        parser.error("--record_dt must be positive.")


_check_args(args_cli)


def _catalog_cmd(path: str) -> list[str]:
    """Command that writes the scenario/task catalog to path (starts a short headless Isaac Sim)."""
    return [sys.executable, os.path.abspath(__file__), "--headless", "--list_scenarios_to", path]


if args_cli.gui:
    if len(sys.argv) > 2:
        print("[WARN] With --gui the other command-line arguments are ignored: set them in the window.")
    try:
        import eval_gui  # tkinter only, no Isaac Sim imports

        _gui_argv = eval_gui.run_gui(_catalog_cmd)
    except ImportError as e:
        sys.exit(f"[ERROR] The GUI needs tkinter ({e}). Install it (conda install tk) or run without --gui.")
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")
    if _gui_argv is None:
        print("[INFO] Cancelled.")
        sys.exit(0)
    # from here on everything works exactly as if these arguments had been typed in the terminal
    sys.argv = [sys.argv[0], *_gui_argv]
    args_cli = parser.parse_args()
    _check_args(args_cli)
    print("[INFO] Equivalent command:\n  python " + shlex.join([sys.argv[0], *_gui_argv]), flush=True)

##
# Parent mode: one child process per (scenario, seed) pair. Isaac Sim is NOT launched in this process.
##


def _child_cmd(extra: list[str]) -> list[str]:
    base = _drop_args(sys.argv[1:], {"--scenarios", "--output", "--list_scenarios_to", "--eval_seed"})
    return [sys.executable, os.path.abspath(__file__), *base, "--internal_child", *extra]


def _open_report(report_path: str, args) -> None:
    """Open the report viewer on the saved report (blocks until its windows are closed)."""
    viewer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.py")
    cmd = [sys.executable, viewer, report_path]
    if args.report_save:
        cmd += ["--save", args.report_save]
    print("[INFO] Opening the report viewer (close its windows to finish) ...", flush=True)
    if subprocess.run(cmd).returncode != 0:
        print(
            "[WARN] The report viewer failed (is matplotlib installed?). The report is saved, view it later with:\n"
            f"  python {viewer} {report_path}",
            flush=True,
        )


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
            data = json.load(f)
        registry = list(data["scenarios"]) if isinstance(data, dict) else list(data)
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
            slot = pooled.setdefault(
                name, {"description": entry["description"], "meta": entry.get("meta"), "seeds": [], "episodes": []}
            )
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
    _dump_json(
        {**meta, "seeds": [int(s) for s in seeds], "speed_override_mps": args.speed, "date": stamp, "scenarios": results},
        out,
    )
    print(f"[INFO] Report saved to: {out}")
    if failed:
        print(f"[WARN] {failed} run(s) failed, their episodes are missing from the report.")
    if not args.no_report:
        _open_report(out, args)
    return 0 if not failed else 1


if not args_cli.list_scenarios_to and not args_cli.internal_child:
    sys.exit(_run_children(args_cli))

##
# Child mode: one scenario, one seed, one Isaac Sim instance (started by the parent above)
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
from unitree_rl_lab.tasks.locomotion.robots.go2.eval_scenarios import (
    EVAL_SCENARIOS,
    apply_eval_scenario,
    find_trunk_contact_terms,
    scenario_meta,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

_UNSET = -1.0e9
_GRAVITY = 9.81


def _setup_trunk_contact(scene, terms: list[dict]) -> dict | None:
    """Resolve the definition of "trunk contact": the sensor, body ids and force threshold of the training
    termination (e.g. base_contact). Falls back to the "base" body of the "contact_forces" sensor."""
    spec = terms[0] if terms else {"sensor": "contact_forces", "body_names": "base", "threshold": 1.0}
    if spec["sensor"] not in scene.sensors:
        return None
    sensor = scene.sensors[spec["sensor"]]
    ids, names = sensor.find_bodies(spec["body_names"] or "base")
    if len(ids) == 0:
        return None
    return {"sensor": sensor, "name": spec["sensor"], "ids": list(ids), "bodies": list(names), "threshold": spec["threshold"]}


def _trunk_force(contact: dict) -> torch.Tensor:
    """Largest trunk contact force (N) over the last physics steps, same measure as the training termination."""
    data = contact["sensor"].data
    history = data.net_forces_w_history
    if history is None:
        history = data.net_forces_w.unsqueeze(1)
    return torch.norm(history[:, :, contact["ids"]], dim=-1).amax(dim=(1, 2))


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
    contact_terms = find_trunk_contact_terms(env_cfg)  # read before they may be removed below
    sc = apply_eval_scenario(
        env_cfg, name, speed=args_cli.speed, contact_termination=not args_cli.no_contact_termination
    )
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
    dir_x, dir_y = sc.course_dir  # world direction of travel: progress = position . dir, lateral = position x dir
    spawn_px, spawn_py = sc.spawn_x * dir_x, sc.spawn_x * dir_y
    is_turn = sc.motion == "turn"
    rot_sign = -1.0 if sc.wz < 0.0 else 1.0

    reward_names = list(rew_mgr.active_terms)
    done_names = list(term_mgr.active_terms)
    contact = _setup_trunk_contact(unwrapped.scene, contact_terms)
    contact_cols = [done_names.index(t["term"]) for t in contact_terms if t["term"] in done_names]
    rec_every_s = args_cli.record_dt
    rows: list[list] = [[] for _ in range(n)]  # per-step rows of the running episodes (see SERIES_CHANNELS)
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
    yaw_total = zeros()  # turn on the spot: total rotation since the start of the episode (rad)
    prev_heading = torch.full((n,), sc.heading, device=dev)
    sums = {k: zeros() for k in ("vel_err_xy", "vel_err_yaw", "fwd_speed", "power", "action_rate", "joint_acc", "tilt")}
    total_reward = zeros()
    term_sums = torch.zeros(n, max(len(reward_names), 1), device=dev)
    prev_actions = None
    episodes: list[dict] = []

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # sanity check of the spawn pose (x relative to tile centre, z in world frame)
    dx0 = (robot.data.root_pos_w[0, 0] - origin_x[0]).item()
    dy0 = (robot.data.root_pos_w[0, 1] - origin_y[0]).item()
    x0 = dx0 * dir_x + dy0 * dir_y
    z0 = robot.data.root_pos_w[0, 2].item()
    goal_txt = "none (fixed duration)" if sc.goal_x is None else (f"{sc.goal_x:.0f} deg" if is_turn else f"{sc.goal_x:.2f} m")
    if sc.motion == "sprint_brake":
        command_txt = f"0 -> {sc.vx:.2f} m/s (maximum of the training range), stop command at {sc.brake_time_s:.0f} s"
    elif is_turn:
        command_txt = f"{sc.wz:.2f} rad/s ({sc.speed_label})"
    else:
        command_txt = f"{sc.primary_command:.2f} m/s ({sc.speed_label})"
    print(f"[INFO] [{name}] seed {seed}: {sc.description}")
    print(
        f"[INFO] [{name}] spawn {x0:.2f} {sc.progress_unit} along the course (rel. to tile centre), "
        f"z={z0:.2f} m (world); goal {goal_txt}"
    )
    print(
        f"[INFO] [{name}] command: {command_txt}; running {args_cli.episodes} episodes, "
        f"up to {sc.episode_length_s:.0f} s each",
        flush=True,
    )
    if contact is None:
        print(f"[WARN] [{name}] trunk contact sensor not found: trunk contact metrics are disabled.", flush=True)
    else:
        print(
            f"[INFO] [{name}] trunk contact = sensor '{contact['name']}', bodies {contact['bodies']}, "
            f"> {contact['threshold']} N; "
            + ("a contact ENDS the episode (as in training)" if contact_cols else "a contact does NOT end the episode"),
            flush=True,
        )

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

        done = dones.bool()
        timeout = extras["time_outs"].bool()
        alive = ~done

        # --- state after the step ---
        px = robot.data.root_pos_w[:, 0] - origin_x  # position relative to the tile centre, world frame
        py = robot.data.root_pos_w[:, 1] - origin_y
        heading = robot.data.heading_w
        if is_turn:
            # progress = total rotation in the commanded direction (deg), lateral = drift from the start point
            yaw_total = torch.where(alive, yaw_total + wrap_to_pi(heading - prev_heading), yaw_total)
            prev_heading = torch.where(alive, heading, prev_heading)
            x = torch.rad2deg(yaw_total) * rot_sign
            y = ((px - spawn_px) ** 2 + (py - spawn_py) ** 2) ** 0.5
            yaw_err = torch.full_like(px, float("nan"))  # the heading is not held: no heading error
        else:
            x = px * dir_x + py * dir_y  # progress along the course
            y = py * dir_x - px * dir_y  # lateral deviation (left of the course is positive)
            yaw_err = wrap_to_pi(heading - sc.heading).abs()
        lin_b, ang_b = robot.data.root_lin_vel_b, robot.data.root_ang_vel_b
        vel_err_xy = torch.norm(cmd[:, :2] - lin_b[:, :2], dim=-1)
        vel_err_yaw = (cmd[:, 2] - ang_b[:, 2]).abs()
        if sc.primary_channel == "vx":  # speed along the commanded direction (backward walk: positive when walking back)
            fwd_speed = lin_b[:, 0] * sc.primary_sign
        elif sc.primary_channel == "vy":
            fwd_speed = lin_b[:, 1] * sc.primary_sign
        else:
            fwd_speed = torch.zeros_like(px)
        power = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(dim=-1)
        action_rate = torch.norm(actions - prev_actions, dim=-1)
        joint_acc = torch.norm(robot.data.joint_acc, dim=-1)
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))

        # flags of the termination terms of THIS step (they survive the automatic reset)
        goal_hit = term_mgr.get_term("goal_reached").bool() if "goal_reached" in done_names else torch.zeros_like(done)
        flags = term_contact = None
        if done.any():
            flags = torch.stack([term_mgr.get_term(nm).bool() for nm in done_names], dim=1)
            term_contact = flags[:, contact_cols].any(dim=1) if contact_cols else torch.zeros_like(done)
        zero = torch.zeros_like(x)

        # rewards of the terminating step belong to the ending episode (they are computed before the reset)
        total_reward = total_reward + rew
        if reward_names:
            term_sums = term_sums + step_reward_terms() * dt

        # kinematic quantities after a reset belong to the next episode, so only use alive envs
        max_x = torch.where(alive, torch.maximum(max_x, x), max_x)
        if not is_turn:
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

        # one row per env and step for the time series. After a reset the state belongs to the next
        # episode, so a terminating step only keeps the command and the contact flag of its termination.
        force = _trunk_force(contact) if contact is not None else torch.zeros_like(x)
        rec = torch.stack(
            [lin_b[:, 0], lin_b[:, 1], ang_b[:, 2], cmd[:, 0], cmd[:, 1], cmd[:, 2], x, y,
             torch.rad2deg(yaw_err), torch.rad2deg(tilt), power, force],
            dim=1,
        ).float().cpu().numpy()  # fmt: skip
        done_np = done.cpu().numpy()
        term_contact_np = term_contact.cpu().numpy() if term_contact is not None else None
        for i in range(n):
            row = np.full(len(SERIES_CHANNELS), np.nan)
            if done_np[i]:
                row[3:6] = rec[i, 3:6]
                row[CONTACT_COL] = float(term_contact_np[i]) if term_contact_np is not None else 0.0
            else:
                row[:FORCE_COL + 1] = rec[i]
                row[CONTACT_COL] = float(contact is not None and rec[i, FORCE_COL] > contact["threshold"])
            rows[i].append(row)

        if done.any():
            for i in done.nonzero().flatten().tolist():
                series, ep_metrics = finalize_episode(
                    rows[i], dt, rec_every_s,
                    brake_time_s=sc.brake_time_s if sc.motion == "sprint_brake" else None, v_cmd=sc.vx,
                )  # fmt: skip
                rows[i] = []
                if contact is None:
                    ep_metrics.update({key: None for key in CONTACT_KEYS})
                # success: the goal line / the full turn, or (sprint) surviving the episode AND coming to a stop
                survived_and_stopped = sc.success_on_timeout and bool(timeout[i]) and ep_metrics.get("stop_time_s") is not None
                success = bool(goal_hit[i]) or survived_and_stopped
                best_x = max_x[i].item() if max_x[i].item() > _UNSET / 2 else sc.spawn_x
                if success and sc.goal_x is not None:  # the pose of the terminating step is already reset
                    best_x = max(best_x, sc.goal_x)
                progress = best_x - sc.spawn_x
                duration = float(steps[i] * dt)
                k = max(float(alive_steps[i]), 1.0)
                mean_power = float(sums["power"][i]) / k
                energy = mean_power * k * dt
                cot = None
                if mass is not None and not is_turn and progress > 0.5:
                    cot = energy / (float(mass[i]) * _GRAVITY * progress)
                reasons = [nm for j, nm in enumerate(done_names) if bool(flags[i, j])]
                ep = {
                    "seed": seed,
                    "success": success,
                    "fell_before_goal": bool((not timeout[i]) and (not success)),
                    "reached_midpoint": success or (bool(reached_mid[i]) if sc.midpoint_x is not None else False),
                    "time_to_goal_s": duration if (success and not sc.success_on_timeout) else None,
                    "duration_s": duration,
                    "progress_m": progress,  # metres along the course; degrees for the turn on the spot (meta.progress_unit)
                    "max_yaw_err_deg": None if is_turn else float(torch.rad2deg(max_yaw_err[i])),
                    "max_lateral_m": float(max_lat[i]),
                    "vel_err_xy_mps": float(sums["vel_err_xy"][i]) / k,
                    "vel_err_yaw_radps": float(sums["vel_err_yaw"][i]) / k,
                    "forward_speed_mps": None if is_turn else float(sums["fwd_speed"][i]) / k,
                    "power_w": mean_power,
                    "cost_of_transport": cot,
                    "action_rate": float(sums["action_rate"][i]) / k,
                    "joint_acc": float(sums["joint_acc"][i]) / k,
                    "mean_tilt_deg": float(torch.rad2deg(sums["tilt"][i] / k)),
                    "max_tilt_deg": float(torch.rad2deg(max_tilt[i])),
                    "total_reward": float(total_reward[i]),
                    "reward_terms_per_s": {nm: float(term_sums[i, j]) / duration for j, nm in enumerate(reward_names)},
                    "end_reason": "+".join(sorted(reasons)) if reasons else ("time_out" if timeout[i] else "unknown"),
                    "command_speed_mps": None if is_turn else sc.primary_command,
                    **ep_metrics,
                    "series": series,
                }
                episodes.append(ep)
                verdict = "SUCCESS" if success else ("FELL" if ep["fell_before_goal"] else "TIMEOUT")
                mid = f" {sc.midpoint_label}={'yes' if ep['reached_midpoint'] else 'no'}" if sc.midpoint_x is not None else ""
                prog_txt = f"{progress:.0f}deg" if is_turn else f"{progress:.2f}m"
                yaw_txt = "n/a" if is_turn else f"{ep['max_yaw_err_deg']:.1f}deg"
                lat_txt = "drift" if is_turn else "lateral"
                sprint_txt = ""
                if ep.get("peak_speed_mps") is not None:
                    stop = ep.get("stop_time_s")
                    sprint_txt = (
                        f" peak={ep['peak_speed_mps']:.2f}m/s stop_time="
                        + (f"{stop:.2f}s" if stop is not None else "not stopped")
                    )
                print(
                    f"[{name}] episode {len(episodes)}/{args_cli.episodes}: {verdict:<8} "
                    f"time={duration:.1f}s progress={prog_txt}{mid}{sprint_txt} yaw_err={yaw_txt} "
                    f"{lat_txt}={ep['max_lateral_m']:.2f}m vel_err={ep['vel_err_xy_mps']:.2f}m/s "
                    f"trunk_contacts={ep['trunk_contact_events'] if ep['trunk_contact_events'] is not None else '-'} "
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
        yaw_total[done] = 0.0
        prev_heading[done] = sc.heading
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
    meta = {
        **scenario_meta(sc),
        "control_dt_s": dt,
        "record_dt_s": episodes[0]["series"]["dt"],
        "trunk_contact": None
        if contact is None
        else {
            "sensor": contact["name"],
            "bodies": contact["bodies"],
            "threshold_n": contact["threshold"],
            "terminates_episode": bool(contact_cols),
            "merge_gap_s": CONTACT_MERGE_GAP_S,
        },
    }
    return {"description": sc.description, "meta": meta, "summary": summarize(episodes), "episodes": episodes}


def main():
    if args_cli.list_scenarios_to:
        tasks = sorted(spec.id for spec in gym.registry.values() if spec.kwargs.get("eval_env_cfg_entry_point"))
        catalog = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tasks": tasks,
            "scenarios": {k: v.description for k, v in EVAL_SCENARIOS.items()},
        }
        with open(args_cli.list_scenarios_to, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=1)
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
        json.dump(report, f)
    print(f"[INFO] Run finished: {name}, seed {seed}.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()