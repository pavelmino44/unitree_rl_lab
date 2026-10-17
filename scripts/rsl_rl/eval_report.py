"""Viewer for the JSON reports written by evaluate.py.

It is started automatically when evaluate.py finishes (use --no_report there to skip it), and it can be
started on its own on any saved report:

    python scripts/rsl_rl/eval_report.py path/to/report.json
    python scripts/rsl_rl/eval_report.py report.json --scenarios "stairs_up_1*,rough_boxes*"
    python scripts/rsl_rl/eval_report.py report.json --save plots/ --no_show

Windows (matplotlib):
  * one window for the whole run: outcomes per scenario, success vs. stair height, speed tracking, trunk
    contacts and a numeric summary of the controller with its weakest scenarios;
  * one window per scenario: averages over all its episodes, aligned by position along the course;
  * one window per episode: forward speed vs. time with the commanded speed, the trunk-contact timeline,
    progress, heading error / tilt / lateral deviation and all numeric metrics of that episode.

This script does not need Isaac Sim, only numpy and matplotlib. Without a display it saves PNG files instead.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import warnings

import numpy as np

# --- look -------------------------------------------------------------------------------------------
BG = "#f3f7fc"
C_SPEED = "#2c6db4"
C_CMD = "#111111"
C_CONTACT = "#d64545"
C_OK = "#2e9e5b"
C_FALL = "#d64545"
C_TIMEOUT = "#e0a030"
C_OBSTACLE = "#8fa8c8"
C_YAW = "#e08a1e"
C_TILT = "#7a4fb0"
C_LAT = "#1d9a9a"
C_GRAY = "#6b7a8c"
OUTCOME_COLORS = {"SUCCESS": C_OK, "FELL": C_FALL, "TIMEOUT": C_TIMEOUT}


##
# Data helpers
##


def num(values) -> np.ndarray:
    """JSON list (with nulls) -> float array (null -> nan)."""
    return np.asarray(values if values is not None else [], dtype=float)


def fmt(value, spec: str = ".2f", unit: str = "", none: str = "-") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return none
    return f"{format(value, spec)}{(' ' + unit) if unit else ''}"


def pct(value, none: str = "-") -> str:
    return fmt(None if value is None else 100.0 * value, ".0f", "%", none)


def mean_of(episodes: list[dict], key: str):
    vals = [e[key] for e in episodes if e.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def worst_of(episodes: list[dict], key: str):
    vals = [e[key] for e in episodes if e.get(key) is not None]
    return float(np.max(vals)) if vals else None


def outcome(ep: dict) -> str:
    if ep.get("success"):
        return "SUCCESS"
    return "FELL" if ep.get("fell_before_goal") else "TIMEOUT"


def forward_fill(x: np.ndarray) -> np.ndarray:
    """Replace nan by the last valid value (leading nans stay)."""
    if x.size == 0:
        return x
    idx = np.where(np.isfinite(x), np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def get_meta(sc: dict, episodes: list[dict]) -> dict:
    """Scenario meta with sane defaults for reports that were written without it."""
    meta = dict(sc.get("meta") or {})
    meta.setdefault("command_speed_mps", mean_of(episodes, "command_speed_mps") or 0.5)
    meta.setdefault("spawn_x", None)
    meta.setdefault("goal_x", None)
    meta.setdefault("midpoint_x", None)
    meta.setdefault("midpoint_label", "top")
    meta.setdefault("obstacle_half_width", None)
    return meta


def contact_threshold(meta: dict):
    tc = meta.get("trunk_contact")
    return None if not tc else tc.get("threshold_n")


##
# Drawing helpers
##


def shade_obstacle_time(ax, t: np.ndarray, x: np.ndarray, half, dt: float):
    """Shade the time spans in which the robot is on the obstacle (|x| <= half)."""
    if half is None or t.size == 0:
        return
    on = np.isfinite(x) & (np.abs(x) <= half)
    for k, (s, e) in enumerate(runs(on)):
        ax.axvspan(t[s] - dt, t[e - 1], color=C_OBSTACLE, alpha=0.16, lw=0, label="on the obstacle" if k == 0 else None)


def style_axes(ax, title: str, ylabel: str = "", xlabel: str = ""):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#1f4e8c")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


def set_window_title(fig, title: str):
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


def text_panel(ax, lines: list[str], size: float = 8.4):
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=size, transform=ax.transAxes)


def kv(label: str, value: str, width: int = 21) -> str:
    return f"{label:<{width}}{value}"


_kv = kv


def pad_top(ax, fraction: float):
    """Leave empty room above the data (for a legend)."""
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * fraction)


##
# Episode window
##


def episode_text(name: str, meta: dict, ep: dict, k: int, n: int) -> list[str]:
    verdict = outcome(ep)
    cmd = meta.get("command_speed_mps")
    speed = ep.get("forward_speed_mps")
    ratio = None if not cmd or speed is None else speed / cmd
    lines = [
        f"{name}",
        f"seed {ep.get('seed', '?')}   episode {k}/{n}",
        "",
        kv("RESULT", f"{verdict}"),
        kv("end reason", str(ep.get("end_reason", "-"))),
        kv("duration", fmt(ep.get("duration_s"), ".1f", "s")),
        kv("time to goal", fmt(ep.get("time_to_goal_s"), ".1f", "s")),
        kv("progress", fmt(ep.get("progress_m"), ".2f", "m")),
        kv(f"reached {meta.get('midpoint_label', 'top')}", "yes" if ep.get("reached_midpoint") else "no")
        if meta.get("midpoint_x") is not None
        else kv("midpoint", "n/a"),
        "",
        "SPEED TRACKING",
        kv("commanded speed", fmt(cmd, ".2f", "m/s")),
        kv("mean forward speed", f"{fmt(speed, '.2f', 'm/s')}  ({pct(ratio)})"),
        kv("vel error xy", fmt(ep.get("vel_err_xy_mps"), ".3f", "m/s")),
        kv("vel error yaw", fmt(ep.get("vel_err_yaw_radps"), ".3f", "rad/s")),
        "",
        "PATH",
        kv("max heading error", fmt(ep.get("max_yaw_err_deg"), ".1f", "deg")),
        kv("max lateral dev.", fmt(ep.get("max_lateral_m"), ".2f", "m")),
        "",
        "TRUNK CONTACTS",
        kv("events", fmt(ep.get("trunk_contact_events"), ".0f")),
        kv("total contact time", fmt(ep.get("trunk_contact_time_s"), ".2f", "s")),
        kv("first contact at", fmt(ep.get("first_trunk_contact_s"), ".2f", "s")),
        kv("max force", fmt(ep.get("max_trunk_contact_force_n"), ".0f", "N")),
        "",
        "STABILITY",
        kv("tilt mean / max", f"{fmt(ep.get('mean_tilt_deg'), '.1f')} / {fmt(ep.get('max_tilt_deg'), '.1f')} deg"),
        "",
        "EFFORT / SMOOTHNESS",
        kv("mean power", fmt(ep.get("power_w"), ".0f", "W")),
        kv("cost of transport", fmt(ep.get("cost_of_transport"), ".2f")),
        kv("action rate", fmt(ep.get("action_rate"), ".2f")),
        kv("joint acceleration", fmt(ep.get("joint_acc"), ".0f")),
        "",
        "REWARD",
        kv("total", fmt(ep.get("total_reward"), ".1f")),
    ]
    terms = sorted((ep.get("reward_terms_per_s") or {}).items(), key=lambda kv_: -abs(kv_[1]))[:5]
    for term, value in terms:
        lines.append(f"  {term[:26]:<26}{value:+.3f}/s")
    return lines


def episode_figure(plt, name: str, meta: dict, ep: dict, k: int, n: int):
    verdict = outcome(ep)
    series = ep.get("series") or {}
    t = num(series.get("t"))
    dt = float(series.get("dt") or 0.1)
    cmd_speed = meta.get("command_speed_mps")
    half = meta.get("obstacle_half_width")

    fig = plt.figure(figsize=(15.6, 9.0), facecolor=BG)
    gs = fig.add_gridspec(
        4, 2, width_ratios=[3.3, 1.3], height_ratios=[2.3, 1.15, 1.6, 1.6],
        left=0.055, right=0.985, top=0.925, bottom=0.07, hspace=0.42, wspace=0.07,
    )  # fmt: skip
    ax_v = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_p = fig.add_subplot(gs[2, 0], sharex=ax_v)
    ax_h = fig.add_subplot(gs[3, 0], sharex=ax_v)
    ax_t = fig.add_subplot(gs[:, 1])
    fig.suptitle(
        f"{name}   |   seed {ep.get('seed', '?')}   |   episode {k}/{n}   |   {verdict}",
        fontsize=13, fontweight="bold", color=OUTCOME_COLORS[verdict], x=0.055, ha="left",
    )  # fmt: skip
    set_window_title(fig, f"{name} - seed {ep.get('seed', '?')} - episode {k}/{n} - {verdict}")
    text_panel(ax_t, episode_text(name, meta, ep, k, n))

    if t.size == 0:
        for ax in (ax_v, ax_c, ax_p, ax_h):
            ax.text(0.5, 0.5, "this report has no time series", ha="center", va="center", transform=ax.transAxes)
        return fig

    x = num(series.get("x"))
    xf = forward_fill(x)
    vx, cmd_vx = num(series.get("vx")), num(series.get("cmd_vx"))
    force, contact = num(series.get("contact_force_n")), num(series.get("contact"))
    t_end = float(t[-1])

    # --- speed ---
    shade_obstacle_time(ax_v, t, xf, half, dt)
    ax_v.plot(t, vx, color=C_SPEED, lw=1.6, label="measured forward speed")
    ax_v.plot(t, cmd_vx, color=C_CMD, lw=1.4, ls="--", label=f"commanded speed ({fmt(cmd_speed, '.2f')} m/s)")
    mean_speed = np.nanmean(vx) if np.isfinite(vx).any() else None
    if mean_speed is not None:
        ax_v.axhline(mean_speed, color=C_SPEED, lw=1.0, ls=":", label=f"mean {mean_speed:.2f} m/s")
    style_axes(ax_v, "Forward speed", "speed, m/s")
    top = max(1.35 * (cmd_speed or 0.0), 1.1 * (np.nanmax(vx) if np.isfinite(vx).any() else 0.0), 0.1)
    ax_v.set_ylim(min(0.0, np.nanmin(vx) if np.isfinite(vx).any() else 0.0), top)
    ax_v.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    # --- trunk contacts ---
    intervals = ep.get("trunk_contact_intervals_s") or []
    thr = contact_threshold(meta)
    min_w = max(0.012 * t_end, 0.05)  # a 0.02 s contact must stay visible on a 30 s axis
    for j, (a, b) in enumerate(intervals, 1):
        b_vis = max(b, a + min_w)
        ax_c.axvspan(a, b_vis, color=C_CONTACT, alpha=0.6, lw=0)
        if len(intervals) <= 25:
            ax_c.text(0.5 * (a + b_vis), 0.97, str(j), transform=ax_c.get_xaxis_transform(), ha="center", va="top",
                      fontsize=7.5, fontweight="bold", color="#8e1f1f")  # fmt: skip
    if np.isfinite(force).any():
        ax_c.plot(t, np.nan_to_num(force, nan=0.0), color="#8e1f1f", lw=1.1, drawstyle="steps-mid")
    if thr:
        ax_c.axhline(thr, color=C_GRAY, lw=0.9, ls="--")
    style_axes(ax_c, "Trunk contacts (red = trunk touches the surface)", "force, N")
    events = ep.get("trunk_contact_events")
    if events is None:
        msg, colour = "trunk contact sensor not available", C_GRAY
    elif events == 0:
        msg, colour = "no trunk contact", C_OK
    else:
        msg = f"{events} event(s), {fmt(ep.get('trunk_contact_time_s'), '.2f', 's')} in contact"
        colour = C_CONTACT
    ax_c.text(0.005, 0.93, msg, transform=ax_c.transAxes, ha="left", va="top", color=colour, fontsize=9, fontweight="bold")
    ax_c.set_ylim(bottom=0)
    ymax = np.nanmax(force) if np.isfinite(force).any() else 0.0
    ax_c.set_ylim(0, max(ymax * 1.15, (thr or 1.0) * 2.0))
    if contact.size and not intervals and events:
        pass  # intervals missing in an old report

    # --- progress ---
    if half is not None:
        ax_p.axhspan(-half, half, color=C_OBSTACLE, alpha=0.16, lw=0, label="obstacle")
    ax_p.plot(t, x, color="#333333", lw=1.4, label="position along the course")
    if meta.get("spawn_x") is not None:
        ax_p.axhline(meta["spawn_x"], color=C_GRAY, lw=0.9, ls=":", label="start")
    if meta.get("midpoint_x") is not None:
        ax_p.axhline(meta["midpoint_x"], color=C_SPEED, lw=0.9, ls="--", label=meta.get("midpoint_label", "midpoint"))
    if meta.get("goal_x") is not None:
        ax_p.axhline(meta["goal_x"], color=C_OK, lw=1.0, ls="--", label="goal")
    style_axes(ax_p, "Progress", "x, m")
    ax_p.legend(loc="upper left", fontsize=7.5, ncol=5, framealpha=0.9)

    # --- heading / tilt / lateral deviation ---
    ax_h.plot(t, num(series.get("yaw_err_deg")), color=C_YAW, lw=1.2, label="heading error, deg")
    ax_h.plot(t, num(series.get("tilt_deg")), color=C_TILT, lw=1.2, label="tilt, deg")
    style_axes(ax_h, "Heading error, tilt and lateral deviation", "deg", "time since episode start, s")
    pad_top(ax_h, 0.35)
    ax_l = ax_h.twinx()
    ax_l.plot(t, num(series.get("y")), color=C_LAT, lw=1.2, label="lateral deviation, m")
    ax_l.set_ylabel("lateral, m", color=C_LAT)
    ax_l.grid(False)
    pad_top(ax_l, 0.35)
    h1, l1 = ax_h.get_legend_handles_labels()
    h2, l2 = ax_l.get_legend_handles_labels()
    ax_h.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3, framealpha=0.9)

    # --- markers shared by the time plots ---
    marks = []
    if ep.get("time_to_goal_s") is not None:
        marks.append((ep["time_to_goal_s"], C_OK, "-", "goal"))
    if meta.get("midpoint_x") is not None and np.isfinite(x).any():
        hit = np.flatnonzero(np.isfinite(x) & (x >= meta["midpoint_x"]))
        if hit.size:
            marks.append((float(t[hit[0]]), C_SPEED, "--", meta.get("midpoint_label", "midpoint")))
    if verdict == "FELL":
        marks.append((t_end, C_FALL, "-", f"ended: {ep.get('end_reason', '?')}"))
    for axis in (ax_v, ax_c, ax_p, ax_h):
        for when, colour, ls, _ in marks:
            axis.axvline(when, color=colour, lw=1.1, ls=ls, alpha=0.9)
    for when, colour, _, label in marks:
        ax_v.text(when, 1.0, f" {label}", transform=ax_v.get_xaxis_transform(), color=colour, fontsize=8, va="bottom", ha="left")
    ax_v.set_xlim(0, t_end * 1.02)
    for axis in (ax_v, ax_c, ax_p):
        plt.setp(axis.get_xticklabels(), visible=False)
    return fig


##
# Scenario window (averages over all episodes of the scenario)
##


def position_stats(episodes: list[dict], key: str, edges: np.ndarray, transform=None):
    """Mean, std (over episodes), max (over samples) and number of episodes per position bin of a channel.

    Every episode is averaged inside each bin first, so that an episode that spends a long time at one place
    (e.g. a robot that got stuck) does not outweigh the others.
    """
    nb = len(edges) - 1
    means, maxima = [], []
    for ep in episodes:
        s_ = ep.get("series") or {}
        x, v = num(s_.get("x")), num(s_.get(key))
        if not x.size or not v.size:
            continue
        if transform is not None:
            v = transform(v)
        ok = np.isfinite(x) & np.isfinite(v)
        idx = np.digitize(x[ok], edges) - 1
        keep = (idx >= 0) & (idx < nb)
        idx, vals = idx[keep], v[ok][keep]
        cnt = np.bincount(idx, minlength=nb)
        tot = np.bincount(idx, weights=vals, minlength=nb)
        mx = np.full(nb, -np.inf)
        np.maximum.at(mx, idx, vals)
        means.append(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan))
        maxima.append(np.where(cnt > 0, mx, np.nan))
    if not means:
        nan = np.full(nb, np.nan)
        return nan, nan.copy(), nan.copy(), np.zeros(nb, int)
    m, mxs = np.vstack(means), np.vstack(maxima)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # bins nobody visited
        return np.nanmean(m, axis=0), np.nanstd(m, axis=0), np.nanmax(mxs, axis=0), np.isfinite(m).sum(axis=0)


def contact_by_position(episodes: list[dict], edges: np.ndarray) -> np.ndarray:
    """Share (0..1) of the episodes that had a trunk contact in each position bin (among those that were there)."""
    nb = len(edges) - 1
    visited, touched = np.zeros(nb), np.zeros(nb)
    for ep in episodes:
        s = ep.get("series") or {}
        x = forward_fill(num(s.get("x")))
        c = num(s.get("contact")) > 0.5
        if not x.size:
            continue
        ok = np.isfinite(x)
        idx = np.digitize(x[ok], edges) - 1
        idx = idx[(idx >= 0) & (idx < nb)]
        visited[np.unique(idx)] += 1
        ci = np.digitize(x[ok & c], edges) - 1
        ci = ci[(ci >= 0) & (ci < nb)]
        touched[np.unique(ci)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(visited > 0, touched / visited, np.nan)


def scenario_text(name: str, meta: dict, sc: dict) -> list[str]:
    s = sc["summary"]
    eps = sc["episodes"]
    kv = lambda label, value: _kv(label, value, 26)  # noqa: E731
    return [
        f"{name}",
        f"{len(eps)} episodes, seeds {sc.get('seeds', '?')}",
        "",
        "OUTCOME",
        kv("success", pct(s.get("success_rate"))),
        kv(f"reached {meta.get('midpoint_label', 'top')}", pct(s.get("midpoint_rate"))),
        kv("fell before goal", pct(s.get("fall_rate"))),
        kv("timeout", pct(s.get("timeout_rate"))),
        kv("mean time to goal", fmt(s.get("mean_time_to_goal_s"), ".1f", "s")),
        kv("mean progress", fmt(s.get("mean_progress_m"), ".2f", "m")),
        "",
        "SPEED TRACKING",
        kv("commanded speed", fmt(meta.get("command_speed_mps"), ".2f", "m/s")),
        kv("mean forward speed", fmt(s.get("mean_forward_speed_mps"), ".2f", "m/s")),
        kv("vel error xy", fmt(s.get("mean_vel_err_xy_mps"), ".3f", "m/s")),
        kv("vel error yaw", fmt(s.get("mean_vel_err_yaw_radps"), ".3f", "rad/s")),
        "",
        "PATH (max per episode)",
        kv("heading err mean / worst", f"{fmt(s.get('mean_max_yaw_err_deg'), '.1f')} / {fmt(s.get('worst_max_yaw_err_deg'), '.1f')} deg"),
        kv("lateral mean / worst", f"{fmt(s.get('mean_max_lateral_m'), '.2f')} / {fmt(s.get('worst_max_lateral_m'), '.2f')} m"),
        "",
        "TRUNK CONTACTS",
        kv("episodes with contact", pct(s.get("trunk_contact_episode_rate"))),
        kv("events per episode", fmt(s.get("mean_trunk_contacts_per_episode"), ".2f")),
        kv("contact time / ep", fmt(s.get("mean_trunk_contact_time_s"), ".2f", "s")),
        kv("worst force", fmt(s.get("worst_trunk_contact_force_n"), ".0f", "N")),
        "",
        "STABILITY / EFFORT",
        kv("tilt mean / worst", f"{fmt(s.get('mean_tilt_deg'), '.1f')} / {fmt(s.get('worst_max_tilt_deg'), '.1f')} deg"),
        kv("mean power", fmt(s.get("mean_power_w"), ".0f", "W")),
        kv("cost of transport", fmt(s.get("mean_cost_of_transport"), ".2f")),
        kv("action rate", fmt(s.get("mean_action_rate"), ".2f")),
        kv("joint acceleration", fmt(s.get("mean_joint_acc"), ".0f")),
        "",
        "REWARD",
        kv("mean total", fmt(s.get("mean_total_reward"), ".1f")),
        "",
        "END REASONS",
        *[f"  {k:<24}{v}" for k, v in sorted((s.get("end_reasons") or {}).items())],
    ]


def scenario_figure(plt, name: str, sc: dict):
    eps = sc["episodes"]
    meta = get_meta(sc, eps)
    cmd_speed = meta.get("command_speed_mps") or 0.5
    half = meta.get("obstacle_half_width")
    rec_dt = (meta.get("record_dt_s") or 0.1)
    spawn = meta.get("spawn_x")
    goal = meta.get("goal_x")
    has_series = any((e.get("series") or {}).get("t") for e in eps)

    fig = plt.figure(figsize=(16.5, 9.2), facecolor=BG)
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1.0, 1.0, 0.72], left=0.05, right=0.99, top=0.925, bottom=0.07, hspace=0.45, wspace=0.36
    )  # fmt: skip
    ax_sp = fig.add_subplot(gs[0, 0])
    ax_ov = fig.add_subplot(gs[0, 1])
    ax_ct = fig.add_subplot(gs[1, 0], sharex=ax_sp)
    ax_oc = fig.add_subplot(gs[1, 1])
    ax_hd = fig.add_subplot(gs[2, 0], sharex=ax_sp)
    ax_tp = fig.add_subplot(gs[2, 1], sharex=ax_sp)
    ax_tx = fig.add_subplot(gs[:, 2])
    fig.suptitle(f"{name}   |   average over {len(eps)} episodes", fontsize=13, fontweight="bold", color="#1f4e8c", x=0.05, ha="left")
    set_window_title(fig, f"{name} - scenario average")
    text_panel(ax_tx, scenario_text(name, meta, sc), size=8.2)

    # --- outcome per episode ---
    ax = ax_oc
    idx = np.arange(1, len(eps) + 1)
    progress = [e.get("progress_m") or 0.0 for e in eps]
    ax.bar(idx, progress, color=[OUTCOME_COLORS[outcome(e)] for e in eps], width=0.7)
    if spawn is not None and goal is not None:
        ax.axhline(goal - spawn, color=C_OK, ls="--", lw=1.0)
        ax.text(len(eps) + 0.45, goal - spawn, "distance to goal ", color=C_OK, fontsize=7.5, ha="right", va="bottom")
        ax.set_ylim(0, (goal - spawn) * 1.18)
    for i, e in zip(idx, eps):
        n_c = e.get("trunk_contact_events")
        label = f"{e.get('duration_s', 0):.0f}s" + (f"\n{n_c}c" if n_c else "")
        ax.text(i, (e.get("progress_m") or 0.0), label, ha="center", va="bottom", fontsize=7, color="#333333")
    if len(eps) <= 30:
        ax.set_xticks(idx)
        seeds = [e.get("seed") for e in eps]
        multi = len(set(seeds)) > 1
        ax.set_xticklabels([f"{i}\ns{sd}" if multi else str(i) for i, sd in zip(idx, seeds)], fontsize=7)
    style_axes(ax, "Outcome per episode (green = success, red = fell, amber = timeout)", "progress, m",
               "episode   (above a bar: duration and 'Nc' = trunk contact events)")  # fmt: skip

    if not has_series:
        for axis in (ax_sp, ax_ov, ax_ct, ax_hd, ax_tp):
            axis.text(0.5, 0.5, "this report has no time series", ha="center", va="center", transform=axis.transAxes)
        return fig

    # position bins
    lo = spawn if spawn is not None else min(np.nanmin(num((e.get("series") or {}).get("x"))) for e in eps if (e.get("series") or {}).get("x"))
    hi = goal if goal is not None else max(np.nanmax(num((e.get("series") or {}).get("x"))) for e in eps if (e.get("series") or {}).get("x"))
    bw = max(0.1, 2.0 * cmd_speed * rec_dt)
    edges = np.arange(lo, hi + bw, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def course_marks(axis):
        if half is not None:
            axis.axvspan(-half, half, color=C_OBSTACLE, alpha=0.16, lw=0)
        if meta.get("midpoint_x") is not None:
            axis.axvline(meta["midpoint_x"], color=C_SPEED, lw=0.9, ls="--")
        if goal is not None:
            axis.axvline(goal, color=C_OK, lw=1.0, ls="--")
        if spawn is not None:
            axis.axvline(spawn, color=C_GRAY, lw=0.9, ls=":")

    # --- speed vs position ---
    mean, std, _, cnt = position_stats(eps, "vx", edges)
    course_marks(ax_sp)
    ax_sp.fill_between(centers, mean - std, mean + std, color=C_SPEED, alpha=0.2, lw=0, label="mean +- std")
    ax_sp.plot(centers, mean, color=C_SPEED, lw=1.8, label="measured forward speed")
    ax_sp.axhline(cmd_speed, color=C_CMD, ls="--", lw=1.3, label=f"commanded ({cmd_speed:.2f} m/s)")
    style_axes(ax_sp, "Forward speed along the course (grey band = obstacle)", "speed, m/s")
    ax_sp.set_ylim(0, max(1.35 * cmd_speed, 1.1 * float(np.nanmax(mean + std)) if np.isfinite(mean).any() else 0.0))
    ax_sp.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    plt.setp(ax_sp.get_xticklabels(), visible=False)

    # --- speed vs time overlay ---
    for e in eps:
        s = e.get("series") or {}
        if s.get("t"):
            ax_ov.plot(num(s["t"]), num(s["vx"]), color=OUTCOME_COLORS[outcome(e)], lw=1.0, alpha=0.55)
    ax_ov.axhline(cmd_speed, color=C_CMD, ls="--", lw=1.3)
    style_axes(ax_ov, "Forward speed vs. time, every episode", "speed, m/s", "time since episode start, s")
    ax_ov.set_ylim(bottom=min(0.0, ax_ov.get_ylim()[0]))

    # --- contacts by position ---
    frac = contact_by_position(eps, edges)
    course_marks(ax_ct)
    ax_ct.bar(centers, 100.0 * np.nan_to_num(frac, nan=0.0), width=bw * 0.92, color=C_CONTACT)
    ax_ct.set_ylim(0, 105)
    style_axes(ax_ct, "Where the trunk touched the surface", "% of episodes with contact")
    total_events = sum(e.get("trunk_contact_events") or 0 for e in eps)
    ax_ct.text(0.995, 0.93, f"{total_events} contact event(s) in {len(eps)} episodes", transform=ax_ct.transAxes, ha="right", va="top",
               color=C_CONTACT if total_events else C_OK, fontsize=9, fontweight="bold")  # fmt: skip
    plt.setp(ax_ct.get_xticklabels(), visible=False)

    # --- heading error / lateral deviation by position ---
    course_marks(ax_hd)
    ym, _, ymax, _ = position_stats(eps, "yaw_err_deg", edges)
    lm, _, lmax, _ = position_stats(eps, "y", edges, transform=np.abs)
    ax_hd.plot(centers, ym, color=C_YAW, lw=1.5, label="heading error, mean")
    ax_hd.plot(centers, ymax, color=C_YAW, lw=0.9, ls="--", label="heading error, max")
    style_axes(ax_hd, "Heading error and lateral deviation", "heading error, deg", "position x along the course, m")
    twin = ax_hd.twinx()
    twin.plot(centers, lm, color=C_LAT, lw=1.5, label="|lateral|, mean")
    twin.plot(centers, lmax, color=C_LAT, lw=0.9, ls="--", label="|lateral|, max")
    twin.set_ylabel("lateral, m", color=C_LAT)
    twin.grid(False)
    h1, l1 = ax_hd.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    pad_top(ax_hd, 0.3)
    pad_top(twin, 0.3)
    ax_hd.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5, ncol=4, framealpha=0.9)

    # --- tilt and power by position ---
    course_marks(ax_tp)
    tm, _, tmax, _ = position_stats(eps, "tilt_deg", edges)
    pm, _, _, _ = position_stats(eps, "power_w", edges)
    ax_tp.plot(centers, tm, color=C_TILT, lw=1.5, label="tilt, mean")
    ax_tp.plot(centers, tmax, color=C_TILT, lw=0.9, ls="--", label="tilt, max")
    style_axes(ax_tp, "Tilt and mechanical power", "tilt, deg", "position x along the course, m")
    twin2 = ax_tp.twinx()
    twin2.plot(centers, pm, color=C_GRAY, lw=1.3, label="power, mean")
    twin2.set_ylabel("power, W", color=C_GRAY)
    twin2.grid(False)
    h1, l1 = ax_tp.get_legend_handles_labels()
    h2, l2 = twin2.get_legend_handles_labels()
    pad_top(ax_tp, 0.3)
    pad_top(twin2, 0.3)
    ax_tp.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5, ncol=3, framealpha=0.9)
    ax_sp.set_xlim(edges[0], edges[-1])
    return fig


##
# Whole-run window
##

STAIRS_RE = re.compile(r"^stairs_up_(\d+)cm$")


def overall_text(report: dict, scenarios: dict) -> list[str]:
    kv = lambda label, value, width=28: _kv(label, value, width)  # noqa: E731
    eps = [e for sc in scenarios.values() for e in sc["episodes"]]
    n = len(eps)
    per_scenario = {k: sc["summary"] for k, sc in scenarios.items()}
    macro = float(np.mean([s["success_rate"] for s in per_scenario.values()])) if per_scenario else None
    speeds = [(get_meta(sc, sc["episodes"]).get("command_speed_mps"), sc) for sc in scenarios.values()]
    ratios = [
        e["forward_speed_mps"] / cmd for cmd, sc in speeds if cmd for e in sc["episodes"] if e.get("forward_speed_mps") is not None
    ]
    contact_eps = [e for e in eps if e.get("trunk_contact_events") is not None]
    ckpt = report.get("checkpoint", "?")
    ckpt = os.path.join(os.path.basename(os.path.dirname(ckpt)), os.path.basename(ckpt))
    ok = [e for e in eps if e.get("success")]
    fell = [e for e in eps if outcome(e) == "FELL"]
    ranked = sorted(per_scenario.items(), key=lambda kv_: (kv_[1]["success_rate"], -kv_[1]["fall_rate"]))
    lines = [
        "CONTROLLER OVERVIEW",
        f"checkpoint {ckpt}",
        f"task {report.get('task', '?')}",
        f"date {report.get('date', '?')}   seeds {report.get('seeds', '?')}",
        f"speed {'scenario default' if report.get('speed_override_mps') is None else str(report['speed_override_mps']) + ' m/s'}",
        f"{len(scenarios)} scenarios, {n} episodes",
        "",
        "TASK",
        kv("success (all episodes)", pct(len(ok) / n if n else None)),
        kv("success (mean of scenarios)", pct(macro)),
        kv("fell before goal", pct(len(fell) / n if n else None)),
        kv("timeout", pct((n - len(ok) - len(fell)) / n if n else None)),
        kv("mean time to goal", fmt(mean_of(ok, "time_to_goal_s"), ".1f", "s")),
        "",
        "SPEED TRACKING",
        kv("forward speed / command", pct(float(np.mean(ratios)) if ratios else None)),
        kv("vel error xy", fmt(mean_of(eps, "vel_err_xy_mps"), ".3f", "m/s")),
        kv("vel error yaw", fmt(mean_of(eps, "vel_err_yaw_radps"), ".3f", "rad/s")),
        "",
        "PATH",
        kv("heading err mean / worst", f"{fmt(mean_of(eps, 'max_yaw_err_deg'), '.1f')} / {fmt(worst_of(eps, 'max_yaw_err_deg'), '.1f')} deg"),
        kv("lateral mean / worst", f"{fmt(mean_of(eps, 'max_lateral_m'), '.2f')} / {fmt(worst_of(eps, 'max_lateral_m'), '.2f')} m"),
        "",
        "TRUNK CONTACTS",
        kv("episodes with contact", pct(sum(e["trunk_contact_events"] > 0 for e in contact_eps) / len(contact_eps) if contact_eps else None)),
        kv("events per episode", fmt(mean_of(eps, "trunk_contact_events"), ".2f")),
        kv("contact time / episode", fmt(mean_of(eps, "trunk_contact_time_s"), ".2f", "s")),
        "",
        "STABILITY / EFFORT",
        kv("tilt mean / worst", f"{fmt(mean_of(eps, 'mean_tilt_deg'), '.1f')} / {fmt(worst_of(eps, 'max_tilt_deg'), '.1f')} deg"),
        kv("mean power", fmt(mean_of(eps, "power_w"), ".0f", "W")),
        kv("cost of transport", fmt(mean_of(eps, "cost_of_transport"), ".2f")),
        kv("action rate", fmt(mean_of(eps, "action_rate"), ".2f")),
        kv("joint acceleration", fmt(mean_of(eps, "joint_acc"), ".0f")),
        kv("mean total reward", fmt(mean_of(eps, "total_reward"), ".1f")),
        "",
    ]
    stairs = sorted(((int(m.group(1)), s) for k, s in per_scenario.items() if (m := STAIRS_RE.match(k))), key=lambda p: p[0])
    if stairs:
        passed = None  # highest height such that every tested height up to it has success >= 80 %
        for cm, s_ in stairs:
            if s_["success_rate"] < 0.8:
                break
            passed = cm
        lines += [
            "STAIRS (passed = success >= 80 %)",
            kv("passed up to", f"{passed} cm" if passed else "none"),
            kv("tested", f"{stairs[0][0]}..{stairs[-1][0]} cm ({len(stairs)} heights)"),
            "",
        ]
    lines.append("WEAKEST SCENARIOS")
    for k, s in ranked[:5]:
        lines.append(f"  {k[:30]:<30}{100 * s['success_rate']:>4.0f}% ok {100 * s['fall_rate']:>4.0f}% fell")
    return lines


def overall_figure(plt, report: dict, scenarios: dict):
    names = list(scenarios)
    nS = len(names)
    height = float(np.clip(0.3 * nS + 4.2, 8.6, 15.0))
    fig = plt.figure(figsize=(16.5, height), facecolor=BG)
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 0.82], height_ratios=[1.0, 1.0], left=0.15, right=0.99, top=0.93, bottom=0.06, hspace=0.32, wspace=0.28
    )  # fmt: skip
    ax_out = fig.add_subplot(gs[0, 0])
    ax_sweep = fig.add_subplot(gs[0, 1])
    ax_trk = fig.add_subplot(gs[1, 0])
    ax_cnt = fig.add_subplot(gs[1, 1], sharey=ax_trk)
    ax_tx = fig.add_subplot(gs[:, 2])
    fig.suptitle("Controller evaluation: whole run", fontsize=13, fontweight="bold", color="#1f4e8c", x=0.15, ha="left")
    set_window_title(fig, "Whole run - controller evaluation")
    text_panel(ax_tx, overall_text(report, scenarios), size=8.2)

    y = np.arange(nS)
    S = [scenarios[k]["summary"] for k in names]
    fs = 7.5 if nS > 14 else 8.5

    # --- outcomes ---
    succ = np.array([s["success_rate"] for s in S]) * 100
    fall = np.array([s["fall_rate"] for s in S]) * 100
    tmo = np.array([s["timeout_rate"] for s in S]) * 100
    ax_out.barh(y, succ, color=C_OK, label="success")
    ax_out.barh(y, fall, left=succ, color=C_FALL, label="fell")
    ax_out.barh(y, tmo, left=succ + fall, color=C_TIMEOUT, label="timeout")
    ax_out.set_yticks(y)
    ax_out.set_yticklabels(names, fontsize=fs)
    ax_out.invert_yaxis()
    ax_out.set_xlim(0, 100)
    style_axes(ax_out, "Outcomes", "", "% of episodes")
    ax_out.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=3, frameon=False)

    # --- stairs sweep ---
    stairs = sorted(((int(m.group(1)), k) for k in names if (m := STAIRS_RE.match(k))))
    if len(stairs) >= 2:
        cms = [c for c, _ in stairs]
        ax_sweep.plot(cms, [scenarios[k]["summary"]["success_rate"] * 100 for _, k in stairs], "o-", color=C_OK, lw=2, label="crossed the whole pyramid")
        ax_sweep.plot(cms, [scenarios[k]["summary"]["midpoint_rate"] * 100 for _, k in stairs], "s--", color=C_SPEED, lw=1.4, label="reached the top")
        ax_sweep.plot(cms, [scenarios[k]["summary"]["fall_rate"] * 100 for _, k in stairs], "x:", color=C_FALL, lw=1.4, label="fell")
        ax_sweep.set_ylim(-3, 103)
        ax_sweep.set_xticks(cms)
        style_axes(ax_sweep, "Capability vs. step height", "% of episodes", "step height, cm")
        ax_sweep.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=3, frameon=False)
        passed = None
        for cm, k in stairs:
            if scenarios[k]["summary"]["success_rate"] < 0.8:
                break
            passed = cm
        if passed is not None:
            ax_sweep.axvspan(cms[0] - 0.5, passed + 0.5, color=C_OK, alpha=0.08, lw=0)
    else:
        mid = np.array([s["midpoint_rate"] for s in S]) * 100
        ax_sweep.barh(y, mid, color=C_SPEED)
        ax_sweep.set_yticks(y)
        ax_sweep.set_yticklabels(names, fontsize=fs)
        ax_sweep.invert_yaxis()
        ax_sweep.set_xlim(0, 100)
        style_axes(ax_sweep, "Reached the midpoint (top / bottom / center)", "", "% of episodes")

    # --- speed tracking ---
    ratio = []
    for k in names:
        cmd = get_meta(scenarios[k], scenarios[k]["episodes"]).get("command_speed_mps")
        v = scenarios[k]["summary"].get("mean_forward_speed_mps")
        ratio.append(np.nan if not cmd or v is None else 100.0 * v / cmd)
    ax_trk.barh(y, np.nan_to_num(ratio), color=C_SPEED)
    ax_trk.axvline(100, color=C_CMD, ls="--", lw=1.2)
    for yi, s in zip(y, S):
        e = s.get("mean_vel_err_xy_mps")
        if e is not None:
            ax_trk.text(1, yi, f"err {e:.2f} m/s", va="center", ha="left", fontsize=fs - 1, color="white")
    ax_trk.set_yticks(y)
    ax_trk.set_yticklabels(names, fontsize=fs)
    ax_trk.invert_yaxis()
    style_axes(ax_trk, "Speed tracking (mean forward speed / command)", "", "% of the commanded speed")

    # --- trunk contacts ---
    events = [s.get("mean_trunk_contacts_per_episode") for s in S]
    rate = [s.get("trunk_contact_episode_rate") for s in S]
    ax_cnt.barh(y, [e or 0.0 for e in events], color=C_CONTACT)
    for yi, e, r in zip(y, events, rate):
        if e is not None:
            ax_cnt.text(max(e, 0.0), yi, f" {pct(r)} of episodes", va="center", ha="left", fontsize=fs - 1, color="#333333")
    plt.setp(ax_cnt.get_yticklabels(), visible=False)
    style_axes(ax_cnt, "Trunk contacts", "", "events per episode")
    ax_cnt.set_xlim(0, max(1.0, max([e or 0.0 for e in events] + [0.0])) * 1.45)
    return fig


##
# Main
##


def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def select_scenarios(report: dict, patterns: str | None) -> dict:
    scenarios = report["scenarios"]
    if not patterns:
        return scenarios
    wanted = [p.strip() for p in patterns.split(",") if p.strip()]
    chosen = {k: v for k, v in scenarios.items() if any(fnmatch.fnmatchcase(k, ("*" if w == "all" else "rough_*" if w == "rough" else w)) for w in wanted)}
    if not chosen:
        sys.exit(f"[ERROR] No scenario of the report matches '{patterns}'. Available: {list(scenarios)}")
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show the results of an evaluate.py report.")
    ap.add_argument("report", help="Path of the JSON report written by evaluate.py.")
    ap.add_argument("--scenarios", default=None, help="Only these scenarios (comma-separated names / glob patterns).")
    ap.add_argument("--no_overall", action="store_true", help="Skip the whole-run window.")
    ap.add_argument("--no_scenarios", action="store_true", help="Skip the per-scenario windows.")
    ap.add_argument("--no_episodes", action="store_true", help="Skip the per-episode windows.")
    ap.add_argument("--max_windows", type=int, default=40, help="Open at most this many windows at once (default 40).")
    ap.add_argument("--save", default=None, help="Save every figure as PNG into this folder.")
    ap.add_argument("--no_show", action="store_true", help="Do not open windows (use with --save).")
    args = ap.parse_args(argv)

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)
    scenarios = select_scenarios(report, args.scenarios)
    if not scenarios:
        sys.exit("[ERROR] The report contains no scenarios.")

    show = not args.no_show
    save_dir = args.save
    if show and not has_display():
        stem = os.path.splitext(os.path.abspath(args.report))[0]
        save_dir = save_dir or stem + "_plots"
        print("[INFO] No display found: the figures are saved as PNG files instead of being shown.")
        show = False
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("[ERROR] matplotlib is required: pip install matplotlib")
    plt.rcParams.update({"font.size": 9, "axes.facecolor": "white", "figure.facecolor": BG, "savefig.facecolor": BG})
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plan = []  # (file name, builder)
    if not args.no_overall:
        plan.append(("overall", lambda: overall_figure(plt, report, scenarios)))
    if not args.no_scenarios:
        for name, sc in scenarios.items():
            plan.append((f"scenario_{safe_name(name)}", lambda name=name, sc=sc: scenario_figure(plt, name, sc)))
    if not args.no_episodes:
        for name, sc in scenarios.items():
            meta = get_meta(sc, sc["episodes"])
            total = len(sc["episodes"])
            for k, ep in enumerate(sc["episodes"], 1):
                fname = f"episode_{safe_name(name)}_s{ep.get('seed', 'x')}_{k:02d}"
                plan.append((fname, lambda name=name, meta=meta, ep=ep, k=k, total=total: episode_figure(plt, name, meta, ep, k, total)))

    opened = skipped = saved = 0
    for fname, build in plan:
        want_window = show and opened < args.max_windows
        if not want_window and not save_dir:
            skipped += 1
            continue
        fig = build()
        if save_dir:
            fig.savefig(os.path.join(save_dir, fname + ".png"), dpi=110)
            saved += 1
        if want_window:
            opened += 1
        else:
            plt.close(fig)
            skipped += show  # only counts as "not shown" when windows were requested
    if save_dir:
        print(f"[INFO] Saved {saved} figure(s) to {os.path.abspath(save_dir)}")
    if skipped:
        print(
            f"[INFO] {skipped} window(s) were not opened (limit --max_windows {args.max_windows}). "
            "Use --scenarios to narrow the selection, --max_windows to raise the limit or --save to write all of them to PNG files."
        )
    if show and opened:
        print(f"[INFO] {opened} window(s) open. Close them all to finish.")
        plt.show()


if __name__ == "__main__":
    main()