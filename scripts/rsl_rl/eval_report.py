"""Viewer for the JSON reports written by evaluate.py.

It is started automatically when evaluate.py finishes (use --no_report there to skip it), and it can be
started on its own on any saved report:

    python scripts/rsl_rl/eval_report.py path/to/report.json
    python scripts/rsl_rl/eval_report.py report.json --scenarios "rough_pyramid_stairs_1*,rough_boxes*"
    python scripts/rsl_rl/eval_report.py report.json --save plots/ --no_show

Windows (matplotlib):
  * one window for the whole run: outcomes per scenario, success vs. stair height, speed tracking, body
    contacts and a numeric summary of the controller with its weakest scenarios;
  * one window per scenario: averages over all its episodes, aligned by position along the course;
  * one window per episode: the speed of the scenario (forward, backward, sideways or the turn rate) vs. time
    with the commanded value, the body-contact timeline (base / head / legs), progress, heading error / tilt / lateral deviation
    and all numeric metrics of that episode; the sprint scenario also shows the stop command and the stop.

This script does not need Isaac Sim, only numpy and matplotlib. Without a display it saves PNG files instead.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import sys
import textwrap
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
C_BASE = "#8e1f1f"  # body contacts: base (ends the episode), head, legs
C_HEAD = C_CONTACT
C_LEGS = "#e08a1e"
CONTACT_LANES = (  # (label, key of the intervals, key of the event count, colour)
    ("base", "base_contact_intervals_s", "base_contact_events", C_BASE),
    ("head", "head_contact_intervals_s", "head_contact_events", C_HEAD),
    ("legs", "leg_contact_intervals_s", "leg_contact_events", C_LEGS),
)


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
    meta.setdefault("motion", "walk")  # walk | turn | sprint_brake
    meta.setdefault("primary_channel", "vx")  # the velocity channel that carries the command: vx | vy | wz
    meta.setdefault("primary_sign", 1.0)  # +1 / -1: plotted value = sign * channel (backward / right are negative)
    meta.setdefault("primary_label", "forward speed")
    meta.setdefault("primary_unit", "m/s")
    meta.setdefault("primary_command", meta.get("command_speed_mps"))
    meta.setdefault("progress_unit", "m")
    meta.setdefault("brake_time_s", None)
    meta.setdefault("spawn_x", None)
    meta.setdefault("goal_x", None)
    meta.setdefault("midpoint_x", None)
    meta.setdefault("midpoint_label", "top")
    meta.setdefault("obstacle_half_width", None)
    return meta


def primary_series(series: dict, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Measured and commanded value of the scenario's main speed, sign-corrected so that "more" is positive."""
    channel, sign = meta["primary_channel"], float(meta["primary_sign"])
    return sign * num(series.get(channel)), sign * num(series.get("cmd_" + channel))


def fmt_progress(value, meta: dict, none: str = "-") -> str:
    """Progress with its unit: metres along the course, or degrees for the turn on the spot."""
    if meta.get("progress_unit") == "deg":
        return fmt(value, ".0f", "deg", none)
    return fmt(value, ".2f", "m", none)


def ints(value, none: str = "-") -> str:
    return none if value is None else f"{value:.0f}"


def contact_lanes(ep: dict) -> list[tuple]:
    """Timeline lanes of an episode: base / head / legs, or a single "trunk" lane for older reports."""
    if "base_contact_intervals_s" not in ep:
        return [("trunk", ep.get("trunk_contact_intervals_s"), ep.get("trunk_contact_events"), C_CONTACT)]
    return [(label, ep.get(iv), ep.get(cnt), colour) for label, iv, cnt, colour in CONTACT_LANES]


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


SUBTITLE_PT = 8  # font size of the grey line(s) under a title


def wrap_to_axes(ax, text: str, font_pt: float = SUBTITLE_PT) -> str:
    """Break ``text`` into lines that fit the current width of the axes (explicit line breaks are kept)."""
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    max_chars = max(24, int(width_in * 72 / (font_pt * 0.6)))
    lines = []
    for part in text.split("\n"):
        lines += textwrap.wrap(part.strip(), max_chars) or [""]
    return "\n".join(lines)


def style_axes(ax, title: str, ylabel: str = "", xlabel: str = "", subtitle: str = ""):
    """Bold title; ``subtitle`` is optional legend hints in smaller grey text below it, wrapped to the axes width."""
    if subtitle:
        subtitle = wrap_to_axes(ax, subtitle)
        ax._subtitle_lines = subtitle.count("\n") + 1
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#1f4e8c", pad=15 + 12 * subtitle.count("\n"))
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=SUBTITLE_PT, color=C_GRAY)
    else:
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#1f4e8c")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


# Screen size in pixels (set in main() when windows are shown) and an optional fixed dpi (--dpi).
SCREEN: dict = {"w": None, "h": None, "dpi": None}
FIT_HEIGHT_CAP_IN = 9.6  # taller figures do not shrink the dpi further (they are clipped to the screen height)
WINDOW_CHROME_PX = 170  # screen height taken by the window title bar, the matplotlib toolbar and desktop panels


def screen_size() -> tuple[int, int] | None:
    """Screen size in pixels, or None if it cannot be read (no display, no tkinter)."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return None


def fit_dpi(width_in: float, height_in: float) -> float:
    """dpi at which a figure of this size fits on the screen. A window that is too big gets squeezed by the
    window manager while the fonts keep their size, which makes the texts overlap; a lower dpi scales the
    whole figure (fonts included) instead."""
    if SCREEN["dpi"]:
        return float(SCREEN["dpi"])
    if not SCREEN["w"]:
        return 100.0
    usable_h = SCREEN["h"] - WINDOW_CHROME_PX  # title bar, toolbar, desktop panels
    dpi = min(100.0, 0.96 * SCREEN["w"] / width_in, usable_h / min(height_in, FIT_HEIGHT_CAP_IN))
    return max(dpi, 50.0)


def new_figure(plt, width_in: float, height_in: float):
    dpi = fit_dpi(width_in, height_in)
    if SCREEN["h"] and not SCREEN["dpi"]:  # never taller than the screen
        height_in = min(height_in, (SCREEN["h"] - WINDOW_CHROME_PX) / dpi)
    return plt.figure(figsize=(width_in, height_in), dpi=dpi, facecolor=BG)


def subtitle_lines(fig) -> int:
    """Largest number of subtitle lines over the axes of a figure."""
    return max([getattr(ax, "_subtitle_lines", 0) for ax in fig.axes] + [0])


def title_zone_in(lines: int) -> float:
    """Height (inches) taken above an axes by its title and ``lines`` subtitle lines."""
    pad_pt = 15 + 12 * max(lines - 1, 0) if lines else 6
    return (pad_pt + 14) / 72


def inch_layout(fig, gs, *, left: float, right: float, top: float, bottom: float, hgap: float, wgap: float):
    """Place the grid with margins and gaps in INCHES, and again whenever the window is resized.

    Fractions of the figure (the matplotlib default) shrink with the window while the fonts keep their size,
    so titles and labels overlap the row above on a small or squeezed window; fixed inches do not.
    """
    nrows, ncols = gs.get_geometry()

    def apply(_event=None):
        w, h = fig.get_size_inches()
        avail_h = max(h - top - bottom - hgap * (nrows - 1), 0.4 * nrows)
        avail_w = max(w - left - right - wgap * (ncols - 1), 0.4 * ncols)
        gs.update(
            left=left / w, right=1.0 - right / w, top=1.0 - top / h, bottom=bottom / h,
            hspace=hgap / (avail_h / nrows) if nrows > 1 else 0.0, wspace=wgap / (avail_w / ncols) if ncols > 1 else 0.0,
        )  # fmt: skip
        if fig._suptitle is not None:
            fig._suptitle.set_position((left / w, 1.0 - 0.12 / h))
            fig._suptitle.set_verticalalignment("top")

    apply()
    try:
        fig.canvas.mpl_connect("resize_event", apply)
    except Exception:
        pass
    return apply


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
    turn = meta.get("motion") == "turn"
    sprint = meta.get("motion") == "sprint_brake"
    unit = meta.get("primary_unit", "m/s")
    cmd = meta.get("primary_command")
    speed = ep.get("forward_speed_mps")
    if turn:  # the mean turn rate comes from the time series
        measured, _ = primary_series(ep.get("series") or {}, meta)
        speed = float(np.nanmean(measured)) if np.isfinite(measured).any() else None
    ratio = None if sprint or not cmd or speed is None else speed / cmd
    label = meta.get("primary_label", "forward speed")
    if sprint and meta.get("brake_time_s") is not None:
        command_txt = f"{fmt(cmd, '.2f')} {unit}, stop {meta['brake_time_s']:.0f} s"
    else:
        command_txt = fmt(cmd, ".2f", unit)
    lines = [
        f"{name}",
        f"seed {ep.get('seed', '?')}   episode {k}/{n}",
        "",
        kv("RESULT", f"{verdict}"),
        kv("end reason", str(ep.get("end_reason", "-"))),
        kv("duration", fmt(ep.get("duration_s"), ".1f", "s")),
        kv("time to goal", fmt(ep.get("time_to_goal_s"), ".1f", "s")),
        kv("progress", fmt_progress(ep.get("progress_m"), meta)),
        kv(f"reached {meta.get('midpoint_label', 'top')}", "yes" if ep.get("reached_midpoint") else "no")
        if meta.get("midpoint_x") is not None
        else kv("midpoint", "n/a"),
        "",
        "SPEED TRACKING",
        kv("command", command_txt),
        kv("mean turn rate" if turn else "mean speed", f"{fmt(speed, '.2f', unit)}" + ("" if ratio is None else f"  ({pct(ratio)})")),
        kv("vel error xy", fmt(ep.get("vel_err_xy_mps"), ".3f", "m/s")),
        kv("vel error yaw", fmt(ep.get("vel_err_yaw_radps"), ".3f", "rad/s")),
        "",
        "PATH",
        kv("max heading error", "not held" if turn else fmt(ep.get("max_yaw_err_deg"), ".1f", "deg")),
        kv("max drift from start" if turn else "max lateral dev.", fmt(ep.get("max_lateral_m"), ".2f", "m")),
        "",
        "BODY CONTACTS (events)",
        kv("trunk (base + head)", ints(ep.get("trunk_contact_events"))),
        kv("  head (nose)", ints(ep.get("head_contact_events"))),
        kv("  base", ints(ep.get("base_contact_events"))),
        kv("legs (not the feet)", ints(ep.get("leg_contact_events"))),
        kv("trunk contact time", fmt(ep.get("trunk_contact_time_s"), ".2f", "s")),
        kv("first trunk contact", fmt(ep.get("first_trunk_contact_s"), ".2f", "s")),
        kv("max force trunk", fmt(ep.get("max_trunk_contact_force_n"), ".0f", "N")),
        kv("max force legs", fmt(ep.get("max_leg_contact_force_n"), ".0f", "N")),
        "",
        *(
            [
                "SPRINT AND STOP",
                kv("peak speed", fmt(ep.get("peak_speed_mps"), ".2f", "m/s")),
                kv("time to 90 % of cmd", fmt(ep.get("time_to_90pct_s"), ".1f", "s")),
                kv("speed at stop cmd", fmt(ep.get("speed_at_brake_mps"), ".2f", "m/s")),
                kv("stop time", fmt(ep.get("stop_time_s"), ".2f", "s")),
                kv("stop distance", fmt(ep.get("stop_distance_m"), ".2f", "m")),
                kv("lowest speed after", fmt(ep.get("min_speed_after_brake_mps"), ".2f", "m/s")),
                "",
            ]
            if sprint
            else []
        ),
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
    cmd_speed = meta.get("primary_command")
    unit = meta.get("primary_unit", "m/s")
    label = meta.get("primary_label", "forward speed")
    turn = meta.get("motion") == "turn"
    half = meta.get("obstacle_half_width")

    fig = new_figure(plt, 15.6, 9.0)
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
    vx, cmd_vx = primary_series(series, meta)
    t_end = float(t[-1])

    # --- speed ---
    shade_obstacle_time(ax_v, t, xf, half, dt)
    sprint = meta.get("motion") == "sprint_brake"
    ax_v.plot(t, vx, color=C_SPEED, lw=1.6, label="measured")
    ax_v.plot(t, cmd_vx, color=C_CMD, lw=1.4, ls="--", label="commanded" + ("" if sprint else f" ({fmt(cmd_speed, '.2f')} {unit})"))
    mean_speed = np.nanmean(vx) if np.isfinite(vx).any() and not sprint else None
    if mean_speed is not None:
        ax_v.axhline(mean_speed, color=C_SPEED, lw=1.0, ls=":", label=f"mean {mean_speed:.2f} {unit}")
    style_axes(ax_v, label[:1].upper() + label[1:], f"{'turn rate' if turn else 'speed'}, {unit}")
    peak = max(1.1 * (np.nanmax(vx) if np.isfinite(vx).any() else 0.0), 1.35 * (cmd_speed or 0.0), 0.1)
    ax_v.set_ylim(min(0.0, np.nanmin(vx) if np.isfinite(vx).any() else 0.0), peak)
    ax_v.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    # --- body contacts: one lane per part (base / head / legs) ---
    lanes = contact_lanes(ep)
    min_w = max(0.012 * t_end, 0.05)  # a 0.02 s contact must stay visible on a 30 s axis
    labels = []
    for row, (part, intervals, count, colour) in enumerate(lanes):
        labels.append(f"{part} ({ints(count, '-')})")
        if count is None:
            ax_c.text(0.01 * t_end, row, "not recorded", va="center", ha="left", fontsize=7.5, color=C_GRAY)
            continue
        spans = [(a, max(b - a, min_w)) for a, b in (intervals or [])]
        if spans:
            ax_c.broken_barh(spans, (row - 0.32, 0.64), facecolors=colour, edgecolors="none")
    ax_c.set_yticks(range(len(lanes)))
    ax_c.set_yticklabels(labels, fontsize=8)
    ax_c.set_ylim(len(lanes) - 0.5, -0.5)
    ax_c.grid(True, axis="x", alpha=0.3)
    ax_c.grid(False, axis="y")
    total = sum(c or 0 for _, _, c, _ in lanes)
    tc = meta.get("trunk_contact") or {}
    ends = "a base contact ends the episode" if tc.get("terminates_episode", True) else "contacts do not end the episode"
    style_axes(ax_c, "Body contacts", "",
               subtitle=("no contact" if total == 0 else "bars = the part touches the surface") + f"; {ends}")  # fmt: skip

    # --- progress ---
    if half is not None:
        ax_p.axhspan(-half, half, color=C_OBSTACLE, alpha=0.16, lw=0, label="obstacle")
    ax_p.plot(t, x, color="#333333", lw=1.4, label="total rotation" if turn else "position along the course")
    if meta.get("spawn_x") is not None:
        ax_p.axhline(meta["spawn_x"], color=C_GRAY, lw=0.9, ls=":", label="start")
    if meta.get("midpoint_x") is not None:
        ax_p.axhline(meta["midpoint_x"], color=C_SPEED, lw=0.9, ls="--", label=meta.get("midpoint_label", "midpoint"))
    if meta.get("goal_x") is not None:
        ax_p.axhline(meta["goal_x"], color=C_OK, lw=1.0, ls="--", label="goal")
    style_axes(ax_p, "Progress", f"{'rotation' if turn else 'x'}, {meta.get('progress_unit', 'm')}")
    ax_p.legend(loc="upper left", fontsize=7.5, ncol=5, framealpha=0.9)

    # --- heading / tilt / lateral deviation ---
    yaw_err = num(series.get("yaw_err_deg"))
    if np.isfinite(yaw_err).any():
        ax_h.plot(t, yaw_err, color=C_YAW, lw=1.2, label="heading error, deg")
    ax_h.plot(t, num(series.get("tilt_deg")), color=C_TILT, lw=1.2, label="tilt, deg")
    style_axes(ax_h, "Tilt and drift from the start point" if turn else "Heading error, tilt and lateral deviation", "deg", "time since episode start, s")
    pad_top(ax_h, 0.35)
    ax_l = ax_h.twinx()
    ax_l.plot(t, num(series.get("y")), color=C_LAT, lw=1.2, label="drift from the start, m" if turn else "lateral deviation, m")
    ax_l.set_ylabel("drift, m" if turn else "lateral, m", color=C_LAT)
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
    brake = meta.get("brake_time_s")
    if brake is not None:
        marks.append((float(brake), C_YAW, "-", "stop command"))
        if ep.get("stop_time_s") is not None:
            marks.append((float(brake) + float(ep["stop_time_s"]), C_OK, ":", "stopped"))
    if verdict == "FELL":
        marks.append((t_end, C_FALL, "-", f"ended: {ep.get('end_reason', '?')}"))
    for axis in (ax_v, ax_c, ax_p, ax_h):
        for when, colour, ls, _ in marks:
            axis.axvline(when, color=colour, lw=1.1, ls=ls, alpha=0.9)
    marks.sort(key=lambda m: m[0])
    for i, (when, colour, _, mark_label) in enumerate(marks):
        # two close markers (e.g. "stop command" and "stopped"): the earlier label goes to the left of its line
        crowded = i + 1 < len(marks) and marks[i + 1][0] - when < 0.12 * t_end
        ax_v.text(when, 1.0, f"{mark_label} " if crowded else f" {mark_label}", transform=ax_v.get_xaxis_transform(),
                  color=colour, fontsize=8, va="bottom", ha="right" if crowded else "left")  # fmt: skip
    ax_v.set_xlim(0, t_end * 1.02)
    for axis in (ax_v, ax_c, ax_p):
        plt.setp(axis.get_xticklabels(), visible=False)
    return fig


##
# Scenario window (averages over all episodes of the scenario)
##


def position_stats(episodes: list[dict], key: str, edges: np.ndarray, transform=None, xkey: str = "x"):
    """Mean, std (over episodes), max (over samples) and number of episodes per bin of a channel.

    The bins are along ``xkey``: "x" = position along the course (or rotation), "t" = time since the episode start.

    Every episode is averaged inside each bin first, so that an episode that spends a long time at one place
    (e.g. a robot that got stuck) does not outweigh the others.
    """
    nb = len(edges) - 1
    means, maxima = [], []
    for ep in episodes:
        s_ = ep.get("series") or {}
        x, v = num(s_.get(xkey)), num(s_.get(key))
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


def contact_by_position(episodes: list[dict], edges: np.ndarray, xkey: str = "x", key: str = "contact") -> np.ndarray:
    """Share (0..1) of the episodes with a contact (channel ``key``) in each bin (among those that were there)."""
    nb = len(edges) - 1
    visited, touched = np.zeros(nb), np.zeros(nb)
    for ep in episodes:
        s = ep.get("series") or {}
        x = forward_fill(num(s.get(xkey)))
        c = num(s.get(key)) > 0.5
        if c.size != x.size:  # channel missing (older report)
            continue
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
    turn = meta.get("motion") == "turn"
    sprint = meta.get("motion") == "sprint_brake"
    unit = meta.get("primary_unit", "m/s")
    cmd = meta.get("primary_command")
    mean_speed = s.get("mean_forward_speed_mps")
    if turn:  # mean turn rate from the series of the episodes
        vals = [float(np.nanmean(primary_series(e.get("series") or {}, meta)[0])) for e in eps if (e.get("series") or {}).get("t")]
        mean_speed = float(np.mean(vals)) if vals else None
    command_txt = (
        f"{fmt(cmd, '.2f')} {unit}, stop {meta['brake_time_s']:.0f} s"
        if sprint and meta.get("brake_time_s") is not None
        else fmt(cmd, ".2f", unit)
    )
    return [
        f"{name}",
        f"{len(eps)} episode{'' if len(eps) == 1 else 's'}, seeds {sc.get('seeds', '?')}",
        "",
        "OUTCOME",
        kv("success", pct(s.get("success_rate"))),
        *([kv(f"reached {meta.get('midpoint_label', 'top')}", pct(s.get("midpoint_rate")))] if meta.get("midpoint_x") is not None else []),
        kv("fell / ended early", pct(s.get("fall_rate"))),
        kv("timeout", pct(s.get("timeout_rate"))),
        kv("mean time to goal", fmt(s.get("mean_time_to_goal_s"), ".1f", "s")),
        kv("mean progress", fmt_progress(s.get("mean_progress_m"), meta)),
        "",
        "SPEED TRACKING",
        kv("command", command_txt),
        kv("mean turn rate" if turn else "mean speed", fmt(mean_speed, ".2f", unit)),
        kv("vel error xy", fmt(s.get("mean_vel_err_xy_mps"), ".3f", "m/s")),
        kv("vel error yaw", fmt(s.get("mean_vel_err_yaw_radps"), ".3f", "rad/s")),
        "",
        "PATH (max per episode)",
        kv("heading err mean / worst", "not held" if turn else f"{fmt(s.get('mean_max_yaw_err_deg'), '.1f')} / {fmt(s.get('worst_max_yaw_err_deg'), '.1f')} deg"),
        kv(("drift" if turn else "lateral") + " mean / worst", f"{fmt(s.get('mean_max_lateral_m'), '.2f')} / {fmt(s.get('worst_max_lateral_m'), '.2f')} m"),
        "",
        *(
            [
                "SPRINT AND STOP",
                kv("peak speed", fmt(s.get("mean_peak_speed_mps"), ".2f", "m/s")),
                kv("time to 90 % of cmd", fmt(s.get("mean_time_to_90pct_s"), ".1f", "s")),
                kv("speed at stop cmd", fmt(s.get("mean_speed_at_brake_mps"), ".2f", "m/s")),
                kv("stopped (episodes)", pct(s.get("stopped_rate"))),
                kv("stop time", fmt(s.get("mean_stop_time_s"), ".2f", "s")),
                kv("stop distance", fmt(s.get("mean_stop_distance_m"), ".2f", "m")),
                kv("lowest speed after", fmt(s.get("lowest_speed_after_brake_mps"), ".2f", "m/s")),
                "",
            ]
            if sprint
            else []
        ),
        "BODY CONTACTS (events / episode)",
        kv("trunk (base + head)", fmt(s.get("mean_trunk_contacts_per_episode"), ".2f")),
        kv("  head (nose)", fmt(s.get("mean_head_contacts_per_episode"), ".2f")),
        kv("  base", fmt(s.get("mean_base_contacts_per_episode"), ".2f")),
        kv("legs (hip/thigh/calf)", fmt(s.get("mean_leg_contacts_per_episode"), ".2f")),
        kv("episodes w/ trunk cont.", pct(s.get("trunk_contact_episode_rate"))),
        kv("episodes w/ leg cont.", pct(s.get("leg_contact_episode_rate"))),
        kv("trunk contact time/ep", fmt(s.get("mean_trunk_contact_time_s"), ".2f", "s")),
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
    motion = meta.get("motion", "walk")
    turn, time_aligned = motion == "turn", motion == "sprint_brake"
    unit = meta.get("primary_unit", "m/s")
    label = meta.get("primary_label", "forward speed")
    sign = float(meta.get("primary_sign", 1.0))
    channel = meta.get("primary_channel", "vx")
    cmd_speed = meta.get("primary_command") or 0.5
    brake = meta.get("brake_time_s")
    half = meta.get("obstacle_half_width")
    rec_dt = (meta.get("record_dt_s") or 0.1)
    xkey = "t" if time_aligned else "x"
    x_label = "time since episode start, s" if time_aligned else ("total rotation, deg" if turn else "position x along the course, m")
    spawn = meta.get("spawn_x")
    goal = meta.get("goal_x")
    has_series = any((e.get("series") or {}).get("t") for e in eps)

    fig = new_figure(plt, 16.5, 9.2)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.72])

    def layout(lines: int):
        # above a row: its title and subtitle; below a row: tick labels (two lines under the outcome bars) and x label
        return inch_layout(fig, gs, left=0.8, right=0.15, top=0.45 + title_zone_in(lines), bottom=0.6,
                           hgap=title_zone_in(lines) + 0.62, wgap=1.45)  # fmt: skip

    layout(2)  # provisional (the subtitles are wrapped to the axes width), redone below with the real line count
    ax_sp = fig.add_subplot(gs[0, 0])
    ax_ov = fig.add_subplot(gs[0, 1])
    ax_ct = fig.add_subplot(gs[1, 0], sharex=ax_sp)
    ax_oc = fig.add_subplot(gs[1, 1])
    ax_hd = fig.add_subplot(gs[2, 0], sharex=ax_sp)
    ax_tp = fig.add_subplot(gs[2, 1], sharex=ax_sp)
    ax_tx = fig.add_subplot(gs[:, 2])
    episodes_txt = f"{len(eps)} episode" + ("" if len(eps) == 1 else "s")
    fig.suptitle(f"{name}   |   average over {episodes_txt}", fontsize=13, fontweight="bold", color="#1f4e8c", ha="left")
    short_label = label.split(" (")[0]  # e.g. "turn rate" of "turn rate (counter-clockwise)"
    label_note = label[len(short_label):].strip(" ()")
    title_label = short_label[:1].upper() + short_label[1:]
    set_window_title(fig, f"{name} - scenario average")
    text_panel(ax_tx, scenario_text(name, meta, sc), size=8.2)

    # --- outcome per episode ---
    ax = ax_oc
    idx = np.arange(1, len(eps) + 1)
    progress = [e.get("progress_m") or 0.0 for e in eps]
    ax.bar(idx, progress, color=[OUTCOME_COLORS[outcome(e)] for e in eps], width=0.7)
    if spawn is not None and goal is not None:
        ax.axhline(goal - spawn, color=C_OK, ls="--", lw=1.0)
        ax.text(1.01, goal - spawn, "full\nturn" if turn else "goal", transform=ax.get_yaxis_transform(),
                color=C_OK, fontsize=7.5, ha="left", va="center", clip_on=False)  # fmt: skip
        ax.set_ylim(0, (goal - spawn) * 1.18)
    elif progress:
        ax.set_ylim(0, max(progress) * 1.25 + 1e-9)
    for i, e in zip(idx, eps):
        n_c = e.get("trunk_contact_events")
        bar_text = f"{e.get('duration_s', 0):.0f}s" + (f"\n{n_c}c" if n_c else "")
        ax.text(i, (e.get("progress_m") or 0.0), bar_text, ha="center", va="bottom", fontsize=7, color="#333333")
    if len(eps) < 5:  # few episodes: do not stretch the bars over the whole width
        ax.set_xlim((len(eps) + 1) / 2 - 2.5, (len(eps) + 1) / 2 + 2.5)
    if len(eps) <= 30:
        ax.set_xticks(idx)
        seeds = [e.get("seed") for e in eps]
        multi = len(set(seeds)) > 1
        ax.set_xticklabels([f"{i}\ns{sd}" if multi else str(i) for i, sd in zip(idx, seeds)], fontsize=7)
    style_axes(ax, "Outcome per episode", f"progress, {meta.get('progress_unit', 'm')}", "episode",
               subtitle="green = success, red = fell, amber = timeout;\nabove a bar: duration, Nc = trunk contacts")  # fmt: skip

    if not has_series:
        for axis in (ax_sp, ax_ov, ax_ct, ax_hd, ax_tp):
            axis.text(0.5, 0.5, "this report has no time series", ha="center", va="center", transform=axis.transAxes)
        layout(subtitle_lines(fig))
        return fig

    # bins: position along the course, rotation angle, or (sprint) time since the start of the episode
    if time_aligned:
        lo, hi, bw = 0.0, max(float(e.get("duration_s") or 0.0) for e in eps), rec_dt
    else:
        xs = [num((e.get("series") or {}).get("x")) for e in eps if (e.get("series") or {}).get("x")]
        lo = spawn if spawn is not None else min(np.nanmin(x) for x in xs)
        hi = goal if goal is not None else max(np.nanmax(x) for x in xs)
        bw = max(5.0, 2.0 * math.degrees(cmd_speed) * rec_dt) if turn else max(0.1, 2.0 * cmd_speed * rec_dt)
    edges = np.arange(lo, hi + bw, bw)
    if time_aligned:
        edges = edges - 0.5 * bw  # the samples lie at multiples of record_dt: centre the bins on them
    centers = 0.5 * (edges[:-1] + edges[1:])

    def course_marks(axis):
        if time_aligned:
            if brake is not None:
                axis.axvline(brake, color=C_YAW, lw=1.1, ls="-")
            return
        if half is not None:
            axis.axvspan(-half, half, color=C_OBSTACLE, alpha=0.16, lw=0)
        if meta.get("midpoint_x") is not None:
            axis.axvline(meta["midpoint_x"], color=C_SPEED, lw=0.9, ls="--")
        if goal is not None:
            axis.axvline(goal, color=C_OK, lw=1.0, ls="--")
        if spawn is not None:
            axis.axvline(spawn, color=C_GRAY, lw=0.9, ls=":")

    # --- speed vs position ---
    scale = lambda v: sign * v  # noqa: E731
    mean, std, _, cnt = position_stats(eps, channel, edges, transform=scale, xkey=xkey)
    cmd_mean, _, _, _ = position_stats(eps, "cmd_" + channel, edges, transform=scale, xkey=xkey)
    course_marks(ax_sp)
    ax_sp.fill_between(centers, mean - std, mean + std, color=C_SPEED, alpha=0.2, lw=0, label="mean ± std")
    ax_sp.plot(centers, mean, color=C_SPEED, lw=1.8, label="measured")
    ax_sp.plot(centers, cmd_mean, color=C_CMD, ls="--", lw=1.3, label="commanded")
    where = "over time" if time_aligned else ("over the turn" if turn else "along the course")
    notes = [label_note] if label_note else []
    if not time_aligned:
        notes.append(f"command {cmd_speed:.2f} {unit}")
    if half is not None:
        notes.append("grey band = obstacle")
    if time_aligned and brake is not None:
        notes.append("orange line = stop command")
    if not notes:
        notes.append("shaded = spread over episodes")
    style_axes(ax_sp, f"{title_label} {where}", f"{'turn rate' if turn else 'speed'}, {unit}", subtitle="; ".join(notes))
    top = max(1.25 * cmd_speed, float(np.nanmax(mean + std)) if np.isfinite(mean).any() else 0.0)
    ax_sp.set_ylim(min(0.0, 1.1 * float(np.nanmin(mean - std)) if np.isfinite(mean).any() else 0.0), top)
    pad_top(ax_sp, 0.3)  # room for the legend above the curves
    ax_sp.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.9, handlelength=1.6, columnspacing=1.0)
    plt.setp(ax_sp.get_xticklabels(), visible=False)

    # --- speed vs time overlay ---
    for e in eps:
        s = e.get("series") or {}
        if s.get("t"):
            ax_ov.plot(num(s["t"]), primary_series(s, meta)[0], color=OUTCOME_COLORS[outcome(e)], lw=1.0, alpha=0.55)
    with_series = [e["series"] for e in eps if (e.get("series") or {}).get("t")]
    first = max(with_series, key=lambda sr: len(sr["t"])) if with_series else None
    if time_aligned and first is not None:
        ax_ov.plot(num(first["t"]), primary_series(first, meta)[1], color=C_CMD, ls="--", lw=1.3)
        if brake is not None:
            ax_ov.axvline(brake, color=C_YAW, lw=1.1)
    else:
        ax_ov.axhline(cmd_speed, color=C_CMD, ls="--", lw=1.3)
    style_axes(ax_ov, f"{title_label} vs. time, every episode", f"{'turn rate' if turn else 'speed'}, {unit}", "time since episode start, s",
               subtitle="colour = outcome of the episode; dashed = command")  # fmt: skip
    ax_ov.set_ylim(bottom=min(0.0, ax_ov.get_ylim()[0]))

    # --- contacts by position ---
    frac = contact_by_position(eps, edges, xkey)
    has_legs = any("leg_contact" in (e.get("series") or {}) for e in eps)
    course_marks(ax_ct)
    w_bar = bw * (0.45 if has_legs else 0.92)
    ax_ct.bar(centers - (w_bar / 2 if has_legs else 0.0), 100.0 * np.nan_to_num(frac, nan=0.0), width=w_bar, color=C_CONTACT)
    if has_legs:
        frac_legs = contact_by_position(eps, edges, xkey, key="leg_contact")
        ax_ct.bar(centers + w_bar / 2, 100.0 * np.nan_to_num(frac_legs, nan=0.0), width=w_bar, color=C_LEGS)
    ax_ct.set_ylim(0, 105)
    total_trunk = sum(e.get("trunk_contact_events") or 0 for e in eps)
    total_legs = sum(e.get("leg_contact_events") or 0 for e in eps)
    where = "time" if time_aligned else ("angle" if turn else "place")
    style_axes(ax_ct, ("When" if time_aligned else "Where") + " the body touched the surface", "% of episodes",
               subtitle=f"share of the episodes with a contact at this {where}:\nred = trunk (base or head)"
               + (", orange = legs" if has_legs else ""))  # fmt: skip
    summary_txt = f"trunk {total_trunk}" + (f", legs {total_legs}" if has_legs else "") + f" contact event(s) in {episodes_txt}"
    ax_ct.text(0.99, 0.95, summary_txt, transform=ax_ct.transAxes, ha="right", va="top",
               color=C_CONTACT if (total_trunk or total_legs) else C_OK, fontsize=9, fontweight="bold")  # fmt: skip
    plt.setp(ax_ct.get_xticklabels(), visible=False)

    # --- heading error / lateral deviation by position ---
    course_marks(ax_hd)
    ym, _, ymax, _ = position_stats(eps, "yaw_err_deg", edges, xkey=xkey)
    lm, _, lmax, _ = position_stats(eps, "y", edges, transform=np.abs, xkey=xkey)
    if np.isfinite(ym).any():
        ax_hd.plot(centers, ym, color=C_YAW, lw=1.5, label="heading err")
        ax_hd.plot(centers, ymax, color=C_YAW, lw=0.9, ls="--", label="heading err, max")
    side = "drift" if turn else "lateral"
    style_axes(ax_hd, "Drift from the start point" if turn else "Heading error and lateral deviation", "heading error, deg", x_label,
               subtitle="solid = mean over episodes, dashed = max")  # fmt: skip
    twin = ax_hd.twinx()
    twin.plot(centers, lm, color=C_LAT, lw=1.5, label=f"|{side}|")
    twin.plot(centers, lmax, color=C_LAT, lw=0.9, ls="--", label=f"|{side}|, max")
    twin.set_ylabel(f"{side}, m", color=C_LAT)
    if not np.isfinite(ym).any():  # nothing on the left axis (heading is not held in a turn)
        ax_hd.set_yticks([])
        ax_hd.set_ylabel("")
    twin.grid(False)
    h1, l1 = ax_hd.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    pad_top(ax_hd, 0.45)
    pad_top(twin, 0.45)
    ax_hd.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, ncol=2, framealpha=0.9, handlelength=1.6, columnspacing=1.0)

    # --- tilt and power by position ---
    course_marks(ax_tp)
    tm, _, tmax, _ = position_stats(eps, "tilt_deg", edges, xkey=xkey)
    pm, _, _, _ = position_stats(eps, "power_w", edges, xkey=xkey)
    ax_tp.plot(centers, tm, color=C_TILT, lw=1.5, label="tilt")
    ax_tp.plot(centers, tmax, color=C_TILT, lw=0.9, ls="--", label="tilt, max")
    style_axes(ax_tp, "Tilt and mechanical power", "tilt, deg", x_label, subtitle="solid = mean over episodes, dashed = max")
    twin2 = ax_tp.twinx()
    twin2.plot(centers, pm, color=C_GRAY, lw=1.3, label="power")
    twin2.set_ylabel("power, W", color=C_GRAY)
    twin2.grid(False)
    h1, l1 = ax_tp.get_legend_handles_labels()
    h2, l2 = twin2.get_legend_handles_labels()
    pad_top(ax_tp, 0.45)
    pad_top(twin2, 0.45)
    ax_tp.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, ncol=3, framealpha=0.9, handlelength=1.6, columnspacing=1.0)
    ax_sp.set_xlim(edges[0], edges[-1])
    layout(subtitle_lines(fig))  # final gaps for the real number of subtitle lines
    return fig


##
# Whole-run window
##

# scenario families that form a sweep (the old stairs_up_<N>cm names of earlier reports still work)
STAIRS_RE = re.compile(r"^(?:stairs_up|rough_pyramid_stairs(_inv)?)_(\d+)cm$")
SLOPE_RE = re.compile(r"^rough_hf_pyramid_slope_(\d+)deg$")
BOXES_RE = re.compile(r"^rough_boxes_d(\d+)$")
PASS_RATE = 0.8  # a scenario counts as passed at this success rate


def find_sweeps(scenarios: dict) -> dict:
    """Group the scenarios of a report into sweeps: {title: {curve label: [(x, scenario name), ...]}}."""
    stairs: dict[str, list] = {"pyramid": [], "pit (inverted)": []}
    slopes, boxes = [], []
    for k in scenarios:
        if m := STAIRS_RE.match(k):
            stairs["pit (inverted)" if m.group(1) else "pyramid"].append((int(m.group(2)), k))
        elif m := SLOPE_RE.match(k):
            slopes.append((int(m.group(1)), k))
        elif m := BOXES_RE.match(k):
            boxes.append((int(m.group(1)), k))
    sweeps = {}
    if any(len(v) >= 2 for v in stairs.values()):
        sweeps["Stairs: step height, cm"] = {label: sorted(v) for label, v in stairs.items() if len(v) >= 2}
    if len(slopes) >= 2:
        sweeps["Slope: angle, deg"] = {"slope": sorted(slopes)}
    if len(boxes) >= 2:
        sweeps["Random blocks: difficulty, %"] = {"boxes": sorted(boxes)}
    return sweeps


def passed_up_to(points: list, scenarios: dict):
    """Highest x such that every tested x up to it has success >= PASS_RATE (None if even the first one fails)."""
    passed = None
    for x, k in points:
        if scenarios[k]["summary"]["success_rate"] < PASS_RATE:
            break
        passed = x
    return passed


def overall_text(report: dict, scenarios: dict) -> list[str]:
    kv = lambda label, value, width=24: _kv(label, value, width)  # noqa: E731
    eps = [e for sc in scenarios.values() for e in sc["episodes"]]
    n = len(eps)
    per_scenario = {k: sc["summary"] for k, sc in scenarios.items()}
    macro = float(np.mean([s["success_rate"] for s in per_scenario.values()])) if per_scenario else None
    walks = [(get_meta(sc, sc["episodes"]), sc) for sc in scenarios.values()]
    ratios = [  # only the walking scenarios have a meaningful "speed / command" ratio
        e["forward_speed_mps"] / m["primary_command"]
        for m, sc in walks
        if m.get("motion") == "walk" and m.get("primary_command")
        for e in sc["episodes"]
        if e.get("forward_speed_mps") is not None
    ]
    contact_eps = [e for e in eps if e.get("trunk_contact_events") is not None]
    ckpt = report.get("checkpoint", "?")
    run_dir, model = os.path.basename(os.path.dirname(ckpt)), os.path.basename(ckpt)
    ok = [e for e in eps if e.get("success")]
    fell = [e for e in eps if outcome(e) == "FELL"]
    ranked = sorted(per_scenario.items(), key=lambda kv_: (kv_[1]["success_rate"], -kv_[1]["fall_rate"]))
    lines = [
        "CONTROLLER OVERVIEW",
        kv("run", run_dir[-22:]),
        kv("model", model[-22:]),
        kv("task", str(report.get("task", "?"))[-22:]),
        kv("date", str(report.get("date", "?"))),
        kv("seeds", str(report.get("seeds", "?"))[:22]),
        kv("speed", "scenario default" if report.get("speed_override_mps") is None else f"{report['speed_override_mps']} m/s"),
        kv("scenarios / episodes", f"{len(scenarios)} / {n}"),
        "",
        "TASK",
        kv("success (all episodes)", pct(len(ok) / n if n else None)),
        kv("success (scenario mean)", pct(macro)),
        kv("fell before goal", pct(len(fell) / n if n else None)),
        kv("timeout", pct((n - len(ok) - len(fell)) / n if n else None)),
        kv("mean time to goal", fmt(mean_of(ok, "time_to_goal_s"), ".1f", "s")),
        "",
        "SPEED TRACKING",
        kv("speed / command (walks)", pct(float(np.mean(ratios)) if ratios else None)),
        kv("vel error xy", fmt(mean_of(eps, "vel_err_xy_mps"), ".3f", "m/s")),
        kv("vel error yaw", fmt(mean_of(eps, "vel_err_yaw_radps"), ".3f", "rad/s")),
        "",
        "PATH",
        kv("heading err mean/worst", f"{fmt(mean_of(eps, 'max_yaw_err_deg'), '.1f')} / {fmt(worst_of(eps, 'max_yaw_err_deg'), '.1f')} deg"),
        kv("lateral mean/worst", f"{fmt(mean_of(eps, 'max_lateral_m'), '.2f')} / {fmt(worst_of(eps, 'max_lateral_m'), '.2f')} m"),
        "",
        "BODY CONTACTS (events / ep.)",
        kv("trunk (base + head)", fmt(mean_of(eps, "trunk_contact_events"), ".2f")),
        kv("  head (nose)", fmt(mean_of(eps, "head_contact_events"), ".2f")),
        kv("legs (hip/thigh/calf)", fmt(mean_of(eps, "leg_contact_events"), ".2f")),
        kv("ep. with trunk contact", pct(sum(e["trunk_contact_events"] > 0 for e in contact_eps) / len(contact_eps) if contact_eps else None)),
        "",
        "STABILITY / EFFORT",
        kv("tilt mean/worst", f"{fmt(mean_of(eps, 'mean_tilt_deg'), '.1f')} / {fmt(worst_of(eps, 'max_tilt_deg'), '.1f')} deg"),
        kv("mean power", fmt(mean_of(eps, "power_w"), ".0f", "W")),
        kv("cost of transport", fmt(mean_of(eps, "cost_of_transport"), ".2f")),
        kv("action rate", fmt(mean_of(eps, "action_rate"), ".2f")),
        kv("joint acceleration", fmt(mean_of(eps, "joint_acc"), ".0f")),
        kv("mean total reward", fmt(mean_of(eps, "total_reward"), ".1f")),
        "",
    ]
    sweeps = find_sweeps(scenarios)
    if sweeps:
        lines.append("SWEEPS (passed: success >= 80 %)" if PASS_RATE == 0.8 else f"SWEEPS (passed: success >= {PASS_RATE:.0%})")
        for title, curves in sweeps.items():
            for label, points in curves.items():
                passed = passed_up_to(points, scenarios)
                unit = title.rsplit(", ", 1)[1]
                tested = f"({points[0][0]}..{points[-1][0]})"
                lines.append(f"  {label[:16]:<16}" + (f"up to {passed} {unit} " if passed is not None else "none passed ") + tested)
        lines.append("")
    for m, sc in walks:
        if m.get("motion") == "sprint_brake":
            sm = sc["summary"]
            lines += [
                "SPRINT AND STOP",
                kv("peak speed", fmt(sm.get("mean_peak_speed_mps"), ".2f", "m/s")),
                kv("stopped (episodes)", pct(sm.get("stopped_rate"))),
                kv("stop time / dist.", f"{fmt(sm.get('mean_stop_time_s'), '.2f')} s / {fmt(sm.get('mean_stop_distance_m'), '.2f')} m"),
                kv("lowest speed after", fmt(sm.get("lowest_speed_after_brake_mps"), ".2f", "m/s")),
                "",
            ]
            break
    lines.append("WEAKEST SCENARIOS (success / fell)")
    for k, s in ranked[:5]:
        short = k if len(k) <= 30 else k[:29] + "~"
        lines.append(f"  {short:<30}{100 * s['success_rate']:>4.0f}%{100 * s['fall_rate']:>5.0f}%")
    return lines


def overall_figure(plt, report: dict, scenarios: dict):
    """Columns: outcomes | speed tracking | body contacts (one row per scenario, full height), then the
    capability sweeps (or the milestone) and the text panel."""
    names = list(scenarios)
    nS = len(names)
    fs = 6.5 if nS > 30 else (7.5 if nS > 14 else 8.5)  # font of the scenario names
    fig = new_figure(plt, 16.5, float(np.clip(0.16 * nS + 2.4, 7.6, 18.0)))
    width, height = fig.get_size_inches()
    sweeps = find_sweeps(scenarios)

    # widths in inches: scenario names, three bar charts, the sweep column, the text panel
    names_in = min(3.0, max(len(n) for n in names) * fs * 0.62 / 72 + 0.15)
    text_in, gap = 3.55, 0.12
    right_col_in = 3.0 if sweeps else 2.2
    bars_in = (width - names_in - text_in - right_col_in - 0.35 - 4 * gap) / 3
    top, bottom = 1.0 - 1.15 / height, 0.62 / height

    def col(x_in, w_in):
        return x_in / width, w_in / width

    x = names_in + 0.1
    axes_bars = []
    for i in range(3):
        cx, cw = col(x, bars_in)
        axes_bars.append(fig.add_axes([cx, bottom, cw, top - bottom], sharey=axes_bars[0] if axes_bars else None))
        x += bars_in + gap
    ax_out, ax_trk, ax_cnt = axes_bars
    x += 0.25
    right_x, right_w = col(x, right_col_in)
    tx, tw = col(width - text_in, text_in - 0.05)
    ax_tx = fig.add_axes([tx, bottom, tw, top - bottom])
    # the milestone chart shares the y axis too: it must exist before the scenario names are set
    ax_mid = None if sweeps else fig.add_axes([right_x, bottom, right_w, top - bottom], sharey=ax_out)
    fig.suptitle("Controller evaluation: whole run", fontsize=13, fontweight="bold", color="#1f4e8c",
                 x=0.1 / width, y=1.0 - 0.15 / height, ha="left")  # fmt: skip
    set_window_title(fig, "Whole run - controller evaluation")
    lines = overall_text(report, scenarios)
    avail_h_pt = (top - bottom) * height * 72
    longest = max(len(line) for line in lines)
    size = min(8.0, avail_h_pt / (len(lines) * 1.2), (text_in - 0.1) * 72 / (longest * 0.62))
    text_panel(ax_tx, lines, size=size)

    y = np.arange(nS)
    S = [scenarios[k]["summary"] for k in names]

    def scenario_axis(ax, labels: bool):
        """The charts share one y axis: ticks, names and order are set once, the other charts only hide them."""
        if labels:
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=fs)
            ax.set_ylim(max(nS, 4) - 0.5, -0.5)  # top to bottom; with few scenarios the bars stay thin
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.tick_params(axis="x", labelsize=7.5)

    # --- outcomes ---
    succ = np.array([s_["success_rate"] for s_ in S]) * 100
    fall = np.array([s_["fall_rate"] for s_ in S]) * 100
    tmo = np.array([s_["timeout_rate"] for s_ in S]) * 100
    ax_out.barh(y, succ, color=C_OK, height=0.72)
    ax_out.barh(y, fall, left=succ, color=C_FALL, height=0.72)
    ax_out.barh(y, tmo, left=succ + fall, color=C_TIMEOUT, height=0.72)
    ax_out.set_xlim(0, 100)
    scenario_axis(ax_out, True)
    style_axes(ax_out, "Outcomes", "", "% of episodes", subtitle="green = success, red = fell,\namber = timeout")

    # --- speed tracking ---
    ratio = []
    for k in names:
        meta = get_meta(scenarios[k], scenarios[k]["episodes"])
        cmd = meta.get("primary_command")
        v = scenarios[k]["summary"].get("mean_forward_speed_mps")
        walk = meta.get("motion") == "walk"  # a turn or a sprint-and-stop has no meaningful mean speed
        ratio.append(np.nan if not walk or not cmd or v is None else 100.0 * v / cmd)
    ax_trk.barh(y, np.nan_to_num(ratio), color=C_SPEED, height=0.72)
    ax_trk.axvline(100, color=C_CMD, ls="--", lw=1.2)
    ax_trk.set_xlim(0, max(105.0, float(np.nanmax(np.nan_to_num(ratio))) * 1.05))
    for yi, s_, r in zip(y, S, ratio):
        e = s_.get("mean_vel_err_xy_mps")
        if not np.isfinite(r):
            ax_trk.text(2, yi, "n/a", va="center", ha="left", fontsize=fs - 1, color="#555555")
        elif e is not None:  # inside the bar if it is long enough, otherwise right of it
            inside = r > 45
            ax_trk.text(2 if inside else r + 2, yi, f"err {e:.2f}", va="center", ha="left", fontsize=fs - 1,
                        color="white" if inside else "#333333")  # fmt: skip
    scenario_axis(ax_trk, False)
    style_axes(ax_trk, "Speed tracking", "", "% of the commanded speed", subtitle="mean speed / command;\nerr = velocity error, m/s")

    # --- body contacts: trunk (upper bar) and legs (lower bar) per scenario ---
    events = [s_.get("mean_trunk_contacts_per_episode") for s_ in S]
    legs = [s_.get("mean_leg_contacts_per_episode") for s_ in S]
    rate = [s_.get("trunk_contact_episode_rate") for s_ in S]
    has_legs = any(v is not None for v in legs)
    h = 0.36 if has_legs else 0.72
    ax_cnt.barh(y - (h / 2 if has_legs else 0.0), [e or 0.0 for e in events], color=C_CONTACT, height=h)
    if has_legs:
        ax_cnt.barh(y + h / 2, [v or 0.0 for v in legs], color=C_LEGS, height=h)
    for yi, e, r in zip(y, events, rate):
        if e is not None and e > 0:
            ax_cnt.text(e, yi - (h / 2 if has_legs else 0.0), f" {pct(r)}", va="center", ha="left", fontsize=fs - 1, color="#333333")
    scenario_axis(ax_cnt, False)
    style_axes(ax_cnt, "Body contacts", "", "events per episode",
               subtitle="red = trunk (base + head), orange = legs;\n% = episodes with a trunk contact")  # fmt: skip
    ax_cnt.set_xlim(0, max(1.0, max([e or 0.0 for e in events] + [v or 0.0 for v in legs] + [0.0])) * 1.45)

    # --- right column: capability sweeps (stairs height, slope angle, block difficulty) or the milestone ---
    if sweeps:
        n_sw = len(sweeps)
        slot = (top - bottom) / n_sw
        palette = [C_OK, C_SPEED, C_YAW]
        for row, (title, curves) in enumerate(sweeps.items()):
            y0 = top - (row + 1) * slot
            ax = fig.add_axes([right_x, y0 + 0.62 / height, right_w, slot - 1.25 / height])
            xs_all = []
            for c, (label, points) in enumerate(curves.items()):
                xs = [x_ for x_, _ in points]
                xs_all += xs
                ok = [scenarios[k]["summary"]["success_rate"] * 100 for _, k in points]
                ax.plot(xs, ok, "os"[c % 2] + "-", color=palette[c % 3], lw=1.8, ms=4, label=label)
                ax.plot(xs, [scenarios[k]["summary"]["fall_rate"] * 100 for _, k in points], ":", color=palette[c % 3], lw=1.1, alpha=0.8)
            ax.set_ylim(-5, 105)
            ax.set_xticks(sorted(set(xs_all)))
            ax.tick_params(labelsize=7)
            style_axes(ax, title.split(":")[0], "%", title.split(": ")[1], subtitle="solid = success, dotted = fell")
            if len(curves) > 1:
                ax.legend(loc="lower left", fontsize=6.5, framealpha=0.85, handlelength=1.4)
    else:
        has_mid = [get_meta(scenarios[k], scenarios[k]["episodes"]).get("midpoint_x") is not None for k in names]
        mid = np.array([s_["midpoint_rate"] if h else 0.0 for s_, h in zip(S, has_mid)]) * 100
        ax_mid.barh(y, mid, color=C_SPEED, height=0.72)
        for yi, h in zip(y, has_mid):
            if not h:  # flat ground, turn, sprint: no milestone on the course
                ax_mid.text(2, yi, "n/a", va="center", ha="left", fontsize=fs - 1, color="#555555")
        ax_mid.set_xlim(0, 100)
        scenario_axis(ax_mid, False)
        style_axes(ax_mid, "Midpoint reached", "", "% of episodes", subtitle="top of a pyramid, bottom\nof a pit, centre of blocks")
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
    ap.add_argument(
        "--dpi", type=float, default=None,
        help="Scale of the windows (default: fitted to the screen, at most 100). Lower = smaller windows.",
    )  # fmt: skip
    args = ap.parse_args(argv)
    SCREEN["dpi"] = args.dpi

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
    if show:
        size = screen_size()
        if size:
            SCREEN["w"], SCREEN["h"] = size
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