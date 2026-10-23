"""Tk GUI for evaluate.py.

Runs BEFORE Isaac Sim is launched (this module never imports isaaclab): it only collects settings and
returns the list of command-line arguments for evaluate.py, so the GUI path and the terminal path
are the same code downstream.

The list of scenarios and tasks lives inside Isaac Sim-dependent code, so it is obtained once by a short
headless child process and cached in ~/.cache/unitree_rl_lab/eval_catalog.json ("Update list" refreshes
it, e.g. after editing eval_scenarios.py or ROUGH_TERRAINS_CFG).

Saved configurations are JSON files (default folder: eval_configs/ next to this file). The last used
settings are stored in eval_configs/_last.json and restored on the next start.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_HERE, "eval_configs")
LAST_CONFIG = os.path.join(CONFIG_DIR, "_last.json")
CATALOG_PATH = os.path.join(os.path.expanduser("~"), ".cache", "unitree_rl_lab", "eval_catalog.json")

# --- look ---
BG = "#dbe8f7"  # window background: light blue
BG_BUTTON = "#c3d7ee"
BG_ACTIVE = "#adc8e6"
BORDER = "#9fb8d4"
FIELD = "#ffffff"
FIELD_READONLY = "#eef4fb"
FG = "#1d2b3a"
FG_HINT = "#5b6f86"
FG_TITLE = "#1f4e8c"
ACCENT = "#2c6db4"
ACCENT_ACTIVE = "#245a96"
SELECT = "#3f7fc4"

# The first available family is used. Set EVAL_GUI_FONT to force another one
# (list the installed ones with: python -c "import tkinter, tkinter.font as f; r = tkinter.Tk(); print(sorted(f.families()))").
FONT_PREFERENCE = ("DejaVu Sans", "Noto Sans", "Liberation Sans", "Ubuntu", "Cantarell", "Helvetica", "Arial", "Lucida Sans")
FONT_SIZE = 10

# Language of Tk's built-in dialogs (file chooser, message boxes). Tk takes it from the system locale by
# default, so without this they would follow e.g. a Russian system. Only the Tcl interpreter of this window
# is affected: no environment variables are changed, so the child processes (Isaac Sim) are not touched.
DIALOG_LOCALE = "en"

DEFAULT_TASK = "Unitree-Go2-Velocity"
MODES = ("headless", "realtime")

DEFAULTS: dict = {
    "task": DEFAULT_TASK,
    "scenarios": [],
    "episodes": 5,
    "seeds": "42",
    "speed": "",  # commanded forward speed in m/s; empty = the scenario's own (0.5)
    "checkpoint": "",
    "load_run": "",
    "output": "",
    "mode": "headless",
    "device": "",
    "extra_args": "",
    "keep_going_after_contact": False,  # --no_contact_termination
    "show_report": True,  # open the report viewer when the runs are finished (False: --no_report)
    "save_plots": False,  # save the figures as PNG (--report_save)
    "plots_dir": "",  # folder for the PNG files; empty = next to the JSON report, named like it
}


##
# Pure logic (no widgets): easy to test
##


def parse_seeds(text: str) -> list[int] | None:
    """'42, 43,44' -> [42, 43, 44]; None if the text is not a comma-separated list of integers."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def parse_episodes(value) -> int | None:
    try:
        n = int(str(value).strip())
    except ValueError:
        return None
    return n if n >= 1 else None


def parse_speed(value) -> float | None:
    """Positive float, or None if the text is empty / not a positive number."""
    try:
        speed = float(str(value).strip())
    except ValueError:
        return None
    return speed if speed > 0.0 else None


def validate(cfg: dict) -> str | None:
    """Return an error message, or None if the configuration can be run."""
    if not cfg["task"].strip():
        return "No task specified."
    if not cfg["scenarios"]:
        return "No scenario selected."
    if parse_episodes(cfg["episodes"]) is None:
        return "Episodes per scenario must be an integer >= 1."
    if parse_seeds(cfg["seeds"]) is None:
        return "Seeds must be integers separated by commas, e.g. 42,43,44."
    if str(cfg["speed"]).strip() and parse_speed(cfg["speed"]) is None:
        return "Speed must be a positive number in m/s (or empty for the scenario default)."
    ckpt = cfg["checkpoint"].strip()
    if ckpt and not os.path.isfile(os.path.expanduser(ckpt)):
        return f"Checkpoint file not found:\n{ckpt}"
    if cfg["mode"] not in MODES:
        return "No run mode selected."
    try:
        shlex.split(cfg["extra_args"])
    except ValueError as e:
        return f"Cannot parse the extra arguments: {e}"
    return None


def build_argv(cfg: dict, order: list[str]) -> list[str]:
    """Arguments for evaluate.py (without the program name). Scenarios keep the registry order."""
    chosen = set(cfg["scenarios"])
    names = [n for n in order if n in chosen] + [n for n in cfg["scenarios"] if n not in order]
    argv = [
        "--task", cfg["task"].strip(),
        "--scenarios", ",".join(names),
        "--episodes", str(parse_episodes(cfg["episodes"]) or cfg["episodes"]),
        "--eval_seed", ",".join(str(s) for s in (parse_seeds(cfg["seeds"]) or [])) or str(cfg["seeds"]),
    ]  # fmt: skip
    if str(cfg["speed"]).strip():
        argv += ["--speed", str(parse_speed(cfg["speed"]) or cfg["speed"]).strip()]
    if cfg["keep_going_after_contact"]:
        argv.append("--no_contact_termination")
    if not cfg["show_report"]:
        argv.append("--no_report")
    if cfg["save_plots"]:  # without a folder evaluate.py uses <report name>/ next to the JSON report
        folder = cfg["plots_dir"].strip()
        argv += ["--report_save", os.path.expanduser(folder)] if folder else ["--report_save"]
    ckpt, run = cfg["checkpoint"].strip(), cfg["load_run"].strip()
    if ckpt:
        argv += ["--checkpoint", os.path.expanduser(ckpt)]
    elif run:  # the run is only used when no explicit checkpoint is given
        argv += ["--load_run", run]
    if cfg["output"].strip():
        argv += ["--output", os.path.expanduser(cfg["output"].strip())]
    argv.append("--headless" if cfg["mode"] == "headless" else "--real-time")
    if cfg["device"].strip():
        argv += ["--device", cfg["device"].strip()]
    try:
        argv += shlex.split(cfg["extra_args"])
    except ValueError:
        pass
    return argv


def matches_filter(name: str, text: str) -> bool:
    """Empty text matches everything. Terms are separated by comma/space; a term with * ? [ is a glob,
    any other term is a case-insensitive substring."""
    terms = [t for t in re.split(r"[,\s]+", text.strip().lower()) if t]
    if not terms:
        return True
    low = name.lower()
    return any(fnmatch.fnmatchcase(low, t) if any(c in t for c in "*?[") else t in low for t in terms)


def load_catalog(path: str | None = None) -> dict | None:
    try:
        with open(path or CATALOG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("scenarios"), dict):
            return data
    except (OSError, ValueError):
        pass
    return None


def discover_runs(root: str = os.path.join("logs", "rsl_rl")) -> list[str]:
    """Run folders found under logs/rsl_rl/<experiment>/ (relative to the current directory), newest first."""
    runs = {os.path.basename(p) for p in glob.glob(os.path.join(root, "*", "*")) if os.path.isdir(p)}
    return sorted(runs, reverse=True)


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    cfg = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in raw:
            cfg[key] = raw[key]
    cfg["scenarios"] = [str(s) for s in cfg["scenarios"]]
    for key in ("task", "seeds", "speed", "checkpoint", "load_run", "output", "device", "extra_args", "plots_dir"):
        cfg[key] = str(cfg[key])
    for key in ("keep_going_after_contact", "show_report", "save_plots"):
        cfg[key] = bool(cfg[key])
    if cfg["mode"] not in MODES:
        cfg["mode"] = "headless"
    return cfg


def write_config(path: str, cfg: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"version": 1, **cfg}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


##
# Window
##


class EvalGui:
    def __init__(self, root: tk.Tk, list_cmd: Callable[[str], list[str]]):
        self.root = root
        self.list_cmd = list_cmd
        self.result: list[str] | None = None

        self.catalog: dict = {"scenarios": {}, "tasks": [], "created": None}
        self.order: list[str] = []
        self.visible: list[str] = []
        self.selected: set[str] = set()
        self.config_path: str | None = None

        self._queue: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._refreshing = False

        self._set_dialog_locale(root)
        self.bold_font = self._setup_look(root)
        root.title("Go2: policy evaluation")
        root.minsize(900, 520)
        root.protocol("WM_DELETE_WINDOW", self._cancel)

        names = ("task", "episodes", "seeds", "speed", "checkpoint", "load_run", "output", "plots_dir", "mode", "device", "extra_args", "filter")
        self.v = {k: tk.StringVar() for k in names}
        self.b = {k: tk.BooleanVar() for k in ("keep_going_after_contact", "show_report", "save_plots")}
        self.desc_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.cmd_var = tk.StringVar()
        self.cfg_name_var = tk.StringVar(value="Configuration: not saved")

        self._build()

        catalog = load_catalog()
        if catalog:
            self._set_catalog(catalog)
        last = None
        if os.path.isfile(LAST_CONFIG):
            try:
                last = read_config(LAST_CONFIG)
            except (OSError, ValueError):
                last = None
        self._apply_cfg(last or DEFAULTS, warn_unknown=False)

        for var in [*self.v.values(), *self.b.values()]:
            var.trace_add("write", self._on_change)
        self._on_change()

        if not catalog:
            root.after(200, self.refresh_catalog)

    ## ---- look ------------------------------------------------------------------------------------

    @staticmethod
    def _set_dialog_locale(root: tk.Tk):
        """Make Tk's built-in dialogs (Open/Save, message boxes) use DIALOG_LOCALE instead of the system locale."""
        try:
            root.tk.call("::msgcat::mclocale", DIALOG_LOCALE)
        except tk.TclError:
            pass  # cosmetic only

    @staticmethod
    def _setup_look(root: tk.Tk):
        """Readable fonts and a light-blue theme. Returns the bold font used for the summary line."""
        # fonts: the Tk default is a poor bitmap font on many Linux installs
        available = {f.lower(): f for f in tkfont.families(root)}
        wanted = os.environ.get("EVAL_GUI_FONT")
        family = available.get(wanted.lower(), wanted) if wanted else next(
            (available[f.lower()] for f in FONT_PREFERENCE if f.lower() in available), "sans-serif"
        )
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont", "TkTooltipFont"):
            tkfont.nametofont(name).configure(family=family, size=FONT_SIZE)
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold", size=FONT_SIZE + 1)

        # colours
        style = ttk.Style(root)
        style.theme_use("clam")
        root.configure(background=BG)
        style.configure(
            ".", background=BG, foreground=FG, fieldbackground=FIELD, bordercolor=BORDER,
            lightcolor=BG, darkcolor=BG, troughcolor=BG_BUTTON, focuscolor=BG,
        )  # fmt: skip
        style.configure("TLabelframe", background=BG, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG, foreground=FG_TITLE, font=bold)
        style.configure("TButton", background=BG_BUTTON, foreground=FG, bordercolor=BORDER, padding=(8, 3))
        style.map("TButton", background=[("pressed", BG_ACTIVE), ("active", BG_ACTIVE)], foreground=[("disabled", FG_HINT)])
        style.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT_ACTIVE, padding=(14, 3), font=bold)
        style.map("Accent.TButton", background=[("pressed", ACCENT_ACTIVE), ("active", ACCENT_ACTIVE)])
        for widget in ("TEntry", "TCombobox", "TSpinbox"):
            style.configure(widget, fieldbackground=FIELD, background=BG_BUTTON, foreground=FG, bordercolor=BORDER, padding=3)
        style.map("TEntry", fieldbackground=[("readonly", FIELD_READONLY)])
        style.map(
            "TCombobox",
            fieldbackground=[("disabled", FIELD_READONLY), ("readonly", FIELD)],
            foreground=[("disabled", FG_HINT)],
            background=[("active", BG_ACTIVE)],
        )
        style.map("TSpinbox", background=[("active", BG_ACTIVE)])
        for widget in ("TRadiobutton", "TCheckbutton"):
            style.configure(widget, background=BG)
            style.map(
                widget,
                background=[("active", BG)],
                indicatorcolor=[("selected", ACCENT), ("!selected", FIELD)],
            )
        style.configure("TScrollbar", background=BG_BUTTON, troughcolor=BG, bordercolor=BORDER, arrowcolor=FG)
        style.map("TScrollbar", background=[("active", BG_ACTIVE)])
        style.configure("Hint.TLabel", foreground=FG_HINT)
        style.configure("Summary.TLabel", foreground=FG_TITLE, font=bold)
        # drop-down lists of the comboboxes are plain Tk listboxes
        root.option_add("*TCombobox*Listbox.background", FIELD)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", SELECT)
        root.option_add("*TCombobox*Listbox.selectForeground", "white")
        return bold

    ## ---- layout ---------------------------------------------------------------------------------

    def _build(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        pad = {"padx": 6, "pady": 3}

        # --- top bar: configurations ---
        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Button(top, text="Load configuration...", command=self._load_config).pack(side="left")
        ttk.Button(top, text="Save configuration...", command=self._save_config).pack(side="left", padx=6)
        ttk.Label(top, textvariable=self.cfg_name_var, style="Hint.TLabel").pack(side="left", padx=10)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", **pad)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # --- left: scenarios ---
        left = ttk.LabelFrame(body, text="Scenarios")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        frow = ttk.Frame(left)
        frow.grid(row=0, column=0, sticky="ew", **pad)
        frow.columnconfigure(1, weight=1)
        ttk.Label(frow, text="Filter:").grid(row=0, column=0)
        ttk.Entry(frow, textvariable=self.v["filter"]).grid(row=0, column=1, sticky="ew", padx=4)

        presets = ttk.Frame(left)
        presets.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Label(presets, text="Quick:").pack(side="left")
        for label, pattern in (("all", ""), ("flat", "flat*"), ("stairs", "rough_pyramid_stairs*"), ("slopes", "*slope*"), ("boxes", "rough_boxes*")):
            ttk.Button(presets, text=label, width=8, command=lambda p=pattern: self.v["filter"].set(p)).pack(side="left", padx=2)

        lbox = ttk.Frame(left)
        lbox.grid(row=2, column=0, sticky="nsew", **pad)
        lbox.columnconfigure(0, weight=1)
        lbox.rowconfigure(0, weight=1)
        self.lb = tk.Listbox(
            lbox,
            selectmode="multiple",
            exportselection=False,
            activestyle="none",
            height=18,
            background=FIELD,
            foreground=FG,
            selectbackground=SELECT,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.lb.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lbox, orient="vertical", command=self.lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.lb.configure(yscrollcommand=sb.set)
        self.lb.bind("<<ListboxSelect>>", self._on_list_select)
        self.lb.bind("<Motion>", self._on_motion)

        btns = ttk.Frame(left)
        btns.grid(row=3, column=0, sticky="ew", **pad)
        ttk.Button(btns, text="Select visible", command=self._select_visible).pack(side="left")
        ttk.Button(btns, text="Deselect visible", command=self._deselect_visible).pack(side="left", padx=4)
        ttk.Button(btns, text="Deselect all", command=self._deselect_all).pack(side="left")
        self.refresh_btn = ttk.Button(btns, text="Refresh list", command=self.refresh_catalog)
        self.refresh_btn.pack(side="right")

        ttk.Label(left, textvariable=self.desc_var, wraplength=420, style="Hint.TLabel").grid(row=4, column=0, sticky="ew", **pad)
        ttk.Label(left, textvariable=self.status_var, style="Hint.TLabel").grid(row=5, column=0, sticky="ew", **pad)

        # --- right: parameters ---
        right = ttk.LabelFrame(body, text="Run settings")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)
        r = 0

        def label(text, hint=None):
            ttk.Label(right, text=text).grid(row=r, column=0, sticky="w", **pad)
            if hint:
                ttk.Label(right, text=hint, style="Hint.TLabel").grid(row=r + 1, column=1, columnspan=2, sticky="w", padx=6)

        label("Task:")
        self.task_cb = ttk.Combobox(right, textvariable=self.v["task"], values=[DEFAULT_TASK])
        self.task_cb.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        r += 1

        label("Episodes per scenario:")
        ttk.Spinbox(right, from_=1, to=1000, textvariable=self.v["episodes"], width=8).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        label("Seeds:", "comma-separated; several seeds are pooled into one result")
        ttk.Entry(right, textvariable=self.v["seeds"]).grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        r += 2

        label("Speed (m/s):", "walking speed; empty = scenario default; ignored by flat_turn_360 and flat_max_speed")
        ttk.Entry(right, textvariable=self.v["speed"], width=10).grid(row=r, column=1, sticky="w", **pad)
        r += 2

        label("Checkpoint:", "empty: the load run or the latest model is used")
        ttk.Entry(right, textvariable=self.v["checkpoint"]).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(right, text="Browse...", width=9, command=self._browse_checkpoint).grid(row=r, column=2, **pad)
        r += 2

        label("Load run:", "used when no checkpoint is set; empty = latest run")
        self.run_cb = ttk.Combobox(right, textvariable=self.v["load_run"], values=discover_runs())
        self.run_cb.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        r += 2

        label("Report file (JSON):", "empty: <checkpoint dir>/eval/eval_<model>_<time>.json")
        ttk.Entry(right, textvariable=self.v["output"]).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(right, text="Choose...", width=9, command=self._browse_output).grid(row=r, column=2, **pad)
        r += 2

        label("Plots (PNG):", "empty folder: next to the JSON report, in a folder with the same name")
        plots = ttk.Frame(right)
        plots.grid(row=r, column=1, sticky="ew", **pad)
        plots.columnconfigure(1, weight=1)
        ttk.Checkbutton(plots, text="Save", variable=self.b["save_plots"]).grid(row=0, column=0, padx=(0, 6))
        self.plots_entry = ttk.Entry(plots, textvariable=self.v["plots_dir"])
        self.plots_entry.grid(row=0, column=1, sticky="ew")
        self.plots_btn = ttk.Button(right, text="Folder...", width=9, command=self._browse_plots)
        self.plots_btn.grid(row=r, column=2, **pad)
        r += 2

        label("Run mode:")
        modes = ttk.Frame(right)
        modes.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Radiobutton(modes, text="Headless (no window, fast)", value="headless", variable=self.v["mode"]).pack(anchor="w")
        ttk.Radiobutton(modes, text="Real-time (simulator window)", value="realtime", variable=self.v["mode"]).pack(anchor="w")
        r += 1

        label("Options:")
        opts = ttk.Frame(right)
        opts.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(opts, text="Keep going after base contact (do not end the episode)", variable=self.b["keep_going_after_contact"]).pack(anchor="w")
        ttk.Checkbutton(opts, text="Show the report windows when finished", variable=self.b["show_report"]).pack(anchor="w")
        r += 1

        label("Device:", "empty = default")
        ttk.Combobox(right, textvariable=self.v["device"], values=["", "cuda:0", "cuda:1", "cpu"]).grid(
            row=r, column=1, columnspan=2, sticky="ew", **pad
        )
        r += 2

        label("Extra arguments:", "e.g. --num_envs 4 --disable_fabric")
        ttk.Entry(right, textvariable=self.v["extra_args"]).grid(row=r, column=1, columnspan=2, sticky="ew", **pad)

        # --- bottom: summary, command, run ---
        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, sticky="ew", **pad)
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.summary_var, style="Summary.TLabel").grid(row=0, column=0, sticky="w")

        cmd = ttk.Frame(bottom)
        cmd.grid(row=1, column=0, sticky="ew", pady=3)
        cmd.columnconfigure(1, weight=1)
        ttk.Label(cmd, text="Equivalent command:").grid(row=0, column=0)
        ttk.Entry(cmd, textvariable=self.cmd_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(cmd, text="Copy", command=self._copy_command).grid(row=0, column=2)

        actions = ttk.Frame(bottom)
        actions.grid(row=0, column=1, rowspan=2, sticky="e", padx=(10, 0))
        ttk.Button(actions, text="Run", style="Accent.TButton", command=self._run).pack(side="left")
        ttk.Button(actions, text="Cancel", command=self._cancel).pack(side="left", padx=4)

    ## ---- catalog ----------------------------------------------------------------------------------

    def refresh_catalog(self):
        if self._refreshing:
            return
        self._refreshing = True
        self.refresh_btn.state(["disabled"])
        self.status_var.set("Loading the scenario list (starts Isaac Sim, 15-40 s)...")
        tmp = os.path.join(tempfile.mkdtemp(prefix="go2_catalog_"), "catalog.json")
        try:
            self._proc = subprocess.Popen(self.list_cmd(tmp), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as e:
            self._finish_refresh(("error", str(e)))
            return

        proc = self._proc

        def work():
            out, _ = proc.communicate()
            if os.path.exists(tmp):
                self._queue.put(("ok", tmp))
            else:
                self._queue.put(("error", "\n".join((out or "").splitlines()[-15:]) or f"exit code {proc.returncode}"))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(300, self._poll)

    def _poll(self):
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(300, self._poll)
            return
        self._finish_refresh(item)

    def _finish_refresh(self, item):
        self._refreshing = False
        self._proc = None
        self.refresh_btn.state(["!disabled"])
        kind, payload = item
        if kind == "ok":
            try:
                os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
                shutil.copyfile(payload, CATALOG_PATH)
                catalog = load_catalog(CATALOG_PATH)
            except OSError:
                catalog = load_catalog(payload)
            if catalog:
                removed = self._set_catalog(catalog)
                if removed:
                    messagebox.showinfo("List updated", "These selected scenarios no longer exist and were deselected:\n" + "\n".join(removed))
                return
            payload = "The scenario list file could not be read."
        self.status_var.set("Could not get the scenario list.")
        messagebox.showerror("Scenario list", f"Could not get the scenario list:\n\n{payload}")

    def _set_catalog(self, catalog: dict) -> list[str]:
        """Install a catalog; returns the selected scenarios that no longer exist (and were deselected)."""
        self.catalog = catalog
        self.order = list(catalog["scenarios"])
        tasks = catalog.get("tasks") or [DEFAULT_TASK]
        self.task_cb.configure(values=tasks)
        removed = sorted(s for s in self.selected if s not in catalog["scenarios"])
        self.selected -= set(removed)
        self.status_var.set(f"Scenarios in the list: {len(self.order)} (updated {catalog.get('created') or '?'})")
        self._apply_filter()
        self._on_change()
        return removed

    ## ---- scenario list -----------------------------------------------------------------------------

    def _apply_filter(self):
        text = self.v["filter"].get()
        self.visible = [n for n in self.order if matches_filter(n, text)]
        self.lb.delete(0, "end")
        for name in self.visible:
            self.lb.insert("end", name)
        self._show_selection()

    def _show_selection(self):
        self.lb.selection_clear(0, "end")
        for i, name in enumerate(self.visible):
            if name in self.selected:
                self.lb.selection_set(i)

    def _on_list_select(self, _event=None):
        current = set(self.lb.curselection())
        for i, name in enumerate(self.visible):
            if i in current:
                self.selected.add(name)
            else:
                self.selected.discard(name)
        self._on_change()

    def _on_motion(self, event):
        i = self.lb.nearest(event.y)
        if 0 <= i < len(self.visible):
            name = self.visible[i]
            self.desc_var.set(f"{name}: {self.catalog['scenarios'].get(name, '')}")

    def _select_visible(self):
        self.selected |= set(self.visible)
        self._show_selection()
        self._on_change()

    def _deselect_visible(self):
        self.selected -= set(self.visible)
        self._show_selection()
        self._on_change()

    def _deselect_all(self):
        self.selected.clear()
        self._show_selection()
        self._on_change()

    ## ---- configuration -----------------------------------------------------------------------------

    def _collect(self) -> dict:
        v = {k: var.get() for k, var in self.v.items()}
        scenarios = [n for n in self.order if n in self.selected] + sorted(n for n in self.selected if n not in self.order)
        return {
            "task": v["task"].strip(),
            "scenarios": scenarios,
            "episodes": v["episodes"].strip(),
            "seeds": v["seeds"].strip(),
            "speed": v["speed"].strip(),
            "checkpoint": v["checkpoint"].strip(),
            "load_run": v["load_run"].strip(),
            "output": v["output"].strip(),
            "mode": v["mode"] or "headless",
            "device": v["device"].strip(),
            "extra_args": v["extra_args"].strip(),
            "keep_going_after_contact": bool(self.b["keep_going_after_contact"].get()),
            "show_report": bool(self.b["show_report"].get()),
            "save_plots": bool(self.b["save_plots"].get()),
            "plots_dir": v["plots_dir"].strip(),
        }

    def _apply_cfg(self, cfg: dict, warn_unknown: bool = True):
        cfg = {**DEFAULTS, **cfg}
        wanted = [str(s) for s in cfg["scenarios"]]
        unknown = [s for s in wanted if self.order and s not in self.catalog["scenarios"]]
        self.selected = {s for s in wanted if s not in unknown}
        for key in ("task", "episodes", "seeds", "speed", "checkpoint", "load_run", "output", "plots_dir", "mode", "device", "extra_args"):
            self.v[key].set(str(cfg[key]))
        for key in self.b:
            self.b[key].set(bool(cfg[key]))
        self._apply_filter()
        self._on_change()
        if unknown and warn_unknown:
            messagebox.showwarning("Configuration", "These scenarios are not in the current list and were skipped:\n" + "\n".join(unknown))

    def _save_config(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save configuration",
            initialdir=CONFIG_DIR,
            initialfile=os.path.basename(self.config_path) if self.config_path else "config.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            write_config(path, self._collect())
        except OSError as e:
            messagebox.showerror("Configuration", f"Could not save:\n{e}")
            return
        self.config_path = path
        self.cfg_name_var.set(f"Configuration: {os.path.basename(path)}")

    def _load_config(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load configuration",
            initialdir=CONFIG_DIR,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            cfg = read_config(path)
        except (OSError, ValueError, TypeError) as e:
            messagebox.showerror("Configuration", f"Could not read the file:\n{e}")
            return
        self._apply_cfg(cfg)
        self.config_path = path
        self.cfg_name_var.set(f"Configuration: {os.path.basename(path)}")

    ## ---- other callbacks ---------------------------------------------------------------------------

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Checkpoint",
            initialdir=os.path.abspath("logs") if os.path.isdir("logs") else os.getcwd(),
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.v["checkpoint"].set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="Report file", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.v["output"].set(path)

    def _browse_plots(self):
        path = filedialog.askdirectory(title="Folder for the plots", mustexist=False)
        if path:
            self.v["plots_dir"].set(path)
            self.b["save_plots"].set(True)

    def _copy_command(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.cmd_var.get())

    def _on_change(self, *_):
        # the filter changes the list, everything else only the summary
        if self.v["filter"].get() != getattr(self, "_last_filter", None):
            self._last_filter = self.v["filter"].get()
            self._apply_filter()
        cfg = self._collect()
        self.run_cb.configure(state="disabled" if cfg["checkpoint"] else "normal")
        plots_state = ["!disabled"] if cfg["save_plots"] else ["disabled"]
        self.plots_entry.state(plots_state)
        self.plots_btn.state(plots_state)

        n = len(cfg["scenarios"])
        seeds = parse_seeds(cfg["seeds"]) or []
        episodes = parse_episodes(cfg["episodes"]) or 0
        runs = n * len(seeds)
        self.summary_var.set(f"Scenarios: {n} x seeds: {len(seeds)} = runs: {runs}; total episodes: {runs * episodes}")
        self.cmd_var.set("python scripts/rsl_rl/evaluate.py " + shlex.join(build_argv(cfg, self.order)))

    def _run(self):
        cfg = self._collect()
        error = validate(cfg)
        if error:
            messagebox.showerror("Check the settings", error)
            return
        runs = len(cfg["scenarios"]) * len(parse_seeds(cfg["seeds"]) or [])
        if cfg["mode"] == "realtime" and runs > 3:
            if not messagebox.askyesno(
                "Real-time",
                f"Real-time mode will perform {runs} runs, each opening its own simulator window.\nContinue?",
            ):
                return
        try:
            write_config(LAST_CONFIG, cfg)
        except OSError:
            pass
        self.result = build_argv(cfg, self.order)
        self._shutdown()

    def _cancel(self):
        self.result = None
        self._shutdown()

    def _shutdown(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self.root.destroy()


def run_gui(list_cmd: Callable[[str], list[str]]) -> list[str] | None:
    """Open the window. Returns the evaluate.py arguments, or None if the window was closed/cancelled.

    list_cmd(path) must return the command that writes the scenario catalog JSON to path.
    """
    try:
        root = tk.Tk()
    except tk.TclError as e:
        raise RuntimeError(f"Could not open a window (no display?): {e}. Run evaluate.py without --gui.") from e
    gui = EvalGui(root, list_cmd)
    root.mainloop()
    return gui.result