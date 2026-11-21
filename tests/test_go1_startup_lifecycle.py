"""Regression for replicated-scene events and errors hidden by Kit shutdown.

The lifecycle check executes _prepare_terms from the real Isaac Lab checkout,
without starting Isaac Sim. Set ISAACLAB_EVENT_MANAGER_SOURCE if necessary.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr
import inspect
import io
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import unittest

import torch

LAB = Path(__file__).resolve().parents[1]
CFG = LAB / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go1/velocity_env_cfg.py"


class EventTerm:
    def __init__(self, *, func, mode, **kw):
        self.func, self.mode = func, mode
        self.min_step_count_between_reset = 0
        self.params = {}
        self.__dict__.update(kw)


def actual_prepare_terms():
    default = LAB.parent / "IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py"
    source = Path(os.environ.get("ISAACLAB_EVENT_MANAGER_SOURCE", str(default)))
    if not source.is_file():
        raise unittest.SkipTest("Isaac Lab event_manager.py not found; set ISAACLAB_EVENT_MANAGER_SOURCE")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EventManager")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_terms")
    context = {"inspect": inspect, "torch": torch, "EventTermCfg": EventTerm,
               "omni": SimpleNamespace(log=SimpleNamespace(warn=lambda *a: None, info=lambda *a: None))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), context)
    return context["_prepare_terms"]


def manager():
    context = {"EventTerm": EventTerm, "configure_go1_mjcf_contacts": lambda *a: None,
               "validate_go1_mjcf_physics": lambda *a: None}
    tree = ast.parse(CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go1EventsCfg")
    assignments = [n for n in cls.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ("mjcf_contacts", "mjcf_physics") for t in n.targets)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(CFG), "exec"), context)
    return SimpleNamespace(cfg=SimpleNamespace(**{k: v for k, v in context.items() if isinstance(v, EventTerm)}),
                           _env=SimpleNamespace(scene=SimpleNamespace(cfg=SimpleNamespace(replicate_physics=True))),
                           _mode_term_names={}, _mode_term_cfgs={}, _mode_class_term_cfgs={},
                           _resolve_common_term_cfg=lambda *a, **kw: None)


class ReplicationLifecycle(unittest.TestCase):
    def test_reproduces_v5_failure_in_real_event_manager(self):
        m = manager()
        m.cfg.mjcf_contacts = EventTerm(func=lambda *a: None, mode="prestartup")
        with self.assertRaisesRegex(RuntimeError, "replication"):
            actual_prepare_terms()(m)

    def test_fixed_events_are_accepted_with_replication_enabled(self):
        m = manager()
        actual_prepare_terms()(m)
        self.assertEqual(m._mode_term_names, {"startup": ["mjcf_physics"]})
        self.assertTrue(m._env.scene.cfg.replicate_physics)


class ExceptionVisibility(unittest.TestCase):
    def run_entrypoint(self, filename, *, shutdown_exits):
        source = LAB / "scripts/project" / filename
        tree = ast.parse(source.read_text())
        entry = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")
        output = io.StringIO()
        closed = []
        def fail():
            raise RuntimeError("fixture failure before shutdown")
        def close():
            # Assert the traceback is already visible even if Kit exits here.
            self.assertIn("fixture failure before shutdown", output.getvalue())
            closed.append(True)
            if shutdown_exits:
                raise SystemExit(0)
        context = {"__name__": "__main__", "main": fail, "app": SimpleNamespace(close=close),
                   "sys": sys, "traceback": traceback}
        with redirect_stderr(output), self.assertRaises(SystemExit) as exc:
            exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), context)
        self.assertTrue(closed)
        self.assertEqual(exc.exception.code, 0 if shutdown_exits else 1)

    def test_tracebacks_survive_immediate_kit_shutdown(self):
        for filename in ("check_go1_physics.py", "check_go1_stand.py"):
            with self.subTest(script=filename):
                self.run_entrypoint(filename, shutdown_exits=True)

    def test_nonzero_exit_when_shutdown_returns(self):
        for filename in ("check_go1_physics.py", "check_go1_stand.py"):
            with self.subTest(script=filename):
                self.run_entrypoint(filename, shutdown_exits=False)


if __name__ == "__main__":
    unittest.main()
