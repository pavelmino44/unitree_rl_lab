"""CPU configuration/tensor regressions, with small Isaac interface fixtures.

Run: python -m unittest discover -s tests -p 'test_go1_training.py' -v
These tests do not launch a simulator or validate rendering/physics.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

LOCOMOTION = (
    Path(__file__).resolve().parents[1]
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion"
)
GO1 = LOCOMOTION / "robots/go1"


class Record:
    def __init__(self, *args, **kwargs):
        if args:
            kwargs["func" if callable(args[0]) else "name"] = args[0]
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        result = copy.deepcopy(self)
        result.__dict__.update(kwargs)
        return result


Record.Ranges = Record


def configclass(cls):
    def init(self, **kwargs):
        for base in reversed(cls.__mro__):
            for key, value in vars(base).items():
                if not key.startswith("_") and (
                    not callable(value) or key == "class_type"
                ):
                    setattr(self, key, copy.deepcopy(value))
        self.__dict__.update(kwargs)

    cls.__init__ = init
    return cls


class FunctionRegistry:
    def __getattr__(self, name):
        fn = lambda *args, **kwargs: None
        setattr(self, name, fn)
        return fn


def load_file(name, path, fixtures=None):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fixtures or {}):
        spec.loader.exec_module(result)
    return result


def load_config_contracts():
    # Execute actual class bodies, excluding simulator-dependent imports/scenes.
    # The small records preserve term functions, weights, parameters and MRO.
    mdp = FunctionRegistry()
    mdp.UniformLevelVelocityCommandCfg = Record
    shared = load_file("go1_shared_curriculum", LOCOMOTION / "mdp/curriculums.py")
    mdp.lin_vel_cmd_levels = shared.lin_vel_cmd_levels
    context = {
        "configclass": configclass,
        "math": math,
        "mdp": mdp,
        "RewTerm": Record,
        "DoneTerm": Record,
        "CurrTerm": Record,
        "SceneEntityCfg": Record,
        "Go1VelocityCommandCfg": Record,
        "base_too_low": lambda *args: None,
    }
    for filename, names in [
        (
            LOCOMOTION / "robots/go2/velocity_env_cfg.py",
            {"CommandsCfg", "RewardsCfg", "TerminationsCfg", "CurriculumCfg"},
        ),
        (
            GO1 / "velocity_env_cfg.py",
            {
                "Go1CommandsCfg",
                "Go1RewardsCfg",
                "Go1TerminationsCfg",
                "Go1CurriculumCfg",
            },
        ),
    ]:
        tree = ast.parse(filename.read_text())
        body = [
            node
            for node in tree.body
            if (isinstance(node, ast.ClassDef) and node.name in names)
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id in {"GO1_MAX_SPEED", "GO1_LINEAR_SPEED_SCALE"}
                    for target in node.targets
                )
            )
        ]
        exec(  # noqa: S102 -- execute selected class bodies from this repository
            compile(ast.Module(body=body, type_ignores=[]), str(filename), "exec"),
            context,
        )
        if filename.parent.name == "go2":
            for name in names:
                context["Go2" + name] = context[name]
    return NS(**context), shared


def normalized(value):
    if isinstance(value, Record):
        return {k: normalized(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return list(map(normalized, value))
    return value


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.shared = load_config_contracts()

    def test_exact_go2_reward_contract_except_contacts(self):
        go1, go2 = self.cfg.Go1RewardsCfg(), self.cfg.RewardsCfg()
        self.assertEqual(set(vars(go1)), set(vars(go2)))
        for name in vars(go2):
            if name != "undesired_contacts":
                self.assertEqual(
                    normalized(getattr(go1, name)), normalized(getattr(go2, name)), name
                )

    def test_go1_contact_penalty_is_preserved(self):
        term = self.cfg.Go1RewardsCfg().undesired_contacts
        self.assertEqual(term.weight, -1)
        self.assertEqual(term.params["threshold"], 1)
        self.assertEqual(
            term.params["sensor_cfg"].body_names,
            ["trunk", ".*_hip", ".*_thigh", ".*_calf"],
        )

    def test_linear_commands_scale_with_maximum_speed(self):
        go1, go2 = (
            self.cfg.Go1CommandsCfg().base_velocity,
            self.cfg.CommandsCfg().base_velocity,
        )
        scale = self.cfg.GO1_MAX_SPEED / go2.limit_ranges.lin_vel_x[1]
        for ranges_name in ("ranges", "limit_ranges"):
            for axis in ("lin_vel_x", "lin_vel_y"):
                expected = [v * scale for v in getattr(getattr(go2, ranges_name), axis)]
                self.assertEqual(
                    list(getattr(getattr(go1, ranges_name), axis)), expected
                )
        self.assertEqual(go1.ranges.lin_vel_x, (-0.06, 0.06))
        self.assertEqual(go1.limit_ranges.lin_vel_x, (-0.6, 0.6))
        self.assertEqual(go1.ranges.ang_vel_z, (-1.0, 1.0))
        self.assertEqual(go1.rel_standing_envs, 0.1)
        self.assertTrue(go1.debug_vis)
        self.assertEqual(go1.resampling_time_range, go2.resampling_time_range)

    def test_curriculum_is_the_go2_implementation(self):
        term = self.cfg.Go1CurriculumCfg()
        self.assertIsNone(term.terrain_levels)
        self.assertIs(
            term.lin_vel_cmd_levels.func,
            self.cfg.CurriculumCfg().lin_vel_cmd_levels.func,
        )

    def test_termination_terms_include_timeout_and_fall_reasons(self):
        terms = self.cfg.Go1TerminationsCfg()
        self.assertEqual(
            set(vars(terms)),
            {"time_out", "base_contact", "bad_orientation", "base_too_low"},
        )
        self.assertTrue(terms.time_out.time_out)
        self.assertAlmostEqual(
            terms.base_too_low.params["minimum_height"], 0.21 * (2 / 3)
        )
        self.assertEqual(terms.bad_orientation.params["limit_angle"], 0.55)
        self.assertEqual(terms.base_contact.params["sensor_cfg"].body_names, "trunk")

    def test_go2_curriculum_requires_full_episode_reward_at_check_step(self):
        command = self.cfg.Go1CommandsCfg().base_velocity
        env = NS(
            device="cpu",
            common_step_counter=1000,
            max_episode_length=1000,
            max_episode_length_s=20.0,
            command_manager=NS(get_term=lambda _: NS(cfg=command)),
            reward_manager=NS(
                get_term_cfg=lambda _: NS(weight=1.5),
                _episode_sums={"track_lin_vel_xy": torch.full((4,), 1.5)},
            ),
        )
        term = self.cfg.Go1CurriculumCfg().lin_vel_cmd_levels
        term.func(env, [0, 1, 2, 3], **term.params)
        self.assertEqual(command.ranges.lin_vel_x, (-0.06, 0.06))
        env.reward_manager._episode_sums["track_lin_vel_xy"].fill_(30.0)
        env.common_step_counter = 1001
        term.func(env, [0, 1, 2, 3], **term.params)
        self.assertEqual(command.ranges.lin_vel_x, (-0.06, 0.06))
        env.common_step_counter = 2000
        state = term.func(env, [0, 1, 2, 3], **term.params)
        self.assertAlmostEqual(float(state), 0.12)
        self.assertAlmostEqual(command.ranges.lin_vel_y[1], 0.12)

    def test_same_relative_progress_as_go2_and_limits_respected(self):
        go1 = self.cfg.Go1CommandsCfg().base_velocity
        go2 = self.cfg.CommandsCfg().base_velocity
        env = NS(
            device="cpu",
            common_step_counter=1000,
            max_episode_length=1000,
            max_episode_length_s=20.0,
            reward_manager=NS(
                get_term_cfg=lambda _: NS(weight=1.5),
                _episode_sums={"track_lin_vel_xy": torch.full((4,), 30.0)},
            ),
        )
        term = self.cfg.Go1CurriculumCfg().lin_vel_cmd_levels
        for _ in range(15):
            env.command_manager = NS(get_term=lambda _: NS(cfg=go1))
            term.func(env, [0, 1, 2, 3], **term.params)
            env.command_manager = NS(get_term=lambda _: NS(cfg=go2))
            self.shared.lin_vel_cmd_levels(env, [0, 1, 2, 3])  # unmodified Go2 default
            self.assertAlmostEqual(
                go1.ranges.lin_vel_x[1] / 0.6, go2.ranges.lin_vel_x[1], places=5
            )
            self.assertAlmostEqual(
                go1.ranges.lin_vel_y[1] / 0.6, go2.ranges.lin_vel_y[1], places=5
            )
            self.assertLessEqual(go1.ranges.lin_vel_x[1], 0.6 + 1e-7)
        self.assertAlmostEqual(go1.ranges.lin_vel_x[1], 0.6, places=6)
        self.assertAlmostEqual(go2.ranges.lin_vel_x[1], 1.0, places=6)

    def test_height_threshold_boundary(self):
        stability = load_file("go1_stability", GO1 / "stability.py")
        position = torch.zeros(3, 3)
        position[:, 2] = torch.tensor([0.1399, 0.14, 0.1401])
        env = NS(scene={"robot": NS(data=NS(root_link_pos_w=position))})
        minimum = self.cfg.Go1TerminationsCfg().base_too_low.params["minimum_height"]
        torch.testing.assert_close(
            stability.base_too_low(env, minimum), torch.tensor([True, False, False])
        )


class Marker:
    def __init__(self, cfg):
        self.cfg, self.visible, self.samples = cfg, False, None

    def set_visibility(self, value):
        self.visible = value

    def visualize(self, position, quat, scale):
        self.samples = (position.clone(), quat.clone(), scale.clone())


class UniformVelocityCommand:
    def __init__(self, cfg, env):
        self.cfg, self.num_envs, self.device = cfg, env.num_envs, env.device
        self.metrics = {
            k: torch.zeros(env.num_envs) for k in ("error_vel_xy", "error_vel_yaw")
        }
        # Upstream enables debug drawing before setting robot and command buffers.
        self._set_debug_vis_impl(cfg.debug_vis)
        self.robot, self.command = env.scene["robot"], torch.zeros(env.num_envs, 3)
        self.xy_callback_count = 0

    def _set_debug_vis_impl(self, value):
        self.xy_visible = value

    def _debug_vis_callback(self, event):
        self.xy_callback_count += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        result = {k: float(v[ids].mean()) for k, v in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0
        return result


def module(name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    return result


def quat_from_euler_xyz(roll, pitch, yaw):
    # Marker term always requests roll=yaw=0; this fixture covers its Y rotation.
    return torch.stack(
        (
            torch.cos(pitch / 2),
            torch.zeros_like(pitch),
            torch.sin(pitch / 2),
            torch.zeros_like(pitch),
        ),
        dim=-1,
    )


def command_type():
    marker_cfg = Record(prim_path="")
    fixtures = {
        "isaaclab.envs.mdp": module(
            "mdp", UniformVelocityCommand=UniformVelocityCommand
        ),
        "isaaclab.markers": module(
            "markers", VisualizationMarkers=Marker, VisualizationMarkersCfg=Record
        ),
        "isaaclab.markers.config": module(
            "config",
            BLUE_ARROW_X_MARKER_CFG=marker_cfg,
            GREEN_ARROW_X_MARKER_CFG=marker_cfg,
        ),
        "isaaclab.utils": module("utils", configclass=configclass),
        "isaaclab.utils.math": module("math", quat_from_euler_xyz=quat_from_euler_xyz),
        "unitree_rl_lab.tasks.locomotion.mdp.commands": module(
            "commands", UniformLevelVelocityCommandCfg=Record
        ),
    }
    return load_file("go1_commands", GO1 / "commands.py", fixtures)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.module = command_type()
        data = NS(
            root_lin_vel_b=torch.zeros(2, 3),
            root_ang_vel_b=torch.zeros(2, 3),
            root_pos_w=torch.zeros(2, 3),
        )
        self.env = NS(
            num_envs=2,
            device="cpu",
            scene={"robot": NS(data=data, is_initialized=True)},
        )
        self.term = self.module.Go1VelocityCommand(
            self.module.Go1VelocityCommandCfg(debug_vis=True), self.env
        )

    def test_episode_mean_and_partial_reset(self):
        self.term.command[:, 0] = 0.3
        self.term.robot.data.root_lin_vel_b[1, 0] = 0.3
        for _ in range(1000):
            self.term._update_metrics()
        torch.testing.assert_close(
            self.term.metrics["error_vel_xy"], torch.tensor([0.3, 0.0])
        )
        self.assertAlmostEqual(self.term.reset([0])["error_vel_xy"], 0.3, places=6)
        self.term.robot.data.root_lin_vel_b[0, 0] = 0.3
        self.term._update_metrics()
        self.assertAlmostEqual(float(self.term.metrics["velocity_x"][0]), 0.3, places=6)
        self.assertEqual(float(self.term._metric_steps[1]), 1001.0)

    def test_yaw_arrow_sign_size_and_zero(self):
        scale, quat = self.term._resolve_yaw_velocity_to_arrow(
            torch.tensor([-1.0, 0.0, 2.0])
        )
        torch.testing.assert_close(scale[:, 0], torch.tensor([0.3, 0.0, 0.6]))
        # Rotate local +X by the computed quaternion: negative yaw points down.
        axial_z = -2 * quat[:, 0] * quat[:, 2]
        torch.testing.assert_close(axial_z, torch.tensor([-1.0, 1.0, 1.0]))

    def test_both_marker_pairs_update_and_visibility_toggles(self):
        self.term.command[:, 2] = 1.0
        self.term.robot.data.root_ang_vel_b[:, 2] = -0.5
        self.term._debug_vis_callback(None)
        self.assertEqual(self.term.xy_callback_count, 1)
        self.assertTrue(self.term.goal_yaw_visualizer.visible)
        self.assertIsNotNone(self.term.current_yaw_visualizer.samples)
        self.term._set_debug_vis_impl(False)
        self.assertFalse(self.term.xy_visible)
        self.assertFalse(self.term.goal_yaw_visualizer.visible)


if __name__ == "__main__":
    unittest.main()
