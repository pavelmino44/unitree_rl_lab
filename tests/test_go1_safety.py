"""CPU regressions against the supplied Safety Block's numerical routines.

No Isaac Sim is launched. Interface fixtures execute the actual Go1 action term.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GO1 = ROOT / 'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go1'
ASSET = ROOT / 'source/unitree_rl_lab/unitree_rl_lab/assets/robots/go1_mjcf'
NAMES = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
         for joint in ('hip', 'thigh', 'calf')]
OFFSET = [-.1, .85, -1.7, .1, .85, -1.7] * 2
PACKAGE = '_go1_safety_test'
package = ModuleType(PACKAGE)
package.__path__ = [str(GO1)]
sys.modules[PACKAGE] = package


def load(name):
    import importlib
    return importlib.import_module(PACKAGE + '.' + name)


safety = load('safety')
limits = load('safety_reference.joint_limits')
validator = load('safety_reference.validator')
torque = load('safety_reference.torque_check')


def module(name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    return result


class JointPositionFixture:
    def __init__(self, cfg, env):
        self.cfg, self.device, self.num_envs = cfg, 'cpu', env.num_envs
        self.action_dim = 12
        self._joint_ids, self._joint_names = list(range(12)), NAMES
        self._asset = env.robot
        self._offset = env.robot.data.default_joint_pos.clone()
        self._raw_actions = torch.zeros_like(self._offset)
        self._processed_actions = torch.zeros_like(self._offset)
        self._scale = torch.tensor([.15, .35, .35] * 4)

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids=None):
        self._raw_actions[slice(None) if env_ids is None else env_ids] = 0


def action_type():
    fixtures = {
        'isaaclab.envs.mdp.actions.actions_cfg': module('cfg', JointPositionActionCfg=object),
        'isaaclab.envs.mdp.actions.joint_actions': module('actions', JointPositionAction=JointPositionFixture),
        'isaaclab.utils': module('utils', configclass=lambda cls: cls),
        'unitree_rl_lab.assets.robots.go1': module('go1', GO1_SDK_JOINTS=NAMES, GO1_ASSET_DIR=ASSET),
    }
    with patch.dict(sys.modules, fixtures):
        name = PACKAGE + '.actions'
        sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(name, GO1 / 'actions.py')
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return result, fixtures


def make_action(count=2, collision='none'):
    mod, fixtures = action_type()
    offset = torch.tensor(OFFSET)[None].repeat(count, 1)
    data = NS(default_joint_pos=offset.clone(), joint_pos=offset.clone(),
              joint_vel=torch.zeros_like(offset),
              default_joint_stiffness=torch.full_like(offset, 100.),
              default_joint_damping=torch.zeros_like(offset))
    env = NS(num_envs=count, robot=NS(data=data))
    cfg = NS(min_target=tuple(limits.Q_MIN), max_target=tuple(limits.Q_MAX),
             max_delta_rad=limits.STEP_LIMIT_RAD, raw_action_clip=math.inf,
             clip=None, collision_check=collision, check_torques=True)
    with patch.dict(sys.modules, fixtures):
        action = mod.RateLimitedJointPositionAction(cfg, env)
    action.reset()
    return action, data


class SafetyParity(unittest.TestCase):
    def test_tensor_clamps_and_whole_command_decisions_match_reference(self):
        generator = torch.Generator().manual_seed(53)
        n = 4096
        q_min, q_max = torch.tensor(limits.Q_MIN), torch.tensor(limits.Q_MAX)
        previous = q_min + torch.rand(n, 12, generator=generator) * (q_max - q_min)
        requested = torch.randn(n, 12, generator=generator) * 3
        measured = previous + torch.randn(n, 12, generator=generator) * .15
        velocity = torch.randn(n, 12, generator=generator)
        kp, kd = torch.full((n, 12), 100.), torch.zeros(n, 12)
        candidate, finite, _, _ = safety.candidate_targets(
            requested, previous, q_min + safety.FLOAT_GUARD_RAD,
            q_max - safety.FLOAT_GUARD_RAD)
        ok, _ = safety.torque_acceptance(candidate, measured, velocity, kp, kd,
                                        torch.tensor(safety.TORQUE_LIMITS))
        output, accepted = safety.accept_targets(candidate, previous, finite, ok, torch.ones(n, dtype=torch.bool))
        expected_q, expected_ok = [], []
        for row in range(n):
            q = validator.enforce_step_limit(validator.clamp_q(requested[row].tolist()), previous[row].tolist())
            tau = torque.compute_torques(q, [0.] * 12, kp[row].tolist(), kd[row].tolist(),
                                         [0.] * 12, measured[row].tolist(), velocity[row].tolist())
            accept = torque.check_torque_limits(tau)
            expected_ok.append(accept)
            expected_q.append(q if accept else previous[row].tolist())
        torch.testing.assert_close(accepted, torch.tensor(expected_ok))
        torch.testing.assert_close(output, torch.tensor(expected_q), atol=2e-6, rtol=0)
        self.assertTrue(accepted.any() and (~accepted).any())

    def test_unbounded_actor_can_cross_old_hip_and_calf_envelopes(self):
        action, data = make_action()
        raw = torch.zeros(2, 12)
        raw[:, 0], raw[:, 2] = 3., -3.
        for _ in range(18):
            action.process_actions(raw)
            data.joint_pos.copy_(action.processed_actions if hasattr(action, 'processed_actions')
                                 else action._processed_actions)
        self.assertGreater(float(action._processed_actions[0, 0]), 0.)
        self.assertLess(float(action._processed_actions[0, 2]), -2.05)
        torch.testing.assert_close(action._raw_actions, raw)
        self.assertLessEqual(float(action._processed_actions[0, 0]), .863)

    def test_single_joint_torque_failure_holds_every_target_and_history(self):
        action, data = make_action()
        original = action._last_target.clone()
        data.joint_pos[0, 0] -= 1.
        action.process_actions(torch.ones(2, 12))
        torch.testing.assert_close(action._processed_actions[0], original[0])
        self.assertTrue(torch.any(action._processed_actions[1] != original[1]))
        self.assertEqual(action.safety_metrics['torque_reject'].tolist(), [1., 0.])
        data.joint_pos[0] = original[0]
        action.process_actions(torch.ones(2, 12))
        self.assertLessEqual(float((action._processed_actions[0] - original[0]).abs().max()), limits.STEP_LIMIT_RAD)

    def test_partial_reset_uses_measured_state_and_preserves_other_env(self):
        action, data = make_action()
        action.process_actions(torch.ones(2, 12))
        untouched = action._last_target[1].clone()
        data.joint_pos[0] += .025
        action.reset([0])
        torch.testing.assert_close(action._last_target[0], data.joint_pos[0])
        torch.testing.assert_close(action._last_target[1], untouched)

    def test_nonfinite_reject_and_five_degree_inclusive_limit(self):
        action, data = make_action()
        raw = torch.full((2, 12), 100.)
        raw[0, 4] = float('nan')
        before = action._last_target.clone()
        action.process_actions(raw)
        torch.testing.assert_close(action._processed_actions[0], before[0])
        self.assertTrue(torch.isfinite(action._processed_actions).all())
        self.assertLessEqual(float((action._processed_actions - before).abs().max()), limits.STEP_LIMIT_RAD)
        self.assertEqual(action.safety_metrics['nonfinite_reject'].tolist(), [1., 0.])

    def test_collision_reject_holds_whole_command(self):
        action, _ = make_action()
        class Reject:
            def check_transition(self, previous, candidate):
                return False
        action._collision_checker = Reject()
        before = action._last_target.clone()
        action.process_actions(torch.ones(2, 12))
        torch.testing.assert_close(action._processed_actions, before)
        self.assertEqual(action.safety_metrics['collision_reject'].tolist(), [1., 1.])


class ExactCollision(unittest.TestCase):
    def test_reference_model_pairs_and_initial_stance(self):
        action, _ = make_action(count=1, collision='mujoco')
        checker = action._collision_checker
        self.assertEqual(checker.joint_names, NAMES)
        self.assertEqual(len(checker.pairs), 655)
        self.assertTrue(checker.check_config(OFFSET))
        colliding = [-.7979998, 3.76787343, -2.56240696, -.70490992,
                     2.14074577, -1.86620678, .69791394, 1.86142027,
                     -1.41367122, -.32139428, 4.45330509, -2.74449792]
        self.assertFalse(checker.check_config(colliding))
        action.process_actions(torch.zeros(1, 12))
        self.assertEqual(action.safety_metrics['accepted'].item(), 1.)
        self.assertEqual(action.safety_metrics['exact_collision_check'].item(), 1.)

    def test_interpolation_includes_intermediate_pose_and_endpoints(self):
        from importlib import import_module
        checker_type = import_module(PACKAGE + '.safety_reference.collision_checker').CollisionChecker
        checker = object.__new__(checker_type)
        samples = []
        checker.check_config = lambda q: samples.append(np.array(q)) or True
        begin, end = np.zeros(12), np.zeros(12)
        end[0] = math.radians(5.)
        self.assertTrue(checker.check_transition(begin, end))
        self.assertGreaterEqual(len(samples), 51)
        np.testing.assert_allclose(samples[0], begin)
        np.testing.assert_allclose(samples[-1], end)
        self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0))), limits.INTERP_STEP_RAD + 1e-12)
        checker.check_config = lambda q: not (math.radians(2.) < q[0] < math.radians(3.))
        self.assertFalse(checker.check_transition(begin, end))


if __name__ == '__main__':
    unittest.main()
