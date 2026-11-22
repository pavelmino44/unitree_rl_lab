"""Exercise the real exporter with minimal Isaac interfaces, without a simulator."""
import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import yaml

from test_go1_safety import NAMES, OFFSET, make_action, limits

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / 'source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py'


class Config(NS):
    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return vars(self).copy()


def exporter(path=EXPORTER):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    context = dict(np=np, os=os, yaml=yaml, Articulation=object,
                   ManagerBasedRLEnv=object, class_to_dict=vars,
                   resolve_matching_names=lambda names, sdk, **kwargs: (list(range(12)), names))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), context)
    return context['export_deploy_cfg']


def export_fixture(path, *, go1):
    action, data = make_action(count=1)
    action.cfg = Config(**vars(action.cfg), class_type=type(action), asset_name='robot',
                        debug_vis=False, preserve_order=True, use_default_offset=True,
                        joint_names=NAMES, scale={'.*': .25}, offset=0.)
    if not go1:
        # A standard Go2 action has no optional full-precision export hook.
        for name in ('min_target', 'max_target', 'max_delta_rad', 'raw_action_clip',
                     'collision_check', 'check_torques'):
            delattr(action.cfg, name)
        action._scale.fill_(.25)
        action = NS(**vars(action))
    data.joint_names = NAMES
    env = NS(
        scene={'robot': NS(data=data)},
        cfg=NS(scene=NS(robot=NS(joint_sdk_names=NAMES)), sim=NS(dt=.002), decimation=10, commands=NS()),
        action_manager=NS(active_terms=['JointPositionAction'], _terms={'JointPositionAction': action}),
        observation_manager=NS(active_terms={'policy': []}, _group_obs_term_cfgs={'policy': []}),
    )
    with tempfile.TemporaryDirectory() as folder:
        exporter(path)(env, folder)
        text = (Path(folder) / 'params/deploy.yaml').read_text()
    return yaml.safe_load(text), text


class ExportTests(unittest.TestCase):
    def test_go1_contract_survives_generic_three_digit_rounding(self):
        result, _ = export_fixture(EXPORTER, go1=True)
        action = result['actions']['JointPositionAction']
        self.assertEqual(action['max_delta_rad'], limits.STEP_LIMIT_RAD)
        self.assertEqual(action['min_target'], limits.Q_MIN)
        self.assertEqual(action['max_target'], limits.Q_MAX)
        self.assertTrue(np.isinf(action['raw_action_clip']))
        self.assertEqual(action['torque_limits_nm'], [23.7, 23.7, 35.55] * 4)
        self.assertEqual(result['step_dt'], .02)

    def test_standard_action_uses_existing_export_precision(self):
        result, _ = export_fixture(EXPORTER, go1=False)
        action = result['actions']['JointPositionAction']
        self.assertNotIn('max_delta_rad', action)
        self.assertNotIn('float_guard_rad', action)
        self.assertNotIn('torque_limits_nm', action)
        self.assertEqual(action['offset'], OFFSET)


if __name__ == '__main__':
    unittest.main()
