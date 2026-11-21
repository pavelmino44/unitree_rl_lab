"""Offline OpenUSD checks for Go1 ground setup before PhysX starts."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

LAB = Path(__file__).resolve().parents[1]
GO1 = LAB / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go1"


class GroundSetup(unittest.TestCase):
    def build(self, *, playing=False, colliders=True, invalid=False):
        stage = Usd.Stage.CreateInMemory()
        calls = []
        class Parent:
            def __init__(self, cfg):
                calls.append("spawn")
                self.cfg = cfg
                UsdGeom.Xform.Define(stage, cfg.prim_path)
                if colliders:
                    cube = UsdGeom.Cube.Define(stage, cfg.prim_path+"/plane")
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        def modify(path, cfg):
            self.assertFalse(playing, "Ground writes must happen before simulation")
            calls.append("write")
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(cfg.contact_offset)
                    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(.01 if invalid else cfg.rest_offset)
        def physics(prim):
            return SimpleNamespace(GetContactOffsetAttr=lambda: prim.GetAttribute("physxCollision:contactOffset"),
                                   GetRestOffsetAttr=lambda: prim.GetAttribute("physxCollision:restOffset"))
        context = dict(TerrainImporter=Parent, Usd=Usd, UsdPhysics=UsdPhysics, json=json,
                       get_current_stage=lambda: stage,
                       PhysxSchema=SimpleNamespace(PhysxCollisionAPI=physics),
                       sim_utils=SimpleNamespace(SimulationContext=SimpleNamespace(instance=lambda: SimpleNamespace(is_playing=lambda: playing)),
                                                 CollisionPropertiesCfg=lambda **kw: SimpleNamespace(**kw),
                                                 modify_collision_properties=modify))
        tree = ast.parse((GO1/"terrain.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "terrain.py", "exec"), context)
        with patch("builtins.print"):
            result = context["Go1TerrainImporter"](SimpleNamespace(prim_path="/World/ground"))
        return result, calls

    def test_ground_written_after_spawn_before_physics(self):
        result, calls = self.build()
        self.assertEqual(calls, ["spawn", "write"])
        self.assertAlmostEqual(result.go1_ground_offsets[0]["contact_offset_m"], .0005)
        self.assertEqual(result.go1_ground_offsets[0]["rest_offset_m"], 0.)

    def test_live_construction_rejected_before_writes(self):
        with self.assertRaisesRegex(RuntimeError, "before starting"):
            self.build(playing=True)

    def test_missing_collider_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No Go1 ground colliders"):
            self.build(colliders=False)

    def test_invalid_readback_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid Go1 ground offsets"):
            self.build(invalid=True)

    def test_configuration_uses_custom_ground_and_no_contact_event(self):
        tree = ast.parse((GO1/"velocity_env_cfg.py").read_text())
        scene = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go1FlatSceneCfg")
        terrain = next(n.value for n in scene.body if isinstance(n, ast.Assign) and n.targets[0].id == "terrain")
        self.assertTrue(any(k.arg == "class_type" and ast.unparse(k.value) == "Go1TerrainImporter" for k in terrain.keywords))
        events = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go1EventsCfg")
        self.assertNotIn("mjcf_contacts", [t.id for n in events.body if isinstance(n, ast.Assign) for t in n.targets])


if __name__ == "__main__":
    unittest.main()
