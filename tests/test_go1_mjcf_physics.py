"""USD/MuJoCo numerical conformance, no Isaac Sim and no simulated training.

Offline: python -m unittest discover -s tests -p test_go1_mjcf_physics.py -v
Dependencies for this developer check: numpy, mujoco==3.3.7, usd-core.
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco as mj
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

LAB = Path(__file__).resolve().parents[1]
ASSET = LAB / "source/unitree_rl_lab/unitree_rl_lab/assets/robots/go1_mjcf"
GO1_CFG = LAB / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go1/velocity_env_cfg.py"


def quat_r(q):
    out = np.empty(9)
    mj.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def usd_mass(prim):
    mass = UsdPhysics.MassAPI(prim)
    q = mass.GetPrincipalAxesAttr().Get()
    r = quat_r(np.array([q.GetReal(), *q.GetImaginary()], dtype=float))
    return float(mass.GetMassAttr().Get()), np.array(mass.GetCenterOfMassAttr().Get()), r @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ r.T


class ModelConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = Usd.Stage.Open(str(ASSET / "go1.usda"))
        cls.model = mj.MjModel.from_xml_path(str(ASSET / "reference/go1.xml"))
        cls.spec = json.loads((ASSET / "physics.json").read_text())

    def test_positive_masses_and_inertias_total(self):
        bodies = [p for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.assertEqual(len(bodies), 17)
        self.assertEqual({p.GetName() for p in bodies}, set(self.spec["bodies"]))
        masses = []
        for p in bodies:
            mass, _, inertia = usd_mass(p)
            masses.append(mass)
            self.assertGreater(mass, 0)
            self.assertGreater(np.linalg.eigvalsh(inertia).min(), 0)
        self.assertAlmostEqual(sum(masses), self.model.body_mass.sum(), places=6)
        self.assertAlmostEqual(sum(masses), 12.743448, places=6)

    def test_full_composite_inertias_against_compiled_mjcf(self):
        for bid in range(1, self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid)
            members = [n for n, b in self.spec["bodies"].items() if b["source_body"] == name]
            parts = []
            for n in members:
                mass, com, inertia = usd_mass(self.stage.GetPrimAtPath("/Go1/"+n))
                if n.endswith("_foot"):
                    com += [0., 0., -.213]
                parts.append((mass, com, inertia))
            mass = sum(m for m, _, _ in parts)
            com = sum(m*c for m, c, _ in parts)/mass
            inertia = np.zeros((3, 3))
            for m, c, i in parts:
                d = c-com
                inertia += i + m*((d@d)*np.eye(3)-np.outer(d, d))
            r = quat_r(self.model.body_iquat[bid])
            expected_i = r @ np.diag(self.model.body_inertia[bid]) @ r.T
            with self.subTest(body=name):
                self.assertAlmostEqual(mass, self.model.body_mass[bid], places=6)
                np.testing.assert_allclose(com, self.model.body_ipos[bid], atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(inertia, expected_i, atol=1e-8, rtol=1e-5)

    def test_all_collision_shapes_and_local_poses(self):
        colliders = [p for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual(len(colliders), 42)
        for g in self.spec["collision_geoms"]:
            prim = self.stage.GetPrimAtPath(g["path"])
            x = np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T
            r = quat_r(np.array(g["quat_wxyz"]))
            size = self.model.geom_size[g["id"]]
            scale = np.eye(3)
            if g["type"] == mj.mjtGeom.mjGEOM_BOX:
                self.assertTrue(prim.IsA(UsdGeom.Cube))
                scale = np.diag(size)
            elif g["type"] == mj.mjtGeom.mjGEOM_SPHERE:
                self.assertTrue(prim.IsA(UsdGeom.Sphere))
                self.assertAlmostEqual(UsdGeom.Sphere(prim).GetRadiusAttr().Get(), .023)
            else:
                cls = UsdGeom.Capsule if g["type"] == mj.mjtGeom.mjGEOM_CAPSULE else UsdGeom.Cylinder
                self.assertTrue(prim.IsA(cls))
                self.assertAlmostEqual(cls(prim).GetRadiusAttr().Get(), size[0])
                self.assertAlmostEqual(cls(prim).GetHeightAttr().Get(), 2*size[1])
            np.testing.assert_allclose(x[:3, :3], r @ scale, atol=1e-8)
            np.testing.assert_allclose(x[:3, 3], g["pos_m"], atol=1e-10)
            np.testing.assert_allclose(g["size"], size)
        # Visual triangles cannot accidentally become massive collision hulls.
        for p in self.stage.Traverse():
            if p.GetName().startswith("visual_"):
                self.assertFalse(p.HasAPI(UsdPhysics.CollisionAPI))

    def test_collision_filters_match_mjcf_pairs(self):
        excluded = {frozenset(p) for p in self.spec["filtered_body_pairs"]}
        authored = set()
        for name in self.spec["bodies"]:
            prim = self.stage.GetPrimAtPath("/Go1/"+name)
            for t in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                authored.add(frozenset([name, t.name]))
        self.assertEqual(excluded, authored)
        checked_pairs = 0
        for ga, gb in itertools.combinations(self.spec["collision_geoms"], 2):
            a = self.model.geom_bodyid[ga["id"]]
            b = self.model.geom_bodyid[gb["id"]]
            source_filtered = a == b or self.model.body_parentid[a] == b or self.model.body_parentid[b] == a
            usd_filtered = ga["body"] == gb["body"] or frozenset([ga["body"], gb["body"]]) in authored
            self.assertEqual(bool(source_filtered), bool(usd_filtered), (ga["id"], gb["id"]))
            checked_pairs += not source_filtered
        self.assertEqual(checked_pairs, 655)  # Published judge/T2 collision pairs.

    def test_kinematics_at_random_motor_configurations(self):
        data = mj.MjData(self.model)
        rng = np.random.default_rng(42)
        for sample in range(8):
            data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
            poses = {"trunk": (np.eye(3), np.zeros(3))}
            for spec in self.spec["joints"]:
                j = UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath("/Go1/joints/"+spec["name"]))
                angle = rng.uniform(*spec["limits_rad"])
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, spec["name"])
                data.qpos[self.model.jnt_qposadr[jid]] = angle
                axis = np.zeros(3)
                axis["XYZ".index(j.GetAxisAttr().Get())] = 1.
                q = np.r_[math.cos(angle/2), axis*math.sin(angle/2)]
                rp, pp = poses[spec["parent"]]
                anchor = np.array(j.GetLocalPos0Attr().Get())
                poses[spec["child"]] = (rp @ quat_r(q), pp+rp @ anchor)
            mj.mj_forward(self.model, data)
            for leg in ("FR", "FL", "RR", "RL"):
                fixed = UsdPhysics.FixedJoint(self.stage.GetPrimAtPath(f"/Go1/joints/{leg}_foot_fixed"))
                r, p = poses[leg+"_calf"]
                actual = p + r @ np.array(fixed.GetLocalPos0Attr().Get())
                sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, leg)
                np.testing.assert_allclose(actual, data.site_xpos[sid], atol=2e-8, err_msg=f"{leg} sample {sample}")

    def test_visual_references_and_asset_integrity(self):
        import hashlib
        for filename in ("go1.usda", "visuals.usdc"):
            self.assertEqual(hashlib.sha256((ASSET/filename).read_bytes()).hexdigest(), self.spec[filename+"_sha256"])
        for p in self.stage.Traverse():
            if p.GetName().startswith("visual_"):
                self.assertTrue(p.IsA(UsdGeom.Xform))
                self.assertTrue(p.IsInstance())
                self.assertFalse(p.IsA(UsdGeom.Gprim))
                mesh = p.GetChild("geometry")
                self.assertTrue(mesh.IsA(UsdGeom.Mesh))
                self.assertTrue(mesh.IsInstanceProxy())
                self.assertGreater(len(UsdGeom.Mesh(mesh).GetPointsAttr().Get()), 0)
        self.assertEqual(hashlib.sha256((ASSET/"reference/go1.xml").read_bytes()).hexdigest(), self.spec["source_sha256"])

    def test_renderable_meshes_are_below_instance_roots(self):
        meshes = [p for p in Usd.PrimRange(self.stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
                  if p.IsA(UsdGeom.Mesh)]
        self.assertEqual(len(meshes), 13)
        for mesh in meshes:
            self.assertFalse(mesh.IsInstance())  # A direct instance root Gprim is skipped by imaging.
            self.assertTrue(mesh.IsInstanceProxy())
            self.assertTrue(mesh.GetParent().IsA(UsdGeom.Xform))
            self.assertEqual(UsdGeom.Imageable(mesh).ComputeVisibility(), "inherited")
            self.assertEqual(UsdGeom.Imageable(mesh).ComputePurpose(), "default")
            self.assertFalse(mesh.HasAPI(UsdPhysics.CollisionAPI))

    def test_visual_points_topology_and_poses_match_compiled_mjcf(self):
        for gid in range(self.model.ngeom):
            if self.model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
                continue
            body = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[gid])
            wrapper = self.stage.GetPrimAtPath(f"/Go1/{body}/visual_{gid}")
            mesh = UsdGeom.Mesh(wrapper.GetChild("geometry"))
            mid = self.model.geom_dataid[gid]
            va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            fa, fn = self.model.mesh_faceadr[mid], self.model.mesh_facenum[mid]
            np.testing.assert_array_equal(np.array(mesh.GetPointsAttr().Get()), self.model.mesh_vert[va:va+vn])
            np.testing.assert_array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()), self.model.mesh_face[fa:fa+fn].ravel())
            x = np.array(UsdGeom.Xformable(wrapper).GetLocalTransformation()).T
            np.testing.assert_allclose(x[:3, :3], quat_r(self.model.geom_quat[gid]), atol=1e-12)
            np.testing.assert_allclose(x[:3, 3], self.model.geom_pos[gid], atol=1e-12)

    def test_controller_config_matches_native_mjcf(self):
        # Evaluate the actual configuration with small Isaac interface records.
        class Record:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        context = {"__file__": str(ASSET.parent/"go1.py"), "Path": Path,
                   "sim_utils": SimpleNamespace(ArticulationRootPropertiesCfg=Record),
                   "IdealPDActuatorCfg": Record, "ArticulationCfg": SimpleNamespace(InitialStateCfg=Record),
                   "UnitreeArticulationCfg": Record, "UnitreeUsdFileCfg": Record}
        tree = ast.parse((ASSET.parent/"go1.py").read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, "go1.py", "exec"), context)
        cfg = context["UNITREE_GO1_CFG"]
        a = cfg.actuators["GO1"]
        self.assertTrue(cfg.spawn.articulation_props.enabled_self_collisions)
        for j in self.spec["joints"]:
            kind = j["name"].split("_")[1]
            self.assertEqual(a.stiffness, j["kp"])
            self.assertEqual(a.damping, j["kv"])
            self.assertEqual(a.effort_limit[f".*_{kind}_joint"], j["force_limit_nm"])
            self.assertEqual(a.viscous_friction[f".*_{kind}_joint"], j["damping_nm_s_rad"])
            self.assertEqual(a.friction, j["frictionloss_nm"])
            self.assertEqual(a.dynamic_friction, j["frictionloss_nm"])
            self.assertEqual(a.armature, j["armature_kgm2"])
        self.assertEqual(a.effort_limit_sim, 1e9)

    def test_v3_control_period_and_no_v4_curriculum(self):
        s = GO1_CFG.read_text()
        self.assertIn("self.sim.dt = 0.002", s)
        self.assertIn("self.decimation = 10", s)
        self.assertIn("func=mdp.lin_vel_cmd_levels", s)
        self.assertNotIn("lin_vel_cmd_levels_with_tracking", s)
        self.assertIn("minimum_height\": 0.14", s)


class StartupChecks(unittest.TestCase):
    def setUp(self):
        import sys
        import torch
        from types import ModuleType
        self.torch = torch
        sim = ModuleType("isaaclab.sim")
        sim.modify_collision_properties = lambda *args: None
        sim.CollisionPropertiesCfg = lambda **kw: SimpleNamespace(**kw)
        isaac = ModuleType("isaaclab")
        isaac.sim = sim
        go1 = ModuleType("unitree_rl_lab.assets.robots.go1")
        go1.GO1_PHYSICS_MANIFEST = ASSET / "physics.json"
        path = GO1_CFG.with_name("physics.py")
        loader = importlib.util.spec_from_file_location("go1_physics_test", path)
        self.module = importlib.util.module_from_spec(loader)
        with patch.dict(sys.modules, {"isaaclab": isaac, "isaaclab.sim": sim, "unitree_rl_lab.assets.robots.go1": go1}):
            loader.loader.exec_module(self.module)
        spec = json.loads((ASSET/"physics.json").read_text())
        stage = Usd.Stage.Open(str(ASSET/"go1.usda"))
        body_names = list(spec["bodies"])
        parts = [usd_mass(stage.GetPrimAtPath("/Go1/"+n)) for n in body_names]
        self.mass = torch.tensor([[m for m, _, _ in parts]], dtype=torch.float32)
        self.inertia = torch.tensor(np.array([[i.ravel() for _, _, i in parts]]), dtype=torch.float32)
        coms = torch.tensor(np.array([[np.r_[c, 0., 0., 0., 1.] for _, c, _ in parts]]), dtype=torch.float32)
        # Deliberately differ from the SDK order: validation must use names.
        joints = list(reversed(spec["joints"]))
        t = lambda values: torch.tensor([values], dtype=torch.float32)
        self.friction = t([[j["frictionloss_nm"], j["frictionloss_nm"], j["damping_nm_s_rad"]] for j in joints])
        view = SimpleNamespace(get_masses=lambda: self.mass, get_coms=lambda: coms, get_inertias=lambda: self.inertia,
                               get_dof_friction_properties=lambda: self.friction,
                               get_dof_armatures=lambda: t([j["armature_kgm2"] for j in joints]),
                               get_dof_limits=lambda: t([j["limits_rad"] for j in joints]),
                               get_dof_stiffnesses=lambda: t([0.]*12), get_dof_dampings=lambda: t([0.]*12))
        robot = SimpleNamespace(is_fixed_base=False, root_physx_view=view, body_names=body_names,
                                joint_names=[j["name"] for j in joints], actuators={"GO1": SimpleNamespace(
                                    stiffness=t([j["kp"] for j in joints]), damping=t([j["kv"] for j in joints]),
                                    effort_limit=t([j["force_limit_nm"] for j in joints]))})
        self.env = SimpleNamespace(scene={"robot": robot}, step_dt=.02, physics_dt=.002)

    def test_correct_loaded_values_pass(self):
        with patch("builtins.print"):
            report = self.module.validate_go1_mjcf_physics(self.env)
        self.assertAlmostEqual(report["mass_kg"], 12.743448, places=5)
        self.assertEqual(report["motors"], 12)

    def test_stale_mass_is_rejected(self):
        self.mass[0, 0] += 8.
        with self.assertRaisesRegex(RuntimeError, "mass"):
            self.module.validate_go1_mjcf_physics(self.env)

    def test_misinterpreted_or_missing_passive_friction_is_rejected(self):
        self.friction[..., 2] = .8
        with self.assertRaisesRegex(RuntimeError, "passive friction"):
            self.module.validate_go1_mjcf_physics(self.env)

    def test_rotated_or_inferred_inertia_is_rejected(self):
        self.inertia[0, 0, 0] += .001
        with self.assertRaisesRegex(RuntimeError, "inertia"):
            self.module.validate_go1_mjcf_physics(self.env)


if __name__ == "__main__":
    unittest.main()
