"""Build the Go1 USD from the unmodified, SHA-pinned hackathon MJCF.

Offline build only: numpy, mujoco==3.3.7, usd-core. Training uses the shipped
USD and does not import MuJoCo. No Isaac importer, mass defaults or hull-based
replacement of capsules are used here. PhysX still has its own contact solver.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ASSET = Path(__file__).resolve().parents[2] / "source/unitree_rl_lab/unitree_rl_lab/assets/robots/go1_mjcf"
SOURCE_SHA = "d5a7466784c8e72fd174cc7fa5a92754a1a0a6d458a1dec79e7ca0abc9525c4a"
ROOT = "/Go1"
FOOT_MASS = 0.06
# The original MJCF lumped inertia includes the URDF's 20 mm mass sphere,
# although its contact sphere has radius 23 mm. Geometry does not override mass.
FOOT_INERTIA = np.eye(3) * 9.6e-6


def quat_matrix(q):
    r = np.empty(9)
    mj.mju_quat2Mat(r, np.asarray(q))
    return r.reshape(3, 3)


def principal(inertia):
    values, axes = np.linalg.eigh(inertia)
    if np.min(values) <= 0 or 2*np.max(values) > values.sum() + 1e-12:
        raise ValueError(f"Invalid rigid-body inertia: {values}")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = np.empty(4)
    mj.mju_mat2Quat(q, axes.ravel())
    return values, q


def shifted_inertia(mass, offset):
    return mass * (np.dot(offset, offset)*np.eye(3) - np.outer(offset, offset))


def split_calf(mass, com, inertia):
    foot_pos = np.array([0., 0., -0.213])
    calf_mass = mass - FOOT_MASS
    calf_com = (mass*com - FOOT_MASS*foot_pos)/calf_mass
    calf_i = (inertia - FOOT_INERTIA - shifted_inertia(FOOT_MASS, foot_pos-com)
              - shifted_inertia(calf_mass, calf_com-com))
    principal(calf_i)  # Reject an invalid split rather than let PhysX infer it.
    return calf_mass, calf_com, calf_i


def attr(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def api(prim, name):
    # Codeless PhysX schemas are resolved when the USD is opened by Isaac Sim.
    prim.AddAppliedSchema(name)


def pose(prim, pos, quat):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))


def make_material(stage, name, mu):
    mat = UsdShade.Material.Define(stage, f"{ROOT}/materials/{name}")
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(mu)
    physics.CreateDynamicFrictionAttr(mu)
    physics.CreateRestitutionAttr(0.)
    api(mat.GetPrim(), "PhysxMaterialAPI")
    attr(mat.GetPrim(), "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, "multiply")
    attr(mat.GetPrim(), "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, "multiply")
    return mat


def build():
    source = ASSET / "reference/go1.xml"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("The reference go1.xml does not match the T2/T3 SHA-256")
    if mj.__version__ != "3.3.7":
        raise RuntimeError("Rebuild with mujoco==3.3.7, the judge version")
    model = mj.MjModel.from_xml_path(str(source))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    # A shared visual layer avoids duplicating the large STL meshes per leg.
    visuals_path = ASSET / "visuals.usdc"
    visuals = Usd.Stage.CreateNew(str(visuals_path))
    visual_root = UsdGeom.Xform.Define(visuals, "/Visuals")
    visuals.SetDefaultPrim(visual_root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(visuals, 1.)
    UsdGeom.SetStageUpAxis(visuals, UsdGeom.Tokens.z)
    for mid in range(model.nmesh):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH, mid)
        # Instance the enclosing Xform, never a leaf Gprim. USD imaging skips
        # directly-instanced Mesh roots; geometry must live below the root.
        UsdGeom.Xform.Define(visuals, f"/Visuals/{name}")
        mesh = UsdGeom.Mesh.Define(visuals, f"/Visuals/{name}/geometry")
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        points = model.mesh_vert[va:va+vn].astype(np.float32)
        faces = model.mesh_face[fa:fa+fn].astype(np.int32).ravel()
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(fn, 3, dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.2, .2, .2)])
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    visuals.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(ASSET / "go1.usda"))
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().customLayerData = {"source_sha256": SOURCE_SHA, "generator": "build_go1_mjcf_asset.py"}
    foot_mat = make_material(stage, "foot", .8)
    body_mat = make_material(stage, "body", 1.)  # MJCF floor/body pair: max(1, .6).
    bodies = {}
    groups = {}
    identity = np.array([1., 0., 0., 0.])
    for bid in range(1, model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        mass = float(model.body_mass[bid])
        com = model.body_ipos[bid].copy()
        r = quat_matrix(model.body_iquat[bid])
        inertia = r @ np.diag(model.body_inertia[bid]) @ r.T
        group = dict(mass_kg=mass, com_m=com.tolist(), inertia_kgm2=inertia.tolist())
        groups[name] = group
        # All USD links are siblings, with joint frames encoding the tree.
        world_pos = data.xpos[bid]-data.xpos[1]
        if name.endswith("_calf"):
            mass, com, inertia = split_calf(mass, com, inertia)
            foot_name = name.removesuffix("_calf") + "_foot"
            bodies[foot_name] = dict(source_body=name, mass_kg=FOOT_MASS, com_m=[0., 0., 0.],
                                     inertia_kgm2=FOOT_INERTIA.tolist(), pos_m=(world_pos+[0., 0., -.213]).tolist())
        bodies[name] = dict(source_body=name, mass_kg=mass, com_m=com.tolist(), inertia_kgm2=inertia.tolist(), pos_m=world_pos.tolist())

    for name, spec in bodies.items():
        body = UsdGeom.Xform.Define(stage, f"{ROOT}/{name}")
        pose(body.GetPrim(), spec["pos_m"], identity)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        values, q = principal(np.array(spec["inertia_kgm2"]))
        mass.CreateMassAttr(spec["mass_kg"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*spec["com_m"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
        api(body.GetPrim(), "PhysxRigidBodyAPI")
        attr(body.GetPrim(), "physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, 0.)
        attr(body.GetPrim(), "physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, 0.)
        if name == "trunk":
            UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
            api(body.GetPrim(), "PhysxArticulationAPI")
            attr(body.GetPrim(), "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, True)

    joints = []
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        bid = model.jnt_bodyid[jid]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        child = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        parent = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.body_parentid[bid])
        dof = model.jnt_dofadr[jid]
        axis = np.argmax(model.jnt_axis[jid])
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/joints/{name}")
        joint.CreateBody0Rel().SetTargets([f"{ROOT}/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"{ROOT}/{child}"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, model.body_pos[bid])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateAxisAttr("XYZ"[axis])
        joint.CreateLowerLimitAttr(math.degrees(model.jnt_range[jid, 0]))
        joint.CreateUpperLimitAttr(math.degrees(model.jnt_range[jid, 1]))
        joint.CreateCollisionEnabledAttr(False)
        # Zero USD drive: the explicit Isaac actuator computes only servo torque.
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.)
        drive.CreateDampingAttr(0.)
        drive.CreateMaxForceAttr(1e9)
        api(joint.GetPrim(), "PhysxJointAPI")
        attr(joint.GetPrim(), "physxJoint:armature", Sdf.ValueTypeNames.Float, float(model.dof_armature[dof]))
        # Runtime ActuatorCfg also sets these through the 5.x tensor API (SI).
        api(joint.GetPrim(), "PhysxJointAxisAPI:angular")
        for key, value in dict(armature=model.dof_armature[dof], staticFrictionEffort=model.dof_frictionloss[dof],
                               dynamicFrictionEffort=model.dof_frictionloss[dof],
                               viscousFrictionCoefficient=model.dof_damping[dof]*math.pi/180).items():
            attr(joint.GetPrim(), "physxJointAxis:angular:"+key, Sdf.ValueTypeNames.Float, float(value))
        joints.append(dict(name=name, parent=parent, child=child, axis=model.jnt_axis[jid].tolist(),
                           anchor_m=model.body_pos[bid].tolist(), limits_rad=model.jnt_range[jid].tolist(),
                           kp=float(model.actuator_gainprm[aid, 0]), kv=float(-model.actuator_biasprm[aid, 2]),
                           force_limit_nm=float(model.actuator_forcerange[aid, 1]),
                           damping_nm_s_rad=float(model.dof_damping[dof]), frictionloss_nm=float(model.dof_frictionloss[dof]),
                           armature_kgm2=float(model.dof_armature[dof])))
    for leg in ("FR", "FL", "RR", "RL"):
        joint = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/joints/{leg}_foot_fixed")
        joint.CreateBody0Rel().SetTargets([f"{ROOT}/{leg}_calf"])
        joint.CreateBody1Rel().SetTargets([f"{ROOT}/{leg}_foot"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0., 0., -.213))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint.CreateCollisionEnabledAttr(False)

    # Reproduce MuJoCo same/welded-body and parent-child collision filtering,
    # including thigh/foot after splitting the calf's welded foot geometry.
    excluded = []
    names = list(bodies)
    for i, a in enumerate(names):
        ga = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bodies[a]["source_body"])
        for b in names[i+1:]:
            gb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bodies[b]["source_body"])
            if ga == gb or model.body_parentid[ga] == gb or model.body_parentid[gb] == ga:
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{ROOT}/{a}")).CreateFilteredPairsRel().AddTarget(f"{ROOT}/{b}")
                excluded.append([a, b])

    geoms = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        original_body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        original_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        foot = original_name in ("FR", "FL", "RR", "RL")
        body = original_name+"_foot" if foot else original_body
        pos = model.geom_pos[gid].copy()
        if foot:
            pos -= np.array([0., 0., -.213])
        q = model.geom_quat[gid]
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            mesh_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH, mid)
            prim = stage.DefinePrim(f"{ROOT}/{body}/visual_{gid}")
            prim.GetReferences().AddReference("./visuals.usdc", f"/Visuals/{mesh_name}")
            prim.SetInstanceable(True)
            pose(prim, pos, q)
            continue
        size = model.geom_size[gid]
        path = f"{ROOT}/{body}/collision_{gid}"
        kind = int(model.geom_type[gid])
        if kind == mj.mjtGeom.mjGEOM_BOX:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(2.)
        elif kind == mj.mjtGeom.mjGEOM_SPHERE:
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(float(size[0]))
        elif kind in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            cls = UsdGeom.Cylinder if kind == mj.mjtGeom.mjGEOM_CYLINDER else UsdGeom.Capsule
            geom = cls.Define(stage, path)
            geom.CreateRadiusAttr(float(size[0]))
            geom.CreateHeightAttr(float(2*size[1]))
            geom.CreateAxisAttr("Z")
        else:
            raise ValueError(f"Unsupported MJCF geometry {gid}: {kind}")
        prim = geom.GetPrim()
        pose(prim, pos, q)
        if kind == mj.mjtGeom.mjGEOM_BOX:
            UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        api(prim, "PhysxCollisionAPI")
        attr(prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, .0005)
        attr(prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(foot_mat if foot else body_mat, materialPurpose="physics")
        geom.CreatePurposeAttr("guide")
        geoms.append(dict(id=gid, path=path, body=body, source_body=original_body, type=kind,
                          pos_m=pos.tolist(), quat_wxyz=q.tolist(), size=size.tolist(),
                          source_friction=model.geom_friction[gid].tolist(), source_condim=int(model.geom_condim[gid]),
                          source_solimp=model.geom_solimp[gid].tolist(), source_solref=model.geom_solref[gid].tolist()))
    stage.GetRootLayer().Save()
    result = dict(source_sha256=SOURCE_SHA, source_mujoco_version=mj.__version__,
                  total_mass_kg=float(model.body_mass.sum()), bodies=bodies, mjcf_groups=groups,
                  joints=joints, collision_geoms=geoms, filtered_body_pairs=excluded,
                  physics_dt_s=.002, policy_dt_s=.02,
                  unmapped_contacts=["MuJoCo solimp/solref and elliptic cone", "condim 1/6 per pair",
                                     "MuJoCo rolling/torsional friction coefficients", "PhysX cylinder cooking differs from MuJoCo analytic cylinders"],
                  controller="Unmodified MJCF position servo; LowCmd bridge overrides must be matched separately")
    for filename in ("go1.usda", "visuals.usdc"):
        result[filename+"_sha256"] = hashlib.sha256((ASSET/filename).read_bytes()).hexdigest()
    (ASSET / "physics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Built {len(bodies)} bodies, {len(joints)} motors, {len(geoms)} collision shapes; {result['total_mass_kg']:.6f} kg")


if __name__ == "__main__":
    build()
