"""Go1-only startup verification of the properties actually loaded by PhysX."""
from __future__ import annotations

import json

import torch

from unitree_rl_lab.assets.robots.go1 import GO1_PHYSICS_MANIFEST


def _close(actual, expected, name, *, atol=2e-6, rtol=2e-5):
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"Go1 MJCF physics mismatch: {name}; actual={actual.tolist()}, expected={expected.tolist()}. "
            "Use the complete physics update on v3 with Isaac Sim >=5.0."
        )


def validate_go1_mjcf_physics(env, env_ids=None):
    """Fail early on stale assets or lost inertias/friction, once at startup.

    The tensor API reports inertias about COM in the BODY frame. Do not rotate
    them by get_coms quaternion again. The COM array uses xyz + xyzw quaternion.
    Only the first identically cloned environment is inspected (no mass DR).
    """
    print("[GO1 MJCF PHYSICS] validating loaded PhysX properties", flush=True)
    spec = json.loads(GO1_PHYSICS_MANIFEST.read_text())
    robot = env.scene["robot"]
    view = robot.root_physx_view
    if robot.is_fixed_base:
        raise RuntimeError("Go1 MJCF asset must have a floating base")
    if set(robot.body_names) != set(spec["bodies"]):
        raise RuntimeError(f"Go1 body set differs from the 17-body MJCF asset: {robot.body_names}")
    masses = view.get_masses()[0].detach().cpu()
    coms = view.get_coms()[0, :, :3].detach().cpu()
    inertias = view.get_inertias()[0].detach().cpu().reshape(-1, 3, 3)
    for i, name in enumerate(robot.body_names):
        b = spec["bodies"][name]
        _close(masses[i], b["mass_kg"], f"{name} mass")
        _close(coms[i], b["com_m"], f"{name} COM")
        _close(inertias[i], b["inertia_kgm2"], f"{name} inertia", atol=3e-8, rtol=2e-4)
    _close(masses.sum(), spec["total_mass_kg"], "total mass", atol=2e-5, rtol=0.)
    if not hasattr(view, "get_dof_friction_properties"):
        raise RuntimeError("Go1 MJCF joint friction requires Isaac Sim >=5.0; legacy friction is dimensionless")
    friction = view.get_dof_friction_properties()[0].detach().cpu()
    armature = view.get_dof_armatures()[0].detach().cpu()
    usd_kp = view.get_dof_stiffnesses()[0].detach().cpu()
    usd_kd = view.get_dof_dampings()[0].detach().cpu()
    limits = view.get_dof_limits()[0].detach().cpu()
    actuator = robot.actuators["GO1"]
    if len(robot.joint_names) != 12 or set(robot.joint_names) != {j["name"] for j in spec["joints"]}:
        raise RuntimeError(f"Unexpected Go1 joints: {robot.joint_names}")
    for j in spec["joints"]:
        i = robot.joint_names.index(j["name"])
        _close(friction[i], [j["frictionloss_nm"], j["frictionloss_nm"], j["damping_nm_s_rad"]], j["name"]+" passive friction")
        _close(armature[i], j["armature_kgm2"], j["name"]+" armature")
        _close(limits[i], j["limits_rad"], j["name"]+" limits")
        _close(actuator.stiffness[0, i].detach().cpu(), j["kp"], j["name"]+" servo kp")
        _close(actuator.damping[0, i].detach().cpu(), j["kv"], j["name"]+" servo kv")
        _close(actuator.effort_limit[0, i].detach().cpu(), j["force_limit_nm"], j["name"]+" motor cap")
    # Explicit PD plus passive joint resistance, with no second hidden drive.
    _close(usd_kp, torch.zeros_like(usd_kp), "USD stiffness must be zero")
    _close(usd_kd, torch.zeros_like(usd_kd), "USD drive damping must be zero")
    if abs(env.step_dt - spec["policy_dt_s"]) > 1e-8:
        raise RuntimeError(f"Go1 policy period changed: {env.step_dt}")
    if abs(env.physics_dt - spec["physics_dt_s"]) > 1e-8:
        raise RuntimeError(f"Go1 physics period changed: {env.physics_dt}")
    report = {
        "mass_kg": float(masses.sum()), "bodies": len(robot.body_names), "motors": len(robot.joint_names),
        "inertias_verified": True, "passive_friction_verified": True,
        "physics_dt_s": float(env.physics_dt), "policy_dt_s": float(env.step_dt),
    }
    env._go1_mjcf_physics_report = report
    print("[GO1 MJCF PHYSICS] " + json.dumps(report), flush=True)
    return report
