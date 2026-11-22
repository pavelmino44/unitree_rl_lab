"""Batched T2 joint/step/PD checks; collision checks live in safety_reference."""

import torch

from .safety_reference.joint_limits import STEP_LIMIT_RAD, tau_limit

# Keep float32 roundoff inside the strict double-precision Safety Block limits.
FLOAT_GUARD_RAD = 1e-6
TORQUE_LIMITS = tuple(tau_limit(index) for index in range(12))


def candidate_targets(requested, previous, q_min, q_max, max_delta=STEP_LIMIT_RAD):
    finite = torch.isfinite(requested).all(dim=-1)
    safe = torch.where(torch.isfinite(requested), requested, previous)
    bounded = safe.clamp(q_min, q_max)
    delta = bounded - previous
    step = max_delta - FLOAT_GUARD_RAD
    candidate = previous + delta.clamp(-step, step)
    bound_clip = (safe != bounded).float().mean(dim=-1)
    step_clip = (delta.abs() > step).float().mean(dim=-1)
    return candidate, finite, bound_clip, step_clip


def torque_acceptance(candidate, measured_q, measured_dq, kp, kd, limits):
    # LowCmd contract: desired joint velocity=0, feedforward torque=0.
    torques = kp * (candidate - measured_q) - kd * measured_dq
    accepted = (torch.isfinite(torques) & (torques.abs() <= limits)).all(dim=-1)
    return accepted, torques


def accept_targets(candidate, previous, finite, torque_ok, collision_ok):
    accepted = finite & torque_ok & collision_ok
    return torch.where(accepted[:, None], candidate, previous), accepted
