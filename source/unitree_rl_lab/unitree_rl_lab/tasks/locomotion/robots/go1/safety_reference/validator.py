import math

from .joint_limits import (
    Q_MIN, Q_MAX, DQ_MIN, DQ_MAX, KP_MIN, KP_MAX, KD_MIN, KD_MAX,
    TAU_MIN, TAU_MAX, STEP_LIMIT_RAD,
)

def extract_cmd_fields(motor_cmd):
    q    = [motor_cmd[i].q    for i in range(12)]
    dq   = [motor_cmd[i].dq   for i in range(12)]
    kp   = [motor_cmd[i].kp   for i in range(12)]
    kd   = [motor_cmd[i].kd   for i in range(12)]
    tau  = [motor_cmd[i].tau  for i in range(12)]
    mode = [motor_cmd[i].mode for i in range(12)]
    return q, dq, kp, kd, tau, mode

def validate_finite(q, dq, kp, kd, tau, mode):
    fields = (q, dq, kp, kd, tau, mode)
    for arr in fields:
        for v in arr:
            if not math.isfinite(float(v)):
                return False
    return True

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def clamp_q(q):  return [_clamp(q[i],  Q_MIN[i], Q_MAX[i]) for i in range(12)]
def clamp_dq(dq): return [_clamp(v, DQ_MIN, DQ_MAX) for v in dq]
def clamp_kp(kp): return [_clamp(v, KP_MIN, KP_MAX) for v in kp]
def clamp_kd(kd): return [_clamp(v, KD_MIN, KD_MAX) for v in kd]
def clamp_tau(tau): return [_clamp(v, TAU_MIN, TAU_MAX) for v in tau]

def enforce_step_limit(q_new, q_last):
    out = []
    for i in range(12):
        lo = q_last[i] - STEP_LIMIT_RAD
        hi = q_last[i] + STEP_LIMIT_RAD
        out.append(_clamp(q_new[i], lo, hi))
    return out
