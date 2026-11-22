import math

JOINT_NAMES = [
    'FR_hip', 'FR_thigh', 'FR_calf',
    'FL_hip', 'FL_thigh', 'FL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
]

Q_MIN = [-0.863, -0.686, -2.818] * 4
Q_MAX = [ 0.863,  4.501, -0.888] * 4

DQ_MIN, DQ_MAX = -20.0, 20.0
KP_MIN, KP_MAX =   0.0, 1000.0
KD_MIN, KD_MAX =   0.0, 100.0
TAU_MIN, TAU_MAX = -100.0, 100.0

TAU_LIMIT_HIP = 23.7
TAU_LIMIT_THIGH = 23.7
TAU_LIMIT_CALF = 35.55

def tau_limit(joint_index: int) -> float:
    t = joint_index % 3
    if t == 0:
        return TAU_LIMIT_HIP
    if t == 1:
        return TAU_LIMIT_THIGH
    return TAU_LIMIT_CALF

STEP_LIMIT_RAD = math.radians(5.0)
INTERP_STEP_RAD = math.radians(0.1)
COLLISION_DEPTH_THRESHOLD = 0.001   # 1 mm

MODE_ACTIVE = 0x0A
