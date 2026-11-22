from .joint_limits import tau_limit

def compute_torques(q_cmd, dq_cmd, kp, kd, tau, q_meas, dq_meas):
    return [
        tau[i] + kp[i] * (q_cmd[i] - q_meas[i]) + kd[i] * (dq_cmd[i] - dq_meas[i])
        for i in range(12)
    ]

def check_torque_limits(torques):
    for i, t in enumerate(torques):
        if abs(t) > tau_limit(i):
            return False
    return True
