import csv
import math
import numpy as np
import mujoco

from .joint_limits import INTERP_STEP_RAD, COLLISION_DEPTH_THRESHOLD

class CollisionChecker:
    def __init__(self, model_xml_path: str, pairs_csv_path: str,
                 threshold: float = COLLISION_DEPTH_THRESHOLD):
        # scene.xml включает go1.xml и пол; geom-id совпадают с CSV
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.threshold = threshold

        # 12 hinge joints
        self.joint_qpos = []
        self.joint_names = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.joint_qpos.append(int(self.model.jnt_qposadr[j]))
                self.joint_names.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                )
        if len(self.joint_qpos) != 12:
            raise RuntimeError(
                f'Expected 12 hinge joints, got {len(self.joint_qpos)}: '
                f'{self.joint_names}'
            )

        # base free joint (для поднятия корпуса над полом в проверке)
        self.base_qpos = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.base_qpos = int(self.model.jnt_qposadr[j])
                break
        if self.base_qpos is None:
            raise RuntimeError('Model has no free base joint')

        self.pairs = self._load_pairs(pairs_csv_path)

    def _load_pairs(self, path: str):
        pairs = set()
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError('CSV has no header')
            if 'geom1_id' not in reader.fieldnames or 'geom2_id' not in reader.fieldnames:
                raise RuntimeError(f'CSV missing geom ids; have: {reader.fieldnames}')
            for row in reader:
                policy = (row.get('diagnostic_policy') or '').strip().lower()
                if policy != 'checked':
                    continue
                g1 = int(row['geom1_id'])
                g2 = int(row['geom2_id'])
                pairs.add((min(g1, g2), max(g1, g2)))
        if not pairs:
            raise RuntimeError('No rows with diagnostic_policy == "checked"')
        return pairs

    def _set_q(self, q12):
        # Поднимаем корпус, чтобы пол не мешал: self-collision от этого не меняется
        if self.base_qpos is not None:
            self.data.qpos[self.base_qpos:self.base_qpos + 3] = (0.0, 0.0, 2.0)
            self.data.qpos[self.base_qpos + 3:self.base_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        for k, adr in enumerate(self.joint_qpos):
            self.data.qpos[adr] = q12[k]
        mujoco.mj_forward(self.model, self.data)

    def check_config(self, q12) -> bool:
        self._set_q(q12)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            key = (min(g1, g2), max(g1, g2))
            if key not in self.pairs:
                continue
            if con.dist < -self.threshold:
                return False
        return True

    def check_transition(self, q_from, q_to) -> bool:
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        delta = q_to - q_from
        max_abs = float(np.max(np.abs(delta)))
        if max_abs <= 1e-9:
            return self.check_config(q_from)
        n = int(math.ceil(max_abs / INTERP_STEP_RAD))
        for k in range(n + 1):
            t = k / n
            q = q_from + t * delta
            if not self.check_config(q):
                return False
        return True
