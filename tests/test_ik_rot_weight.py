"""IKParams.rot_weight: with the wrist roll at its limit, a low weight keeps the TCP in place."""
import math
import unittest

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik
from reBotArm_control_py.kinematics.robot_model import (
    get_arm_joint_count, get_end_effector_frame_id, load_robot_model,
)
from reBotArm_control_py.kinematics.forward_kinematics import compute_fk


class RotWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = load_robot_model()
        ee_id = get_end_effector_frame_id(model)
        ee = model.frames[ee_id]
        offset = pin.SE3(np.eye(3), np.array([-0.15, 0.0, 0.0]))   # quest teleop TCP
        cls.fid = model.addFrame(pin.Frame("tcp", ee.parentJoint, ee_id, ee.placement * offset,
                                           pin.FrameType.OP_FRAME))
        cls.model, cls.data = model, model.createData()
        cls.n = get_arm_joint_count(model, cls.fid)

    def pose(self, q_arm):
        q = pin.neutral(self.model)
        q[:self.n] = q_arm
        p, R, _ = compute_fk(self.model, q, "tcp")
        pin.forwardKinematics(self.model, self.data, q)
        roll_axis = self.data.oMi[self.model.getJointId("joint6")].rotation @ np.array([0, 0, 1.0])
        return p, R, roll_axis

    def solve(self, q0, target, w):
        r = solve_ik(self.model, self.data, self.fid, target, q0,
                     IKParams(max_iter=500, rot_weight=w), controlled_joints=self.n)
        p, R, _ = compute_fk(self.model, np.r_[r.q, np.zeros(self.model.nq - self.n)], "tcp")
        return r, p, R

    def test_roll_past_limit_keeps_tcp_with_low_weight(self):
        q0 = np.array([0.0, -1.0, -1.2, 0.0, 0.3, math.radians(190.0)])
        p0, R0, axis = self.pose(q0)
        target = pin.SE3(pin.exp3(axis * math.radians(20.0)) @ R0, p0)   # needs joint6 at 210°
        _, p_w, _ = self.solve(q0, target, 0.02)
        _, p_1, _ = self.solve(q0, target, 1.0)
        self.assertLess(np.linalg.norm(p_w - p0), 0.002)       # stays put
        self.assertGreater(np.linalg.norm(p_1 - p0), 0.02)     # unweighted drags the TCP

    def test_reachable_target_still_reached(self):
        q0 = np.array([0.3, -1.1, -1.1, 0.2, -0.3, 1.0])
        p0, R0, _ = self.pose(q0)
        axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        target = pin.SE3(pin.exp3(axis * math.radians(25.0)) @ R0, p0 + np.array([0.03, -0.02, 0.02]))
        r, p, R = self.solve(q0, target, 0.02)
        self.assertTrue(r.success)
        self.assertLess(np.linalg.norm(p - target.translation), 1e-3)
        self.assertLess(np.linalg.norm(pin.log3(R.T @ target.rotation)), math.radians(0.5))


if __name__ == "__main__":
    unittest.main()
