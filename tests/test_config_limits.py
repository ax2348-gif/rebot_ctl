"""Optional q_min_deg / q_max_deg: joint-space limits on q, alongside the URDF's."""
import math
import unittest

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from reBotArm_control_py.actuator import RebotArm, joint_limits, load_cfg
from reBotArm_control_py.actuator.rebotarm import JointCfg, JointLimitError, load_joint_limits

DM_CONFIGS = ("rebotarm_dm.yaml", "rebotarm_dm_soft.yaml", "rebotarm_dm_safe.yaml")


def cfg_with(**limits):
    """A config whose joint1 / gripper carry the given q_min / q_max (rad)."""
    cfg = load_cfg("rebotarm_dm_safe.yaml")
    joints = []
    for jc in cfg["joints"]:
        extra = limits.get(jc.name, {})
        joints.append(JointCfg(**{**jc.__dict__, "q_min": None, "q_max": None, **extra}))
    return {**cfg, "joints": joints}


class ConfigLimitTests(unittest.TestCase):
    def test_yaw_bounded_at_minus_39_in_every_damiao_config(self):
        for name in DM_CONFIGS:
            cfg = load_cfg(name)
            urdf = load_joint_limits(cfg["urdf_path"])
            lim = joint_limits(cfg)
            self.assertAlmostEqual(math.degrees(lim["joint1"][0]), -39.0, msg=name)
            self.assertEqual(lim["joint1"][1], urdf["joint1"][1], name)
            for j in ("joint2", "joint3", "joint4", "joint5", "joint6"):
                self.assertEqual(lim[j], urdf[j], f"{name}: {j} unchanged")

    def test_no_fields_means_urdf_limits(self):
        cfg = cfg_with()
        self.assertEqual(joint_limits(cfg), load_joint_limits(cfg["urdf_path"]))

    def test_urdf_limit_still_applies_when_config_is_wider(self):
        cfg = cfg_with(joint1={"q_min": math.radians(-400), "q_max": math.radians(400)})
        self.assertEqual(joint_limits(cfg)["joint1"], load_joint_limits(cfg["urdf_path"])["joint1"])

    def test_limit_on_a_joint_without_urdf_limits(self):
        cfg = cfg_with(gripper={"q_max": 1.0})
        self.assertEqual(joint_limits(cfg)["gripper"], (-math.inf, 1.0))

    def test_contradictory_limits_rejected(self):
        cfg = cfg_with(joint1={"q_min": 1.0, "q_max": 0.5})
        with self.assertRaises(ValueError):
            joint_limits(cfg)

    def test_setpoint_limits_vs_connect_check(self):
        robot = RebotArm("rebotarm_dm_safe.yaml")
        self.assertAlmostEqual(math.degrees(robot.setpoint_limits["joint1"][0]), -39.0)
        urdf = load_joint_limits(load_cfg("rebotarm_dm_safe.yaml")["urdf_path"])
        self.assertEqual(robot._joint_limits, urdf, "connect() checks the URDF limits only")
        # -60° is outside the config limit but inside the URDF's: connect() must allow it;
        # -260° is outside the URDF's (beyond the 0.2 rad margin) and is refused.
        for deg, ok in ((-60.0, True), (-270.0, False)):
            state = {n: SimpleNamespace(pos=0.0) for n in robot.joint_names}
            state["joint2"] = state["joint3"] = SimpleNamespace(pos=-0.5)
            state["joint1"] = SimpleNamespace(pos=math.radians(deg))
            robot._motor_map = {n: SimpleNamespace(request_feedback=lambda: None,
                                                   get_state=lambda n=n: state[n])
                                for n in robot.joint_names}
            robot._ctrl_map = {}
            if ok:
                robot.check_joint_limits(timeout=0.0)
            else:
                with self.assertRaises(JointLimitError) as e:
                    robot.check_joint_limits(timeout=0.0)
                self.assertIn("joint1", str(e.exception))


class RecMotor:
    def __init__(self):
        self.sent = []

    def send_mit(self, p, v, kp, kd, tau):
        self.sent.append(("mit", p, v, kp))

    def send_pos_vel(self, p, vlim):
        self.sent.append(("pv", p))


class SetpointEnforcementTests(unittest.TestCase):
    """RebotArm's groups refuse position setpoints outside setpoint_limits, for every caller."""

    def setUp(self):
        self.robot = RebotArm("rebotarm_dm_safe.yaml")
        self.motors = {n: RecMotor() for n in self.robot.joint_names}
        self.robot._motor_map.update(self.motors)  # the groups share this dict
        self.arm = self.robot.arm
        self.lo = self.robot.setpoint_limits["joint1"][0]  # -39°
        self.j1 = self.motors["joint1"].sent

    def q(self, j1):
        q = np.full(self.arm.num_joints, -0.5)
        q[0] = j1
        return q

    def test_mit_past_the_limit_is_refused_and_the_last_setpoint_held(self):
        self.arm.send_mit(self.q(self.lo + 0.01), vel=np.full(6, 0.3))
        self.arm.send_mit(self.q(self.lo - 0.05), vel=np.full(6, 0.3))
        self.assertEqual(len(self.j1), 2)
        self.assertAlmostEqual(self.j1[1][1], self.lo + 0.01, msg="held, not the illegal one")
        self.assertEqual(self.j1[1][2], 0.0, "no velocity feedforward while held")
        self.assertAlmostEqual(self.motors["joint2"].sent[1][1], -0.5, msg="other joints unaffected")
        self.arm.send_mit(self.q(self.lo + 0.02))
        self.assertAlmostEqual(self.j1[2][1], self.lo + 0.02, msg="legal again: sent")

    def test_pos_vel_is_enforced_too(self):
        self.arm.send_pos_vel(self.q(0.0))
        self.arm.send_pos_vel(self.q(math.radians(-45)))
        self.assertEqual([s[1] for s in self.j1], [0.0, 0.0])

    def test_no_correction_a_joint_outside_may_only_come_back(self):
        # Never commanded before and the first setpoint is illegal: nothing is sent at all.
        self.arm.send_mit(self.q(math.radians(-50)))
        self.assertEqual(self.j1, [])
        self.assertEqual(len(self.motors["joint2"].sent), 1)
        # Slack (kp = 0): position ignored, so not checked and not remembered.
        z = np.zeros(6)
        self.arm.send_mit(self.q(math.radians(-50)), kp=z, kd=z)
        self.assertEqual(len(self.j1), 1)
        # Once a setpoint outside was legal (no further out), going back toward legal is fine,
        # going further out is not.
        self.arm._q_last[0] = math.radians(-50)
        self.arm.send_mit(self.q(math.radians(-45)))
        self.arm.send_mit(self.q(math.radians(-48)))
        self.assertAlmostEqual(math.degrees(self.j1[1][1]), -45)
        self.assertAlmostEqual(math.degrees(self.j1[2][1]), -45, msg="further out: held")

    def test_nan_setpoint_is_refused(self):
        self.arm.send_mit(self.q(0.1))
        self.arm.send_mit(self.q(float("nan")))
        self.assertEqual(self.j1[1][1], 0.1)

    def test_added_groups_get_limits(self):
        g = self.robot.add_group("yaw", ["joint1"])
        g.send_pos_vel(np.array([self.lo - 0.1]), vlim=np.array([1.0]))
        self.assertEqual(self.j1, [])


if __name__ == "__main__":
    unittest.main()
