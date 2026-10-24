"""Exercise RebotArm's connect-time joint-limit check with fake motors (no hardware)."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reBotArm_control_py.actuator import JointLimitError, RebotArm


class FakeMotor:
    def __init__(self, pos, timeout_reg=2000):
        self.pos = pos
        self.timeout_reg = timeout_reg  # Damiao TIMEOUT register (2000 = 100 ms)
        self.request_feedback = Mock()

    def get_state(self):
        return None if self.pos is None else SimpleNamespace(pos=self.pos, status_code=0)

    def get_register_u32(self, rid, timeout_ms):
        assert rid == 9
        return self.timeout_reg


class JointLimitCheckTests(unittest.TestCase):
    def connect_with(self, readings, **kwargs):
        """Connect a DM arm whose motors report `readings` (joint name -> rad, None = silent)."""
        robot = RebotArm("rebotarm_dm.yaml")
        ctrl = Mock()

        def fake_setup():
            robot._ctrl_map["damiao"] = ctrl
            for jc in robot._all_joints:
                robot._motor_map[jc.name] = FakeMotor(readings.get(jc.name, 0.0))

        with patch.object(robot, "_setup_motors", side_effect=fake_setup), \
                patch("reBotArm_control_py.actuator.rebotarm.time.sleep"):
            robot.connect(**kwargs)
        return robot, ctrl

    def test_limits_loaded_from_urdf(self):
        robot = RebotArm("rebotarm_dm.yaml")
        self.assertEqual(robot._joint_limits["joint2"], (-3.14, 0.0))

    def test_in_range_readings_connect(self):
        robot, ctrl = self.connect_with({"joint2": -1.0, "joint3": 0.05})
        self.assertTrue(robot._connected)
        ctrl.shutdown.assert_not_called()

    def test_two_pi_offset_names_joint_and_turns(self):
        with self.assertRaises(JointLimitError) as cm:
            self.connect_with({"joint2": 2 * math.pi - 0.01})
        msg = str(cm.exception)
        self.assertIn("joint2", msg)
        self.assertIn("+1×2π", msg)
        self.assertNotIn("joint3", msg)

    def test_disconnects_on_failure(self):
        robot = RebotArm("rebotarm_dm.yaml")
        ctrl = Mock()

        def fake_setup():
            robot._ctrl_map["damiao"] = ctrl
            for jc in robot._all_joints:
                robot._motor_map[jc.name] = FakeMotor(2 * math.pi if jc.name == "joint2" else 0.0)

        with patch.object(robot, "_setup_motors", side_effect=fake_setup), \
                patch("reBotArm_control_py.actuator.rebotarm.time.sleep"):
            with self.assertRaises(JointLimitError):
                robot.connect()
        self.assertFalse(robot._connected)
        ctrl.shutdown.assert_called_once()

    def test_missing_feedback_fails(self):
        with self.assertRaises(JointLimitError) as cm:
            self.connect_with({"joint1": None}, check_limits=True)
        self.assertIn("joint1: no position feedback", str(cm.exception))

    def test_check_limits_false_skips_check(self):
        robot, _ = self.connect_with({"joint2": 2 * math.pi}, check_limits=False)
        self.assertTrue(robot._connected)
        with self.assertRaises(JointLimitError):
            robot.check_joint_limits(timeout=0.0)


if __name__ == "__main__":
    unittest.main()
