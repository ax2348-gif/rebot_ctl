"""RebotArm.connect() must refuse motors whose CAN watchdog isn't CAN_TIMEOUT_MS (no hardware)."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from motorbridge import CallError
from reBotArm_control_py.actuator import CAN_TIMEOUT_MS, CanTimeoutError, RebotArm


class FakeMotor:
    def __init__(self, timeout_reg):
        self.timeout_reg = timeout_reg  # int, or an Exception to raise
        self.request_feedback = Mock()
        # JointGroup.enable()/disable()/clear_error() act per motor
        self.enable, self.disable, self.clear_error = Mock(), Mock(), Mock()

    def get_state(self):
        return SimpleNamespace(pos=0.0, status_code=0)

    def get_register_u32(self, rid, timeout_ms):
        assert rid == 9
        if isinstance(self.timeout_reg, Exception):
            raise self.timeout_reg
        return self.timeout_reg


class CanTimeoutCheckTests(unittest.TestCase):
    def connect_with(self, regs, vendors=None, **kwargs):
        robot = RebotArm("rebotarm_dm.yaml")
        ctrl = Mock()
        for jc in robot._all_joints:
            if vendors and jc.name in vendors:
                jc.vendor = vendors[jc.name]

        def fake_setup():
            for jc in robot._all_joints:
                robot._ctrl_map[jc.vendor] = ctrl
                robot._motor_map[jc.name] = FakeMotor(regs.get(jc.name, 2000))

        with patch.object(robot, "_setup_motors", side_effect=fake_setup), \
                patch("reBotArm_control_py.actuator.rebotarm.time.sleep"):
            robot.connect(**kwargs)
        return robot, ctrl

    def test_expected_value_is_100ms(self):
        self.assertEqual(CAN_TIMEOUT_MS, 100)

    def test_correct_watchdog_connects(self):
        robot, ctrl = self.connect_with({})
        self.assertTrue(robot._connected)
        ctrl.shutdown.assert_not_called()

    def test_wrong_value_refuses_and_disconnects(self):
        with self.assertRaises(CanTimeoutError) as cm:
            self.connect_with({"joint4": 0, "gripper": 4000})
        msg = str(cm.exception)
        self.assertIn("joint4: TIMEOUT = 0 (0 ms)", msg)
        self.assertIn("gripper: TIMEOUT = 4000 (200 ms)", msg)
        self.assertNotIn("joint1", msg)
        self.assertIn("persist_can_timeout.py", msg)

    def test_disconnects_on_failure(self):
        robot = RebotArm("rebotarm_dm.yaml")
        ctrl = Mock()

        def fake_setup():
            robot._ctrl_map["damiao"] = ctrl
            for jc in robot._all_joints:
                robot._motor_map[jc.name] = FakeMotor(0)

        with patch.object(robot, "_setup_motors", side_effect=fake_setup), \
                patch("reBotArm_control_py.actuator.rebotarm.time.sleep"):
            with self.assertRaises(CanTimeoutError):
                robot.connect()
        self.assertFalse(robot._connected)
        ctrl.shutdown.assert_called_once()

    def test_unreadable_register_refuses(self):
        with self.assertRaises(CanTimeoutError) as cm:
            self.connect_with({"joint2": CallError("register 9 not received")})
        self.assertIn("joint2: could not read TIMEOUT", str(cm.exception))

    def test_non_damiao_motor_refuses(self):
        with self.assertRaises(CanTimeoutError) as cm:
            self.connect_with({}, vendors={"joint1": "robstride"})
        self.assertIn("joint1: vendor 'robstride', watchdog cannot be read back", str(cm.exception))

    def test_check_limits_false_does_not_skip_watchdog_check(self):
        with self.assertRaises(CanTimeoutError):
            self.connect_with({"joint1": 0}, check_limits=False)


if __name__ == "__main__":
    unittest.main()
