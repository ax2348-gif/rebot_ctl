"""connect(check_gripper=True): close the gripper, require it to read near zero (no hardware)."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from reBotArm_control_py.actuator import GripperCheckError, RebotArm


def closed_at(pos):
    return patch.object(RebotArm, "close_gripper",
                        return_value=None if pos is None else SimpleNamespace(pos=pos))


class GripperCheckTests(unittest.TestCase):
    def setUp(self):
        self.robot = RebotArm("rebotarm_dm.yaml")

    def test_near_zero_passes(self):
        for deg in (0.0, 7.8, -19.9, 19.9):
            with closed_at(math.radians(deg)):
                self.assertAlmostEqual(self.robot.check_gripper_closed(), math.radians(deg))

    def test_two_pi_wrap_is_named(self):
        with closed_at(2 * math.pi + 0.1), self.assertRaises(GripperCheckError) as e:
            self.robot.check_gripper_closed()
        self.assertIn("+1×2π", str(e.exception))

    def test_beyond_20_deg_refused_without_rezeroing(self):
        with closed_at(math.radians(25.0)), \
                patch.object(RebotArm, "set_zero") as set_zero, \
                self.assertRaises(GripperCheckError) as e:
            self.robot.check_gripper_closed()
        self.assertIn("zero has moved", str(e.exception))
        set_zero.assert_not_called()

    def test_no_feedback_refused(self):
        with closed_at(None), self.assertRaises(GripperCheckError):
            self.robot.check_gripper_closed()


class ConnectGripperCheckTests(unittest.TestCase):
    def connect(self, **kwargs):
        robot = RebotArm("rebotarm_dm.yaml")
        with patch.object(robot, "_setup_motors"), \
                patch.object(robot, "check_can_timeout"), \
                patch.object(robot, "check_joint_limits"), \
                patch.object(robot, "check_gripper_closed") as check, \
                patch.object(robot, "disconnect") as disconnect:
            robot.connect(**kwargs)
        return check, disconnect

    def test_on_by_default(self):
        check, _ = self.connect()
        check.assert_called_once_with()

    def test_can_be_skipped(self):
        check, _ = self.connect(check_gripper=False)
        check.assert_not_called()

    def test_failure_disconnects(self):
        robot = RebotArm("rebotarm_dm.yaml")
        with patch.object(robot, "_setup_motors"), \
                patch.object(robot, "check_can_timeout"), \
                patch.object(robot, "check_joint_limits"), \
                patch.object(robot, "check_gripper_closed", side_effect=GripperCheckError("x")), \
                patch.object(robot, "disconnect") as disconnect:
            with self.assertRaises(GripperCheckError):
                robot.connect()
        disconnect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
