"""JointGroup enable / disable / clear_error act on the group's own motors only.

The groups share one controller per vendor; its enable_all() would also enable the other
groups (enabling the arm used to enable the gripper, which then tripped its CAN watchdog).
"""
import unittest
from unittest.mock import Mock

from reBotArm_control_py.actuator.rebotarm import JointCfg, JointGroup


class GroupEnableTests(unittest.TestCase):
    def setUp(self):
        names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
        self.joints = [JointCfg(name=n, motor_id=i + 1, feedback_id=0x11 + i, model="4310")
                       for i, n in enumerate(names)]
        self.motors = {n: Mock() for n in names}
        self.ctrl = Mock()
        self.arm = JointGroup("arm", names[:6], self.joints, self.motors, {"damiao": self.ctrl})
        self.gripper = JointGroup("gripper", ["gripper"], self.joints, self.motors,
                                  {"damiao": self.ctrl})

    def test_arm_enable_leaves_gripper_alone(self):
        self.arm.enable()
        for i in range(1, 7):
            self.motors[f"joint{i}"].enable.assert_called_once_with()
        self.motors["gripper"].enable.assert_not_called()
        self.ctrl.enable_all.assert_not_called()

    def test_gripper_ops_touch_only_the_gripper(self):
        for op in ("enable", "disable", "clear_error"):
            getattr(self.gripper, op)()
            getattr(self.motors["gripper"], op).assert_called_once_with()
            for i in range(1, 7):
                getattr(self.motors[f"joint{i}"], op).assert_not_called()
        self.ctrl.enable_all.assert_not_called()
        self.ctrl.disable_all.assert_not_called()


if __name__ == "__main__":
    unittest.main()
