"""Exercise RebotArm.get_state(position_source=...) with fake motors (no hardware)."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from motorbridge import CallError
from reBotArm_control_py.actuator import RebotArm


class FakeMotor:
    def __init__(self, pos, xout):
        self.pos = pos
        self.xout = xout
        self.request_feedback = Mock()
        self.register_reads = []

    def get_state(self):
        return SimpleNamespace(pos=self.pos, vel=0.5, torq=0.25, status_code=1)

    def get_register_f32(self, rid, timeout_ms):
        self.register_reads.append(rid)
        if self.xout is None:
            raise CallError("register read timeout")
        return self.xout


class PositionSourceTests(unittest.TestCase):
    def make_robot(self, xout=lambda i: 6.0 + i):
        robot = RebotArm("rebotarm_dm.yaml")
        robot._ctrl_map["damiao"] = Mock()
        for i, jc in enumerate(robot._all_joints):
            robot._motor_map[jc.name] = FakeMotor(pos=0.1 * i, xout=xout(i))
        return robot

    def test_default_is_feedback_and_reads_no_registers(self):
        robot = self.make_robot()
        pos, vel, torq = robot.get_state()
        np.testing.assert_allclose(pos, 0.1 * np.arange(len(robot._all_joints)))
        np.testing.assert_allclose(robot.get_positions(), pos)
        for m in robot._motor_map.values():
            self.assertEqual(m.register_reads, [])

    def test_output_reads_xout_for_damiao_and_keeps_vel_torq(self):
        robot = self.make_robot()
        pos, vel, torq = robot.get_state(position_source="output")
        np.testing.assert_allclose(pos, 6.0 + np.arange(len(robot._all_joints)))
        np.testing.assert_allclose(vel, 0.5)
        np.testing.assert_allclose(torq, 0.25)
        np.testing.assert_allclose(robot.get_positions(position_source="output"), pos)
        for m in robot._motor_map.values():
            self.assertTrue(set(m.register_reads) == {81})

    def test_output_read_failure_raises_instead_of_zero(self):
        robot = self.make_robot(xout=lambda i: None if i == 5 else 1.0)
        with self.assertRaises(CallError):
            robot.get_state(position_source="output")

    def test_unknown_source_rejected(self):
        robot = self.make_robot()
        with self.assertRaises(ValueError):
            robot.get_state(position_source="rotor")


if __name__ == "__main__":
    unittest.main()
