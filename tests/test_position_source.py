"""Exercise RebotArm.get_state(position_source=...) with fake motors (no hardware)."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from motorbridge import CallError
from reBotArm_control_py.actuator import RebotArm


class OldFakeMotor:
    """Damiao-like motor on a motorbridge without the non-blocking register API."""

    def __init__(self, pos, xout):
        self.pos = pos
        self.xout = xout
        self.request_feedback = Mock()
        self.requests = []        # register ids requested (non-blocking)
        self.blocking_reads = []  # register ids read with the blocking call

    def get_state(self):
        return SimpleNamespace(pos=self.pos, vel=0.5, torq=0.25, status_code=1)

    def deliver(self):
        pass

    def get_register_f32(self, rid, timeout_ms):
        self.blocking_reads.append(rid)
        if self.xout is None:
            raise CallError("register read timeout")
        return self.xout


class FakeMotor(OldFakeMotor):
    """Adds the non-blocking API; replies arrive when the controller polls."""

    def __init__(self, pos, xout):
        super().__init__(pos, xout)
        self._pending = False
        self._cached = None  # (value, fresh)

    def request_register(self, rid):
        self.requests.append(rid)
        self._pending = True
        if self._cached is not None:
            self._cached = (self._cached[0], False)

    def deliver(self):
        if self._pending and self.xout is not None:
            self._cached = (self.xout, True)
            self._pending = False

    def get_cached_register_f32(self, rid, fresh_only=True):
        if self._cached is None or (fresh_only and not self._cached[1]):
            return None
        return self._cached[0], 0.001


class PositionSourceTests(unittest.TestCase):
    def make_robot(self, xout=lambda i: 6.0 + i, old=False):
        robot = RebotArm("rebotarm_dm.yaml")
        motors = []
        cls = OldFakeMotor if old else FakeMotor
        for i, jc in enumerate(robot._all_joints):
            m = cls(pos=0.1 * i, xout=xout(i))
            robot._motor_map[jc.name] = m
            motors.append(m)
        ctrl = Mock()
        ctrl.poll_feedback_once.side_effect = lambda: [m.deliver() for m in motors]
        robot._ctrl_map["damiao"] = ctrl
        return robot, motors

    def test_default_is_feedback_and_reads_no_registers(self):
        robot, motors = self.make_robot()
        pos, vel, torq = robot.get_state()
        np.testing.assert_allclose(pos, 0.1 * np.arange(len(motors)))
        np.testing.assert_allclose(robot.get_positions(), pos)
        for m in motors:
            self.assertEqual((m.requests, m.blocking_reads), ([], []))

    def test_output_uses_batched_requests_and_keeps_vel_torq(self):
        robot, motors = self.make_robot()
        pos, vel, torq = robot.get_state(position_source="output")
        np.testing.assert_allclose(pos, 6.0 + np.arange(len(motors)))
        np.testing.assert_allclose(vel, 0.5)
        np.testing.assert_allclose(torq, 0.25)
        for m in motors:
            self.assertEqual(m.requests, [81])
            self.assertEqual(m.blocking_reads, [])

    def test_second_read_never_returns_previous_value(self):
        robot, motors = self.make_robot()
        robot.get_state(position_source="output")
        motors[5].xout = None  # joint6 stops answering
        with self.assertRaises(CallError) as cm:
            robot.get_state(position_source="output")
        self.assertIn("joint6", str(cm.exception))
        self.assertNotIn("joint5", str(cm.exception))

    def test_missing_reply_times_out(self):
        robot, _ = self.make_robot(xout=lambda i: None if i == 2 else 1.0)
        with self.assertRaises(CallError) as cm:
            robot.get_state(position_source="output")
        self.assertIn("joint3", str(cm.exception))

    def test_falls_back_to_blocking_read_on_old_motorbridge(self):
        robot, motors = self.make_robot(old=True)
        pos = robot.get_positions(position_source="output")
        np.testing.assert_allclose(pos, 6.0 + np.arange(len(motors)))
        for m in motors:
            self.assertEqual(m.blocking_reads, [81])

    def test_unknown_source_rejected(self):
        robot, _ = self.make_robot()
        with self.assertRaises(ValueError):
            robot.get_state(position_source="rotor")


if __name__ == "__main__":
    unittest.main()
