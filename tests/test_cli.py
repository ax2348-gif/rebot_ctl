"""Exercise the rebot_* command-line tools against fakes (no hardware)."""
import contextlib
import io
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from reBotArm_control_py.actuator.rebotarm import dm_timeout_register_value, CAN_TIMEOUT_MS
from reBotArm_control_py.cli import setup as setup_cli
from reBotArm_control_py.cli import vis as vis_cli
from reBotArm_control_py.cli import zero as zero_cli

HW = "rebotarm_dm.yaml"
GOOD_WD = dm_timeout_register_value(CAN_TIMEOUT_MS)


def run(fn, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(*args)
    return code, out.getvalue()


# ---------------------------------------------------------------- rebot_setup


class FakeMotor:
    def __init__(self, status=0, watchdog=GOOD_WD, pos=0.0, answers=True):
        self.status, self.watchdog, self.pos, self.answers = status, watchdog, pos, answers
        self.writes, self.stores = [], 0

    def request_feedback(self):
        pass

    def get_state(self):
        return SimpleNamespace(status_code=self.status, pos=self.pos) if self.answers else None

    def get_register_u32(self, rid, timeout_ms):
        return self.watchdog if rid == 9 else 1

    def get_register_f32(self, rid, timeout_ms):
        return 1.0

    def write_register_u32(self, rid, value):
        self.writes.append((rid, value))

    def store_parameters(self):
        self.stores += 1

    def close(self):
        pass


class SetupCheckTests(unittest.TestCase):
    def check(self, motors):
        it = iter(motors)
        ctrl = Mock()
        ctrl.add_damiao_motor.side_effect = lambda *a: next(it)
        with patch("motorbridge.Controller.from_dm_serial", return_value=ctrl):
            code, out = run(setup_cli.main, [HW])
        for name in ("enable_all", "disable_all", "shutdown"):
            getattr(ctrl, name).assert_not_called()
        for m in motors:
            self.assertEqual((m.writes, m.stores), ([], 0), "check mode must not write")
        return code, out

    def test_all_good(self):
        code, out = self.check([FakeMotor() for _ in range(7)])
        self.assertEqual(code, 0)
        self.assertIn("OK: every motor answers", out)

    def test_wrong_watchdog_points_to_store(self):
        motors = [FakeMotor() for _ in range(7)]
        motors[2].watchdog = 0
        code, out = self.check(motors)
        self.assertEqual(code, 1)
        self.assertIn("joint3: watchdog is 0 ms", out)
        self.assertIn("rebot_setup --store-watchdog", out)

    def test_missing_motor_and_fault_reported(self):
        motors = [FakeMotor() for _ in range(7)]
        motors[0].answers = False
        motors[4].status = 0xA
        code, out = self.check(motors)
        self.assertEqual(code, 1)
        self.assertIn("joint1: no feedback", out)
        self.assertIn("joint5: FAULT overcurrent", out)

    def test_channel_override(self):
        motors = iter([FakeMotor() for _ in range(7)])
        ctrl = Mock()
        ctrl.add_damiao_motor.side_effect = lambda *a: next(motors)
        with patch("motorbridge.Controller.from_dm_serial", return_value=ctrl) as opened:
            code, out = run(setup_cli.main, [HW, "--channel", "/dev/ttyACM0"])
        self.assertEqual(code, 0)
        opened.assert_called_once_with("/dev/ttyACM0", 921600)
        self.assertIn("on /dev/ttyACM0", out)

    def test_out_of_limits_points_to_zero(self):
        motors = [FakeMotor() for _ in range(7)]
        motors[1].pos = 7.0  # far outside any joint limit
        code, out = self.check(motors)
        self.assertEqual(code, 1)
        self.assertIn("joint2: reads", out)
        self.assertIn("rebot_zero", out)


# ---------------------------------------------------------------- RebotArm fake


class FakeGroup:
    def __init__(self, n):
        self.num_joints = n
        self.sent = []
        self.mode = None

    def mode_mit(self):
        self.mode = "mit"
        return True

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None):
        self.sent.append((pos, vel, kp, kd, tau))


class FakeArm:
    """Records every call in order in ``calls``."""

    def __init__(self, hw_yaml=None, n_arm=6, status=0x1, positions=None):
        self.hardware_yaml = hw_yaml or "fake.yaml"
        self._channel = "/dev/fake"
        self.calls = []
        self.arm, self.gripper = FakeGroup(n_arm), FakeGroup(1)
        names = [f"joint{i + 1}" for i in range(n_arm)] + ["gripper"]
        self.joint_names = names
        self._all_joints = [SimpleNamespace(name=n, vendor="damiao") for n in names]
        self.status = status
        self._motor_map = {n: SimpleNamespace(clear_error=lambda n=n: self.calls.append(("clear_error", n)),
                                              get_state=lambda: SimpleNamespace(status_code=self.status))
                           for n in names}
        self.positions = positions if positions is not None else np.zeros(len(names))
        self.controller = None
        self.reads = 0

    def __getattr__(self, name):
        if name in ("connect", "disable_all", "enable_all", "set_zero", "check_joint_limits",
                    "disconnect"):
            return lambda *a, **k: self.calls.append((name, a, k))
        raise AttributeError(name)

    def start_control_loop(self, fn, rate=None):
        self.calls.append(("start_control_loop", rate))
        self.controller = fn
        fn(self, 0.01)

    def get_positions(self, **kwargs):
        self.reads += 1
        return self.positions

    def names(self):
        return [c[0] for c in self.calls]


# ---------------------------------------------------------------- rebot_zero


class ZeroTests(unittest.TestCase):
    def run_zero(self, answer, argv=(HW,)):
        arm = FakeArm()
        with patch.object(zero_cli, "RebotArm", return_value=arm), \
                patch("builtins.input", return_value=answer):
            code, out = run(zero_cli.main, list(argv))
        self.assertNotIn("enable_all", arm.names(), "rebot_zero must never enable")
        self.assertEqual(arm.names()[-1], "disconnect")
        connect = next(c for c in arm.calls if c[0] == "connect")
        self.assertEqual(connect[2], {"check_limits": False})
        return code, out, arm

    def test_needs_confirmation(self):
        code, out, arm = self.run_zero("yes")
        self.assertEqual(code, 1)
        self.assertNotIn("set_zero", arm.names())
        self.assertIn("nothing changed", out)

    def test_zeroes_and_rechecks_limits(self):
        code, out, arm = self.run_zero("ZERO")
        self.assertEqual(code, 0)
        names = arm.names()
        self.assertLess(names.index("set_zero"), names.index("check_joint_limits"))


# ---------------------------------------------------------------- rebot_vis


class VisTests(unittest.TestCase):
    def run_vis(self, arm, stop_after=3):
        viewer = Mock(port=7000)

        def update(pos):
            if viewer.update.call_count >= stop_after:
                signal.raise_signal(signal.SIGINT)

        viewer.update.side_effect = update
        old = signal.getsignal(signal.SIGINT)
        try:
            with patch.object(vis_cli, "RebotArm", return_value=arm), \
                    patch.object(vis_cli, "ArmViewer", return_value=viewer), \
                    patch.object(vis_cli.time, "sleep"):
                code, out = run(vis_cli.main, [HW])
        finally:
            signal.signal(signal.SIGINT, old)
        return code, out, viewer

    def test_slack_stream_starts_before_enable(self):
        arm = FakeArm()
        code, out, viewer = self.run_vis(arm)
        self.assertEqual(code, 0)
        names = arm.names()
        self.assertLess(names.index("start_control_loop"), names.index("enable_all"))
        self.assertEqual(names[-1], "disconnect")
        self.assertEqual(arm.arm.mode, "mit")
        for group in (arm.arm, arm.gripper):
            self.assertTrue(group.sent)
            for pos, vel, kp, kd, tau in group.sent:
                for v in (kp, kd, tau):
                    self.assertTrue(np.all(np.asarray(v) == 0), "every gain and torque must be 0")
        self.assertGreaterEqual(viewer.update.call_count, 3)
        self.assertIn(":7000/static/", out)

    def test_motor_fault_stops(self):
        arm = FakeArm(status=0x8)
        code, out, viewer = self.run_vis(arm, stop_after=100)
        self.assertEqual(code, 1)
        self.assertIn("Motor fault", out)
        self.assertEqual(arm.names()[-1], "disconnect")

    def test_error_still_disconnects(self):
        arm = FakeArm()
        arm.get_positions = Mock(side_effect=RuntimeError("bus gone"))
        with self.assertRaises(RuntimeError):
            self.run_vis(arm)
        self.assertEqual(arm.names()[-1], "disconnect")


if __name__ == "__main__":
    unittest.main()
