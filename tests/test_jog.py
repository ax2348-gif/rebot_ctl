"""rebot_jog: e-stop, control loop, move checks and startup flow (no hardware)."""
import contextlib
import io
import math
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reBotArm_control_py.actuator import load_cfg
from reBotArm_control_py.actuator.rebotarm import load_joint_limits
from reBotArm_control_py.cli import jog
from reBotArm_control_py.cli._estop import EStop

SAFE = jog.DEFAULT_HW
CFG = load_cfg(SAFE)
LIMITS = load_joint_limits(CFG["urdf_path"])
NAMES = [f"joint{i}" for i in range(1, 7)]
# A comfortable pose inside every joint's limits (joint1's range is shifted on mount_right).
Q0 = np.array([-1.57, -0.6, -0.9, 0.0, 0.3, 0.0])


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- fakes


class FakeMotor:
    def __init__(self, pos=0.0):
        self.pos, self.status, self.present = pos, 0x1, True

    def get_state(self):
        return SimpleNamespace(pos=self.pos, status_code=self.status) if self.present else None

    def request_feedback(self):
        pass

    def clear_error(self):
        pass


class FakeGroup:
    def __init__(self, names, kp, kd):
        self.joint_names, self.num_joints = list(names), len(names)
        self._mit_kp, self._mit_kd = np.array(kp, float), np.array(kd, float)
        self.sent, self.mit_ok, self.fail_send = [], True, False

    def mode_mit(self):
        return self.mit_ok

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None):
        if self.fail_send:
            raise RuntimeError("bus gone")
        self.sent.append(dict(pos=np.array(pos), vel=np.array(vel), kp=np.array(kp),
                              kd=np.array(kd), tau=np.array(tau)))


class FakeArm:
    def __init__(self, q=Q0, kp=12.0, kd=8.0):
        self.arm = FakeGroup(NAMES, [kp] * 6, [kd] * 6)
        self.gripper = FakeGroup(["gripper"], [8.0], [1.0])
        self._motor_map = {n: FakeMotor(v) for n, v in zip(NAMES, q)}
        self._motor_map["gripper"] = FakeMotor(0.0)
        self._ctrl_map = {"damiao": SimpleNamespace(poll_feedback_once=lambda: None)}
        self.calls, self.loop, self.rate = [], None, 200.0
        self.hardware_yaml, self.channel = SAFE, "/dev/fake"

    @property
    def control_loop_active(self):
        return self.loop is not None

    def start_control_loop(self, fn, rate=None):
        self.calls.append("start_control_loop")
        self.loop = fn
        fn(self, 0.005)

    def __getattr__(self, name):
        if name in ("connect", "enable_all", "disable_all", "disconnect"):
            return lambda *a, **k: self.calls.append(name)
        raise AttributeError(name)


def controller(arm=None, clock=None, gravity=None, estop=None):
    arm = arm or FakeArm()
    estop = estop or EStop(grace_s=None)
    clock = clock or Clock()
    ctrl = jog.JogController(arm, estop, Q0.copy(), np.zeros(1), LIMITS, gravity, 0.35, clock=clock)
    return arm, estop, clock, ctrl


# ---------------------------------------------------------------- e-stop


class EStopTests(unittest.TestCase):
    def test_signal_trips_from_the_watcher_thread(self):
        e = EStop(grace_s=None).install()
        try:
            e.quiet()  # no KeyboardInterrupt: the trip must not depend on the main thread
            os.kill(os.getpid(), signal.SIGINT)
            self.assertTrue(e.wait(1.0), "watcher tripped the stop")
            self.assertEqual(e.reason, "SIGINT received")
        finally:
            e.uninstall()

    def test_main_thread_gets_keyboard_interrupt(self):
        e = EStop(grace_s=None).install()
        try:
            with self.assertRaises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(1.0)
            self.assertTrue(e.wait(1.0))
        finally:
            e.uninstall()

    def run_child(self, body, timeout=5.0):
        code = ("import os, signal, time\n"
                "from reBotArm_control_py.cli._estop import EStop\n" + body)
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stderr, time.monotonic() - t0

    def test_second_signal_hard_exits(self):
        rc, err, dt = self.run_child(
            "e = EStop(grace_s=60).install(); e.quiet()\n"
            "os.kill(os.getpid(), signal.SIGINT); time.sleep(0.2)\n"
            "os.kill(os.getpid(), signal.SIGINT); time.sleep(10)\n")
        self.assertEqual(rc, 130)
        self.assertIn("second signal", err)
        self.assertLess(dt, 4.0)

    def test_hung_shutdown_hard_exits_after_grace(self):
        rc, err, dt = self.run_child(
            "e = EStop(grace_s=0.5).install()\n"
            "e.trip('test'); time.sleep(10)\n")
        self.assertEqual(rc, 1)
        self.assertIn("shutdown took over 0.5 s", err)
        self.assertLess(dt, 4.0)


# ---------------------------------------------------------------- control loop


class ControllerTests(unittest.TestCase):
    def test_before_enable_only_zero_gains_at_measured_pose(self):
        arm, estop, clock, ctrl = controller()
        ctrl(arm, 0.005)
        cmd = arm.arm.sent[-1]
        self.assertTrue(np.allclose(cmd["pos"], Q0))
        for k in ("kp", "kd", "tau", "vel"):
            self.assertTrue(np.all(cmd[k] == 0), k)
        self.assertTrue(np.all(arm.gripper.sent[-1]["kp"] == 0))

    def test_gains_ramp_in_after_enable(self):
        arm, estop, clock, ctrl = controller()
        ctrl.mark_enabled()
        clock.t += jog.GAIN_RAMP_S / 2
        ctrl(arm, 0.005)
        mid = arm.arm.sent[-1]["kp"]
        self.assertTrue(np.all((mid > 0) & (mid < 12.0)))
        clock.t += jog.GAIN_RAMP_S
        ctrl(arm, 0.005)
        self.assertTrue(np.allclose(arm.arm.sent[-1]["kp"], 12.0))
        self.assertTrue(np.allclose(arm.arm.sent[-1]["pos"], Q0), "still holding the start pose")

    def test_nothing_sent_once_tripped(self):
        arm, estop, clock, ctrl = controller()
        estop.trip("test")
        ctrl(arm, 0.005)
        self.assertEqual(arm.arm.sent, [])
        self.assertEqual(arm.gripper.sent, [])

    def assert_trips(self, arm, ctrl, estop, fragment):
        before = len(arm.arm.sent)
        ctrl(arm, 0.005)
        self.assertTrue(estop.tripped)
        self.assertIn(fragment, estop.reason)
        self.assertEqual(len(arm.arm.sent), before, "no command on the faulting tick")
        ctrl(arm, 0.005)
        self.assertEqual(len(arm.arm.sent), before, "nor afterwards")

    def test_motor_fault_trips(self):
        arm, estop, clock, ctrl = controller()
        arm._motor_map["joint3"].status = 0xA
        self.assert_trips(arm, ctrl, estop, "joint3: FAULT overcurrent")

    def test_missing_feedback_trips(self):
        arm, estop, clock, ctrl = controller()
        arm._motor_map["gripper"].present = False
        self.assert_trips(arm, ctrl, estop, "stopped reporting")

    def test_outside_limits_trips(self):
        arm, estop, clock, ctrl = controller()
        arm._motor_map["joint2"].pos = 3.0
        self.assert_trips(arm, ctrl, estop, "joint2")

    def test_tracking_error_trips_after_ramp(self):
        arm, estop, clock, ctrl = controller()
        ctrl.mark_enabled()
        arm._motor_map["joint4"].pos = Q0[3] + 0.5
        ctrl(arm, 0.005)  # during the ramp: no tracking check yet
        self.assertFalse(estop.tripped)
        clock.t += jog.GAIN_RAMP_S + 0.01
        self.assert_trips(arm, ctrl, estop, "joint4 is")

    def test_exception_in_loop_trips_instead_of_dying(self):
        arm, estop, clock, ctrl = controller()
        arm.arm.fail_send = True
        ctrl(arm, 0.005)  # must not raise
        self.assertTrue(estop.tripped)
        self.assertIn("control loop error", estop.reason)

    def test_gravity_torque_ramps_in(self):
        g = lambda q: np.full(6, 2.0)
        arm, estop, clock, ctrl = controller(gravity=g)
        ctrl(arm, 0.005)
        self.assertTrue(np.all(arm.arm.sent[-1]["tau"] == 0), "no torque before enable")
        ctrl.mark_enabled()
        clock.t += jog.GAIN_RAMP_S
        ctrl(arm, 0.005)
        self.assertTrue(np.allclose(arm.arm.sent[-1]["tau"], 2.0))

    def test_segment_follows_min_jerk(self):
        arm, estop, clock, ctrl = controller()
        ctrl.mark_enabled()
        clock.t += jog.GAIN_RAMP_S
        q1 = Q0 + 0.1
        ctrl.go(q1, 1.0)
        clock.t += 0.5
        mid = ctrl.setpoint()
        self.assertTrue(np.allclose(mid, Q0 + 0.05, atol=1e-9))
        clock.t += 0.6
        self.assertTrue(np.allclose(ctrl.setpoint(), q1))
        self.assertFalse(ctrl.moving())


# ---------------------------------------------------------------- move checks


class JoggerTests(unittest.TestCase):
    def setUp(self):
        self.kin = jog.Kinematics(CFG["urdf_path"], CFG["end_effector_frame"], 6)
        self.arm, self.estop, self.clock, self.ctrl = controller()
        self.ctrl.mark_enabled()
        self.clock.t += jog.GAIN_RAMP_S
        self.args = jog.build_parser().parse_args([])

    def jogger(self, **overrides):
        for k, v in overrides.items():
            setattr(self.args, k, v)
        return jog.Jogger(self.ctrl, self.kin, self.args)

    def test_step_moves_the_end_effector_by_one_step(self):
        j = self.jogger()
        start = np.array(self.kin.fk(Q0).translation)
        msg = j.request("x", +1)
        self.assertTrue(msg.startswith("x +"), msg)
        end = np.array(self.kin.fk(self.ctrl.goal()).translation)
        self.assertTrue(np.allclose(end - start, [0.01, 0, 0], atol=2e-4), end - start)

    def test_duration_respects_speed_limits(self):
        j = self.jogger(cart_speed=0.01, joint_speed=1.0)
        j.request("z", +1)
        seg = self.ctrl._seg
        self.assertGreaterEqual(seg.T, jog.MIN_JERK_PEAK * 0.01 / 0.01 - 1e-9)
        peak = jog.MIN_JERK_PEAK * np.max(np.abs(seg.q1 - seg.q0)) / seg.T
        self.assertLessEqual(peak, 1.0 + 1e-9)

    def test_refuses_below_z_min(self):
        z = float(self.kin.fk(Q0).translation[2])
        j = self.jogger(z_min=z - 0.005)
        msg = j.request("z", -1)
        self.assertIn("below --z-min", msg)
        self.assertTrue(np.allclose(self.ctrl.goal(), Q0), "no motion planned")

    def test_refuses_large_joint_jump(self):
        j = self.jogger(max_jump=0.001)
        msg = j.request("x", +1)
        self.assertIn("would jump", msg)
        self.assertTrue(np.allclose(self.ctrl.goal(), Q0))

    def test_refuses_near_joint_limit(self):
        lo4 = LIMITS["joint4"][0]
        q = Q0.copy()
        q[3] = lo4 + jog.LIMIT_MARGIN_RAD / 2  # already inside the margin
        _, _, clock, ctrl = controller()
        ctrl._seg = jog.Segment(q, q, clock(), 0.0)
        ctrl.mark_enabled()
        clock.t += jog.GAIN_RAMP_S
        msg = jog.Jogger(ctrl, self.kin, self.args).request("x", +1)
        self.assertIn("refused", msg)

    def test_no_moves_while_ramping(self):
        _, _, clock, ctrl = controller()
        ctrl.mark_enabled()
        self.assertIn("ramping", jog.Jogger(ctrl, self.kin, self.args).request("x", +1))

    def test_hard_ceilings(self):
        p = jog.build_parser()
        for bad in (["--step", "0.2"], ["--cart-speed", "1"], ["--joint-speed", "3"],
                    ["--rot-step", "45"], ["--step", "nan"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                p.parse_args(bad)
        self.assertEqual(p.parse_args([]).hardware_yaml, SAFE)


# ---------------------------------------------------------------- startup flow


class FakeKeyboard:
    keys = []

    def __init__(self):
        self._keys = list(type(self).keys)

    def read(self, timeout):
        if not self._keys:
            return "q"
        k = self._keys.pop(0)
        if k == "^C":
            raise KeyboardInterrupt
        return k

    def restore(self):
        pass


class MainTests(unittest.TestCase):
    def run_main(self, arm, argv=("--no-gravity",), answer="JOG", keys=(), tty=True):
        FakeKeyboard.keys = list(keys)
        out = io.StringIO()
        with patch.object(jog, "RebotArm", return_value=arm), \
                patch.object(jog, "Keyboard", FakeKeyboard), \
                patch.object(jog.sys.stdin, "isatty", return_value=tty), \
                patch("builtins.input", return_value=answer), \
                contextlib.redirect_stdout(out):
            code = jog.main(list(argv))
        return code, out.getvalue()

    def test_needs_a_terminal(self):
        arm = FakeArm()
        code, out = self.run_main(arm, tty=False)
        self.assertEqual(code, 2)
        self.assertEqual(arm.calls, [])

    def test_refuses_stiff_config_before_connecting(self):
        arm = FakeArm(kp=120.0)
        code, out = self.run_main(arm)
        self.assertEqual(code, 2)
        self.assertIn("exceed rebot_jog's ceilings", out)
        self.assertEqual(arm.calls, [])

    def test_no_confirmation_enables_nothing(self):
        arm = FakeArm()
        code, out = self.run_main(arm, answer="yes")
        self.assertEqual(code, 1)
        self.assertNotIn("enable_all", arm.calls)
        self.assertNotIn("start_control_loop", arm.calls)
        self.assertEqual(arm.calls[-1], "disconnect")

    def test_no_mit_mode_enables_nothing(self):
        arm = FakeArm()
        arm.gripper.mit_ok = False
        code, out = self.run_main(arm)
        self.assertEqual(code, 1)
        self.assertNotIn("enable_all", arm.calls)

    def test_session_order_and_clean_stop(self):
        arm = FakeArm()
        code, out = self.run_main(arm, keys=["w", "p", "q"])
        self.assertEqual(code, 0, out)
        c = arm.calls
        self.assertLess(c.index("connect"), c.index("start_control_loop"))
        self.assertLess(c.index("start_control_loop"), c.index("enable_all"))
        self.assertEqual(c[-2:], ["disable_all", "disconnect"])
        self.assertIn("stopped by user", out)

    def test_ctrl_c_stops_commands_then_disables(self):
        arm = FakeArm()
        code, out = self.run_main(arm, keys=["w", "^C"])
        self.assertEqual(code, 1)
        sent = len(arm.arm.sent)
        arm.loop(arm, 0.005)  # the control thread's next tick after Ctrl+C
        self.assertEqual(len(arm.arm.sent), sent, "no commands after Ctrl+C")
        self.assertEqual(arm.calls[-2:], ["disable_all", "disconnect"])


if __name__ == "__main__":
    unittest.main()
