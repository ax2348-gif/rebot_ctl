"""rebot_jog: e-stop, control loop, follower, move checks, keyboard and startup flow.

Uses simulated motors (no hardware) and, where it matters, a real control thread.
"""
import contextlib
import io
import math
import os
import pty
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reBotArm_control_py.actuator import joint_limits, load_cfg
from reBotArm_control_py.actuator.rebotarm import load_joint_limits
from reBotArm_control_py.cli import jog
from reBotArm_control_py.cli._estop import EStop

SAFE = jog.DEFAULT_HW
CFG = load_cfg(SAFE)
URDF_LIMITS = load_joint_limits(CFG["urdf_path"])
SETPOINT_LIMITS = joint_limits(CFG)
NAMES = [f"joint{i}" for i in range(1, 7)]
# A comfortable pose inside every joint's limits, joint1 inside its config range (>= -39°).
Q0 = np.array([-0.3, -0.6, -0.9, 0.0, 0.3, 0.0])


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- simulated hardware


class SimMotor:
    """Follows its MIT setpoint when enabled with kp > 0; replies to register reads."""

    def __init__(self, pos=0.0, clock=time.monotonic):
        self.pos, self.status, self.clock = float(pos), 0x0, clock
        self.cmd = None
        self.sends = 0
        self.last_request = None
        self.dead = False          # stops replying (bus lost)
        self.fail_send = False
        self.enable_fails = False

    def get_state(self):
        return SimpleNamespace(pos=self.pos, status_code=self.status)

    def request_feedback(self):
        pass

    def clear_error(self):
        pass

    def request_register(self, rid):
        if not self.dead:
            self.last_request = self.clock()

    def get_cached_register_f32(self, rid, fresh_only=True):
        if self.last_request is None:
            return None
        return (self.pos, self.clock() - self.last_request)

    def send_mit(self, pos, vel, kp, kd, tau):
        if self.fail_send:
            raise RuntimeError("bus gone")
        self.sends += 1
        self.cmd = (pos, vel, kp, kd, tau)
        if self.status == 0x1 and kp > 0:
            self.pos += (pos - self.pos) * 0.5  # crude, fast servo


class SimGroup:
    def __init__(self, arm, names, kp, kd):
        self._arm = arm
        self.joint_names, self.num_joints = list(names), len(names)
        self._mit_kp, self._mit_kd = np.array(kp, float), np.array(kd, float)
        self.mit_ok = True

    def mode_mit(self):
        return self.mit_ok


class SimArm:
    def __init__(self, q=Q0, kp=12.0, kd=8.0, clock=time.monotonic, threaded=True):
        self.arm = SimGroup(self, NAMES, [kp] * 6, [kd] * 6)
        self.gripper = SimGroup(self, ["gripper"], [8.0], [1.0])
        self._motor_map = {n: SimMotor(v, clock) for n, v in zip(NAMES, q)}
        self._motor_map["gripper"] = SimMotor(0.0, clock)
        self._ctrl_map = {"damiao": SimpleNamespace(poll_feedback_once=lambda: None)}
        self._joint_limits = URDF_LIMITS
        self.setpoint_limits = SETPOINT_LIMITS
        self.calls, self.loop, self.rate = [], None, 200.0
        self.hardware_yaml, self.channel = SAFE, "/dev/fake"
        self._connected = False
        self._threaded = threaded
        self._running = False
        self._thread = None
        self.sent_log = []  # (time, arm setpoints) of every tick that sent

    @property
    def control_loop_active(self):
        return self._thread is not None and self._thread.is_alive() if self._threaded else self.loop is not None

    def connect(self, check_limits=True):
        self.calls.append("connect")
        self._connected = True

    def enable_all(self):
        self.calls.append("enable_all")
        for m in self._motor_map.values():
            if not m.enable_fails:
                m.status = 0x1

    def disable_all(self):
        self.calls.append("disable_all")
        for m in self._motor_map.values():
            m.status = 0x0

    def disconnect(self):
        self.calls.append("disconnect")
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)
        self._connected = False

    def start_control_loop(self, fn, rate=None):
        self.calls.append("start_control_loop")
        self.loop = fn
        if not self._threaded:
            return
        self._running = True

        def run():
            while self._running:
                before = self._motor_map["joint1"].sends
                fn(self, 1.0 / self.rate)
                if self._motor_map["joint1"].sends != before:
                    self.sent_log.append((time.monotonic(),
                                          np.array([self._motor_map[n].cmd[0] for n in NAMES])))
                time.sleep(1.0 / self.rate)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def names(self):
        return list(self.calls)


def make_ctrl(arm=None, clock=None, gravity=None, joint_speed=0.3, joint_accel=1.0):
    clock = clock or Clock()
    arm = arm or SimArm(clock=clock, threaded=False)
    estop = EStop(grace_s=None)
    ctrl = jog.JogController(arm, estop, URDF_LIMITS, gravity, 0.35, joint_speed, joint_accel,
                             200.0, clock=clock)
    return arm, estop, clock, ctrl


def tick(arm, ctrl, clock, n=1, dt=0.005):
    for _ in range(n):
        clock.t += dt
        ctrl(arm, dt)


def enable_and_ramp(arm, ctrl, clock):
    tick(arm, ctrl, clock)
    arm.enable_all()
    ctrl.mark_enabled()
    tick(arm, ctrl, clock, 2)
    assert ctrl.verified.is_set()
    tick(arm, ctrl, clock, int(jog.GAIN_RAMP_S / 0.005) + 2)


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
        rc, err, dt = self.run_child("e = EStop(grace_s=0.5).install()\ne.trip('test'); time.sleep(10)\n")
        self.assertEqual(rc, 1)
        self.assertIn("shutdown took over 0.5 s", err)
        self.assertLess(dt, 4.0)


# ---------------------------------------------------------------- control loop


class ControllerTests(unittest.TestCase):
    def test_before_enable_zero_gains_tracking_the_measured_pose(self):
        arm, estop, clock, ctrl = make_ctrl()
        tick(arm, ctrl, clock)
        pos, vel, kp, kd, tau = arm._motor_map["joint2"].cmd
        self.assertAlmostEqual(pos, Q0[1])
        self.assertEqual((vel, kp, kd, tau), (0.0, 0.0, 0.0, 0.0))
        arm._motor_map["joint2"].pos = Q0[1] + 0.3  # someone moves the arm before enabling
        tick(arm, ctrl, clock)
        self.assertAlmostEqual(arm._motor_map["joint2"].cmd[0], Q0[1] + 0.3, msg="no stale hold pose")

    def test_ramp_starts_around_the_pose_at_enable(self):
        arm, estop, clock, ctrl = make_ctrl()
        tick(arm, ctrl, clock)
        arm._motor_map["joint3"].pos = Q0[2] + 0.2
        arm.enable_all()
        ctrl.mark_enabled()
        tick(arm, ctrl, clock, 3)
        self.assertTrue(ctrl.verified.is_set())
        self.assertAlmostEqual(ctrl.snapshot()[0][2], Q0[2] + 0.2)
        tick(arm, ctrl, clock, 100)
        kp = arm._motor_map["joint3"].cmd[2]
        self.assertTrue(0 < kp < 12.0, "gains ramping")
        tick(arm, ctrl, clock, 200)
        self.assertAlmostEqual(arm._motor_map["joint3"].cmd[2], 12.0)
        self.assertAlmostEqual(arm._motor_map["joint1"].cmd[3], 5.0, msg="kd clamped to the MIT maximum")

    def test_enable_must_be_confirmed(self):
        arm, estop, clock, ctrl = make_ctrl()
        arm._motor_map["joint5"].enable_fails = True
        tick(arm, ctrl, clock)
        arm.enable_all()
        ctrl.mark_enabled()
        tick(arm, ctrl, clock, int(jog.ENABLE_TIMEOUT_S / 0.005) + 5)
        self.assertTrue(estop.tripped)
        self.assertIn("joint5", estop.reason)
        self.assertFalse(ctrl.verified.is_set())

    def test_motor_dropping_to_disabled_trips(self):
        arm, estop, clock, ctrl = make_ctrl()
        enable_and_ramp(arm, ctrl, clock)
        arm._motor_map["joint2"].status = 0x0
        tick(arm, ctrl, clock)
        self.assertTrue(estop.tripped)
        self.assertIn("joint2", estop.reason)

    def test_stale_feedback_trips(self):
        arm, estop, clock, ctrl = make_ctrl()
        enable_and_ramp(arm, ctrl, clock)
        arm._motor_map["joint4"].dead = True
        tick(arm, ctrl, clock, int(jog.STALE_S / 0.005) + 3)
        self.assertTrue(estop.tripped)
        self.assertIn("joint4: no reply", estop.reason)

    def test_send_error_trips(self):
        arm, estop, clock, ctrl = make_ctrl()
        enable_and_ramp(arm, ctrl, clock)
        arm._motor_map["joint1"].fail_send = True
        tick(arm, ctrl, clock)  # must not raise
        self.assertTrue(estop.tripped)
        self.assertIn("control loop error", estop.reason)

    def test_nothing_sent_once_tripped(self):
        arm, estop, clock, ctrl = make_ctrl()
        enable_and_ramp(arm, ctrl, clock)
        sends = {n: m.sends for n, m in arm._motor_map.items()}
        estop.trip("test")
        tick(arm, ctrl, clock, 5)
        self.assertEqual(sends, {n: m.sends for n, m in arm._motor_map.items()})

    def test_fault_limits_and_tracking_trip(self):
        for setup, fragment in (
                (lambda a: setattr(a._motor_map["joint3"], "status", 0xA), "FAULT overcurrent"),
                (lambda a: setattr(a._motor_map["joint2"], "pos", 3.0), "outside its limits"),
                (lambda a: setattr(a._motor_map["joint4"], "pos", Q0[3] + 0.5), "from its target")):
            arm, estop, clock, ctrl = make_ctrl()
            enable_and_ramp(arm, ctrl, clock)
            setup(arm)
            before = arm._motor_map["joint1"].sends
            tick(arm, ctrl, clock)
            self.assertTrue(estop.tripped, fragment)
            self.assertIn(fragment, estop.reason)
            self.assertEqual(arm._motor_map["joint1"].sends, before, "no command on the faulting tick")

    def test_follower_respects_speed_and_acceleration(self):
        arm, estop, clock, ctrl = make_ctrl(joint_speed=0.3, joint_accel=1.0)
        enable_and_ramp(arm, ctrl, clock)
        ctrl.set_goal(Q0 + np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0]), vmax=0.3)
        prev_q, prev_v, vmax_seen, amax_seen = ctrl.snapshot()[0], 0.0, 0.0, 0.0
        for _ in range(600):
            tick(arm, ctrl, clock)
            q = ctrl.snapshot()[0]
            v = float(np.max(np.abs(q - prev_q))) / 0.005
            vmax_seen = max(vmax_seen, v)
            amax_seen = max(amax_seen, abs(v - prev_v) / 0.005)
            prev_q, prev_v = q, v
        self.assertLessEqual(vmax_seen, 0.3 + 1e-6)
        self.assertLessEqual(amax_seen, 1.0 + 0.05)
        self.assertTrue(np.allclose(ctrl.snapshot()[0], ctrl.snapshot()[1]), "reached the goal")

    def run_and_measure(self, arm, ctrl, clock, ticks):
        """Max per-joint velocity change per tick (rad/s²) and the setpoints visited."""
        prev_v, worst, qs = ctrl.qd_set.copy(), 0.0, []
        for _ in range(ticks):
            tick(arm, ctrl, clock)
            v = ctrl.qd_set.copy()
            worst = max(worst, float(np.max(np.abs(v - prev_v))) / 0.005)
            prev_v = v
            qs.append(ctrl.snapshot()[0])
        return worst, np.array(qs)

    def test_reversing_mid_move_stays_acceleration_limited(self):
        """The re-review's N1: w w s flipped the velocity in one tick (45 rad/s²)."""
        arm, estop, clock, ctrl = make_ctrl(joint_speed=0.3, joint_accel=1.0)
        enable_and_ramp(arm, ctrl, clock)
        g1 = Q0 + np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        ctrl.set_goal(g1, vmax=0.3)
        a1, _ = self.run_and_measure(arm, ctrl, clock, 120)  # up to speed
        self.assertGreater(ctrl.speed, 0.2)
        ctrl.set_goal(Q0 - np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]), vmax=0.3)  # reverse
        a2, qs = self.run_and_measure(arm, ctrl, clock, 800)
        self.assertLessEqual(max(a1, a2), 1.0 + 1e-6)
        self.assertLessEqual(float(np.max(qs[:, 2])), g1[2] + 1e-9, "never past the old goal")
        self.assertAlmostEqual(ctrl.snapshot()[0][2], Q0[2] - 0.1, places=6)

    def test_turning_mid_move_stays_acceleration_limited(self):
        arm, estop, clock, ctrl = make_ctrl(joint_speed=1.0, joint_accel=1.0)
        enable_and_ramp(arm, ctrl, clock)
        g1 = Q0 + np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0])
        ctrl.set_goal(g1, vmax=1.0)
        self.run_and_measure(arm, ctrl, clock, 120)
        g2 = ctrl.snapshot()[0] + np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])  # turn
        ctrl.set_goal(g2, vmax=1.0)
        a, qs = self.run_and_measure(arm, ctrl, clock, 1500)
        self.assertLessEqual(a, 1.0 + 1e-6)
        lo = np.minimum(np.minimum(Q0, g1), g2) - 1e-9
        hi = np.maximum(np.maximum(Q0, g1), g2) + 1e-9
        self.assertTrue(np.all((qs >= lo) & (qs <= hi)), "stays inside the checked box")
        self.assertTrue(np.allclose(ctrl.snapshot()[0], g2, atol=1e-6))

    def test_hold_decelerates(self):
        arm, estop, clock, ctrl = make_ctrl()
        enable_and_ramp(arm, ctrl, clock)
        ctrl.set_goal(Q0 + 0.25, vmax=0.3)
        tick(arm, ctrl, clock, 100)
        speed = ctrl.speed
        self.assertGreater(speed, 0.2)
        ctrl.hold()
        tick(arm, ctrl, clock)
        self.assertGreater(ctrl.speed, speed - 1.0 * 0.005 - 1e-9, "no instant stop")
        tick(arm, ctrl, clock, 200)
        self.assertEqual(ctrl.speed, 0.0)

    def test_clock_jump_does_not_jump_the_setpoint(self):
        arm, estop, clock, ctrl = make_ctrl()
        enable_and_ramp(arm, ctrl, clock)
        ctrl.set_goal(Q0 + 0.25, vmax=0.3)
        tick(arm, ctrl, clock, 100)
        q_before = ctrl.snapshot()[0]
        tick(arm, ctrl, clock, 1, dt=5.0)  # a 5 s stall
        step = float(np.max(np.abs(ctrl.snapshot()[0] - q_before)))
        self.assertLessEqual(step, 0.3 * ctrl.max_dt + 1e-9)


# ---------------------------------------------------------------- moves


class JoggerTests(unittest.TestCase):
    def setUp(self):
        self.kin = jog.Kinematics(CFG["urdf_path"], CFG["end_effector_frame"], 6)
        self.arm, self.estop, self.clock, self.ctrl = make_ctrl()
        enable_and_ramp(self.arm, self.ctrl, self.clock)
        self.args = jog.build_parser().parse_args([])

    def jogger(self, **overrides):
        for k, v in overrides.items():
            setattr(self.args, k, v)
        return jog.Jogger(self.ctrl, self.kin, SETPOINT_LIMITS, self.args)

    def test_step_moves_the_end_effector_by_one_step(self):
        start = self.kin.ee_pos(Q0)
        msg = self.jogger().request("x", +1)
        self.assertEqual(msg, "x +")
        end = self.kin.ee_pos(self.ctrl.snapshot()[1])
        self.assertTrue(np.allclose(end - start, [0.01, 0, 0], atol=2e-4), end - start)

    def test_key_mashing_cannot_exceed_speed_caps(self):
        """The review's runaway: auto-repeat at 30 Hz for 0.8 s, then release."""
        j = self.jogger(joint_speed=0.3, cart_speed=0.03)
        qs = [self.ctrl.snapshot()[0]]
        accepted = 0
        for k in range(int(3.0 / 0.005)):
            if k * 0.005 < 0.8 and k % 7 == 0:  # ~30 Hz key repeat
                accepted += j.request("x", +1) == "x +"
            tick(self.arm, self.ctrl, self.clock)
            qs.append(self.ctrl.snapshot()[0])
        qs = np.array(qs)
        joint_speed = np.max(np.abs(np.diff(qs, axis=0))) / 0.005
        ee = np.array([self.kin.ee_pos(q) for q in qs[::4]])
        ee_speed = np.max(np.linalg.norm(np.diff(ee, axis=0), axis=1)) / (4 * 0.005)
        self.assertLessEqual(joint_speed, 0.3 + 1e-6)
        self.assertLessEqual(ee_speed, 0.03 * 1.15)  # straight joint lines bend slightly in space
        lead = float(np.max(np.abs(self.ctrl.snapshot()[1] - qs[int(0.8 / 0.005)])))
        self.assertLessEqual(lead, self.args.max_jump + 1e-9, "bounded coasting after release")
        self.assertLess(accepted, 24, "requests beyond the lead limit were refused")

    def test_refuses_below_z_min_at_target_or_halfway(self):
        z = float(self.kin.ee_pos(Q0)[2])
        msg = self.jogger(z_min=z - 0.005).request("z", -1)
        self.assertIn("below --z-min", msg)
        self.assertTrue(np.allclose(self.ctrl.snapshot()[1], Q0, atol=1e-9), "no motion planned")

    def test_refuses_large_joint_jump(self):
        msg = self.jogger(max_jump=0.001).request("x", +1)
        self.assertIn("would jump", msg)

    def test_config_q_limit_refuses_beyond_and_allows_back(self):
        lo1 = SETPOINT_LIMITS["joint1"][0]
        j = self.jogger()
        q = Q0.copy()
        q[0] = lo1 + jog.LIMIT_MARGIN_RAD + 0.001
        self.assertIsNone(j._limit_problem(q, q))
        beyond = q.copy(); beyond[0] = lo1 - 0.01
        self.assertIn("beyond its limits", j._limit_problem(q, beyond))
        out = q.copy(); out[0] = lo1 - 0.3            # starting outside the config range
        back = out.copy(); back[0] = lo1 - 0.2          # moving back towards it
        further = out.copy(); further[0] = lo1 - 0.4
        self.assertIsNone(j._limit_problem(out, back))
        self.assertIsNotNone(j._limit_problem(out, further))

    def test_no_moves_while_ramping(self):
        arm, estop, clock, ctrl = make_ctrl()
        self.assertIn("ramping", jog.Jogger(ctrl, self.kin, SETPOINT_LIMITS, self.args).request("x", +1))

    def test_argument_ceilings(self):
        p = jog.build_parser()
        for bad in (["--step", "0.2"], ["--cart-speed", "1"], ["--joint-speed", "3"],
                    ["--rot-step", "45"], ["--step", "nan"], ["--z-min", "nan"],
                    ["--z-min", "inf"], ["--joint-accel", "10"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                p.parse_args(bad)
        self.assertEqual(p.parse_args([]).hardware_yaml, SAFE)


# ---------------------------------------------------------------- keyboard


class KeyboardTests(unittest.TestCase):
    def test_each_key_is_delivered_immediately(self):
        master, slave = pty.openpty()
        try:
            kb = jog.Keyboard(fd=slave)
            os.write(master, b"www")
            got = [kb.read(0.2) for _ in range(3)]
            self.assertEqual(got, ["w", "w", "w"], "no key waits for the next one")
            self.assertIsNone(kb.read(0.05))
            import termios
            self.assertFalse(termios.tcgetattr(slave)[0] & termios.IXON, "Ctrl+S cannot freeze output")
            kb.restore()
        finally:
            os.close(master)
            os.close(slave)


# ---------------------------------------------------------------- startup flow


class TimedKeyboard:
    """Feeds keys from a script: (seconds_after_start, key); '^C' sends a real SIGINT."""

    script = []

    def __init__(self):
        self._t0 = time.monotonic()
        self._script = list(type(self).script)

    def read(self, timeout):
        time.sleep(min(timeout, 0.01))
        if not self._script:
            return "q"
        at, key = self._script[0]
        if time.monotonic() - self._t0 < at:
            return None
        self._script.pop(0)
        if key == "^C":
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)
            return None
        return key

    def restore(self):
        pass


class MainTests(unittest.TestCase):
    def run_main(self, arm, argv=("--no-gravity",), answer="JOG", keys=(), tty=True):
        TimedKeyboard.script = list(keys)
        out = io.StringIO()
        old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with patch.object(jog, "RebotArm", return_value=arm), \
                    patch.object(jog, "Keyboard", TimedKeyboard), \
                    patch.object(jog.sys.stdin, "isatty", return_value=tty), \
                    patch("builtins.input", side_effect=answer if callable(answer) else (lambda *_: answer)), \
                    contextlib.redirect_stdout(out):
                code = jog.main(list(argv))
        finally:
            for s, h in old.items():
                signal.signal(s, h)
        return code, out.getvalue()

    def test_needs_a_terminal(self):
        arm = SimArm()
        code, _ = self.run_main(arm, tty=False)
        self.assertEqual((code, arm.calls), (2, []))

    def test_refuses_stiff_config_before_connecting(self):
        arm = SimArm(kp=120.0)
        code, out = self.run_main(arm)
        self.assertEqual(code, 2)
        self.assertIn("exceed rebot_jog's ceilings", out)
        self.assertEqual(arm.calls, [])

    def test_no_confirmation_enables_nothing(self):
        arm = SimArm()
        code, _ = self.run_main(arm, answer="yes")
        self.assertEqual(code, 1)
        self.assertNotIn("enable_all", arm.calls)
        self.assertEqual(arm.calls[-1], "disconnect")

    def test_no_mit_mode_enables_nothing(self):
        arm = SimArm()
        arm.gripper.mit_ok = False
        code, _ = self.run_main(arm)
        self.assertEqual(code, 1)
        self.assertNotIn("enable_all", arm.calls)

    def test_real_ctrl_c_at_the_prompt(self):
        arm = SimArm()

        def blocked_prompt(*_):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(2.0)  # the KeyboardInterrupt must unwind this
            return "JOG"

        t0 = time.monotonic()
        code, out = self.run_main(arm, answer=blocked_prompt)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(code, 1)
        self.assertNotIn("enable_all", arm.calls)
        self.assertRegex(out, r"Stopping \((SIGINT received|interrupted)\)")

    def test_session_moves_then_stops_cleanly(self):
        arm = SimArm()
        code, out = self.run_main(arm, keys=[(1.4, "w"), (1.45, "w"), (2.4, "q")])
        self.assertEqual(code, 0, out)
        c = arm.calls
        self.assertLess(c.index("connect"), c.index("start_control_loop"))
        self.assertLess(c.index("start_control_loop"), c.index("enable_all"))
        self.assertEqual(c[-2:], ["disable_all", "disconnect"])
        qs = np.array([q for _, q in arm.sent_log])
        self.assertGreater(np.max(np.abs(qs - qs[0])), 1e-3, "a move was executed")
        ts = np.array([t for t, _ in arm.sent_log])
        v = np.max(np.abs(np.diff(qs, axis=0)) / np.maximum(np.diff(ts), 1e-4)[:, None])
        self.assertLessEqual(v, 0.3 * 1.5)  # real-thread timing jitter, still far below the review's 5 rad/s
        self.assertIn("stopped by user", out)

    def test_real_ctrl_c_while_running_stops_commands(self):
        """The trip alone must stop commands: count sends while the control thread is
        still running (disable_all is made slow, and runs before disconnect joins it)."""
        arm = SimArm()
        during = {}
        real_disable = arm.disable_all

        def slow_disable():
            alive = arm._thread is not None and arm._thread.is_alive()
            before = arm._motor_map["joint1"].sends
            time.sleep(0.2)
            during["alive"], during["sends"] = alive, arm._motor_map["joint1"].sends - before
            real_disable()

        arm.disable_all = slow_disable
        code, out = self.run_main(arm, keys=[(1.3, "w"), (1.5, "^C")])
        self.assertEqual(code, 1)
        self.assertRegex(out, r"Stopping \((SIGINT received|interrupted)\)")
        self.assertTrue(during["alive"], "the control thread was still running")
        self.assertEqual(during["sends"], 0, "no commands after Ctrl+C, before any disable")
        self.assertEqual(arm.calls[-2:], ["disable_all", "disconnect"])

    def test_pose_changed_at_the_prompt_is_held_not_undone(self):
        arm = SimArm()

        def moved_while_prompting(*_):
            arm._motor_map["joint2"].pos = Q0[1] + 0.25
            return "JOG"

        code, out = self.run_main(arm, answer=moved_while_prompting, keys=[(1.4, "q")])
        self.assertEqual(code, 0, out)
        held = np.array([q[1] for _, q in arm.sent_log])
        self.assertTrue(np.allclose(held, Q0[1] + 0.25, atol=1e-6), "never pulled back to the old pose")


if __name__ == "__main__":
    unittest.main()
