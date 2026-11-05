"""rebot_jog: jog the arm's end effector from the keyboard (IK, MIT impedance control).

Defaults to config/rebotarm_dm_safe.yaml. Each key press moves the goal pose by one step;
IK turns it into a joint goal, and a velocity- and acceleration-limited follower moves the
setpoint towards it along a straight joint-space line.

Safety design:
  - startup: connect() verifies the CAN watchdog and joint limits; the configured gains
    must be below hard ceilings; the pose, settings and keys are shown and you must type
    JOG; every motor must confirm MIT mode;
  - enabling never pulls the arm anywhere: until every motor reports ENABLED the setpoint
    tracks the measured pose with zero gains; stiffness and gravity torque then ramp in
    from zero around the pose the arm is in at that moment;
  - motion: speed is computed from the current setpoint each tick and never exceeds the
    joint and end-effector speed caps (accelerating and decelerating at --joint-accel);
    a new step is refused while the goal is already --max-jump ahead, so key repeat cannot
    pile up distance; holding decelerates instead of stopping dead;
  - every step is checked first: IK must converge, no joint may move more than --max-jump
    (rejects IK branch flips), targets must respect the joint limits including the
    config's q limits (a step that brings a joint back towards its range is allowed), and
    the end effector must stay above --z-min at the target and halfway there;
  - while running, the control loop stops on: a motor fault or any motor no longer
    ENABLED, a failed send, feedback older than 0.15 s (every motor is polled for its
    output-shaft position, whose replies carry their age), a joint outside its URDF
    limits, a tracking error above --tracking-limit, or any exception;
  - stopping never moves the arm: Ctrl+C (also SIGTERM, SIGHUP, SIGQUIT, Ctrl+Z), q, Esc,
    a fault or an error stops all commands immediately (the 100 ms motor watchdog then
    disables the motors even if this process hangs), then disables them explicitly. A
    second Ctrl+C, or a shutdown taking over 3 s, ends the process at once. The arm goes
    limp when stopped: keep it supported or in a safe place.
"""
from __future__ import annotations

import math
import os
import select
import sys
import threading
import time
from typing import Optional

import numpy as np

from ..actuator import RebotArm, load_cfg, load_gravity_compensation_config
from ._common import confirm, describe_status, fmt_deg, parser
from ._estop import EStop
from .vis import GravityTorque

DEFAULT_HW = "rebotarm_dm_safe.yaml"

# Hard ceilings: command-line values above these are refused.
MAX_STEP_M = 0.05
MAX_ROT_STEP_DEG = 15.0
MAX_CART_SPEED = 0.10       # m/s
MAX_JOINT_SPEED = 1.0       # rad/s
MAX_JOINT_ACCEL = 3.0       # rad/s²
MAX_ARM_KP = 30.0           # refuse configs stiffer than this (rebotarm_dm.yaml has 120)
MAX_ARM_KD = 5.0
MAX_GRIPPER_KP = 30.0
DAMIAO_KD_MAX = 5.0         # the MIT frame cannot carry more; larger configured values clamp

GAIN_RAMP_S = 1.0           # stiffness and gravity torque fade in over this time
ENABLE_TIMEOUT_S = 0.5      # every motor must report ENABLED within this time
LIVENESS_PERIOD_S = 0.05    # output-shaft position requested this often per motor
STALE_S = 0.15              # newest reply older than this: the motor (or the bus) is gone
LIMIT_MARGIN_RAD = 0.05     # move targets stay this far inside the limits
RUN_LIMIT_MARGIN_RAD = 0.15  # a measured position beyond URDF limits + this stops the arm
DM_RID_XOUT = 81
STATUS_ENABLED = 0x1

KEYS = {
    "w": ("x", +1), "s": ("x", -1), "a": ("y", +1), "d": ("y", -1), "r": ("z", +1), "f": ("z", -1),
    "u": ("roll", +1), "o": ("roll", -1), "i": ("pitch", +1), "k": ("pitch", -1),
    "j": ("yaw", +1), "l": ("yaw", -1),
}
HELP = """keys:  w/s ±x   a/d ±y   r/f ±z        (base frame)
       u/o ±roll i/k ±pitch j/l ±yaw   (about the base axes, at the end effector)
       + / -  bigger / smaller steps    space  stop here (decelerates)    p  print state
       q / Esc / Ctrl+C  stop: commands cease and the motors are disabled (arm goes limp)"""


def _bounded(lo: float, hi: float):
    def parse(text: str) -> float:
        v = float(text)
        if not (math.isfinite(v) and lo <= v <= hi):
            raise ValueError
        return v
    parse.__name__ = f"number in [{lo:g}, {hi:g}]"
    return parse


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------- kinematics


class Kinematics:
    def __init__(self, urdf_path: Optional[str], frame: str, n_arm: int):
        import pinocchio as pin

        from ..kinematics import load_robot_model
        from ..kinematics.robot_model import pad_q_for_model

        self.pin = pin
        self.model = load_robot_model(urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(frame):
            raise ValueError(f"end-effector frame {frame!r} not in the URDF")
        self.frame_id = self.model.getFrameId(frame)
        self.n = n_arm
        self._pad = pad_q_for_model

    def fk(self, q_arm: np.ndarray):
        pin = self.pin
        q = self._pad(self.model, np.asarray(q_arm, dtype=np.float64), self.n)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id].copy()

    def ee_pos(self, q_arm: np.ndarray) -> np.ndarray:
        return np.array(self.fk(q_arm).translation, dtype=np.float64)

    def ik(self, target, q_seed: np.ndarray):
        from ..kinematics import solve_ik
        from ..kinematics.inverse_kinematics import IKParams

        res = solve_ik(self.model, self.data, self.frame_id, target, np.asarray(q_seed),
                       IKParams(max_iter=300, tolerance=1e-4, step_size=0.5, damping=1e-6),
                       controlled_joints=self.n)
        return (np.asarray(res.q[: self.n], dtype=np.float64) if res.success else None), res.error


# --------------------------------------------------------------------------- controller


class JogController:
    """Control-loop callback. Every tick: stop check first, then safety checks, then one
    MIT command per motor. Thread-safe against the main thread's goal changes.

    Phases: before ``mark_enabled()`` (and until every motor reports ENABLED) the setpoint
    tracks the measured pose and every gain is zero; then gains and gravity torque ramp in
    over GAIN_RAMP_S around the pose at that moment; then the follower runs.
    """

    def __init__(self, arm: RebotArm, estop: EStop, urdf_limits: dict,
                 gravity: Optional[GravityTorque], tracking_limit: float,
                 joint_speed: float, joint_accel: float, rate: float,
                 clock=time.monotonic):
        self.arm, self.estop, self.clock = arm, estop, clock
        self.names = arm.arm.joint_names
        self.grip_names = arm.gripper.joint_names
        self.all_names = self.names + self.grip_names
        self.kp_cfg = np.asarray(arm.arm._mit_kp, dtype=np.float64).copy()
        self.kd_cfg = np.minimum(np.asarray(arm.arm._mit_kd, dtype=np.float64), DAMIAO_KD_MAX)
        ng = len(self.grip_names)
        self.gkp_cfg = np.asarray(getattr(arm.gripper, "_mit_kp", np.zeros(ng)), dtype=np.float64)[:ng]
        self.gkd_cfg = np.minimum(np.asarray(getattr(arm.gripper, "_mit_kd", np.zeros(ng)),
                                             dtype=np.float64)[:ng], DAMIAO_KD_MAX)
        self.lo = np.array([urdf_limits.get(n, (-np.inf, np.inf))[0] for n in self.names])
        self.hi = np.array([urdf_limits.get(n, (-np.inf, np.inf))[1] for n in self.names])
        self.gravity = gravity
        self.tracking_limit = tracking_limit
        self.joint_speed = joint_speed
        self.joint_accel = joint_accel
        self.max_dt = 2.0 / rate

        self._lock = threading.Lock()
        n = len(self.names)
        self.q_set = np.zeros(n)
        self.q_goal = np.zeros(n)
        self.q_grip = np.zeros(ng)
        self.speed = 0.0            # rad/s of the fastest joint along the current line
        self.vmax = joint_speed     # current line's speed cap (joint and Cartesian caps)
        self.qd_set = np.zeros(n)
        self.q_meas = np.zeros(n)
        self.last_tau = np.zeros(n)

        self.t_enable_cmd: Optional[float] = None
        self.t_ramp: Optional[float] = None
        self.verified = threading.Event()
        self._t_last: Optional[float] = None
        self._t_liveness = -math.inf
        self._requested = False

    # -- main-thread API ---------------------------------------------------

    def mark_enabled(self) -> None:
        """Call right after enable_all(): verification and the gain ramp start from here."""
        self.t_enable_cmd = self.clock()

    def ramped(self) -> bool:
        return self.t_ramp is not None and self.clock() - self.t_ramp >= GAIN_RAMP_S

    def snapshot(self):
        with self._lock:
            return self.q_set.copy(), self.q_goal.copy()

    def lead(self) -> float:
        with self._lock:
            return float(np.max(np.abs(self.q_goal - self.q_set)))

    def set_goal(self, q_goal: np.ndarray, vmax: float) -> None:
        with self._lock:
            self.q_goal = np.asarray(q_goal, dtype=np.float64).copy()
            self.vmax = min(self.joint_speed, max(1e-6, vmax))

    def hold(self) -> None:
        """Stop where the current speed allows: the goal becomes the braking point."""
        with self._lock:
            e = self.q_goal - self.q_set
            dist = float(np.max(np.abs(e)))
            if dist < 1e-12 or self.speed <= 0.0:
                self.q_goal = self.q_set.copy()
                return
            brake = min(dist, self.speed ** 2 / (2.0 * self.joint_accel))
            self.q_goal = self.q_set + e / dist * brake

    # -- control loop ------------------------------------------------------

    def __call__(self, r: RebotArm, dt: float) -> None:
        if self.estop.tripped:
            return  # send nothing: the motor watchdog disables the motors within 100 ms
        try:
            self._tick(r)
        except Exception as e:  # never let the loop die quietly
            self.estop.trip(f"control loop error: {e!r}")

    def _trip(self, reason: str) -> None:
        self.estop.trip(reason)

    def _tick(self, r: RebotArm) -> None:
        now = self.clock()
        dt = 0.0 if self._t_last is None else min(max(0.0, now - self._t_last), self.max_dt)
        self._t_last = now
        for c in r._ctrl_map.values():
            c.poll_feedback_once()

        motors = [r._motor_map[n] for n in self.all_names]
        states = [m.get_state() for m in motors]
        if any(st is None for st in states):
            return self._trip("a motor is not reporting feedback")
        codes = [int(st.status_code) for st in states]
        q = np.array([st.pos for st in states[: len(self.names)]], dtype=np.float64)
        self.q_meas = q
        qg = np.array([st.pos for st in states[len(self.names):]], dtype=np.float64)

        # Liveness: each motor's newest output-shaft reply must be recent once verified.
        if now - self._t_liveness >= LIVENESS_PERIOD_S:
            self._t_liveness = now
            for m in motors:
                m.request_register(DM_RID_XOUT)
            self._requested = True
        stale = []
        for n, m in zip(self.all_names, motors):
            reply = m.get_cached_register_f32(DM_RID_XOUT, fresh_only=False)
            if reply is None or reply[1] > STALE_S:
                stale.append(n)
        if stale and self.verified.is_set():
            return self._trip(f"{stale[0]}: no reply for over {STALE_S * 1000:.0f} ms "
                              "(bus or motor lost)")

        outside = (q < self.lo - RUN_LIMIT_MARGIN_RAD) | (q > self.hi + RUN_LIMIT_MARGIN_RAD)
        if np.any(outside):
            j = int(np.argmax(outside))
            return self._trip(f"{self.names[j]} at {math.degrees(q[j]):+.1f}° is outside its limits")

        if not self.verified.is_set():
            # Zero gains: keep the setpoint on the measured pose so nothing pulls on enable.
            with self._lock:
                self.q_set, self.q_goal, self.q_grip = q.copy(), q.copy(), qg.copy()
                self.speed, self.qd_set = 0.0, np.zeros(len(self.names))
            if self.t_enable_cmd is not None:
                if all(c == STATUS_ENABLED for c in codes) and not stale:
                    self.t_ramp = now
                    self.verified.set()
                elif now - self.t_enable_cmd > ENABLE_TIMEOUT_S:
                    bad = [f"{n} {describe_status('damiao', c)}" for n, c in zip(self.all_names, codes)
                           if c != STATUS_ENABLED] + [f"{n} not answering" for n in stale]
                    return self._trip("not every motor enabled: " + ", ".join(bad))
            ramp = 0.0
        else:
            for n, c in zip(self.all_names, codes):
                if c != STATUS_ENABLED:
                    return self._trip(f"{n}: {describe_status('damiao', c)}")
            ramp = _smoothstep((now - self.t_ramp) / GAIN_RAMP_S)
            with self._lock:
                if ramp >= 1.0:
                    self._follow(dt)
                q_set, qd_set = self.q_set.copy(), self.qd_set.copy()
            if ramp >= 1.0:
                err = np.abs(q - q_set)
                if np.any(err > self.tracking_limit):
                    j = int(np.argmax(err))
                    return self._trip(f"{self.names[j]} is {math.degrees(err[j]):.1f}° from its "
                                      f"target (limit {math.degrees(self.tracking_limit):.0f}°): "
                                      "blocked or pushed?")

        with self._lock:
            q_set, qd_set, q_grip = self.q_set.copy(), self.qd_set.copy(), self.q_grip.copy()
        tau = np.zeros(len(self.names))
        if self.gravity is not None and ramp > 0.0:
            g = self.gravity(q)
            if np.all(np.isfinite(g)):
                tau = g * ramp
        self.last_tau = tau
        if self.estop.tripped:  # re-check right before sending
            return
        # Per motor, so a failed send is seen (JointGroup.send_mit would swallow it).
        for i, n in enumerate(self.names):
            r._motor_map[n].send_mit(float(q_set[i]), float(qd_set[i] * ramp), float(self.kp_cfg[i] * ramp),
                                     float(self.kd_cfg[i] * ramp), float(tau[i]))
        for i, n in enumerate(self.grip_names):
            r._motor_map[n].send_mit(float(q_grip[i]), 0.0, float(self.gkp_cfg[i] * ramp),
                                     float(self.gkd_cfg[i] * ramp), 0.0)

    def _follow(self, dt: float) -> None:
        """Advance q_set towards q_goal (lock held). Straight joint-space line; the fastest
        joint moves at ``speed``, which accelerates/decelerates at joint_accel and never
        exceeds vmax or the speed from which it can still stop at the goal."""
        e = self.q_goal - self.q_set
        dist = float(np.max(np.abs(e)))
        if dist < 1e-9:
            self.q_set = self.q_goal.copy()
            self.speed, self.qd_set = 0.0, np.zeros_like(self.q_set)
            return
        a = self.joint_accel
        # Fastest speed from which it can still stop at the goal, one tick of travel
        # included (s²/2a + s·dt = dist); speed never changes by more than a·dt per tick.
        stop_speed = a * (-dt + math.sqrt(dt * dt + 2.0 * dist / a)) if dt > 0 else math.sqrt(2.0 * a * dist)
        target = min(self.vmax, stop_speed)
        if target > self.speed:
            self.speed = min(target, self.speed + a * dt)
        else:
            self.speed = max(target, self.speed - a * dt)
        step = self.speed * dt
        if step >= dist:
            self.q_set = self.q_goal.copy()
            self.speed, self.qd_set = 0.0, np.zeros_like(self.q_set)
            return
        direction = e / dist
        self.q_set = self.q_set + direction * step
        self.qd_set = direction * self.speed


# --------------------------------------------------------------------------- moves


class Jogger:
    """Turns jog requests into checked joint goals."""

    def __init__(self, ctrl: JogController, kin: Kinematics, setpoint_limits: dict, args):
        self.ctrl, self.kin, self.args = ctrl, kin, args
        names = ctrl.names
        self.lo = np.array([setpoint_limits.get(n, (-np.inf, np.inf))[0] for n in names]) + LIMIT_MARGIN_RAD
        self.hi = np.array([setpoint_limits.get(n, (-np.inf, np.inf))[1] for n in names]) - LIMIT_MARGIN_RAD
        self.step = args.step
        self.rot_step = math.radians(args.rot_step)

    def _limit_problem(self, q_from: np.ndarray, q_to: np.ndarray) -> Optional[str]:
        """A target outside the limits is allowed only if it is no further outside than
        where the move starts (so a joint can be brought back into range)."""
        def outside(q):
            return np.maximum(self.lo - q, 0.0) + np.maximum(q - self.hi, 0.0)
        worse = outside(q_to) > outside(q_from) + 1e-9
        if np.any(worse):
            j = int(np.argmax(worse))
            return (f"refused: {self.ctrl.names[j]} would be at {math.degrees(q_to[j]):+.1f}°, "
                    "beyond its limits")
        return None

    def request(self, axis: str, sign: int) -> str:
        pin = self.kin.pin
        if not self.ctrl.ramped():
            return "wait: still ramping in"
        q_set, q_from = self.ctrl.snapshot()  # chain from the goal, not the sagging measurement
        T_from = self.kin.fk(q_from)
        pos = np.array(T_from.translation, dtype=np.float64)
        rot = np.array(T_from.rotation, dtype=np.float64)
        if axis in ("x", "y", "z"):
            pos[("x", "y", "z").index(axis)] += sign * self.step
        else:
            w = np.zeros(3)
            w[("roll", "pitch", "yaw").index(axis)] = sign * self.rot_step
            rot = pin.exp3(w) @ rot  # about the base axes, at the end effector
        T_to = pin.SE3(rot, pos)

        q_to, err = self.kin.ik(T_to, q_from)
        if q_to is None:
            return f"refused: IK did not converge (error {err:.1e}); try a smaller step"
        dq = np.abs(q_to - q_from)
        if np.max(dq) > self.args.max_jump:
            j = int(np.argmax(dq))
            return (f"refused: {self.ctrl.names[j]} would jump {math.degrees(dq[j]):.1f}° "
                    f"(> --max-jump {math.degrees(self.args.max_jump):.0f}°)")
        if float(np.max(np.abs(q_to - q_set))) > self.args.max_jump:
            return "busy: the arm is still catching up"  # bounds how far it coasts after a key
        problem = self._limit_problem(q_from, q_to)
        if problem:
            return problem
        for label, q in (("target", q_to), ("halfway", 0.5 * (q_from + q_to))):
            z = float(self.kin.ee_pos(q)[2])
            if z < self.args.z_min:
                return f"refused: end effector {label} z {z:+.3f} m is below --z-min {self.args.z_min:g}"

        # Speed cap for the line from the current setpoint to the new goal: joint cap, and
        # the joint speed at which the end effector moves at cart_speed.
        dq_line = float(np.max(np.abs(q_to - q_set)))
        dx_line = float(np.linalg.norm(self.kin.ee_pos(q_to) - self.kin.ee_pos(q_set)))
        vmax = self.args.joint_speed
        if dx_line > 1e-9 and dq_line > 1e-9:
            vmax = min(vmax, self.args.cart_speed * dq_line / dx_line)
        self.ctrl.set_goal(q_to, vmax)
        return f"{axis} {'+' if sign > 0 else '-'}"

    def scale_steps(self, factor: float) -> str:
        self.step = min(MAX_STEP_M, max(0.001, self.step * factor))
        self.rot_step = min(math.radians(MAX_ROT_STEP_DEG), max(math.radians(0.5), self.rot_step * factor))
        return f"step {self.step * 1000:.1f} mm, {math.degrees(self.rot_step):.1f}°"


# --------------------------------------------------------------------------- terminal


class Keyboard:
    """Unbuffered single-key input. cbreak keeps Ctrl+C as SIGINT; XON/XOFF is disabled so
    Ctrl+S cannot freeze the output. restore() never waits on the terminal."""

    def __init__(self, fd: Optional[int] = None):
        import termios
        import tty

        self._termios = termios
        self.fd = sys.stdin.fileno() if fd is None else fd
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] &= ~(termios.IXON | termios.IXOFF)
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)

    def read(self, timeout: float) -> Optional[str]:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd, 1)  # unbuffered: each key is seen when it arrives
        if not data:
            raise EOFError
        return data.decode("latin-1")

    def restore(self) -> None:
        self._termios.tcsetattr(self.fd, self._termios.TCSANOW, self._saved)


# --------------------------------------------------------------------------- main


def _check_gains(arm: RebotArm) -> Optional[str]:
    kp, kd = np.asarray(arm.arm._mit_kp), np.asarray(arm.arm._mit_kd)
    gkp = np.asarray(getattr(arm.gripper, "_mit_kp", []))
    if np.any(kp > MAX_ARM_KP) or np.any(np.minimum(kd, DAMIAO_KD_MAX) > MAX_ARM_KD) \
            or np.any(gkp > MAX_GRIPPER_KP):
        return (f"the config's gains (arm kp {kp.tolist()}, gripper kp {gkp.tolist()}) exceed "
                f"rebot_jog's ceilings (arm kp ≤ {MAX_ARM_KP:g}, gripper kp ≤ {MAX_GRIPPER_KP:g}); "
                f"use a safe config such as {DEFAULT_HW}")
    return None


def _read_positions(arm: RebotArm, names, timeout: float = 1.0) -> Optional[np.ndarray]:
    """Current positions, or None unless every motor answered a request made now.

    get_state() is a cache without timestamps, so freshness is proven by each motor
    answering a new register read (whose replies are tracked per request)."""
    motors = [arm._motor_map[n] for n in names]
    for m in motors:
        m.request_register(DM_RID_XOUT)
        m.request_feedback()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for c in arm._ctrl_map.values():
            c.poll_feedback_once()
        if all(m.get_cached_register_f32(DM_RID_XOUT, fresh_only=True) is not None for m in motors):
            states = [m.get_state() for m in motors]
            if all(st is not None for st in states):
                return np.array([st.pos for st in states], dtype=np.float64)
        time.sleep(0.02)
    return None


def build_parser():
    p = parser("Jog the arm's end effector from the keyboard (IK, MIT impedance control).")
    p.set_defaults(hardware_yaml=DEFAULT_HW)
    p.add_argument("--step", type=_bounded(0.001, MAX_STEP_M), default=0.01,
                   help=f"translation step in m (default 0.01, max {MAX_STEP_M:g})")
    p.add_argument("--rot-step", type=_bounded(0.5, MAX_ROT_STEP_DEG), default=5.0,
                   help=f"rotation step in degrees (default 5, max {MAX_ROT_STEP_DEG:g})")
    p.add_argument("--cart-speed", type=_bounded(0.005, MAX_CART_SPEED), default=0.03,
                   help=f"peak end-effector speed in m/s (default 0.03, max {MAX_CART_SPEED:g})")
    p.add_argument("--joint-speed", type=_bounded(0.05, MAX_JOINT_SPEED), default=0.3,
                   help=f"peak joint speed in rad/s (default 0.3, max {MAX_JOINT_SPEED:g})")
    p.add_argument("--joint-accel", type=_bounded(0.1, MAX_JOINT_ACCEL), default=1.0,
                   help=f"joint acceleration in rad/s² (default 1, max {MAX_JOINT_ACCEL:g})")
    p.add_argument("--max-jump", type=_bounded(0.02, 0.5), default=0.25,
                   help="refuse a step that moves any joint more than this many rad, and refuse "
                        "new steps while the goal is this far ahead (default 0.25)")
    p.add_argument("--tracking-limit", type=_bounded(0.05, 0.6), default=0.35,
                   help="stop if any joint is this many rad from its target (default 0.35)")
    p.add_argument("--z-min", type=_bounded(-1.0, 2.0), default=0.0,
                   help="refuse end-effector targets below this height in m, base frame (default 0)")
    p.add_argument("--no-gravity", action="store_true",
                   help="no gravity torque (with the safe gains the arm sags and may trip the "
                        "tracking limit)")
    p.add_argument("--tau-limit", type=_bounded(0.0, 20.0), default=12.0,
                   help="clip each joint's gravity torque to this many N·m (default 12)")
    p.add_argument("--profile", default="default", help="gravity_compensation profile")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not sys.stdin.isatty():
        print("rebot_jog needs an interactive terminal.")
        return 2

    arm = RebotArm(args.hardware_yaml, channel=args.channel)
    cfg = load_cfg(args.hardware_yaml)
    problem = _check_gains(arm)
    if problem:
        print(f"Refusing: {problem}.")
        return 2
    kin = Kinematics(cfg.get("urdf_path"), cfg.get("end_effector_frame") or "end_link", arm.arm.num_joints)
    gravity = None
    if not args.no_gravity:
        gravity = GravityTorque(cfg.get("urdf_path"),
                                load_gravity_compensation_config(args.profile, args.hardware_yaml),
                                arm.arm.num_joints, args.tau_limit)

    estop = EStop(grace_s=3.0)
    keyboard: Optional[Keyboard] = None
    started = False
    try:
        estop.install()
        print(f"Config: {arm.hardware_yaml} on {arm.channel}")
        started = True
        arm.connect()  # verifies the CAN watchdog and joint limits
        names, grip = arm.arm.joint_names, arm.gripper.joint_names
        q = _read_positions(arm, names + grip)
        if q is None:
            print("Not every motor reported its position; nothing enabled.")
            return 1
        q0 = q[: len(names)]
        T0 = kin.fk(q0)
        rpy = np.degrees(kin.pin.rpy.matrixToRpy(T0.rotation))
        kd_eff = np.minimum(np.asarray(arm.arm._mit_kd, dtype=np.float64), DAMIAO_KD_MAX)

        print("\n  " + "  ".join(f"{n:>8}" for n in names))
        print("  " + fmt_deg(q0) + "   (deg, now)")
        print(f"  end effector: [{T0.translation[0]:+.3f} {T0.translation[1]:+.3f} "
              f"{T0.translation[2]:+.3f}] m, rpy [{rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}]°")
        print(f"  arm gains: kp {np.round(arm.arm._mit_kp, 1).tolist()}  kd {np.round(kd_eff, 1).tolist()}")
        print(f"  steps {args.step * 1000:.0f} mm / {args.rot_step:g}°, speed ≤ {args.cart_speed:g} m/s "
              f"and ≤ {args.joint_speed:g} rad/s per joint, gravity "
              f"{'off' if gravity is None else f'on (≤ {args.tau_limit:g} N·m)'}")
        print("\n" + HELP)
        print("\nThe arm will be enabled and hold the pose it is in when enabled (stiffness ramps "
              f"in over {GAIN_RAMP_S:g} s).\nWhen it stops for any reason the motors are disabled "
              "and the arm goes limp.")
        if not confirm("Keep the workspace clear and the arm supported.", "JOG"):
            print("Not confirmed; nothing enabled.")
            return 1

        for n in names + grip:
            arm._motor_map[n].clear_error()
        if not (arm.arm.mode_mit() and arm.gripper.mode_mit()):
            print("Not every motor confirmed MIT mode; nothing enabled.")
            return 1
        ctrl = JogController(arm, estop, arm._joint_limits, gravity, args.tracking_limit,
                             args.joint_speed, args.joint_accel, arm.rate)
        jog = Jogger(ctrl, kin, arm.setpoint_limits, args)
        arm.start_control_loop(ctrl, rate=arm.rate)
        time.sleep(0.1)  # zero-gain frames at the measured pose are flowing before enabling
        if estop.tripped:
            return 1
        arm.enable_all()
        ctrl.mark_enabled()
        if not ctrl.verified.wait(ENABLE_TIMEOUT_S + 0.5) and not estop.tripped:
            estop.trip("motors did not report ENABLED")
        if estop.tripped:
            return 1

        keyboard = Keyboard()
        estop.on_hard_exit(keyboard.restore)
        print("\nEnabled. " + jog.scale_steps(1.0))
        last_print = 0.0
        while not estop.tripped:
            if not arm.control_loop_active:
                estop.trip("control loop stopped")
                break
            key = keyboard.read(0.05)
            msg = None
            if key in ("q", "\x1b"):
                estop.trip("stopped by user")
                break
            if key in KEYS:
                msg = jog.request(*KEYS[key])
            elif key in ("+", "="):
                msg = jog.scale_steps(2.0)
            elif key in ("-", "_"):
                msg = jog.scale_steps(0.5)
            elif key == " ":
                ctrl.hold()
                msg = "stopping here"
            elif key in ("h", "?"):
                msg = "\n" + HELP
            now = time.monotonic()
            if msg or key == "p" or now - last_print > 0.5:
                last_print = now
                q_set, _ = ctrl.snapshot()
                ee = kin.ee_pos(q_set)
                if msg:
                    print(f"\r  {msg}" + " " * 20)
                print(f"\r  q {fmt_deg(ctrl.q_meas)}  ee [{ee[0]:+.3f} {ee[1]:+.3f} {ee[2]:+.3f}]",
                      end="", flush=True)
    except (KeyboardInterrupt, EOFError):
        estop.trip("interrupted")
    except Exception as e:
        estop.trip(f"error: {e!r}")
        raise
    finally:
        estop.quiet()  # from here a Ctrl+C no longer interrupts cleanup (a second one hard-exits)
        estop.trip("exiting")  # no-op if already tripped: stop commanding first, whatever happened
        if keyboard is not None:
            keyboard.restore()
        print(f"\n\nStopping ({estop.reason}): commands stopped; disabling all motors.")
        for _attempt in range(2):  # a stray KeyboardInterrupt cannot skip the explicit disable
            try:
                if started and getattr(arm, "_connected", False):
                    try:
                        arm.disable_all()
                    finally:
                        arm.disconnect()
                break
            except KeyboardInterrupt:
                continue
        estop.uninstall()
    return 0 if estop.reason == "stopped by user" else 1


if __name__ == "__main__":
    sys.exit(main())
