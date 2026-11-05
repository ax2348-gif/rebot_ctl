"""rebot_jog: jog the arm's end effector from the keyboard (IK, MIT impedance control).

Defaults to config/rebotarm_dm_safe.yaml (low stiffness on the big joints). Each key press
moves the target pose by one step; IK finds the joint target, which the arm then follows
along a min-jerk joint-space segment limited in joint and Cartesian speed.

Safety design:
  - startup: the CAN watchdog and joint limits are verified (connect()); the configured
    arm gains must be below hard ceilings; the arm's current pose, the settings and the
    keys are shown, and you must type JOG before anything is enabled;
  - the arm first holds its *measured* pose, with stiffness and gravity torque ramping in
    from zero, so enabling never pulls it anywhere; every motor must confirm MIT mode;
  - every move is checked before it starts: IK must converge, the target must stay inside
    the joint limits (with margin) and above --z-min, and no joint may jump more than
    --max-jump in one step (rejects elbow flips); speed is capped per joint and in
    Cartesian space, both with hard ceilings;
  - while running, the control loop stops the arm on a motor fault, a joint outside its
    limits, a tracking error above --tracking-limit, or a missing reading;
  - stopping never moves the arm: Ctrl+C (also SIGTERM, SIGHUP, SIGQUIT, Ctrl+Z), q, Esc,
    a fault or an error stops all commands immediately (the 100 ms motor watchdog disables
    the motors even if this process hangs), then disables the motors explicitly. A second
    Ctrl+C, or a shutdown taking over 3 s, ends the process at once. The arm goes limp
    when stopped: keep it supported or in a safe place.
"""
from __future__ import annotations

import math
import select
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..actuator import RebotArm, load_cfg, load_gravity_compensation_config
from ..actuator.rebotarm import load_joint_limits
from ._common import confirm, describe_status, fmt_deg, parser
from ._estop import EStop
from .vis import GravityTorque

DEFAULT_HW = "rebotarm_dm_safe.yaml"

# Hard ceilings: command-line values above these are refused.
MAX_STEP_M = 0.05
MAX_ROT_STEP_DEG = 15.0
MAX_CART_SPEED = 0.10       # m/s
MAX_JOINT_SPEED = 1.0       # rad/s
MAX_ARM_KP = 30.0           # refuse configs stiffer than this (rebotarm_dm.yaml has 120)
MAX_ARM_KD = 10.0

GAIN_RAMP_S = 1.0           # stiffness and gravity torque fade in over this time
MIN_SEGMENT_S = 0.2
MIN_JERK_PEAK = 1.875       # peak velocity of a min-jerk segment / average velocity
LIMIT_MARGIN_RAD = 0.05     # move targets stay this far inside the URDF limits
RUN_LIMIT_MARGIN_RAD = 0.15  # a measured position beyond limits + this stops the arm

KEYS = {
    "w": ("x", +1), "s": ("x", -1), "a": ("y", +1), "d": ("y", -1), "r": ("z", +1), "f": ("z", -1),
    "u": ("roll", +1), "o": ("roll", -1), "i": ("pitch", +1), "k": ("pitch", -1),
    "j": ("yaw", +1), "l": ("yaw", -1),
}
HELP = """keys:  w/s ±x   a/d ±y   r/f ±z        (base frame)
       u/o ±roll i/k ±pitch j/l ±yaw   (about the base axes, at the end effector)
       + / -  bigger / smaller steps    space  stop here    p  print state
       q / Esc / Ctrl+C  stop: commands cease and the motors are disabled (arm goes limp)"""


def _bounded(lo: float, hi: float):
    def parse(text: str) -> float:
        v = float(text)
        if not (lo <= v <= hi and math.isfinite(v)):
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

    def ik(self, target, q_seed: np.ndarray):
        from ..kinematics import solve_ik
        from ..kinematics.inverse_kinematics import IKParams

        res = solve_ik(self.model, self.data, self.frame_id, target, np.asarray(q_seed),
                       IKParams(max_iter=300, tolerance=1e-4, step_size=0.5, damping=1e-6),
                       controlled_joints=self.n)
        return (np.asarray(res.q[: self.n], dtype=np.float64) if res.success else None), res.error


# --------------------------------------------------------------------------- controller


@dataclass
class Segment:
    q0: np.ndarray
    q1: np.ndarray
    t0: float
    T: float

    def at(self, t: float):
        if self.T <= 0:
            return self.q1.copy(), np.zeros_like(self.q1)
        s = min(1.0, max(0.0, (t - self.t0) / self.T))
        blend = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        dblend = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / self.T if s < 1.0 else 0.0
        d = self.q1 - self.q0
        return self.q0 + d * blend, d * dblend

    def done(self, t: float) -> bool:
        return t >= self.t0 + self.T


class JogController:
    """Control-loop callback. Every tick: stop check first, then safety checks, then one
    MIT command per joint. Thread-safe against the main thread's move requests."""

    def __init__(self, arm: RebotArm, estop: EStop, q0: np.ndarray, q_grip0: np.ndarray,
                 limits: dict, gravity: Optional[GravityTorque], tracking_limit: float,
                 clock=time.monotonic):
        self.arm, self.estop, self.clock = arm, estop, clock
        self.names = arm.arm.joint_names
        self.grip_names = arm.gripper.joint_names
        self.kp_cfg = np.asarray(arm.arm._mit_kp, dtype=np.float64).copy()
        self.kd_cfg = np.asarray(arm.arm._mit_kd, dtype=np.float64).copy()
        self.gkp_cfg = np.asarray(getattr(arm.gripper, "_mit_kp", np.zeros(len(self.grip_names))),
                                  dtype=np.float64).copy()
        self.gkd_cfg = np.asarray(getattr(arm.gripper, "_mit_kd", np.zeros(len(self.grip_names))),
                                  dtype=np.float64).copy()
        self.q_grip0 = np.asarray(q_grip0, dtype=np.float64).copy()
        self.lo = np.array([limits.get(n, (-np.inf, np.inf))[0] for n in self.names])
        self.hi = np.array([limits.get(n, (-np.inf, np.inf))[1] for n in self.names])
        self.gravity = gravity
        self.tracking_limit = tracking_limit
        self._lock = threading.Lock()
        self._seg = Segment(q0.copy(), q0.copy(), clock(), 0.0)
        self.t_enabled: Optional[float] = None
        self.q_meas = q0.copy()
        self.last_tau = np.zeros(len(self.names))

    # -- main-thread API ---------------------------------------------------

    def mark_enabled(self) -> None:
        self.t_enabled = self.clock()

    def setpoint(self) -> np.ndarray:
        with self._lock:
            return self._seg.at(self.clock())[0]

    def goal(self) -> np.ndarray:
        with self._lock:
            return self._seg.q1.copy()

    def moving(self) -> bool:
        with self._lock:
            return not self._seg.done(self.clock())

    def ramped(self) -> bool:
        return self.t_enabled is not None and self.clock() - self.t_enabled >= GAIN_RAMP_S

    def go(self, q1: np.ndarray, T: float) -> None:
        """Start a segment from the current setpoint to ``q1`` lasting ``T`` seconds."""
        with self._lock:
            now = self.clock()
            q_now, _ = self._seg.at(now)
            self._seg = Segment(q_now, np.asarray(q1, dtype=np.float64).copy(), now, max(T, 0.0))

    def hold(self) -> None:
        with self._lock:
            now = self.clock()
            q_now, _ = self._seg.at(now)
            self._seg = Segment(q_now, q_now, now, 0.0)

    # -- control loop ------------------------------------------------------

    def __call__(self, r: RebotArm, dt: float) -> None:
        if self.estop.tripped:
            return  # send nothing: the motor watchdog disables the motors within 100 ms
        try:
            self._tick(r)
        except Exception as e:  # never let the loop die quietly
            self.estop.trip(f"control loop error: {e!r}")

    def _tick(self, r: RebotArm) -> None:
        now = self.clock()
        states = [r._motor_map[n].get_state() for n in self.names + self.grip_names]
        if any(st is None for st in states):
            self.estop.trip("a motor stopped reporting feedback")
            return
        for n, st in zip(self.names + self.grip_names, states):
            if int(st.status_code) >= 0x8:
                self.estop.trip(f"{n}: {describe_status('damiao', int(st.status_code))}")
                return
        q = np.array([st.pos for st in states[: len(self.names)]], dtype=np.float64)
        self.q_meas = q
        outside = (q < self.lo - RUN_LIMIT_MARGIN_RAD) | (q > self.hi + RUN_LIMIT_MARGIN_RAD)
        if np.any(outside):
            j = int(np.argmax(outside))
            self.estop.trip(f"{self.names[j]} at {math.degrees(q[j]):+.1f}° is outside its limits")
            return

        with self._lock:
            q_set, qd_set = self._seg.at(now)
        ramp = 0.0 if self.t_enabled is None else _smoothstep((now - self.t_enabled) / GAIN_RAMP_S)
        if ramp >= 1.0:
            err = np.abs(q - q_set)
            if np.any(err > self.tracking_limit):
                j = int(np.argmax(err))
                self.estop.trip(f"{self.names[j]} is {math.degrees(err[j]):.1f}° from its target "
                                f"(limit {math.degrees(self.tracking_limit):.0f}°): blocked or "
                                f"pushed?")
                return

        tau = np.zeros(len(self.names))
        if self.gravity is not None and ramp > 0.0:
            g = self.gravity(q)
            if np.all(np.isfinite(g)):
                tau = g * ramp
        self.last_tau = tau
        if self.estop.tripped:  # re-check right before sending
            return
        r.arm.send_mit(pos=q_set, vel=qd_set * ramp, kp=self.kp_cfg * ramp, kd=self.kd_cfg * ramp,
                       tau=tau)
        if self.grip_names:
            z = np.zeros(len(self.grip_names))
            r.gripper.send_mit(pos=self.q_grip0, vel=z, kp=self.gkp_cfg * ramp,
                               kd=self.gkd_cfg * ramp, tau=z)


# --------------------------------------------------------------------------- moves


class Jogger:
    """Turns jog requests into checked joint-space segments."""

    def __init__(self, ctrl: JogController, kin: Kinematics, args):
        self.ctrl, self.kin, self.args = ctrl, kin, args
        self.step = args.step
        self.rot_step = math.radians(args.rot_step)

    def request(self, axis: str, sign: int) -> str:
        pin = self.kin.pin
        if not self.ctrl.ramped():
            return "wait: still ramping in"
        q_from = self.ctrl.goal()  # chain from the commanded goal, not the sagging measurement
        T_from = self.kin.fk(q_from)
        pos = np.array(T_from.translation, dtype=np.float64)
        rot = np.array(T_from.rotation, dtype=np.float64)
        if axis in ("x", "y", "z"):
            pos[("x", "y", "z").index(axis)] += sign * self.step
        else:
            w = np.zeros(3)
            w[("roll", "pitch", "yaw").index(axis)] = sign * self.rot_step
            rot = pin.exp3(w) @ rot  # about the base axes, at the end effector
        if pos[2] < self.args.z_min:
            return f"refused: z would be {pos[2]:+.3f} m, below --z-min {self.args.z_min:g}"
        T_to = pin.SE3(rot, pos)

        q_to, err = self.kin.ik(T_to, q_from)
        if q_to is None:
            return f"refused: IK did not converge (error {err:.1e}); try a smaller step"
        lo, hi = self.ctrl.lo + LIMIT_MARGIN_RAD, self.ctrl.hi - LIMIT_MARGIN_RAD
        bad = np.where((q_to < lo) | (q_to > hi))[0]
        if bad.size:
            j = int(bad[0])
            return f"refused: {self.ctrl.names[j]} would be at {math.degrees(q_to[j]):+.1f}°, near its limit"
        dq = np.abs(q_to - q_from)
        if np.max(dq) > self.args.max_jump:
            j = int(np.argmax(dq))
            return (f"refused: {self.ctrl.names[j]} would jump {math.degrees(dq[j]):.1f}° "
                    f"(> --max-jump {math.degrees(self.args.max_jump):.0f}°)")

        dist = float(np.linalg.norm(pos - np.array(T_from.translation)))
        T = max(MIN_SEGMENT_S,
                MIN_JERK_PEAK * float(np.max(dq)) / self.args.joint_speed,
                MIN_JERK_PEAK * dist / self.args.cart_speed)
        self.ctrl.go(q_to, T)
        return f"{axis} {'+' if sign > 0 else '-'}  ({T:.2f} s)"

    def scale_steps(self, factor: float) -> str:
        self.step = min(MAX_STEP_M, max(0.001, self.step * factor))
        self.rot_step = min(math.radians(MAX_ROT_STEP_DEG), max(math.radians(0.5), self.rot_step * factor))
        return f"step {self.step * 1000:.1f} mm, {math.degrees(self.rot_step):.1f}°"


# --------------------------------------------------------------------------- terminal


class Keyboard:
    """Single-key input (cbreak: Ctrl+C still raises SIGINT). Restores the terminal on exit."""

    def __init__(self):
        import termios
        import tty

        self._termios = termios
        self.fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def read(self, timeout: float) -> Optional[str]:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch == "":
            raise EOFError
        return ch

    def restore(self) -> None:
        self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self._saved)


# --------------------------------------------------------------------------- main


def _check_gains(arm: RebotArm) -> Optional[str]:
    kp = np.asarray(arm.arm._mit_kp)
    kd = np.asarray(arm.arm._mit_kd)
    if np.any(kp > MAX_ARM_KP) or np.any(kd > MAX_ARM_KD):
        return (f"the config's arm gains (kp {kp.tolist()}, kd {kd.tolist()}) exceed rebot_jog's "
                f"ceilings (kp ≤ {MAX_ARM_KP:g}, kd ≤ {MAX_ARM_KD:g}); use a safe config such as "
                f"{DEFAULT_HW}")
    return None


def _read_positions(arm: RebotArm, names, timeout: float = 1.0) -> Optional[np.ndarray]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for n in names:
            arm._motor_map[n].request_feedback()
        for c in arm._ctrl_map.values():
            c.poll_feedback_once()
        states = [arm._motor_map[n].get_state() for n in names]
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
    p.add_argument("--max-jump", type=_bounded(0.02, 0.5), default=0.25,
                   help="refuse a step that moves any joint more than this many rad (default 0.25)")
    p.add_argument("--tracking-limit", type=_bounded(0.05, 0.6), default=0.35,
                   help="stop if any joint is this many rad from its target (default 0.35)")
    p.add_argument("--z-min", type=float, default=0.0,
                   help="refuse targets below this height in m, base frame (default 0.0)")
    p.add_argument("--no-gravity", action="store_true",
                   help="no gravity-compensation torque (the arm will sag against its low stiffness)")
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
    limits = load_joint_limits(cfg.get("urdf_path"))
    gravity = None
    if not args.no_gravity:
        gravity = GravityTorque(cfg.get("urdf_path"),
                                load_gravity_compensation_config(args.profile, args.hardware_yaml),
                                arm.arm.num_joints, args.tau_limit)

    estop = EStop(grace_s=3.0).install()
    keyboard: Optional[Keyboard] = None
    started = False
    print(f"Config: {arm.hardware_yaml} on {arm.channel}")
    try:
        arm.connect()  # verifies the CAN watchdog and joint limits
        started = True
        names, grip = arm.arm.joint_names, arm.gripper.joint_names
        q = _read_positions(arm, names + grip)
        if q is None:
            print("Not every motor reported its position; nothing enabled.")
            return 1
        q0, qg0 = q[: len(names)], q[len(names):]
        T0 = kin.fk(q0)
        rpy = np.degrees(kin.pin.rpy.matrixToRpy(T0.rotation))

        print("\n  " + "  ".join(f"{n:>8}" for n in names))
        print("  " + fmt_deg(q0) + "   (deg, now)")
        print(f"  end effector: [{T0.translation[0]:+.3f} {T0.translation[1]:+.3f} "
              f"{T0.translation[2]:+.3f}] m, rpy [{rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}]°")
        print(f"  arm gains: kp {np.round(arm.arm._mit_kp, 1).tolist()}  "
              f"kd {np.round(arm.arm._mit_kd, 1).tolist()}")
        print(f"  steps {args.step * 1000:.0f} mm / {args.rot_step:g}°, speed ≤ {args.cart_speed:g} m/s "
              f"and ≤ {args.joint_speed:g} rad/s per joint, gravity "
              f"{'off' if gravity is None else f'on (≤ {args.tau_limit:g} N·m)'}")
        print("\n" + HELP)
        print("\nThe arm will be enabled and hold its current pose (stiffness ramps in over "
              f"{GAIN_RAMP_S:g} s).\nWhen it stops for any reason the motors are disabled and the "
              "arm goes limp.")
        if not confirm("Keep the workspace clear and the arm supported.", "JOG"):
            print("Not confirmed; nothing enabled.")
            return 1

        for n in names + grip:
            arm._motor_map[n].clear_error()
        if not (arm.arm.mode_mit() and arm.gripper.mode_mit()):
            print("Not every motor confirmed MIT mode; nothing enabled.")
            return 1
        ctrl = JogController(arm, estop, q0, qg0, limits, gravity, args.tracking_limit)
        jog = Jogger(ctrl, kin, args)
        arm.start_control_loop(ctrl, rate=arm.rate)
        time.sleep(0.05)  # zero-gain hold frames are flowing before anything is enabled
        if estop.tripped:
            return 1
        arm.enable_all()
        ctrl.mark_enabled()

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
                msg = "holding here"
            elif key in ("h", "?"):
                msg = "\n" + HELP
            now = time.monotonic()
            if msg or key == "p" or now - last_print > 0.5:
                last_print = now
                T = kin.fk(ctrl.setpoint())
                line = (f"\r  q {fmt_deg(ctrl.q_meas)}  ee [{T.translation[0]:+.3f} "
                        f"{T.translation[1]:+.3f} {T.translation[2]:+.3f}]")
                if msg:
                    print(f"\r  {msg}" + " " * 20)
                print(line, end="", flush=True)
    except (KeyboardInterrupt, EOFError):
        estop.trip(estop.reason or "interrupted")
    except Exception as e:
        estop.trip(f"error: {e!r}")
        raise
    finally:
        estop.trip(estop.reason or "exiting")  # stop commanding first, whatever happened
        estop.quiet()  # further Ctrl+C no longer interrupts the cleanup (a second one hard-exits)
        if keyboard is not None:
            keyboard.restore()
        print(f"\n\nStopping ({estop.reason}): commands stopped; disabling all motors.")
        if started:
            try:
                arm.disable_all()
            finally:
                arm.disconnect()
        estop.uninstall()
    return 0 if estop.reason in ("stopped by user",) else 1


if __name__ == "__main__":
    sys.exit(main())
