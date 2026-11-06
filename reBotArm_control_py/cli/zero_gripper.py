"""rebot_zero_gripper: drive the gripper closed against its stop; with --zero, zero it there.

The gripper motor (only) runs in MIT mode as a pure velocity loop: kp = 0, tau = 0, target
velocity CLOSE_VEL and damping CLOSE_KD, so the motor torque is

    tau = CLOSE_KD * (CLOSE_VEL - v)

Moving freely it closes at about CLOSE_VEL (less by friction / CLOSE_KD); against the
closed stop (v = 0) it pushes with CLOSE_KD * CLOSE_VEL. A torque feed-forward alone would
be poorly defined while the gripper moves freely; this loop is not.

Steps:
  1. connect (CAN watchdog verified; the arm's joint-limit check is skipped, since the arm
     is never enabled);
  2. clear the gripper's faults, put it in MIT mode, show its reading;
  3. enable the gripper and drive it closed for --duration seconds (default 0.5);
  4. show where it ended up, then disable it;
  5. with --zero: set that position as the gripper's zero (fully closed, with the gearbox
     backlash taken up in the closing direction), and show the new reading (~0).

The arm motors stay disabled throughout. CLOSE_VEL and CLOSE_KD are hard-coded on purpose:
tune them here (rebot_ctl is installed editable, so edits take effect on the next run).
Closing is the positive direction (closed = 0, open is negative; tools/quest_teleop.py
GRIPPER_OPEN_RAD).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ..actuator import RebotArm
from ._common import describe_status, parser

CLOSE_VEL = 3.0     # rad/s target closing speed (+ = closing)
CLOSE_KD = 0.3      # Nm·s/rad; stall push at the stop = CLOSE_KD * CLOSE_VEL = 0.9 Nm
RATE_HZ = 200.0     # command rate; must be well above the 100 ms motor CAN watchdog


def _state(motor):
    st = motor.get_state()
    if st is None:
        return "no feedback"
    return (f"pos {st.pos:+.3f} rad ({np.degrees(st.pos):+.1f}°)  vel {st.vel:+.2f} rad/s  "
            f"torque {st.torq:+.2f} Nm  status {describe_status('damiao', st.status_code)}")


def _read(group, motor, timeout=1.0):
    """Fresh feedback for the gripper (requests until it answers)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        group.get_positions()   # sends a feedback request and processes replies
        if motor.get_state() is not None:
            return motor.get_state()
        time.sleep(0.02)
    return None


def _wait_disabled(group, motor, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        group.get_positions()
        st = motor.get_state()
        if st is not None and st.status_code == 0:
            return True
        time.sleep(0.05)
    return False


def main(argv=None) -> int:
    p = parser("Drive the gripper closed against its stop (MIT velocity loop); "
               "optionally set that position as its zero.")
    p.add_argument("--duration", type=float, default=0.5,
                   help="seconds to drive closed (default 0.5)")
    p.add_argument("--zero", action="store_true",
                   help="after driving, disable the gripper and set its position as zero")
    args = p.parse_args(argv)
    if not 0.0 < args.duration <= 10.0:
        p.error("--duration must be in (0, 10] s")

    arm = RebotArm(args.hardware_yaml, channel=args.channel)
    if not arm.has_gripper:
        print(f"{arm.hardware_yaml} defines no gripper group.")
        return 1
    print(f"Config: {arm.hardware_yaml} on {arm.channel}")
    print(f"Drive: {args.duration:.2f} s at {CLOSE_VEL:+.2f} rad/s, kd {CLOSE_KD:.3f} "
          f"(stall push {CLOSE_KD * CLOSE_VEL:.2f} Nm), kp 0, tau 0")
    arm.connect(check_limits=False)   # still verifies the CAN watchdog
    g = arm.gripper
    motor = arm._motor_map[g.joint_names[0]]
    try:
        arm.disable_all()
        g.clear_error()
        if not g.mode_mit():
            print("Could not put the gripper in MIT mode; nothing driven.")
            return 1
        if _read(g, motor) is None:
            print("No feedback from the gripper; nothing driven.")
            return 1
        print(f"\nBefore: {_state(motor)}")

        g.enable()
        n = 1
        zero, vel, kp, kd = np.zeros(n), np.full(n, CLOSE_VEL), np.zeros(n), np.full(n, CLOSE_KD)
        period = 1.0 / RATE_HZ
        t_start = time.monotonic()
        while (t := time.monotonic() - t_start) < args.duration:
            t0 = time.monotonic()
            g.send_mit(zero, vel=vel, kp=kp, kd=kd, tau=zero)
            g.get_positions()
            print(f"\r  {t:4.2f} s  {_state(motor)}   ", end="", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        print(f"\nAfter:  {_state(motor)}")
        g.disable()

        if not args.zero:
            return 0
        if not _wait_disabled(g, motor):
            print("The gripper did not report disabled; zero NOT set.")
            return 1
        motor.set_zero_position()
        time.sleep(0.2)
        st = _read(g, motor)
        print(f"Zeroed: {_state(motor)}")
        if st is None or abs(st.pos) > 0.05:
            print("The gripper does not read ~0 after setting zero; check it.")
            return 1
        return 0
    finally:
        arm.disconnect()   # disables every motor


if __name__ == "__main__":
    sys.exit(main())
