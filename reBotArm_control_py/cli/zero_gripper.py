"""rebot_zero_gripper: drive the gripper closed against its stop; with --zero, zero it there.

The gripper motor (only) runs in MIT mode as a pure velocity loop: kp = 0, tau = 0, target
velocity GRIPPER_CLOSE_VEL and damping GRIPPER_CLOSE_KD, so the motor torque is

    tau = GRIPPER_CLOSE_KD * (GRIPPER_CLOSE_VEL - v)

Moving freely it closes at about GRIPPER_CLOSE_VEL (less by friction / GRIPPER_CLOSE_KD);
against the closed stop (v = 0) it pushes with GRIPPER_CLOSE_KD * GRIPPER_CLOSE_VEL. A torque
feed-forward alone would be poorly defined while the gripper moves freely; this loop is not.

This is the same drive RebotArm.connect() runs by default to check the gripper's zero
(RebotArm.close_gripper); its constants are in actuator/rebotarm.py.

Steps:
  1. connect (CAN watchdog verified; the arm's joint-limit check and the gripper closing
     check are skipped: the arm is never enabled, and this tool is the fix for a bad zero);
  2. clear the gripper's faults, put it in MIT mode, show its reading;
  3. enable the gripper and drive it closed for --duration seconds (default
     GRIPPER_CLOSE_S, 0.5);
  4. show where it ended up, then disable it;
  5. with --zero: set that position as the gripper's zero (fully closed, with the gearbox
     backlash taken up in the closing direction), and show the new reading (~0).

The arm motors stay disabled throughout. Closing is the positive direction (closed = 0,
open is negative; tools/quest_teleop.py GRIPPER_OPEN_RAD).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ..actuator import GripperCheckError, RebotArm
from ..actuator.rebotarm import GRIPPER_CLOSE_KD, GRIPPER_CLOSE_S, GRIPPER_CLOSE_VEL
from ._common import describe_status, parser


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
    p.add_argument("--duration", type=float, default=GRIPPER_CLOSE_S,
                   help=f"seconds to drive closed (default {GRIPPER_CLOSE_S})")
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
    print(f"Drive: {args.duration:.2f} s at {GRIPPER_CLOSE_VEL:+.2f} rad/s, kd {GRIPPER_CLOSE_KD:.3f} "
          f"(stall push {GRIPPER_CLOSE_KD * GRIPPER_CLOSE_VEL:.2f} Nm), kp 0, tau 0")
    arm.connect(check_limits=False, check_gripper=False)   # still verifies the CAN watchdog
    g = arm.gripper
    motor = arm._motor_map[g.joint_names[0]]
    try:
        arm.disable_all()
        if _read(g, motor) is None:
            print("No feedback from the gripper; nothing driven.")
            return 1
        print(f"\nBefore: {_state(motor)}")
        try:
            arm.close_gripper(args.duration, on_sample=lambda t, st: print(
                f"\r  {t:4.2f} s  {_state(motor)}   ", end="", flush=True))
        except GripperCheckError as e:
            print(f"\n{e}")
            return 1
        print(f"\nAfter:  {_state(motor)}")   # read while still pushing; it is disabled now

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
