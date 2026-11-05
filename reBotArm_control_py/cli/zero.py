"""rebot_zero: set the arm's current pose as every joint's zero.

Steps:
  1. connect (the motor CAN watchdog is verified as always; the joint-limit check is
     skipped, since a wrong zero is exactly what it would trip on);
  2. show the current readings and ask you to type ZERO;
  3. set zero on every motor, with all motors disabled throughout;
  4. re-run the joint-limit check and show the new readings, which should be ~0.

No motor is ever enabled. Put the arm in its calibrated zero pose before running.
"""
from __future__ import annotations

import sys
import threading
import time

from ..actuator import RebotArm
from ._common import confirm, fmt_deg, parser


def _set_zero_with_progress(arm: RebotArm) -> None:
    # Setting zero takes up to ~20 s; show progress so it does not look stuck.
    done = threading.Event()

    def progress() -> None:
        start = time.monotonic()
        while not done.wait(1.0):
            print(f"\r  setting zero... {time.monotonic() - start:.0f} s", end="", flush=True)

    threading.Thread(target=progress, daemon=True).start()
    try:
        arm.set_zero()
    finally:
        done.set()
        print("\r  zero set." + " " * 30)


def main(argv=None) -> int:
    p = parser("Set the arm's current pose as every joint's zero (motors stay disabled).")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = p.parse_args(argv)

    arm = RebotArm(args.hardware_yaml, channel=args.channel)
    print(f"Config: {arm.hardware_yaml} on {arm._channel}")
    arm.connect(check_limits=False)  # still verifies the CAN watchdog
    try:
        arm.disable_all()
        names = arm.joint_names
        print("\nJoint readings before (deg):")
        print("  " + "  ".join(f"{n:>8}" for n in names))
        print("  " + fmt_deg(arm.get_positions()))
        print("\nEvery joint's current position becomes its zero. Put the arm in its"
              " calibrated zero pose first.")
        if not (args.yes or confirm("Is the arm in its zero pose?", "ZERO")):
            print("Not confirmed; nothing changed.")
            return 1

        _set_zero_with_progress(arm)
        time.sleep(0.2)
        print("\nJoint readings after (deg):")
        print("  " + "  ".join(f"{n:>8}" for n in names))
        print("  " + fmt_deg(arm.get_positions()))
        arm.check_joint_limits()
        print("\nDone: every joint is zeroed and reads inside its limits.")
        return 0
    finally:
        arm.disconnect()  # disables all motors (they were never enabled)


if __name__ == "__main__":
    sys.exit(main())
