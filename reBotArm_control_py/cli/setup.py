"""rebot_setup: check, and if needed perform, the arm's one-time setup.

The one setup step the arm needs is the motor CAN watchdog: every motor must have
TIMEOUT = CAN_TIMEOUT_MS (100 ms) stored in flash, so an enabled motor that stops
receiving CAN frames disables itself. RebotArm.connect() refuses to run otherwise.

    rebot_setup                    read-only check: every motor answers, is disabled or
                                   fault-free, has the watchdog set, and reads inside its
                                   joint limits. Changes nothing.
    rebot_setup --store-watchdog   write TIMEOUT and store it to every Damiao motor's flash
                                   (asks you to type STORE first)

Neither mode enables a motor, switches modes, sets zero or sends a control command, and
neither uses RebotArm (whose connect() is what checks the watchdog).
"""
from __future__ import annotations

import math
import sys
from typing import List, Optional

from motorbridge.damiao_registers import DAMIAO_RW_REGISTERS

from ..actuator.rebotarm import (
    CAN_TIMEOUT_MS,
    DM_RID_TIMEOUT,
    RebotArm,
    dm_timeout_register_value,
    load_cfg,
    joint_limits,
    load_joint_limits,
    resolve_channel,
)
from ._common import (describe_status, make_controller, optional, parser, print_joint_table,
                      wait_for_feedback)

READ_TIMEOUT_MS = 500
READ_ATTEMPTS = 3
LIMIT_MARGIN_RAD = RebotArm.LIMIT_MARGIN_RAD


def _close(ctrl, motors: dict) -> None:
    # Free handles only: nothing here enables a motor, and shutdown() would send
    # extra disable frames.
    for m in motors.values():
        try:
            m.close()
        except Exception:
            pass
    ctrl.close()


def check(hw_yaml: Optional[str], channel: Optional[str] = None) -> int:
    """Read-only check of the arm's setup. Returns 0 if everything is in order."""
    cfg = load_cfg(hw_yaml)
    cfg["channel"] = resolve_channel(cfg, channel)
    joints = cfg["joints"]
    urdf_limits = load_joint_limits(cfg.get("urdf_path"))  # what connect() checks
    limits = joint_limits(cfg)  # adds the config's q limits (setpoints only)
    target = dm_timeout_register_value(CAN_TIMEOUT_MS)
    print(f"Config: {cfg['name']} on {cfg['channel']}, {len(joints)} joints")

    ctrl = make_controller(cfg["channel"])
    motors = {}
    damiao = {}
    try:
        for jc in joints:
            motors[jc.name] = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model) \
                if jc.vendor == "damiao" else None
            if motors[jc.name] is not None:
                damiao[jc.name] = motors[jc.name]
        states = wait_for_feedback(ctrl, damiao)

        rows = {"status": [], "watchdog": [], "position °": [], "limits °": []}
        problems: List[str] = []
        notes: List[str] = []
        fixes = set()
        for jc in joints:
            if motors[jc.name] is None:
                for r in rows.values():
                    r.append("n/a")
                problems.append(f"{jc.name}: vendor {jc.vendor!r} cannot be checked "
                                "(only Damiao motors report the watchdog)")
                continue
            st = states.get(jc.name)
            if st is None:
                rows["status"].append("NO REPLY")
                rows["watchdog"].append("—")
                rows["position °"].append("—")
                rows["limits °"].append("—")
                problems.append(f"{jc.name}: no feedback (power, wiring or IDs "
                                f"0x{jc.motor_id:X}/0x{jc.feedback_id:X})")
                continue
            code = int(st.status_code)
            rows["status"].append(describe_status("damiao", code).split(" (")[0])
            if code >= 0x8:
                problems.append(f"{jc.name}: {describe_status('damiao', code)}")

            wd, err = None, None
            for _ in range(READ_ATTEMPTS):  # a single reply can be lost right after feedback
                try:
                    wd = int(motors[jc.name].get_register_u32(DM_RID_TIMEOUT, READ_TIMEOUT_MS))
                    break
                except Exception as e:
                    err = e
            if wd is None:
                problems.append(f"{jc.name}: could not read the watchdog register "
                                f"in {READ_ATTEMPTS} attempts ({err})")
            rows["watchdog"].append(optional(None if wd is None else wd / 20, "g") +
                                    ("" if wd is None else " ms"))
            if wd is not None and wd != target:
                problems.append(f"{jc.name}: watchdog is {wd / 20:g} ms, must be {CAN_TIMEOUT_MS} ms")
                fixes.add("watchdog")

            q = float(st.pos)
            rows["position °"].append(f"{math.degrees(q):+.1f}")
            if jc.name in limits:
                lo, hi = limits[jc.name]
                rows["limits °"].append(f"{math.degrees(lo):+.0f}..{math.degrees(hi):+.0f}")
                ulo, uhi = urdf_limits.get(jc.name, (-math.inf, math.inf))
                if not ulo - LIMIT_MARGIN_RAD <= q <= uhi + LIMIT_MARGIN_RAD:
                    problems.append(f"{jc.name}: reads {math.degrees(q):+.1f}°, outside its limits")
                    fixes.add("limits")
                elif not lo <= q <= hi:
                    notes.append(f"{jc.name}: reads {math.degrees(q):+.1f}°, outside the config's "
                                 "q limits (setpoints there are refused; nothing else changes)")
            else:
                rows["limits °"].append("none")

        print()
        print_joint_table([jc.name for jc in joints], rows)
        print()
        for n in notes:
            print(f"Note: {n}")
        if not problems:
            print(f"OK: every motor answers, has the {CAN_TIMEOUT_MS} ms watchdog stored and "
                  "reads inside its limits.")
            return 0
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
        if "watchdog" in fixes:
            print("\nFix the watchdog: power-cycle the arm, then run  rebot_setup --store-watchdog")
        if "limits" in fixes:
            print("\nA joint outside its limits usually picked the wrong encoder turn at power-on:"
                  " power-cycle with the joint near its zero stop. If the zero itself is wrong,"
                  " put the arm in its zero pose and run  rebot_zero")
        return 1
    finally:
        _close(ctrl, damiao)


def _read_rw_registers(motor) -> dict:
    out = {}
    for rid, spec in sorted(DAMIAO_RW_REGISTERS.items()):
        try:
            v = (motor.get_register_u32(rid, READ_TIMEOUT_MS) if spec.data_type == "u32"
                 else motor.get_register_f32(rid, READ_TIMEOUT_MS))
            out[spec.variable] = f"{v}" if spec.data_type == "u32" else f"{v:.6g}"
        except Exception:
            out[spec.variable] = "ERR"
    return out


def store_watchdog(hw_yaml: Optional[str], channel: Optional[str] = None) -> int:
    """Write the CAN watchdog to every Damiao motor and store it in flash.

    The store command saves *every* parameter currently in RAM, not just the
    timeout, so the arm must be power-cycled first (RAM == flash) with nothing run
    in between. Motors must be disabled. Writes only register 9 (TIMEOUT), then
    stores and reads it back on each motor, after printing every read/write
    register and asking the user to type STORE.
    """
    cfg = load_cfg(hw_yaml)
    cfg["channel"] = resolve_channel(cfg, channel)
    others = [jc.name for jc in cfg["joints"] if jc.vendor != "damiao"]
    if others:
        print(f"Only Damiao motors are supported; aborting (non-Damiao: {', '.join(others)}).")
        return 1
    joints = cfg["joints"]
    target = dm_timeout_register_value(CAN_TIMEOUT_MS)

    ctrl = make_controller(cfg["channel"])
    motors = {}
    try:
        for jc in joints:
            motors[jc.name] = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)

        # Every motor must answer and be disabled
        states = wait_for_feedback(ctrl, motors)
        bad = [f"{n}: {'no feedback' if st is None else f'status 0x{st.status_code:X}'}"
               for n, st in states.items() if st is None or st.status_code != 0]
        if bad:
            print("Every motor must answer and be disabled (status 0x0); aborting:")
            print("\n".join(f"  {b}" for b in bad))
            return 1

        print("Read/write registers that the store command will save to flash:\n")
        snapshot = {n: _read_rw_registers(m) for n, m in motors.items()}
        names = [jc.name for jc in joints]
        w = 11
        print(f"{'':<11}" + "".join(f"{n:>{w}}" for n in names))
        for var in next(iter(snapshot.values())):
            print(f"{var:<11}" + "".join(f"{snapshot[n][var]:>{w}}" for n in names))
        if any(v == "ERR" for regs in snapshot.values() for v in regs.values()):
            print("\nSome registers could not be read; aborting.")
            return 1

        print(f"\nTIMEOUT will be set to {target} ({CAN_TIMEOUT_MS} ms) on every motor, then"
              " every register above is stored to flash.")
        print("The arm must have been power-cycled since anything last changed its settings.")
        if input("Type STORE to continue: ").strip() != "STORE":
            print("Not confirmed; nothing written.")
            return 1

        for n, m in motors.items():
            m.write_register_u32(DM_RID_TIMEOUT, target)
            if m.get_register_u32(DM_RID_TIMEOUT, READ_TIMEOUT_MS) != target:
                print(f"{n}: TIMEOUT write did not read back as {target}; aborting before store.")
                return 1
        for n, m in motors.items():
            m.store_parameters()
            readback = m.get_register_u32(DM_RID_TIMEOUT, READ_TIMEOUT_MS)
            print(f"  {n}: stored, TIMEOUT reads {readback}")
            if readback != target:
                print(f"{n}: TIMEOUT reads {readback} after store; stopping.")
                return 1

        print("\nDone. Power-cycle the arm, then run rebot_setup: the watchdog must read"
              f" {CAN_TIMEOUT_MS} ms on every motor (that confirms it is in flash).")
        return 0
    finally:
        _close(ctrl, motors)


def main(argv=None) -> int:
    p = parser("Check the arm's one-time setup (motor CAN watchdog, feedback, joint limits).")
    p.add_argument("--store-watchdog", action="store_true",
                   help=f"write the {CAN_TIMEOUT_MS} ms CAN watchdog to every motor's flash "
                        "(power-cycle the arm first; asks for confirmation)")
    args = p.parse_args(argv)
    if args.store_watchdog:
        return store_watchdog(args.hardware_yaml, args.channel)
    return check(args.hardware_yaml, args.channel)


if __name__ == "__main__":
    sys.exit(main())
