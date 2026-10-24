#!/usr/bin/env python3
"""将 CAN 通讯看门狗（CAN_TIMEOUT_MS）永久写入每个 Damiao 电机的 flash。

Store the CAN watchdog (CAN_TIMEOUT_MS, 100 ms) permanently in every Damiao
motor's flash. RebotArm.connect() refuses to run unless every motor reports it.

注意 / Important:
  - 存储命令会把电机当前 RAM 中的 *全部* 参数写入 flash，而不仅是超时值。
    运行前请先给机械臂断电重启，使 RAM 与 flash 一致，且之后不要运行其他脚本。
    The store command saves *every* parameter currently in RAM, not just the
    timeout. Power-cycle the arm first so RAM matches flash, and run nothing
    else in between.
  - 电机必须处于失能状态。脚本只写寄存器 9（TIMEOUT），然后逐个存储并回读校验。
    Motors must be disabled. The script writes only register 9 (TIMEOUT), then
    stores and reads it back on each motor.
  - 存储前会打印每个电机全部可读写寄存器，需输入 STORE 确认。
    Before storing it prints every motor's read/write registers and asks you
    to type STORE.

不使用 RebotArm（其 connect() 正是要校验这个值）。
Does not use RebotArm, whose connect() is what checks this value.

用法 / Usage::

    python tools/persist_can_timeout.py [hardware.yaml]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motorbridge import Controller
from motorbridge.damiao_registers import DAMIAO_RW_REGISTERS

from reBotArm_control_py.actuator.rebotarm import (
    CAN_TIMEOUT_MS,
    DM_RID_TIMEOUT,
    dm_timeout_register_value,
    load_cfg,
)

READ_TIMEOUT_MS = 500
STARTUP_TIMEOUT_S = 2.0


def make_controller(channel: str) -> Controller:
    # Same transport selection as RebotArm._make_controller
    if channel.startswith("/dev/tty"):
        return Controller.from_dm_serial(channel, 921600)
    return Controller(channel)


def read_rw_registers(motor) -> dict:
    out = {}
    for rid, spec in sorted(DAMIAO_RW_REGISTERS.items()):
        try:
            v = (motor.get_register_u32(rid, READ_TIMEOUT_MS) if spec.data_type == "u32"
                 else motor.get_register_f32(rid, READ_TIMEOUT_MS))
            out[spec.variable] = f"{v}" if spec.data_type == "u32" else f"{v:.6g}"
        except Exception:
            out[spec.variable] = "ERR"
    return out


def wait_for_feedback(ctrl, motors: dict) -> dict:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        for m in motors.values():
            m.request_feedback()
        ctrl.poll_feedback_once()
        states = {n: m.get_state() for n, m in motors.items()}
        if all(st is not None for st in states.values()):
            return states
        time.sleep(0.02)
    return {n: m.get_state() for n, m in motors.items()}


def main() -> int:
    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else None)
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

        # 每个电机必须应答且处于失能状态 / Every motor must answer and be disabled
        states = wait_for_feedback(ctrl, motors)
        bad = [f"{n}: {'no feedback' if st is None else f'status 0x{st.status_code:X}'}"
               for n, st in states.items() if st is None or st.status_code != 0]
        if bad:
            print("Every motor must answer and be disabled (status 0x0); aborting:")
            print("\n".join(f"  {b}" for b in bad))
            return 1

        print("Read/write registers that the store command will save to flash:\n")
        snapshot = {n: read_rw_registers(m) for n, m in motors.items()}
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

        print("\nDone. Power-cycle the arm, then run tools/dump_motor_regs.py: the TIMEOUT row"
              f" must read {target} on every motor (that confirms it is in flash).")
        return 0
    finally:
        # Free handles only: nothing here enables a motor, and shutdown() would
        # send extra disable frames.
        for m in motors.values():
            try:
                m.close()
            except Exception:
                pass
        ctrl.close()


if __name__ == "__main__":
    sys.exit(main())
