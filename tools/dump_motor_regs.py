#!/usr/bin/env python3
"""只读寄存器转储：并排打印每个 Damiao 电机的 KT_Value / Gr / PMAX / VMAX / TMAX / sw_ver。

Read-only register dump: prints KT_Value, Gr, PMAX, VMAX, TMAX and sw_ver of
every Damiao motor side by side, so motors of the same model can be compared.

只发送寄存器读取帧（0x33）。不使能、不失能、不切换模式、不写寄存器、不设零、
不发送任何控制指令，也不使用 RebotArm（其 disconnect() 会失能电机）。

Only register-read frames (0x33) go on the bus. It never enables, disables,
switches mode, writes registers, sets zero or sends a control command, and it
does not use RebotArm (whose disconnect() disables all motors).

用法 / Usage::

    python tools/dump_motor_regs.py
      自动从 config/rebotarm.yaml 的 hardware_yaml 读取配置
      Auto-load config from config/rebotarm.yaml's hardware_yaml field

    python tools/dump_motor_regs.py rebotarm_dm.yaml
      强制指定硬件配置文件 / Force a specific hardware config file
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motorbridge import Controller

from reBotArm_control_py.actuator.rebotarm import load_cfg

READ_TIMEOUT_MS = 500

# (rid, name, type, format)
REGISTERS = [
    (1, "KT_Value", "f32", "{:.4f}"),
    (20, "Gr", "f32", "{:.3f}"),
    (21, "PMAX", "f32", "{:.3f}"),
    (22, "VMAX", "f32", "{:.3f}"),
    (23, "TMAX", "f32", "{:.3f}"),
    (14, "sw_ver", "u32", "{}"),
    # RAM 值在电机重启后恢复为 flash 值，可用于判断电机是否重启过
    # RAM-only values revert to flash on reboot, so these reveal whether a motor reset
    (9, "TIMEOUT", "u32", "{}"),
    (10, "CTRL_MODE", "u32", "{}"),
]


def make_controller(channel: str) -> Controller:
    # Same transport selection as RebotArm._make_controller
    if channel.startswith("/dev/"):  # any device path is the serial bridge; CAN interfaces are bare names (can0)
        return Controller.from_dm_serial(channel, 921600)
    return Controller(channel)


def read_registers(motor) -> dict:
    out = {}
    for rid, name, typ, fmt in REGISTERS:
        try:
            value = (motor.get_register_f32(rid, READ_TIMEOUT_MS) if typ == "f32"
                     else motor.get_register_u32(rid, READ_TIMEOUT_MS))
            out[name] = fmt.format(value)
        except Exception as e:
            out[name] = "ERR"
            print(f"[{name}] read failed: {e}")
    return out


def render(joints, values) -> str:
    # 同型号电机读数不一致的寄存器标 '*' / Mark '*' where motors of the same model disagree
    by_model: dict = {}
    for jc in joints:
        by_model.setdefault(jc.model, []).append(jc.name)

    w = 12
    lines = [f"{'':<10}" + "".join(f"{jc.name:>{w}}" for jc in joints),
             f"{'model':<10}" + "".join(f"{jc.model:>{w}}" for jc in joints),
             f"{'motor_id':<10}" + "".join(f"{hex(jc.motor_id):>{w}}" for jc in joints),
             ""]
    for _, name, _, _ in REGISTERS:
        cells = []
        for jc in joints:
            v = values[jc.name][name]
            peers = {values[n][name] for n in by_model[jc.model]}
            cells.append(f"{v + ('*' if len(peers) > 1 else ' '):>{w}}")
        lines.append(f"{name:<10}" + "".join(cells))
    lines += ["", "* = differs from another motor of the same model"]
    return "\n".join(lines)


def main():
    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else None)
    joints = [jc for jc in cfg["joints"] if jc.vendor == "damiao"]
    skipped = [jc.name for jc in cfg["joints"] if jc.vendor != "damiao"]
    if skipped:
        print(f"跳过非 Damiao 电机 / Skipping non-Damiao motors: {', '.join(skipped)}\n")
    if not joints:
        sys.exit("No Damiao motors in config.")

    ctrl = make_controller(cfg["channel"])
    motors = {}
    try:
        values = {}
        for jc in joints:
            motors[jc.name] = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)
            print(f"读取 / Reading {jc.name} ({hex(jc.motor_id)})...")
            values[jc.name] = read_registers(motors[jc.name])
        print()
        print(render(joints, values))
    finally:
        # Free handles only. No disable_all()/shutdown(): nothing was enabled
        # here, and shutdown() disables every motor on the bus.
        for m in motors.values():
            try:
                m.close()
            except Exception:
                pass
        ctrl.close()


if __name__ == "__main__":
    main()
