#!/usr/bin/env python3
"""实机随动 + MeshCat 实时可视化。

机械臂处于随动状态（MIT，kp=0, kd=0），可手动自由移动，
MeshCat 实时同步显示当前姿态，并在终端打印末端位置/姿态。

Real-arm free-drive + live MeshCat visualization.
The arm is in free-drive (MIT, kp=0, kd=0) and can be moved by hand; MeshCat
mirrors its pose in real time and the end-effector pose is printed.

用法 / Usage::

    python tools/vis_bot.py
      自动从 config/rebotarm.yaml 的 hardware_yaml 读取配置
      Auto-load config from config/rebotarm.yaml's hardware_yaml field

    python tools/vis_bot.py rebotarm_dm.yaml
      强制指定硬件配置文件 / Force a specific hardware config file

    python tools/vis_bot.py --zero
      启动时将当前姿态设为零点（会覆盖原有零点！）
      Set the current pose as zero on startup (overwrites existing zero!)

    python tools/vis_bot.py --can-timeout-ms 100
      设置电机 CAN 通讯超时（仅写 RAM，不保存）。拔掉 CAN 线后电机应自动退出使能。
      Set the motors' CAN watchdog (RAM only, not stored). Unplugging CAN should
      make the motors drop out of enabled mode on their own.
"""
import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.kinematics import compute_fk, get_arm_joint_count, pad_q_for_model
from example.sim.visualizer import Visualizer

# 可视化/打印刷新频率 / Visualization & print refresh rate
VIS_RATE_HZ = 30.0

# Damiao 反馈帧 ERR 状态码 / Damiao feedback ERR nibble (raw status_code)
DM_STATUS = {
    0x0: "disabled",
    0x1: "ENABLED",
    0x8: "FAULT overvoltage",
    0x9: "FAULT undervoltage",
    0xA: "FAULT overcurrent",
    0xB: "FAULT MOS overtemp",
    0xC: "FAULT rotor overtemp",
    0xD: "FAULT comm loss",
    0xE: "FAULT overload",
}

should_exit = False


def signal_handler(sig, frame):
    global should_exit
    should_exit = True


def set_zero_with_progress(rebotarm: RebotArm) -> None:
    # 零点校准约需 20s，期间实时打印进度，避免误以为卡死
    # Zero calibration takes ~20s; print live progress so it doesn't look frozen.
    done = threading.Event()

    def _progress() -> None:
        start = time.time()
        while not done.is_set():
            print(f"\r正在设置零点 / Setting zero... {time.time() - start:.0f}s",
                  end="", flush=True)
            done.wait(1.0)

    threading.Thread(target=_progress, daemon=True).start()
    try:
        rebotarm.set_zero()
    finally:
        done.set()
    print("\r--- 零点已设置 / Zero set ---" + " " * 30 + "\n")


def set_can_timeout(rebotarm: RebotArm, timeout_ms: int) -> None:
    # 寄存器单位均为 50us（20 = 1ms）；只写 RAM，断电即恢复。
    # motorbridge 的 set_can_timeout_ms 对 RobStride 未做 ×20 换算，故直接写参数 0x7028。
    # Both vendors use 50us units (20 = 1ms); RAM only, reverts on power cycle.
    # motorbridge's set_can_timeout_ms skips the ×20 for RobStride, so write 0x7028 directly.
    for jc in rebotarm._all_joints:
        mot = rebotarm._motor_map[jc.name]
        if jc.vendor == "damiao":
            mot.set_can_timeout_ms(timeout_ms)
        elif jc.vendor == "robstride":
            mot.robstride_write_param_u32(0x7028, timeout_ms * 20)
        else:
            print(f"[{jc.name}] CAN timeout not supported for vendor {jc.vendor}, skipped")
            continue
        # 清除上次测试残留的通讯丢失故障 / Clear a LOST_COMM fault left by a previous test
        mot.clear_error()
    print(f"--- CAN 超时 / CAN timeout: {timeout_ms} ms ---\n")


class StatusMonitor:
    """电机状态码变化时立即打印（如 ENABLED -> FAULT overvoltage）。

    Prints a line the moment any motor's status code changes, e.g.
    ENABLED -> FAULT overvoltage. Reads the cached feedback, so call it right
    after something has polled the bus (e.g. get_positions()).
    """

    def __init__(self, rebotarm: RebotArm) -> None:
        self._r = rebotarm
        self._last: dict = {}
        self._t0 = time.monotonic()

    @staticmethod
    def _describe(vendor: str, code: int) -> str:
        if vendor == "damiao":
            return f"{DM_STATUS.get(code, 'unknown')} (0x{code:X})"
        return f"status 0x{code:X}"

    def check(self) -> None:
        for jc in self._r._all_joints:
            st = self._r._motor_map[jc.name].get_state()
            if st is None:
                continue
            code = int(st.status_code)
            prev = self._last.get(jc.name)
            if code != prev:
                was = "—" if prev is None else self._describe(jc.vendor, prev)
                print(f"\n[{time.monotonic() - self._t0:7.2f}s] {jc.name}: "
                      f"{was} -> {self._describe(jc.vendor, code)}", flush=True)
                self._last[jc.name] = code


def make_free_drive_controller(n_arm: int):
    # 随动模式：MIT 零刚度，机械臂可自由手动移动
    # Free-drive mode: MIT zero stiffness, arm can be freely moved by hand
    zeros_arm = np.zeros(n_arm)
    ones_arm = np.ones(n_arm)
    zeros_grip = np.zeros(1)
    kp_arm = None #ones_arm*120.0
    kd_arm = None #ones_arm*8.0
    kp_grip = np.ones(1)

    def controller(r: RebotArm, dt: float) -> None:
        r.arm.send_mit(pos=zeros_arm, vel=zeros_arm, kp=kp_arm, kd=kd_arm, tau=zeros_arm)
        r.gripper.send_mit(pos=zeros_grip, vel=zeros_grip, kp=kp_grip, kd=zeros_grip, tau=zeros_grip)

    return controller


def main():
    parser = argparse.ArgumentParser(description="Free-drive the real arm and mirror it in MeshCat.")
    parser.add_argument("hardware_yaml", nargs="?", default=None, help="hardware config file")
    parser.add_argument("--zero", action="store_true",
                        help="set the current pose as zero on startup (overwrites existing zero)")
    parser.add_argument("--can-timeout-ms", type=int, default=100,
                        help="motor CAN watchdog in ms (RAM only; 0 disables)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    # 先加载可视化器，失败时不会留下已使能的电机
    # Load the visualizer first so a failure here never leaves motors enabled.
    print("加载可视化器 / Loading visualizer...")
    viz = Visualizer()
    n_model = get_arm_joint_count(viz.model)

    rebotarm = RebotArm(args.hardware_yaml)
    print(f"使用配置 / Using config: {rebotarm.hardware_yaml}")
    try:
        # 设零时跳过限位检查（设零本身就是修复手段），设零后再检查
        # Skip the limit check when zeroing (zeroing is the fix), then check afterwards
        rebotarm.connect(check_limits=not args.zero)
        print("--- 连接成功 / Connection OK ---\n")

        if args.zero:
            set_zero_with_progress(rebotarm)
            rebotarm.check_joint_limits()
        if args.can_timeout_ms is not None:
            set_can_timeout(rebotarm, args.can_timeout_ms)

        n_arm = rebotarm.arm.num_joints
        rebotarm.arm.mode_mit()
        rebotarm.gripper.mode_mit()
        rebotarm.enable_all()
        rebotarm.start_control_loop(make_free_drive_controller(n_arm))

        print("--- 随动模式 / Free-drive mode ---")
        print("手动移动机械臂，MeshCat 实时同步。按 Ctrl+C 退出。")
        print("Move the arm by hand; MeshCat follows. Press Ctrl+C to exit.\n")

        status = StatusMonitor(rebotarm)
        period = 1.0 / VIS_RATE_HZ
        while not should_exit:
            t0 = time.time()
            pos = rebotarm.get_positions(position_source = "output")
            # print(rebotarm.get_torques())
            q = pad_q_for_model(viz.model,pos , n_model)
            status.check()
            viz.update(q)
            
            pos, rot, _ = compute_fk(viz.model, q)
            euler = np.degrees(pin.rpy.matrixToRpy(rot))
            # print(f"\r末端位置 / pos: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m  "
            #       f"姿态 / rpy: [{euler[0]:+7.2f}, {euler[1]:+7.2f}, {euler[2]:+7.2f}] deg",
            #       end="", flush=True)

            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        print("\n退出 / Exiting.")
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
