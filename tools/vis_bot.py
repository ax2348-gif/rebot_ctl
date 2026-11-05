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

CAN 看门狗（100 ms）已存储在电机 flash 中，由 RebotArm.connect() 校验，此处不可修改。
The CAN watchdog (100 ms) is stored in motor flash and verified by
RebotArm.connect(); this tool cannot change it.
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
from motorbridge import CallError
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.kinematics import compute_fk, get_arm_joint_count, pad_q_for_model
from reBotArm_control_py.dynamics import compute_generalized_gravity, load_dynamics_model, compute_coriolis_matrix, compute_mass_matrix
from reBotArm_control_py.dynamics.robot_model import create_data
from reBotArm_control_py.kinematics.robot_model import get_end_effector_frame_id
from reBotArm_control_py.kinematics import pad_q_for_model

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


def clear_faults(rebotarm: RebotArm) -> None:
    # 清除上次运行残留的故障（如看门狗触发后的通讯丢失），否则可能无法使能
    # Clear faults left by a previous run (e.g. comm loss after a watchdog trip),
    # which could otherwise stop the motors from enabling
    for jc in rebotarm._all_joints:
        rebotarm._motor_map[jc.name].clear_error()


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


class ReadTimer:
    """每秒打印一次位置读取耗时（中位数/最大值）与超时次数。

    Prints once a second how long the position reads took (median / max) and
    how many timed out. Read times that creep upward second after second mean
    the CAN bus is backing up; timeouts mean it is saturated.
    """

    def __init__(self, label: str, interval: float = 1.0) -> None:
        self._label = label
        self._interval = interval
        self._times: list = []
        self._timeouts = 0
        self._t_report = time.monotonic()

    def record(self, ms: float) -> None:
        self._times.append(ms)
        self._maybe_report()

    def timeout(self, err: Exception) -> None:
        self._timeouts += 1
        print(f"\n[{self._label}] {err}", flush=True)
        self._maybe_report()

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._t_report < self._interval:
            return
        if self._times:
            t = sorted(self._times)
            stats = f"median {t[len(t) // 2]:5.2f} ms  max {t[-1]:6.2f} ms"
        else:
            stats = "no successful reads"
        print(f"\n[{self._label}] n={len(self._times)}  {stats}  timeouts {self._timeouts}",
              flush=True)
        self._times.clear()
        self._timeouts = 0
        self._t_report = now

def make_test_controller(n_arm: int):
    # 随动模式：MIT 零刚度，机械臂可自由手动移动
    # Free-drive mode: MIT zero stiffness, arm can be freely moved by hand
    zeros_arm = np.zeros(n_arm)
    ones_arm = np.ones(n_arm)
    zeros_grip = np.zeros(1)
    kp_arm = ones_arm*1.0
    kd_arm = None #ones_arm*8.0
    kp_grip = np.ones(1)

    def controller(r: RebotArm, dt: float) -> None:
        r.arm.send_mit(pos=zeros_arm, vel=zeros_arm, kp=kp_arm, kd=kd_arm, tau=zeros_arm)
        r.gripper.send_mit(pos=zeros_grip, vel=zeros_grip, kp=kp_grip, kd=zeros_grip, tau=zeros_grip)
    return controller


pos_arm = None

def make_comp_controller(n_arm: int, model,data):
    zeros_arm = np.zeros(n_arm)
    ones_arm = np.ones(n_arm)
    pos_arm = zeros_arm

    zeros_grip = np.zeros(1)
    kp_arm = zeros_arm #None#ones_arm*2.0
    kd_arm = zeros_arm#None#ones_arm*0.1

    kp_grip = np.ones(1)*0.0

    RF = pin.LOCAL_WORLD_ALIGNED
    fid = get_end_effector_frame_id(model)
    def compute_qdd(model,q,v,data,xdd_des = np.zeros(6)):
        pin.computeJointJacobians(model, data, q)
        pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        J        = pin.getFrameJacobian(model, data, fid, RF)                       # 6 x nv
        Jdot_qd  = pin.getFrameClassicalAcceleration(model, data, fid, RF).vector   # 6

        qdd = np.linalg.pinv(J) @ (xdd_des - Jdot_qd)
        return qdd
    
    def controller(r: RebotArm, dt: float) -> None:
        pos = r.get_positions(position_source="output",request_feedback=True)
        pos,vel,torque = r.get_state(request_feedback=False,position_source="output")
        # pvel = np.zeros([8])
        # pvel[:6] = vel[:6]
        # cmat = compute_coriolis_matrix(model,pos,pvel,data)[:6,:6]
        # mmat = compute_mass_matrix(model,pos,data)
        
        # ctau = cmat @ vel[:6]*0.9
        gtau = compute_generalized_gravity(model,pos,data)[:6]
        # xtau = mmat @ compute_qdd(model,pad_q_for_model(model,pos,6),pvel,create_data(model))
        # xtau = np.clip(xtau[:6],-20,20)

        gtau = np.clip(gtau, -20,20)*0.89
        # ctau = np.clip(ctau,-5,5) # this will make the robot joints try to rotate at contant velocity when disturbed.

        # print(np.max(np.abs(xtau)))
        tau = gtau
        # tau *= 0
        r.arm.send_mit(pos=pos_arm, vel=zeros_arm, kp=kp_arm, kd=kd_arm, tau=tau)
        r.gripper.send_mit(pos=zeros_grip, vel=zeros_grip, kp=kp_grip, kd=zeros_grip, tau=zeros_grip)
        
    return controller




def make_free_drive_controller(n_arm: int):
    # 随动模式：MIT 零刚度，机械臂可自由手动移动
    # Free-drive mode: MIT zero stiffness, arm can be freely moved by hand
    zeros_arm = np.zeros(n_arm)
    ones_arm = np.ones(n_arm)
    zeros_grip = np.zeros(1)

    def controller(r: RebotArm, dt: float) -> None:
        r.arm.send_mit(pos=zeros_arm, vel=zeros_arm, kp=zeros_arm, kd=zeros_arm, tau=zeros_arm)
        r.gripper.send_mit(pos=zeros_grip, vel=zeros_grip, kp=zeros_arm, kd=zeros_grip, tau=zeros_grip)
    return controller


def main():
    parser = argparse.ArgumentParser(description="Free-drive the real arm and mirror it in MeshCat.")
    parser.add_argument("hardware_yaml", nargs="?", default=None, help="hardware config file")
    parser.add_argument("--zero", action="store_true",
                        help="set the current pose as zero on startup (overwrites existing zero)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    # 先加载可视化器，失败时不会留下已使能的电机
    # Load the visualizer first so a failure here never leaves motors enabled.
    print("加载可视化器 / Loading visualizer...")
    viz = Visualizer()
    n_model = get_arm_joint_count(viz.model)
    dynamics_model = load_dynamics_model()
    dynamics_data = create_data(dynamics_model)

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
        clear_faults(rebotarm)

        n_arm = rebotarm.arm.num_joints
        rebotarm.arm.mode_mit()
        rebotarm.gripper.mode_mit()
        rebotarm.enable_all()
        gcomp_controller = make_comp_controller(n_arm,dynamics_model,dynamics_data)
        test_controller = make_test_controller(n_arm)
        rebotarm.start_control_loop(gcomp_controller)

        print("--- 随动模式 / Free-drive mode ---")
        print("手动移动机械臂，MeshCat 实时同步。按 Ctrl+C 退出。")
        print("Move the arm by hand; MeshCat follows. Press Ctrl+C to exit.\n")

        status = StatusMonitor(rebotarm)
        read_timer = ReadTimer("xout")
        period = 1.0 / VIS_RATE_HZ
        while not should_exit:
            t0 = time.time()
            # 控制循环的每条 MIT 指令都会带回反馈帧，无需再显式请求（节省总线带宽）
            # Every MIT command in the control loop already returns a feedback
            # frame, so skip the explicit feedback requests (saves bus bandwidth)
            t_read = time.perf_counter()
            try:
                line = input("> ").split("#", 1)[0].strip()
                if not line:
                    raise
            except (EOFError, KeyboardInterrupt):
                break


            
            try:
                pos = rebotarm.get_positions(position_source="output", request_feedback=False)
            except CallError as e:
                # 超时说明总线饱和：记录后跳过本帧，保持运行
                # A timeout means the bus is saturated: log it, skip this frame, keep running
                read_timer.timeout(e)
                status.check()
                time.sleep(max(0.0, period - (time.time() - t0)))
                continue
            read_timer.record((time.perf_counter() - t_read) * 1e3)
            # print(rebotarm.get_torques())
            q = pad_q_for_model(viz.model,pos , n_model)
            status.check()
            viz.update(q)
            print(pos)
            pos, rot, _ = compute_fk(viz.model, q)

            # gtau = compute_generalized_gravity(dynamics_model,pos,dynamics_data)
            # print(np.sign(rebotarm.get_torques()[:6]*gtau[:6]))
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
