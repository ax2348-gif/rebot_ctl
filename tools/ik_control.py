#!/usr/bin/env python3
"""
RebotArmEndPose 交互控制示例（IK 模式）。
输入: x y z [roll pitch yaw] 目标末端位置（米 / 弧度）
      g <pos>               设置夹爪目标位置

RebotArmEndPose interactive control example (IK mode).
Input: x y z [roll pitch yaw]  target end-effector pose (meters / radians)
       g <pos>                set gripper target position

用法 / Usage:
    python example/7_arm_ik_control.py

退出 / Exit: q / quit / exit /ctrl+c
状态 / State: state, end_state
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import joint_to_pose
import argparse
def clear_faults(rebotarm: RebotArm) -> None:
    # 清除上次运行残留的故障（如看门狗触发后的通讯丢失），否则可能无法使能
    # Clear faults left by a previous run (e.g. comm loss after a watchdog trip),
    # which could otherwise stop the motors from enabling
    for jc in rebotarm._all_joints:
        rebotarm._motor_map[jc.name].clear_error()

def main() -> None:
    parser = argparse.ArgumentParser(description="Free-drive the real arm and mirror it in MeshCat.")
    parser.add_argument("hardware_yaml", nargs="?", default=None, help="hardware config file")
    args = parser.parse_args()
    rebotarm = RebotArm(args.hardware_yaml)
    ctrl = RebotArmEndPose(rebotarm)

    try:
        ctrl.start()
        clear_faults(rebotarm)
        interactive(ctrl, rebotarm)
    finally:
        ctrl.end()
    print("\n完成 / Done.")


def interactive(ctrl, rebotarm):
    print("--- 已启动末端位置控制器 ---\n")
    print("--- End-effector pose controller started ---\n")

    while True:
        try:
            line = input("> ").split("#", 1)[0].strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break

        if line.lower() == "state":
            q, _, _ = rebotarm.get_state()
            print(f"  机械臂 / Arm (rad): {[f'{v:+.3f}' for v in q[:rebotarm.arm.num_joints]]}")
            if rebotarm.has_gripper:
                print(f"  夹爪 / Gripper (rad): {q[rebotarm.arm.num_joints]:+.3f}")
            continue

        if line.lower() == "end_state":
            q, _, _ = rebotarm.get_state()
            pos, rpy = joint_to_pose(q[:rebotarm.arm.num_joints])
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            rx, ry, rz = float(rpy[0]), float(rpy[1]), float(rpy[2])
            print(f"  pos=[{px:+.3f} {py:+.3f} {pz:+.3f}] m  rpy=[{rx:+.2f} {ry:+.2f} {rz:+.2f}] rad")
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "g" and len(parts) >= 2:
            try:
                pos = float(parts[1])
                ctrl.set_gripper_target(pos)
                print(f"  夹爪 / Gripper -> {pos:.3f} rad")
            except ValueError:
                print("  用法 / Usage: g <pos>")
            continue
        if cmd == "e":
            rebotarm.enable_all()
            continue
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            print("  格式 / Format: x y z [roll pitch yaw]")
            continue

        if len(vals) not in (3, 6) or not np.all(np.isfinite(vals)):
            print("  需要有效的位置/姿态输入 / Invalid pose input")
            continue
        x, y, z = vals[0], vals[1], vals[2]
        if len(vals) >= 6:
            roll, pitch, yaw = vals[3:6]
        else:
            # q, _, _ = rebotarm.get_state()
            # _, rpy = joint_to_pose(q[:rebotarm.arm.num_joints])
            # roll, pitch, yaw = map(float, rpy)
            roll,pitch,yaw = 0, 0, 0 

        # ok = ctrl.move_to_ik(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        ok = ctrl.move_to_traj(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            duration=1.0,
        )
        print(f"  -> ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
              f"rpy=[{roll:+.2f} {pitch:+.2f} {yaw:+.2f}]  "
              f"{'ok' if ok else 'failed'}")



if __name__ == "__main__":
    main()
