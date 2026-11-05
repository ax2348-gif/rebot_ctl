#!/usr/bin/env python3
"""Quest 遥操作实机 — 握持离合时使能并跟踪，松开立即失能；MeshCat 显示实机状态。

离合协议、TCP 偏移与坐标系可视化与 example/sim/quest_teleop_sim.py 相同，
区别在于锚点取自实机测得的关节角，指令经 MIT（重力前馈）发给电机:
    - 新的握持: 使能电机，以实测 TCP 作为锚点，从实测关节角开始跟踪
    - 握持期间: IK 求解目标，关节指令每步限幅后下发
    - 松开 / 断流 / 失焦 / 读数失败: 立即 disable_all()（机械臂失去力矩，会在重力下落下）

可视化 (红 X / 绿 Y / 蓝 Z): 实机姿态；base、end_link（小）、ee = 实测 TCP；
握持时另显示 anchor（锚点）与 target（离合目标）。

用法 / Usage:
    conda run -n lerobot python tools/quest_teleop.py [hardware_yaml] [--host 100.72.77.83 --port 9391 --hand right]

退出 / Exit: Ctrl+C（失能并断开）

The Quest app serves the stream on every interface, port 9391 (quest_env 447e80f and
later); connect to the headset's address directly, no relay needed. Check with
nc 100.72.77.83 9391
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from motorbridge import CallError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import compute_generalized_gravity
from reBotArm_control_py.kinematics import compute_fk, get_arm_joint_count, pad_q_for_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik
from example.sim.quest_teleop_sim import TCP_OFFSET, Clutch, QuestStream, add_tcp_frame, show_frame
from example.sim.visualizer import Visualizer

RATE_HZ = 100.0                   # 必须远快于电机 100 ms CAN 看门狗
MAX_JOINT_STEP = 6.0 / RATE_HZ    # 每步关节指令变化上限 (rad)
GRIPPER_OPEN_RAD = -3.1          # 夹爪张开位置 (rad，0 = 闭合)，需按实物标定；None = 不控制夹爪


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hardware_yaml", nargs="?", default=None, help="hardware config file")
    parser.add_argument("--host", default="100.72.77.83")
    parser.add_argument("--port", type=int, default=9391)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    args = parser.parse_args()

    # 先加载可视化器，失败时不会留下已使能的电机
    viz = Visualizer()
    model = viz.model
    frame_id = add_tcp_frame(model, TCP_OFFSET)
    data = model.createData()
    n = get_arm_joint_count(model, frame_id)
    ik_params = IKParams(max_iter=30)
    viz.meshcat["/Cameras/default/rotated/<object>"].set_property("position", [0.6, 0.5, 0.45])
    show_frame(viz, "frames/base", np.zeros(3), np.eye(3), scale=0.15)

    stream = QuestStream(args.host, args.port)
    clutch = Clutch()
    rebotarm = RebotArm(args.hardware_yaml)
    groups = [rebotarm.arm] + ([rebotarm.gripper] if GRIPPER_OPEN_RAD is not None else [])

    try:
        rebotarm.connect()
        for jc in rebotarm._all_joints:   # 清除上次残留故障（如看门狗触发）
            rebotarm._motor_map[jc.name].clear_error()
        for g in groups:
            g.mode_mit()
        rebotarm.disable_all()
        enabled = False
        q_cmd = np.zeros(n)
        print(f"连接 Quest {args.host}:{args.port}，{args.hand} 手握持键离合（松开即失能）。Ctrl-C 退出。")

        while True:
            t0 = time.monotonic()
            try:
                q_meas = pad_q_for_model(model, rebotarm.arm.get_positions(position_source="output"), n)
                tcp = compute_fk(model, q_meas, "tcp")[:2]
                sample = stream.fresh()
            except CallError:   # 读数失败：本帧视为无效输入
                q_meas, tcp, sample = None, None, None
            target = clutch.update(sample, args.hand, tcp)
            if q_meas is None:
                clutch.state = "电机读数失败"   # 与 Quest 断流区分：检查电机供电 / CAN 连线

            if target is None:
                rebotarm.arm.send_mit(q_cmd, tau=compute_generalized_gravity(model, q_meas, data)[:n])
                rebotarm.gripper.send_mit(np.array([GRIPPER_OPEN_RAD]))

                if enabled:
                    # rebotarm.disable_all()
                    enabled = False
            else:
                if not enabled:
                    for g in groups:
                        g.enable()
                    q_cmd = q_meas[:n].copy()   # 从实测姿态开始，避免跳变
                    enabled = True
                result = solve_ik(model, data, frame_id, pin.SE3(target[1], target[0]),
                                  q_cmd, ik_params, controlled_joints=n)
                q_cmd += np.clip(result.q - q_cmd, -MAX_JOINT_STEP, MAX_JOINT_STEP)
                if np.linalg.norm(result.q-q_cmd)>0.0001:
                    print("clipped")
                rebotarm.arm.send_mit(q_cmd, tau=compute_generalized_gravity(model, q_meas, data)[:n])
                if GRIPPER_OPEN_RAD is not None:
                    trigger = sample["control_intent"]["hands"][args.hand]["gripper"]
                    rebotarm.gripper.send_mit(np.array([GRIPPER_OPEN_RAD * (1 - trigger)]))

            # 可视化：实机姿态 + 坐标系
            if q_meas is not None:
                viz.update(q_meas)
                show_frame(viz, "frames/ee", *tcp)
                show_frame(viz, "frames/end_link", *compute_fk(model, q_meas)[:2], scale=0.03)
            viz.meshcat["frames/anchor"].set_property("visible", enabled)
            viz.meshcat["frames/target"].set_property("visible", enabled)
            if enabled:
                show_frame(viz, "frames/anchor", *clutch.anchor, scale=0.05)
                show_frame(viz, "frames/target", *target)

            err = np.linalg.norm(target[0] - tcp[0]) * 1000 if enabled else 0.0
            print(f"\r[{clutch.state:4}] {'使能' if enabled else '失能'} 位置误差={err:5.1f}mm   ",
                  end="", flush=True)
            time.sleep(max(0.0, 1 / RATE_HZ - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\n失能并断开 / Disabling and disconnecting.")
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
