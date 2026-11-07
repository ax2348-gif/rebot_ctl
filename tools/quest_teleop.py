#!/usr/bin/env python3
"""Quest 遥操作实机 — 握持离合时使能并跟踪，松开立即失能；MeshCat 显示实机状态。

离合协议、TCP 偏移与坐标系可视化与 example/sim/quest_teleop_sim.py 相同，
区别在于锚点取自实机测得的关节角，指令经 MIT（重力前馈）发给电机:
    - 新的握持: 使能电机，以实测 TCP 作为锚点，从实测关节角开始跟踪
    - 握持期间: IK 求解目标，关节指令每步限幅后下发
    - 松开 / 断流 / 失焦 / 读数失败: 立即 disable_all()（机械臂失去力矩，会在重力下落下）

夹爪 / Gripper — independent of the clutch, same Y/X enable as the base (rev_ctl
examples/quest_teleop.py), so with quest_teleop.sh one press serves both:
    - Y (left controller): enable; the gripper follows the --hand trigger (0 = open).
      Every Y press that starts following clears the gripper's faults and re-enables it,
      which also recovers a tripped CAN watchdog.
    - X: disable; the gripper holds its last command (a grasp is kept).
    - 断流 / 失焦 / 手柄无效 / 离合松开: also stop following but keep holding.
    - Only Ctrl-C (disconnect) releases it.
    - At startup it is closed and must read within 20° of zero (RebotArm.check_gripper_closed;
      up to --connect-tries attempts, default 2); the zero is never changed.

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

from reBotArm_control_py.actuator import GripperCheckError, RebotArm
from reBotArm_control_py.dynamics import compute_generalized_gravity
from reBotArm_control_py.kinematics import compute_fk, get_arm_joint_count, pad_q_for_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik
from example.sim.quest_teleop_sim import TCP_OFFSET, Clutch, QuestStream, add_tcp_frame, show_frame
from example.sim.visualizer import Visualizer

RATE_HZ = 100.0                   # 必须远快于电机 100 ms CAN 看门狗
# 每步关节指令变化上限 (rad)，逐关节：joint1-3（臂）6 rad/s，joint4-6（腕）18 rad/s
# Per-joint command step limit (rad per step): arm joints 1-3 at 6 rad/s, wrist joints 4-6 at 18 rad/s
MAX_JOINT_STEP = np.array([6.0, 6.0, 6.0, 18.0, 18.0, 18.0]) / RATE_HZ
IK_ROT_WEIGHT = 0.02              # IK 姿态权重：腕部 roll 到限位时保持末端位置，不用其它关节凑姿态
                                  # (IKParams.rot_weight; same as example/sim/quest_teleop_sim.py)
GRIPPER_OPEN_RAD = -3.1          # 夹爪张开位置 (rad，0 = 闭合)，需按实物标定；None = 不控制夹爪


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hardware_yaml", nargs="?", default=None, help="hardware config file")
    parser.add_argument("--host", default="100.72.77.83")
    parser.add_argument("--port", type=int, default=9391)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--connect-tries", type=int, default=2,
                        help="gripper closing check attempts at startup (default 2): "
                             "each attempt closes the gripper again, so one that started far open "
                             "usually passes the second time")
    args = parser.parse_args()

    # 先加载可视化器，失败时不会留下已使能的电机
    viz = Visualizer()
    model = viz.model
    frame_id = add_tcp_frame(model, TCP_OFFSET)
    data = model.createData()
    n = get_arm_joint_count(model, frame_id)
    assert len(MAX_JOINT_STEP) == n, f"MAX_JOINT_STEP has {len(MAX_JOINT_STEP)} entries for {n} arm joints"
    ik_params = IKParams(max_iter=30, rot_weight=IK_ROT_WEIGHT)
    viz.meshcat["/Cameras/default/rotated/<object>"].set_property("position", [0.6, 0.5, 0.45])
    show_frame(viz, "frames/base", np.zeros(3), np.eye(3), scale=0.15)

    stream = QuestStream(args.host, args.port)
    clutch = Clutch()
    rebotarm = RebotArm(args.hardware_yaml)
    groups = [rebotarm.arm] + ([rebotarm.gripper] if GRIPPER_OPEN_RAD is not None else [])

    try:
        # 夹爪闭合检查最多 --connect-tries 次（不断开重连：串口在 disconnect 后短时间内仍被占用）
        # The gripper closing check, up to --connect-tries times. It is repeated while connected
        # (connect() would disconnect on failure, and the serial port stays busy for a moment
        # after a disconnect, so reconnecting at once fails).
        rebotarm.connect(check_gripper=False)
        for attempt in range(1, max(1, args.connect_tries) + 1):
            try:
                rebotarm.check_gripper_closed()
                break
            except GripperCheckError as e:
                if attempt >= args.connect_tries:
                    raise
                print(f"{e}\n夹爪检查失败，重试 / Gripper check failed; retrying "
                      f"({attempt + 1}/{args.connect_tries})...")
        for jc in rebotarm._all_joints:   # 清除上次残留故障（如看门狗触发）
            rebotarm._motor_map[jc.name].clear_error()
        for g in groups:
            g.mode_mit()
        rebotarm.disable_all()
        enabled = False
        q_cmd = None              # 首次握持前不向机械臂发送任何指令 / nothing is sent to the arm before the first clutch
        gripper_enabled = False   # Y 使能 / X 停用，与底盘相同；与离合无关
        gripper_cmd = None        # 最近一次夹爪指令；首次使能后每帧重发（看门狗 100 ms），保持夹持
        print(f"连接 Quest {args.host}:{args.port}，{args.hand} 手握持键离合；Y/X 使能/停用夹爪。Ctrl-C 退出。")

        while True:
            t0 = time.monotonic()
            quest = stream.fresh()   # 夹爪只看 Quest 输入，不受电机读数失败影响
            try:
                q_meas = pad_q_for_model(model, rebotarm.arm.get_positions(position_source="output"), n)
                tcp = compute_fk(model, q_meas, "tcp")[:2]
                sample = quest
            except CallError:   # 读数失败：本帧视为无效输入
                q_meas, tcp, sample = None, None, None
            target = clutch.update(sample, args.hand, tcp)
            if q_meas is None:
                clutch.state = "电机读数失败"   # 与 Quest 断流区分：检查电机供电 / CAN 连线

            if target is None:
                if q_meas is not None and q_cmd is not None:   # 读数失败时跳过一帧（重力项需要实测姿态），不让脚本崩溃松开夹爪
                    rebotarm.arm.send_mit(q_cmd, tau=compute_generalized_gravity(model, q_meas, data)[:n])

                if enabled:
                    # rebotarm.disable_all()
                    enabled = False
            else:
                if not enabled:
                    q_cmd = q_meas[:n].copy()   # 从实测姿态开始，避免跳变
                    rebotarm.arm.clear_error()  # 例如看门狗触发后 / e.g. after a tripped CAN watchdog
                    rebotarm.arm.enable()       # 只使能机械臂；夹爪由 Y 单独使能 / arm motors only
                    enabled = True
                result = solve_ik(model, data, frame_id, pin.SE3(target[1], target[0]),
                                  q_cmd, ik_params, controlled_joints=n)
                q_cmd += np.clip(result.q - q_cmd, -MAX_JOINT_STEP, MAX_JOINT_STEP)
                if np.linalg.norm(result.q-q_cmd)>0.0001:
                    print("clipped")
                rebotarm.arm.send_mit(q_cmd, tau=compute_generalized_gravity(model, q_meas, data)[:n])

            # 夹爪：Y 使能跟随扳机，X 停用；无效输入只停止跟随，仍保持最近指令
            if GRIPPER_OPEN_RAD is not None:
                hands = quest and quest.get("hands")
                if not (hands and quest["input_focused"] and hands["left"]["active"]
                        and hands[args.hand]["active"]):
                    gripper_enabled = False
                elif hands["left"]["buttons"]["primary"]:      # X
                    gripper_enabled = False
                elif hands["left"]["buttons"]["secondary"]:    # Y
                    if not gripper_enabled:   # 每次开始跟随都清故障并重新使能 / every press that starts following
                        rebotarm.gripper.clear_error()
                        rebotarm.gripper.enable()
                    gripper_enabled = True
                if gripper_enabled:
                    trigger = min(1.0, max(0.0, hands[args.hand]["trigger"]))
                    gripper_cmd = GRIPPER_OPEN_RAD * (1 - trigger)
                if gripper_cmd is not None:
                    rebotarm.gripper.send_mit(np.array([gripper_cmd]))

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
            gripper_state = "跟随" if gripper_enabled else ("保持" if gripper_cmd is not None else "未用")
            print(f"\r[{clutch.state:4}] {'使能' if enabled else '失能'} 位置误差={err:5.1f}mm 夹爪={gripper_state}   ",
                  end="", flush=True)
            time.sleep(max(0.0, 1 / RATE_HZ - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\n失能并断开 / Disabling and disconnecting.")
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
