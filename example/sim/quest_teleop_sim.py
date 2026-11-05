#!/usr/bin/env python3
"""Quest 遥操作仿真 — 读取 Quest 的 control_intent，离合 (clutch) 驱动 MeshCat 机械臂。

用法:
    conda run -n lerobot python example/sim/quest_teleop_sim.py [--host 100.72.77.83 --port 9392 --hand right]

Quest 端 (quest_env) 已完成离合判定，每帧发送 teleop.operator_intent.v1:
    engaged / needs_release / engagement_id、delta_position、delta_rotation (operator_base_flu)。
本端按 quest_env/DEMO.md 的远端消费者约定实现:
    - 每次新的握持 (engagement)：记录当前测得的末端位姿 (e0, E0) 作为锚点
    - 握持期间：目标 = (e0 + dp, dR · E0)，机械臂基座系 = X 前 Y 左 Z 上
    - 松开 / 超时 (200ms) / 失焦 / session 或 A 键重新对齐：保持关节不动
    - 中途连上（未看到握持开始）：必须先松开再握持

控制 (Quest):
    握持键 (grip)    按住 = 离合；松开 = 保持
    扳机 (trigger)   夹爪闭合
    A (双手松开时)   以当前头部朝向重新定义“前方”

可视化坐标系 (红 X / 绿 Y / 蓝 Z):
    base    机械臂基座 (= 操作者 FLU 坐标轴)
    end_link  URDF 末端（小）
    ee      当前控制点 TCP = end_link · TCP_OFFSET（遥操作、IK、离合都作用于它）
    anchor  握持开始时的末端锚点（仅握持时）
    target  离合目标 + 小球（绿=握持中，灰=保持）；黄线 = 锚点→目标的平移量
"""

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import meshcat.geometry as mcg
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik
from reBotArm_control_py.kinematics.robot_model import (
    get_end_effector_frame_id, get_arm_joint_count,
)
from reBotArm_control_py.kinematics.forward_kinematics import compute_fk
from example.sim.visualizer import Visualizer

STALE_S = 0.2           # 超过该时间未收到新帧视为断流
RATE_HZ = 60.0
MAX_JOINT_STEP = 2.5 / RATE_HZ   # 每帧关节步长上限 (rad)，避免奇异处跳变
GRIPPER_OPEN = 0.0285   # finger_left 上限；finger_right 取负

# 控制点 (TCP) 相对 URDF end_link 的固定变换，在 end_link 坐标系中表示。
# end_link 位于指尖附近，X 轴指向夹爪外侧；沿 -X 后移即把控制点移向手腕。
# 也可加旋转，例如 pin.SE3(pin.rpy.rpyToMatrix(0, 0, np.pi / 2), ...)
TCP_OFFSET = pin.SE3(np.eye(3), np.array([-0.15, 0.0, 0.0]))


def add_tcp_frame(model, offset):
    """在 end_link 上挂一个固定的 "tcp" 帧，返回其 frame id。"""
    ee_id = get_end_effector_frame_id(model)
    ee = model.frames[ee_id]
    return model.addFrame(pin.Frame("tcp", ee.parentJoint, ee_id, ee.placement * offset, pin.FrameType.OP_FRAME))


class QuestStream:
    """后台线程读取 JSON 行，只保留最新一帧及其到达时间（本机时钟）。"""

    def __init__(self, host, port):
        self.address = (host, port)
        self.latest, self.arrived = None, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                with socket.create_connection(self.address, timeout=2) as sock:
                    sock.settimeout(1.0)   # 连接静默超过 1 s 视为断开并重连，避免永久阻塞
                    for line in sock.makefile("rb"):
                        self.latest, self.arrived = json.loads(line), time.monotonic()
            except (OSError, ValueError):
                time.sleep(1.0)  # 断开后重连

    def fresh(self):
        return self.latest if time.monotonic() - self.arrived < STALE_S else None


class Clutch:
    """远端离合消费者：把 Quest 的相对位姿增量施加到本端测得的末端锚点上。"""

    def __init__(self):
        self.anchor = None        # (e0, E0)，握持期间有效
        self.engagement = None    # 当前锚点对应的 engagement_id
        self.armed = False        # 已观察到松开，下一次握持可以捕获锚点
        self.key = None           # (session_id, operator_frame_revision)
        self.state = "断流"

    def update(self, sample, side, measured_ee):
        """返回目标 (p, R)，不应运动时返回 None。"""
        intent = sample and sample.get("control_intent")
        hand = intent and intent["hands"][side]
        if not (hand and sample["input_focused"] and hand["valid"]):
            self._reset()
            self.state = "无效" if hand else "断流"
            return None

        key = (intent["session_id"], intent["operator_frame_revision"])
        if key != self.key:   # 新 session 或 A 键重新对齐：旧锚点作废，需重新松开
            self._reset()
            self.key = key

        if not hand["engaged"]:
            self.anchor, self.armed, self.state = None, True, "保持"
            return None
        if hand["engagement_id"] != self.engagement:
            if not self.armed:  # 中途接入：不对新锚点施加已有的非零增量
                self.state = "需先松开"
                return None
            self.anchor, self.engagement, self.armed = measured_ee, hand["engagement_id"], False

        self.state = "握持"
        e0, E0 = self.anchor
        dp = np.array(hand["delta_position"])
        dR = np.array(hand["delta_rotation"]).reshape(3, 3)
        return e0 + dp, dR @ E0

    def _reset(self):
        self.anchor, self.engagement, self.armed = None, None, False


def show_frame(viz, name, p, R, scale=0.08):
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = R, p
    viz.meshcat[name].set_object(mcg.triad(scale))
    viz.meshcat[name].set_transform(H)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="100.72.77.83")
    parser.add_argument("--port", type=int, default=9392)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--tcp-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="覆盖 TCP_OFFSET 的平移 (米，end_link 坐标系)")
    args = parser.parse_args()

    viz = Visualizer()
    model = viz.model
    if args.tcp_offset:
        TCP_OFFSET.translation = np.array(args.tcp_offset)
    frame_id = add_tcp_frame(model, TCP_OFFSET)
    data = model.createData()   # 必须在 addFrame 之后创建
    arm_nq = get_arm_joint_count(model, frame_id)
    fingers = [model.joints[model.getJointId(n)].idx_q for n in ("finger_left", "finger_right")]
    ik_params = IKParams(max_iter=30)

    q = pin.neutral(model)
    gripper = 0.0
    stream = QuestStream(args.host, args.port)
    clutch = Clutch()

    viz.meshcat["/Cameras/default/rotated/<object>"].set_property("position", [0.6, 0.5, 0.45])
    show_frame(viz, "frames/base", np.zeros(3), np.eye(3), scale=0.15)
    viz.meshcat["frames/target/ball"].set_object(mcg.Sphere(0.012), mcg.MeshLambertMaterial(color=0x888888))
    print(f"连接 Quest {args.host}:{args.port}，{args.hand} 手握持键离合。Ctrl-C 退出。")

    try:
        while True:
            t0 = time.monotonic()
            sample = stream.fresh()
            ee_p, ee_R, _ = compute_fk(model, q, "tcp")
            target = clutch.update(sample, args.hand, (ee_p, ee_R))

            if target is not None:
                result = solve_ik(model, data, frame_id, pin.SE3(target[1], target[0]),
                                  q[:arm_nq], ik_params, controlled_joints=arm_nq)
                step = np.clip(result.q - q[:arm_nq], -MAX_JOINT_STEP, MAX_JOINT_STEP)
                q[:arm_nq] += step  # 不可达时尽力逼近，松开后重新握持即可
            hand = sample and sample["control_intent"]["hands"][args.hand]
            if hand and hand["valid"]:  # 夹爪：仅在指令有效时更新，否则保持
                gripper = hand["gripper"]
            q[fingers[0]], q[fingers[1]] = GRIPPER_OPEN * (1 - gripper), -GRIPPER_OPEN * (1 - gripper)
            viz.update(q)

            # 坐标系可视化
            ee_p, ee_R, _ = compute_fk(model, q, "tcp")
            show_frame(viz, "frames/ee", ee_p, ee_R)
            show_frame(viz, "frames/end_link", *compute_fk(model, q)[:2], scale=0.03)
            engaged = target is not None
            viz.meshcat["frames/anchor"].set_property("visible", engaged)
            viz.meshcat["frames/delta"].set_property("visible", engaged)
            viz.meshcat["frames/target/ball"].set_property("color", [0, 0.8, 0.2, 1] if engaged else [0.5, 0.5, 0.5, 1])
            if engaged:
                (e0, E0), (tp, tR) = clutch.anchor, target
                show_frame(viz, "frames/anchor", e0, E0, scale=0.05)
                show_frame(viz, "frames/target", tp, tR)
                viz.meshcat["frames/delta"].set_object(mcg.Line(
                    mcg.PointsGeometry(np.array([e0, tp], dtype=np.float32).T),
                    mcg.LineBasicMaterial(color=0xFFCC00)))
                err = np.linalg.norm(tp - ee_p)
            else:
                show_frame(viz, "frames/target", ee_p, ee_R)
                err = 0.0

            print(f"\r[{clutch.state:4}] 位置误差={err * 1000:5.1f}mm 夹爪={gripper:.2f} "
                  f"关节(deg)={np.array2string(np.degrees(q[:arm_nq]), precision=0, suppress_small=True)}   ",
                  end="", flush=True)
            time.sleep(max(0.0, 1 / RATE_HZ - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
