"""reBotArm 分组控制系统 — JointGroup 架构。

配置驱动的硬件抽象层：
  - 所有参数均在 config/rebotarm.yaml 中定义（hardware_yaml 指定硬件配置文件）
  - 关节按 groups 分组，每组独立控制模式
  - 统一 loop 中按组顺序同步发送，防止总线争用

使用示例::

    # arm 组 POS_VEL，gripper 组 MIT（解耦混合控制）
    arm = RebotArm()
    arm.connect()
    arm.arm.enable()
    arm.gripper.enable()
    arm.arm.mode_pos_vel()
    arm.gripper.mode_mit()

    def loop(ref, dt):
        ref.arm.send_pos_vel(joint_pos)
        ref.gripper.send_mit(gripper_pos)

    arm.start_control_loop(loop)

    # 全部组 MIT（纯测试）
    arm.arm.mode_mit()
    arm.gripper.mode_mit()

    arm.disconnect()
"""
from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml

from motorbridge import Controller, Mode, CallError

_CFG_DIR = Path(__file__).parent.parent.parent / "config"
_GLOBAL_CFG = _CFG_DIR / "rebotarm.yaml"


def _resolve_hw_cfg_path(hw_yaml: str | None = None) -> Path:
    if hw_yaml is None:
        if not _GLOBAL_CFG.exists():
            raise FileNotFoundError(f"{_GLOBAL_CFG} not found")
        data = yaml.safe_load(_GLOBAL_CFG.read_text())
        hw_yaml = data.get("hardware_yaml") if data else None
        if not hw_yaml:
            raise ValueError("hardware_yaml not set in rebotarm.yaml")

    p = Path(hw_yaml)
    if p.is_absolute():
        return p
    path = _CFG_DIR / hw_yaml
    if path.exists():
        return path
    raise FileNotFoundError(f"hardware config not found: {path}")


# --------------------------------------------------------------------------
# 配置加载
# --------------------------------------------------------------------------

@dataclass
class JointCfg:
    name: str
    motor_id: int
    feedback_id: int
    model: str
    vendor: str = "damiao"
    kp: float = 0.0
    kd: float = 0.0
    vel_kp: float = 0.0
    vel_ki: float = 0.0
    pos_kp: float = 0.0
    pos_ki: float = 0.0
    vlim: float = 0.0


def load_cfg(hw_yaml: str | None = None) -> dict:
    hw_path = _resolve_hw_cfg_path(hw_yaml)

    with open(hw_path, "r") as f:
        data = yaml.safe_load(f)

    joints = []
    for j in data.get("joints", []):
        mc = j.get("MIT", {})
        pc = j.get("POS_VEL", {})
        joints.append(JointCfg(
            name=j["name"],
            motor_id=int(j["motor_id"]),
            feedback_id=int(j["feedback_id"]),
            model=str(j.get("model", "4340P")),
            vendor=str(j.get("vendor", "damiao")).lower(),
            kp=float(mc.get("kp", 0.0)),
            kd=float(mc.get("kd", 0.0)),
            vel_kp=float(pc.get("vel_kp", 0.0)),
            vel_ki=float(pc.get("vel_ki", 0.0)),
            pos_kp=float(pc.get("pos_kp", 0.0)),
            pos_ki=float(pc.get("pos_ki", 0.0)),
            vlim=float(pc.get("vlim", 2.0)),
        ))

    return {
        "name": data.get("name", "reBotArm"),
        "channel": data.get("channel", "/dev/ttyACM0"),
        "rate": float(data.get("rate", 500.0)),
        "groups": data.get("groups", {}),
        "joints": joints,
        "arm_control_mode": str(data.get("arm_control_mode", "posvel")),
        "urdf_path": data.get("urdf_path"),
    }


def load_joint_limits(urdf_path: str | None) -> Dict[str, tuple[float, float]]:
    """从 URDF 读取 revolute/prismatic 关节的 [lower, upper] 限位。

    Read [lower, upper] limits of revolute/prismatic joints from the URDF.
    Relative paths resolve against the project root, like the kinematics loader.
    """
    if not urdf_path:
        return {}
    path = Path(urdf_path)
    if not path.is_absolute():
        path = _CFG_DIR.parent / path
    if not path.exists():
        raise FileNotFoundError(f"URDF not found (needed for joint-limit check): {path}")

    limits: Dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        lim = joint.find("limit")
        if lim is None or lim.get("lower") is None or lim.get("upper") is None:
            continue
        limits[joint.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return limits


# 电机 CAN 通讯看门狗：使能状态下超过该时间未收到 CAN 帧，电机自行失能（进入阻尼）。
# 该值由 tools/persist_can_timeout.py 写入电机 flash；connect() 会校验，不一致则拒绝连接。
# Motor CAN watchdog: an enabled motor that receives no CAN frame for this long
# disables itself (damping). Stored in motor flash by tools/persist_can_timeout.py;
# connect() verifies it and refuses to continue on a mismatch.
CAN_TIMEOUT_MS = 100
DM_RID_TIMEOUT = 9  # Damiao register: TIMEOUT, units of 50 µs


def dm_timeout_register_value(timeout_ms: int) -> int:
    """Damiao TIMEOUT 寄存器值（50 µs 为单位）/ Damiao TIMEOUT register value (50 µs units)."""
    return timeout_ms * 20


class CanTimeoutError(RuntimeError):
    """电机的 CAN 看门狗未设置为 CAN_TIMEOUT_MS（或无法校验）。

    A motor's CAN watchdog is not set to CAN_TIMEOUT_MS, or cannot be verified.
    """


class JointLimitError(RuntimeError):
    """上电读数超出关节限位（常见原因：单圈编码器上电时多算/少算一圈 2π）。

    A joint reads outside its URDF limits at connect time. The usual cause on
    Damiao motors is a 2π power-on wrap of the single-turn output encoder.
    """


def load_gravity_compensation_config(
    profile: str,
    hw_yaml: str | None = None,
) -> dict:
    """Load one gravity-compensation profile from the active hardware YAML.

    Profile values override the legacy/default keys directly under
    ``gravity_compensation``. Hardware YAML files without ``profiles`` remain
    compatible and return their legacy/default gravity-compensation mapping.
    """
    hw_path = _resolve_hw_cfg_path(hw_yaml)
    with open(hw_path, "r") as f:
        data = yaml.safe_load(f) or {}

    gravity_cfg = data.get("gravity_compensation", {}) or {}
    if not isinstance(gravity_cfg, dict):
        raise TypeError("gravity_compensation must be a mapping")

    profiles = gravity_cfg.get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise TypeError("gravity_compensation.profiles must be a mapping")

    if not profiles:
        return {
            key: value
            for key, value in gravity_cfg.items()
            if key != "profiles"
        }
    if profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise KeyError(
            f"Unknown gravity-compensation profile {profile!r}; "
            f"available: {available}"
        )

    profile_cfg = profiles[profile]
    if not isinstance(profile_cfg, dict):
        raise TypeError(
            f"gravity_compensation.profiles.{profile} must be a mapping"
        )

    merged = {
        key: value
        for key, value in gravity_cfg.items()
        if key != "profiles"
    }
    merged.update(profile_cfg)
    return merged


# --------------------------------------------------------------------------
# 输出轴位置（xout）读取 — RebotArm 与 JointGroup 共用
# Output-shaft position (xout) reads, shared by RebotArm and JointGroup
# --------------------------------------------------------------------------

# Damiao 只读寄存器：输出轴位置 / Damiao read-only register: output shaft position
DM_RID_XOUT = 81
POSITION_SOURCES = ("feedback", "output")
# 等待全部 xout 回复的上限 / Upper bound on waiting for all xout replies
XOUT_TIMEOUT_S = 0.05


def _check_position_source(position_source: str) -> None:
    if position_source not in POSITION_SOURCES:
        raise ValueError(
            f"position_source must be one of {POSITION_SOURCES}, got {position_source!r}"
        )


def _request_xout(motor_map: Dict[str, any], names: List[str]) -> bool:
    """向各电机发出 xout 读请求（不等待）；旧版 motorbridge 不支持时返回 False。

    Send the xout read request to each motor without waiting. Returns False
    (and sends nothing) on a motorbridge without the non-blocking API.
    """
    if not all(hasattr(motor_map[n], "request_register") for n in names):
        return False
    for n in names:
        motor_map[n].request_register(DM_RID_XOUT)
    return True


def _collect_xout(
    motor_map: Dict[str, any],
    ctrls: List[Controller],
    names: List[str],
    batched: bool,
    timeout_s: float,
) -> Dict[str, float]:
    """等待 _request_xout 已发请求的新鲜回复；未分批时逐个阻塞读取。

    Await fresh replies to the requests _request_xout sent; a reply to an
    older request never counts. A motor that hasn't answered within timeout_s
    raises CallError naming it. Without batching (old motorbridge), fall back
    to one blocking read per motor (~2.5 ms each).
    """
    if not batched:
        return {n: float(motor_map[n].get_register_f32(DM_RID_XOUT, 500)) for n in names}
    out: Dict[str, float] = {}
    deadline = time.monotonic() + timeout_s
    while True:
        for name in names:
            if name not in out:
                # fresh_only (default): a reply to an older request never counts
                cached = motor_map[name].get_cached_register_f32(DM_RID_XOUT)
                if cached is not None:
                    out[name] = float(cached[0])
        if len(out) == len(names):
            return out
        if time.monotonic() > deadline:
            missing = ", ".join(n for n in names if n not in out)
            raise CallError(
                f"no xout (register {DM_RID_XOUT}) reply within "
                f"{timeout_s * 1e3:.0f} ms from: {missing}"
            )
        for ctrl in ctrls:
            ctrl.poll_feedback_once()


# --------------------------------------------------------------------------
# NoOpGroup — 无执行器时的空操作桩
# --------------------------------------------------------------------------

class NoOpGroup:
    """当配置中不存在 gripper 组时的空实现。

    所有属性和方法与 JointGroup 接口兼容，但不对电机发送任何指令，
    方便用户代码在有/无夹爪时共用同一套逻辑，无需条件判断。
    """

    name: str = "gripper"
    _mode: str = "mit"

    @property
    def num_joints(self) -> int:
        return 0

    @property
    def joint_names(self) -> List[str]:
        return []

    @property
    def mode(self) -> str:
        return "mit"

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

    def mode_mit(self, kp=None, kd=None) -> bool:
        self._mode = "mit"
        return True

    def mode_pos_vel(self, vlim=None) -> bool:
        self._mode = "pos_vel"
        return True

    def mode_vel(self) -> bool:
        self._mode = "vel"
        return True

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None) -> None:
        pass

    def send_pos_vel(self, pos, vlim=None) -> None:
        pass

    def send_vel(self, vel) -> None:
        pass

    def get_positions(self, request_feedback: bool = True,
                      position_source: str = "feedback") -> np.ndarray:
        _check_position_source(position_source)
        return np.array([], dtype=np.float64)

    def get_velocities(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def __repr__(self) -> str:
        return "NoOpGroup(gripper, no actuator)"


# --------------------------------------------------------------------------
# JointGroup — 单组关节控制
# --------------------------------------------------------------------------

class JointGroup:
    """一组关节的独立控制器。

    每组拥有独立的控制模式（MIT / POS_VEL）、PID 参数和电机列表，
    可单独使能、切换模式、发送命令。

    由 RebotArm 通过 __getattr__ 代理访问，例如 arm.arm / arm.gripper。
    组内关节数量、顺序由配置决定。
    """

    def __init__(
        self,
        name: str,
        joint_names: List[str],
        all_joints: List[JointCfg],
        motor_map: Dict[str, any],
        ctrl_map: Dict[str, Controller],
    ) -> None:
        self.name = name
        self._jn: List[str] = joint_names
        self._jcfgs: List[JointCfg] = [
            next(j for j in all_joints if j.name == n) for n in joint_names
        ]
        self._mm: Dict[str, any] = motor_map
        self._cm: Dict[str, Controller] = ctrl_map
        self._mode: str = "mit"
        self._mit_kp: np.ndarray = np.array([j.kp for j in self._jcfgs], dtype=np.float64)
        self._mit_kd: np.ndarray = np.array([j.kd for j in self._jcfgs], dtype=np.float64)
        self._pv_vlim: np.ndarray = np.array([j.vlim for j in self._jcfgs], dtype=np.float64)

    # ── 属性 ────────────────────────────────────────────────────────────

    @property
    def num_joints(self) -> int:
        return len(self._jn)

    @property
    def joint_names(self) -> List[str]:
        return list(self._jn)

    @property
    def mode(self) -> str:
        return self._mode

    # ── 使能 / 失能 ────────────────────────────────────────────────────

    def enable(self) -> None:
        by_vendor: Dict[str, List[str]] = {}
        for jc in self._jcfgs:
            by_vendor.setdefault(jc.vendor, []).append(jc.name)
        for vendor in by_vendor:
            try:
                self._cm[vendor].enable_all()
            except CallError as e:
                print(f"[{self.name}/enable] {e}")
            time.sleep(0.05)

    def disable(self) -> None:
        by_vendor: Dict[str, List[str]] = {}
        for jc in self._jcfgs:
            by_vendor.setdefault(jc.vendor, []).append(jc.name)
        for vendor in by_vendor:
            try:
                self._cm[vendor].disable_all()
            except CallError as e:
                print(f"[{self.name}/disable] {e}")
            time.sleep(0.05)

    # ── 模式切换 ────────────────────────────────────────────────────────

    def _write_pv_params(self, jc: JointCfg) -> None:
        m = self._mm[jc.name]
        try:
            if jc.vendor == "robstride":
                m.robstride_write_param_f32(0x7017, jc.vlim)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x701F, jc.vel_kp)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x7020, jc.vel_ki)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x701E, jc.pos_kp)
            elif jc.vendor == "damiao":
                m.write_register_f32(25, jc.vel_kp)
                m.write_register_f32(26, jc.vel_ki)
                m.write_register_f32(27, jc.pos_kp)
                m.write_register_f32(28, jc.pos_ki)
            time.sleep(0.02)
        except Exception as e:
            print(f"[{self.name}/pv_params/{jc.name}] {e}")

    def mode_mit(
        self,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> bool:
        self._mode = "mit"
        if kp is not None:
            self._mit_kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        if kd is not None:
            self._mit_kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        ok = True
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].ensure_mode(Mode.MIT, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_mit/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    def mode_pos_vel(
        self,
        vlim: Optional[np.ndarray] = None,
    ) -> bool:
        self._mode = "pos_vel"
        if vlim is not None:
            self._pv_vlim = np.asarray(vlim, dtype=np.float64).reshape(-1)
        ok = True
        for jc in self._jcfgs:
            self._write_pv_params(jc)
            try:
                self._mm[jc.name].ensure_mode(Mode.POS_VEL, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_pos_vel/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    def mode_vel(self) -> bool:
        self._mode = "vel"
        ok = True
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].ensure_mode(Mode.VEL, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_vel/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    # ── MIT 发送 ────────────────────────────────────────────────────────

    def send_mit(
        self,
        pos: np.ndarray,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
    ) -> None:
        n = self.num_joints
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if vel is None:
            vel = np.zeros(n)
        if tau is None:
            tau = np.zeros(n)
        if kp is None:
            kp = self._mit_kp
        if kd is None:
            kd = self._mit_kd

        for i, jc in enumerate(self._jcfgs):
            try:
                self._mm[jc.name].send_mit(
                    float(pos[i]),
                    float(vel[i]),
                    float(kp[i]),
                    float(kd[i]),
                    float(tau[i]),
                )
            except CallError:
                pass

    # ── POS_VEL 发送 ───────────────────────────────────────────────────

    def send_pos_vel(
        self,
        pos: np.ndarray,
        vlim: Optional[np.ndarray] = None,
    ) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if vlim is None:
            vlim = self._pv_vlim
        vlim = np.asarray(vlim, dtype=np.float64).reshape(-1)
        for i in range(min(len(pos), len(vlim))):
            try:
                self._mm[self._jcfgs[i].name].send_pos_vel(
                    float(pos[i]),
                    float(vlim[i]),
                )
            except CallError:
                pass

    # ── VEL 发送 ───────────────────────────────────────────────────────

    def send_vel(self, vel: np.ndarray) -> None:
        vel = np.asarray(vel, dtype=np.float64).reshape(-1)
        for i in range(min(len(vel), self.num_joints)):
            try:
                self._mm[self._jcfgs[i].name].send_vel(float(vel[i]))
            except CallError:
                pass

    # ── 状态读取 ───────────────────────────────────────────────────────

    def _poll_feedback(self) -> None:
        """仅处理 CAN 接收队列中的反馈帧（快速，无总线发送）。"""
        seen: set[str] = set()
        for jc in self._jcfgs:
            if jc.vendor not in seen:
                seen.add(jc.vendor)
                try:
                    self._cm[jc.vendor].poll_feedback_once()
                except Exception:
                    pass

    def _request_feedback(self) -> None:
        """发送显式反馈请求帧 + 处理接收队列（慢，有总线发送）。"""
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].request_feedback()
            except Exception:
                pass
        self._poll_feedback()

    XOUT_TIMEOUT_S = XOUT_TIMEOUT_S

    def get_positions(self, request_feedback: bool = True,
                      position_source: str = "feedback") -> np.ndarray:
        """本组关节位置 / This group's joint positions.

        position_source 与 RebotArm.get_state() 相同："output" 时本组 Damiao 关节
        改读寄存器 81（xout，输出轴位置），超时或失败抛出 CallError。

        position_source works as in RebotArm.get_state(): with "output",
        Damiao joints in this group read register 81 (xout, output-shaft
        position) instead of the feedback frame; a missing reply (after
        XOUT_TIMEOUT_S) or failed read raises CallError, never 0 or a stale
        value. Feedback requests are sent as before either way.
        """
        _check_position_source(position_source)
        xout_names: List[str] = []
        batched = False
        if position_source == "output":
            # 先发 xout 读请求，与反馈请求的往返重叠
            # Send xout reads first so their round trip overlaps the feedback request's
            xout_names = [jc.name for jc in self._jcfgs if jc.vendor == "damiao"]
            batched = _request_xout(self._mm, xout_names)

        # 始终发送显式请求帧 + 处理接收队列
        # motorbridge 内部会针对 RS/DM 分别处理
        self._request_feedback()
        
        out: list[float] = []
        for jc in self._jcfgs:
            m = self._mm[jc.name]
            st = m.get_state()
            if st is not None:
                out.append(st.pos)
            else:
                # 缓存为空时回退到 SDO 读取（安全兜底）
                if jc.vendor == "robstride":
                    try:
                        out.append(float(m.robstride_get_param_f32(0x7019)))
                        continue
                    except CallError:
                        pass
                out.append(0.0)
        if xout_names:
            ctrls = [self._cm[v] for v in {jc.vendor for jc in self._jcfgs if jc.name in xout_names}]
            xout = _collect_xout(self._mm, ctrls, xout_names, batched, self.XOUT_TIMEOUT_S)
            for i, jc in enumerate(self._jcfgs):
                if jc.name in xout:
                    out[i] = xout[jc.name]
        return np.array(out, dtype=np.float64)

    def get_velocities(self, request_feedback: bool = True) -> np.ndarray:
        # NOTE (RobStride): the cached state has the same staleness problem as
        # get_positions, and the mechVel param (0x701A) was measured NOT to be
        # rad/s on RS firmware (inconsistent scale/sign vs dq/dt, 2026-07-17).
        # For a live velocity on RobStride, finite-difference get_positions().
        if request_feedback:
            self._request_feedback()
        return np.array([
            self._mm[jc.name].get_state().vel
            if self._mm[jc.name].get_state() is not None else 0.0
            for jc in self._jcfgs
        ], dtype=np.float64)

    def __repr__(self) -> str:
        return f"JointGroup({self.name!r}, joints={self.num_joints}, mode={self._mode})"


# --------------------------------------------------------------------------
# RebotArm — 分组控制器容器
# --------------------------------------------------------------------------

class RebotArm:
    """reBotArm 分组控制系统。

    持有多个 JointGroup，每组独立控制模式，独立发送命令，
    在同一个控制循环中按组顺序同步发送，防止总线争用。

    按组访问（通过 __getattr__）::

        arm.arm       # 机械臂关节组
        arm.gripper   # 夹爪关节组（如果有）

    也可以通过 groups 字典::

        arm.groups["arm"]
        arm.groups["gripper"]

    手动添加组::

        arm.add_group("custom", ["joint1", "joint2"])
    """

    def __init__(self, hw_yaml: str | None = None, channel: str | None = None) -> None:
        """channel overrides the config's ``channel`` (the serial port or CAN interface
        differs per machine, e.g. /dev/tty.usbmodem* on macOS, /dev/ttyACM0 on Linux)."""
        self._hw_yaml = _resolve_hw_cfg_path(hw_yaml).name
        cfg = load_cfg(hw_yaml)

        self._name: str = cfg["name"]
        self._channel: str = channel or cfg["channel"]
        self._rate: float = cfg["rate"]
        self._all_joints: List[JointCfg] = cfg["joints"]
        self._groups_def: dict = cfg["groups"]
        self._arm_control_mode: str = cfg.get("arm_control_mode", "posvel")
        self._joint_limits = load_joint_limits(cfg.get("urdf_path"))

        self._ctrl_map: Dict[str, Controller] = {}
        self._motor_map: Dict[str, any] = {}
        self._groups: Dict[str, JointGroup] = {}

        self._running = False
        self._ctrl_thread: Optional[threading.Thread] = None
        self._ctrl_fn: Optional[Callable] = None
        self._ctrl_rate: float = self._rate
        self._connected: bool = False

        self._build_groups()

    # 读数允许超出 URDF 限位的余量（rad），容纳硬限位处的压缩/回差
    # Allowed overshoot past URDF limits (rad), for compression/backlash at hard stops
    LIMIT_MARGIN_RAD = 0.2

    def connect(self, check_limits: bool = True) -> None:
        """连接总线、注册电机。模式切换需在 connect 后调用。

        始终校验每个电机的 CAN 看门狗为 CAN_TIMEOUT_MS，否则断开并抛出 CanTimeoutError
        （不可跳过）。默认还会检查各关节读数是否在 URDF 限位内，不在则抛出 JointLimitError；
        仅零点校准脚本应传 check_limits=False（重新设零正是修复手段）。

        Always verifies that every motor's CAN watchdog is CAN_TIMEOUT_MS and
        raises CanTimeoutError otherwise; this check cannot be skipped. By
        default also checks every joint reading against its URDF limits
        (JointLimitError); only zeroing scripts should pass check_limits=False,
        since re-zeroing is the fix. On any failure the arm is disconnected first.
        """
        if self._connected:
            return
        self._setup_motors()
        self._connected = True
        try:
            self.check_can_timeout()
            if check_limits:
                self.check_joint_limits()
        except Exception:
            self.disconnect()
            raise

    def check_can_timeout(self) -> None:
        """读取每个电机的 CAN 看门狗设置，不等于 CAN_TIMEOUT_MS 则抛出 CanTimeoutError。

        Read every motor's CAN watchdog and raise CanTimeoutError unless it is
        CAN_TIMEOUT_MS. Only Damiao motors can be read back; any other vendor
        raises too, since its watchdog cannot be verified.
        """
        expected = dm_timeout_register_value(CAN_TIMEOUT_MS)
        problems: List[str] = []
        for jc in self._all_joints:
            if jc.vendor != "damiao":
                problems.append(f"  {jc.name}: vendor {jc.vendor!r}, watchdog cannot be read back")
                continue
            try:
                value = int(self._motor_map[jc.name].get_register_u32(DM_RID_TIMEOUT, 200))
            except Exception as e:
                problems.append(f"  {jc.name}: could not read TIMEOUT register ({e})")
                continue
            if value != expected:
                problems.append(f"  {jc.name}: TIMEOUT = {value} ({value / 20:g} ms), "
                                f"expected {expected} ({CAN_TIMEOUT_MS} ms)")
        if problems:
            raise CanTimeoutError(
                "CAN watchdog check failed; refusing to continue:\n"
                + "\n".join(problems)
                + f"\nFix: with the arm powered and disabled, run "
                  f"tools/persist_can_timeout.py to store {CAN_TIMEOUT_MS} ms in every motor."
            )

    def check_joint_limits(self, timeout: float = 1.0) -> None:
        """读取各关节位置，若无反馈或超出 URDF 限位则抛出 JointLimitError。

        Read every joint's position; raise JointLimitError if any joint gives no
        feedback or reads outside its URDF limits (± LIMIT_MARGIN_RAD).
        Joints without URDF limits (e.g. the gripper) are skipped.
        """
        names = [jc.name for jc in self._all_joints if jc.name in self._joint_limits]
        if not names:
            return

        # 上电后的首帧反馈可能尚未到达，轮询直到全部关节有状态或超时
        # The first feedback frames may not have arrived yet; poll until all report.
        positions: Dict[str, float] = {}
        deadline = time.monotonic() + timeout
        while True:
            for name in names:
                if name not in positions:
                    try:
                        self._motor_map[name].request_feedback()
                    except Exception:
                        pass
            for ctrl in self._ctrl_map.values():
                try:
                    ctrl.poll_feedback_once()
                except Exception:
                    pass
            for name in names:
                st = self._motor_map[name].get_state()
                if st is not None:
                    positions[name] = float(st.pos)
            if len(positions) == len(names) or time.monotonic() > deadline:
                break
            time.sleep(0.02)

        problems: List[str] = []
        m = self.LIMIT_MARGIN_RAD
        for name in names:
            lo, hi = self._joint_limits[name]
            if name not in positions:
                problems.append(f"  {name}: no position feedback within {timeout:.1f}s")
                continue
            q = positions[name]
            if lo - m <= q <= hi + m:
                continue
            msg = (f"  {name}: reads {q:+.4f} rad ({math.degrees(q):+.1f}°), "
                   f"limits [{lo:+.3f}, {hi:+.3f}] rad")
            turns = round((q - (lo + hi) / 2) / (2 * math.pi))
            if turns and lo - m <= q - turns * 2 * math.pi <= hi + m:
                msg += (f" — off by {turns:+d}×2π; the encoder picked the wrong turn at power-on")
            problems.append(msg)

        if problems:
            raise JointLimitError(
                "Joint position check failed; refusing to continue:\n"
                + "\n".join(problems)
                + "\nFix: power off, put the arm in the pose it normally reads correctly in "
                  "(e.g. press the joint firmly onto its zero stop), power on and retry. "
                  "Re-zeroing also clears it, but only do that with the arm at the "
                  "calibrated zero pose."
            )

    def _make_controller(self, vendor: str) -> Controller:
        if self._channel.startswith("/dev/"):  # any device path is the serial bridge; CAN interfaces are bare names (can0)
            return Controller.from_dm_serial(self._channel, 921600)
        return Controller(self._channel)

    def _setup_motors(self) -> None:
        for jc in self._all_joints:
            vendor = jc.vendor
            if vendor not in self._ctrl_map:
                self._ctrl_map[vendor] = self._make_controller(vendor)
            ctrl = self._ctrl_map[vendor]

            if vendor == "damiao":
                mot = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "robstride":
                mot = ctrl.add_robstride_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "myactuator":
                mot = ctrl.add_myactuator_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "hightorque":
                mot = ctrl.add_hightorque_motor(jc.motor_id, jc.feedback_id, jc.model)
            else:
                raise ValueError(f"Unsupported vendor: {vendor}")

            self._motor_map[jc.name] = mot

    def _build_groups(self) -> None:
        for gname, gdef in self._groups_def.items():
            joints_def = gdef.get("joints", [])
            g = JointGroup(
                name=gname,
                joint_names=joints_def,
                all_joints=self._all_joints,
                motor_map=self._motor_map,
                ctrl_map=self._ctrl_map,
            )
            self._groups[gname] = g
        if "gripper" not in self._groups:
            self._groups["gripper"] = NoOpGroup()

    # ── 属性 ────────────────────────────────────────────────────────────

    @property
    def num_joints(self) -> int:
        return len(self._all_joints)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self._all_joints]

    @property
    def groups(self) -> Dict[str, JointGroup]:
        return self._groups

    @property
    def control_loop_active(self) -> bool:
        t = getattr(self, "_ctrl_thread", None)
        return t is not None and t.is_alive()

    @property
    def rate(self) -> float:
        return self._ctrl_rate

    @property
    def has_gripper(self) -> bool:
        return not isinstance(self._groups.get("gripper", None), NoOpGroup)

    @property
    def arm_control_mode(self) -> str:
        """从硬件配置文件读取的默认 arm 控制模式（"mit" 或 "posvel"）。"""
        return self._arm_control_mode

    @property
    def hardware_yaml(self) -> str:
        return self._hw_yaml

    def __getattr__(self, name: str) -> any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._groups:
            return self._groups[name]
        raise AttributeError(name)

    # ── 手动添加组 ────────────────────────────────────────────────────

    def add_group(self, name: str, joint_names: List[str]) -> JointGroup:
        if name in self._groups:
            raise ValueError(f"组 {name!r} 已存在")
        g = JointGroup(
            name=name,
            joint_names=joint_names,
            all_joints=self._all_joints,
            motor_map=self._motor_map,
            ctrl_map=self._ctrl_map,
        )
        self._groups[name] = g
        return g

    # ── 全局使能 / 失能 ────────────────────────────────────────────────

    def enable_all(self) -> None:
        for g in self._groups.values():
            g.enable()

    def disable_all(self) -> None:
        for g in self._groups.values():
            g.disable()

    # ── 零点 ────────────────────────────────────────────────────────────

    def set_zero(self, poll_max: int = 200, poll_interval: float = 0.05) -> None:
        self.disable_all()
        time.sleep(0.3)
        for jc in self._all_joints:
            for _ in range(poll_max):
                for m in self._motor_map.values():
                    try:
                        m.request_feedback()
                    except Exception:
                        pass
                for ctrl in self._ctrl_map.values():
                    try:
                        ctrl.poll_feedback_once()
                    except Exception:
                        pass
                st = self._motor_map[jc.name].get_state()
                if st is not None and st.status_code == 0:
                    break
                time.sleep(poll_interval)
            try:
                self._motor_map[jc.name].set_zero_position()
            except CallError as e:
                print(f"[set_zero] {jc.name}: {e}")
            time.sleep(0.1)

    # ── 全局状态读取 ───────────────────────────────────────────────────

    _DM_RID_XOUT = DM_RID_XOUT
    POSITION_SOURCES = POSITION_SOURCES
    XOUT_TIMEOUT_S = XOUT_TIMEOUT_S

    def get_state(
        self,
        request_feedback: bool = True,
        position_source: str = "feedback",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (pos, vel, torq)。

        position_source:
          - "feedback"（默认）：位置取自反馈帧，与控制指令同一坐标系。
          - "output"：Damiao 关节的位置改为读取寄存器 81（xout，输出轴编码器位置，
            rad，与反馈位置同一零点），可反映减速器回差，适合作为本体感知。
            所有电机的读请求一次性发出、统一等待（7 个电机约 2 ms），每个电机
            每次多占 2 帧总线带宽。超时或读取失败抛出 CallError，不会以 0 或
            旧值代替。非 Damiao 关节仍用反馈帧。

        Returns (pos, vel, torq).

        position_source:
          - "feedback" (default): position from the feedback frame, the same
            frame control commands use.
          - "output": Damiao joints read register 81 (xout, the output-shaft
            encoder; rad, same zero as the feedback position) instead. It
            reflects gearbox backlash, so it is the better proprioception
            signal. Requests go to all motors at once and are awaited together
            (about 2 ms for 7 motors over dm-serial); each read costs 2 extra
            bus frames per motor. A missing reply (after XOUT_TIMEOUT_S) or a
            failed read raises CallError: never 0 or a stale value.
            Non-Damiao joints keep the feedback-frame position.
            Note the motor's own MIT servo still runs on its internal position,
            so closing a host-side position loop on xout can hunt in the
            backlash; using it as an observation is fine.
        vel and torq always come from the feedback frame.
        """
        _check_position_source(position_source)
        xout_names: List[str] = []
        batched = False
        if position_source == "output":
            # 先发 xout 读请求，与反馈请求的往返重叠
            # Send xout reads first so their round trip overlaps the feedback request's
            xout_names = [jc.name for jc in self._all_joints if jc.vendor == "damiao"]
            batched = _request_xout(self._motor_map, xout_names)
        if request_feedback:
            for m in self._motor_map.values():
                try:
                    m.request_feedback()
                except Exception:
                    pass
        for ctrl in self._ctrl_map.values():
            try:
                ctrl.poll_feedback_once()
            except Exception:
                pass
        pos, vel, torq = [], [], []
        for jc in self._all_joints:
            st = self._motor_map[jc.name].get_state()
            if st is not None:
                pos.append(st.pos)
                vel.append(st.vel)
                torq.append(st.torq)
            else:
                pos.append(0.0)
                vel.append(0.0)
                torq.append(0.0)
        if position_source == "output":
            xout = _collect_xout(self._motor_map, list(self._ctrl_map.values()),
                                 xout_names, batched, self.XOUT_TIMEOUT_S)
            for i, jc in enumerate(self._all_joints):
                if jc.name in xout:
                    pos[i] = xout[jc.name]
        return (
            np.array(pos, dtype=np.float64),
            np.array(vel, dtype=np.float64),
            np.array(torq, dtype=np.float64),
        )

    def get_positions(self, position_source: str = "feedback", request_feedback: bool = True) -> np.ndarray:
        return self.get_state(request_feedback=request_feedback, position_source=position_source)[0]

    def get_velocities(self) -> np.ndarray:
        return self.get_state()[1]

    def get_torques(self) -> np.ndarray:
        return self.get_state()[2]

    # ── 生命周期 ────────────────────────────────────────────────────────

    def disconnect(self) -> None:
        if not self._connected:
            return
        self.stop_control_loop()
        self.disable_all()
        time.sleep(0.5)
        for ctrl in self._ctrl_map.values():
            ctrl.shutdown()
            time.sleep(0.1)
            ctrl.close()
        self._ctrl_map.clear()
        self._motor_map.clear()
        self._connected = False

    def estop(self) -> None:
        self.disable_all()

    def reconnect(
        self,
        init_delay: float = 1.0,
        post_setup_delay: float = 0.5,
    ) -> None:
        self.disconnect()
        time.sleep(init_delay)
        for vendor in set(j.vendor for j in self._all_joints):
            self._ctrl_map[vendor] = self._make_controller(vendor)
        self._motor_map.clear()
        for jc in self._all_joints:
            ctrl = self._ctrl_map[jc.vendor]
            if jc.vendor == "damiao":
                mot = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif jc.vendor == "robstride":
                mot = ctrl.add_robstride_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif jc.vendor == "myactuator":
                mot = ctrl.add_myactuator_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif jc.vendor == "hightorque":
                mot = ctrl.add_hightorque_motor(jc.motor_id, jc.feedback_id, jc.model)
            else:
                raise ValueError(f"Unsupported vendor: {jc.vendor}")
            self._motor_map[jc.name] = mot
            time.sleep(0.05)
        self._build_groups()
        time.sleep(post_setup_delay)
        print("[reconnect] 控制器和电机已重新初始化")

    # ── 控制循环 ────────────────────────────────────────────────────────

    def start_control_loop(
        self,
        control_fn: Callable[["RebotArm", float], None],
        rate: Optional[float] = None,
    ) -> None:
        if self.control_loop_active:
            raise RuntimeError("控制循环已在运行，请先调用 stop_control_loop()")
        self._running = True
        self._ctrl_rate = rate if rate is not None else self._rate
        self._ctrl_fn = control_fn
        self._ctrl_thread = threading.Thread(
            target=self._control_loop_impl,
            name="rebotarm-control-loop",
            daemon=True,
        )
        self._ctrl_thread.start()

    def _control_loop_impl(self) -> None:
        dt = 1.0 / self._ctrl_rate
        while self._running:
            t0 = time.perf_counter()
            try:
                self._ctrl_fn(self, dt)
            except Exception:
                if self._running:
                    raise
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop_control_loop(self) -> None:
        self._running = False
        t = getattr(self, "_ctrl_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=5.0)

    # ── 上下文管理器 ───────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        gs = ", ".join(f"{k}({g.num_joints}j)" for k, g in self._groups.items())
        return f"RebotArm({self._name!r}, [{gs}], rate={self._ctrl_rate}Hz)"
