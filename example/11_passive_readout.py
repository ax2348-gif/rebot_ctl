#!/usr/bin/env python3
"""被动读数 + MeshCat 实时可视化（不发送任何控制指令）。

Passive readout + live MeshCat visualization. Sends NO control commands.

The only frames put on the bus are feedback requests. This script never
enables, disables, switches mode, sets zero, writes registers or sends a
position/torque command, and it does not use RebotArm (whose disconnect()
disables all motors and calls the native controller shutdown).

Use it to sanity-check the setup before anything moves the arm:
  - every motor answers, with the expected (disabled) status
  - joint signs: move one joint by hand and watch the model follow it
  - zero: the folded rest pose should read ~0 on every joint
  - readings inside the URDF joint limits
  - near-zero torque reading while unpowered

用法 / Usage::

    python example/11_passive_readout.py
      配置取自 config/rebotarm.yaml 的 hardware_yaml（URDF 与末端帧也来自此处，
      故不支持命令行覆盖，以免读数与模型不一致）
      Config comes from config/rebotarm.yaml's hardware_yaml. The URDF and
      end-effector frame are loaded from there too, so no CLI override is
      offered (it would let the readout and the model disagree).

退出 / Exit: Ctrl+C
"""
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motorbridge import Controller

from reBotArm_control_py.actuator.rebotarm import load_cfg
from reBotArm_control_py.kinematics import compute_fk, get_arm_joint_count, pad_q_for_model
from example.sim.visualizer import Visualizer

POLL_HZ = 50.0
PRINT_HZ = 5.0
STARTUP_TIMEOUT = 2.0
LIMIT_MARGIN = np.deg2rad(2.0)

# Damiao feedback ERR nibble (raw status_code)
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


def make_controller(channel: str) -> Controller:
    # Same transport selection as RebotArm._make_controller
    if channel.startswith("/dev/tty"):
        return Controller.from_dm_serial(channel, 921600)
    return Controller(channel)


def poll(ctrl: Controller, motors: dict, errors: dict) -> None:
    for name, m in motors.items():
        try:
            m.request_feedback()
        except Exception:
            errors[name] += 1
    for _ in range(3):
        try:
            ctrl.poll_feedback_once()
        except Exception:
            pass


def main() -> None:
    cfg = load_cfg()
    if any(j.vendor != "damiao" for j in cfg["joints"]):
        sys.exit("This readout only supports damiao joints (status decoding, single bus).")

    arm_names = list(cfg["groups"]["arm"]["joints"])
    all_cfgs = {j.name: j for j in cfg["joints"]}

    viz = Visualizer()
    model = viz.model
    n_arm = get_arm_joint_count(model)
    if n_arm != len(arm_names):
        sys.exit(f"URDF arm chain has {n_arm} joints, config arm group has {len(arm_names)}.")

    # Map config joints onto URDF joints by name, not by position.
    q_idx = []
    for name in arm_names:
        if not model.existJointName(name):
            sys.exit(f"Config joint {name!r} not found in URDF.")
        q_idx.append(model.joints[model.getJointId(name)].idx_q)
    if q_idx != list(range(n_arm)):
        sys.exit(f"Config arm joint order {arm_names} does not match URDF order.")
    lo = model.lowerPositionLimit[:n_arm]
    hi = model.upperPositionLimit[:n_arm]

    ctrl = make_controller(cfg["channel"])
    motors = {}
    try:
        for name, jc in all_cfgs.items():
            motors[name] = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)

        errors = {name: 0 for name in motors}

        # Every motor must answer before anything is shown; a missing reply
        # is never treated as 0.
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            poll(ctrl, motors, errors)
            if all(m.get_state() is not None for m in motors.values()):
                break
            time.sleep(1.0 / POLL_HZ)
        silent = [n for n, m in motors.items() if m.get_state() is None]
        if silent:
            sys.exit(f"No feedback from: {', '.join(silent)}. Check power, wiring and IDs.")

        # get_state() is a cache with no timestamp, so a motor that stops
        # answering keeps its last value. Track when each reading last
        # changed as a (weak) liveness hint.
        last_vals = {n: None for n in motors}
        last_change = {n: time.monotonic() for n in motors}

        t_print = 0.0
        while True:
            t0 = time.monotonic()
            poll(ctrl, motors, errors)
            states = {n: m.get_state() for n, m in motors.items()}
            for n, st in states.items():
                vals = (st.pos, st.vel, st.torq, st.t_mos, st.status_code)
                if vals != last_vals[n]:
                    last_vals[n] = vals
                    last_change[n] = t0

            q_arm = np.array([states[n].pos for n in arm_names])
            # Gripper motor angle -> finger displacement mapping is unknown;
            # fingers stay at 0 m, as in example/sim/fk_sim.py.
            q = pad_q_for_model(model, q_arm, n_arm)
            viz.update(q)

            if t0 - t_print >= 1.0 / PRINT_HZ:
                t_print = t0
                render(states, arm_names, q_arm, q, lo, hi, model,
                       last_change, errors, t0, viz.meshcat.url())

            sleep = 1.0 / POLL_HZ - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        # Free handles only. No disable_all()/shutdown(): nothing was
        # enabled here, and the native shutdown's bus behaviour is unknown.
        for m in motors.values():
            try:
                m.close()
            except Exception:
                pass
        ctrl.close()
        print("\n完成 / Done.")


def render(states, arm_names, q_arm, q, lo, hi, model, last_change, errors, now, url) -> None:
    lines = [f"MeshCat: {url}    Ctrl+C to exit    (passive: no commands sent)", ""]
    lines.append(f"{'joint':<8}{'deg':>9}{'limits deg':>18}{'vel':>8}{'torq':>8}"
                 f"{'T_mos':>7}{'T_rot':>7}  {'status':<20}{'unchanged':>10}{'req err':>8}")
    flags = []
    for n, st in states.items():
        status = DM_STATUS.get(st.status_code, f"UNKNOWN 0x{st.status_code:X}")
        if n in arm_names:
            i = arm_names.index(n)
            lim = f"[{np.rad2deg(lo[i]):+7.1f},{np.rad2deg(hi[i]):+7.1f}]"
            if st.pos < lo[i] - LIMIT_MARGIN or st.pos > hi[i] + LIMIT_MARGIN:
                flags.append(f"{n}: {np.rad2deg(st.pos):+.1f} deg is outside URDF limits {lim}")
        else:
            lim = ""
        if st.status_code != 0x0:
            flags.append(f"{n}: status {status} (expected disabled)")
        if abs(st.torq) > 0.5:
            flags.append(f"{n}: torque reading {st.torq:+.2f} Nm while it should be unpowered")
        lines.append(
            f"{n:<8}{np.rad2deg(st.pos):>+9.2f}{lim:>18}{st.vel:>+8.2f}{st.torq:>+8.2f}"
            f"{st.t_mos:>7.0f}{st.t_rotor:>7.0f}  {status:<20}"
            f"{now - last_change[n]:>9.1f}s{errors[n]:>8}"
        )

    pos, rot, _ = compute_fk(model, q)
    rpy = np.rad2deg(pin.rpy.matrixToRpy(rot))
    lines += [
        "",
        f"end effector  pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}] m"
        f"  rpy=[{rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}] deg",
        f"max |q| = {np.rad2deg(np.max(np.abs(q_arm))):.2f} deg  (folded rest should be ~0)",
        "",
    ]
    lines += ["RED FLAGS:"] + [f"  ! {f}" for f in flags] if flags else ["no red flags"]
    lines.append("'unchanged' is a weak liveness hint: a still, unpowered joint may legitimately not change.")
    print("\033[H\033[J" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
