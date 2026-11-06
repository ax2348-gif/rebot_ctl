# reBotArm Control

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a> &nbsp;|&nbsp;
    <a href="./README_JP.md">日本語</a>&nbsp;|&nbsp;
    <a href="./README_Fr.md">français</a>&nbsp;|&nbsp;
    <a href="./README_es.md">Español</a>
  </strong>
</p>

For complete usage instructions, please refer to the official wiki:

- **DM Version (Damiao Motors)**: [https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/](https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/)
- **RS Version (RobStride Motors)**: [https://wiki.seeedstudio.com/rebot_arm_b601_rs_pinocchio_meshcat/](https://wiki.seeedstudio.com/rebot_arm_b601_rs_pinocchio_meshcat/)

## Command-line tools

Installing the package (`pip install -e .`; keep it editable, it reads `config/` and `urdf/`
from this repo) puts these on `PATH`:

| Tool | What it does | Moves the arm? |
|---|---|---|
| `rebot_setup` | Read-only check: every motor answers, has the 100 ms CAN watchdog stored, and reads inside its joint limits. `--store-watchdog` writes the watchdog to motor flash (power-cycle first; asks you to type `STORE`). | no |
| `rebot_zero` | Sets the current pose as every joint's zero (asks you to type `ZERO`), then re-checks the joint limits. Motors stay disabled. This zeroes the gripper too, wherever it is; follow with `rebot_zero_gripper --zero`. | no |
| `rebot_zero_gripper` | Drives the gripper closed against its stop for `--duration` s (default 0.5) as an MIT velocity loop (kp = 0, target velocity and kd hard-coded at the top of `cli/zero_gripper.py`, stall push = kd × velocity), then shows where it ended up. `--zero` then disables it and sets that position as its zero: fully closed, backlash taken up. The Damiao gripper motor's encoder is single-turn, and its range (open ≈ −3.1 rad) sits near ±π, so a reading can come up 2π off after a power cycle; this closes it and shows the reading. Arm motors stay disabled. | gripper only |
| `rebot_vis` | All joints slack (MIT, kp = kd = tau = 0) with the live pose in MeshCat, served on every network interface; the addresses are printed. Support the arm: it does not hold itself up. `-g` adds gravity-compensation torque (still zero stiffness; clipped by `--tau-limit`), so it floats but drops when stopped. Ctrl+C disables the motors. | enables; zero torque, or gravity torque with `-g` |
| `rebot_jog` | Jogs the end effector from the keyboard (IK, MIT impedance). Defaults to `rebotarm_dm_safe.yaml` and refuses stiffer configs; asks you to type `JOG`; holds the current pose with gains ramping in; every step is checked (IK, joint limits, joint jump, `--z-min`) and speed-limited. Ctrl+C, `q` or any fault stops all commands at once (the motor watchdog disables the motors even if the process hangs) and the arm goes limp. | yes |

Each takes an optional hardware config (default: `hardware_yaml` in `config/rebotarm.yaml`)
and `--channel` to override the config's serial port or CAN interface for this machine.
First-time order: `rebot_setup` (and `--store-watchdog` if it says so), `rebot_zero` with the
arm in its zero pose, then `rebot_vis`.
