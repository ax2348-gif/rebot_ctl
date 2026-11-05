"""rebot_vis: all joints slack (MIT, zero gains) with the live pose in MeshCat.

Every joint is commanded MIT with kp = kd = tau = 0, so the arm can be moved freely by
hand (support it: it does not hold itself up), and MeshCat mirrors the pose. The MeshCat
page is served on all network interfaces, so open it from any device on the same network;
the addresses are printed at startup.

With -g the arm joints also get a gravity-compensation torque tau = g(q) (Pinocchio, from
the config's URDF, scaled by its gravity_compensation tau_scale / joint_direction), still
with kp = kd = 0: the arm floats where you leave it but has no stiffness. The torque fades
in after enabling, is clipped per joint (--tau-limit), and is zero for any cycle without a
fresh reading from every arm motor. The gripper stays slack.

Safety:
  - connect() verifies the 100 ms motor CAN watchdog and the joint limits first;
  - every motor must confirm MIT mode, or nothing is enabled;
  - the command stream starts *before* the motors are enabled, with zero torque until
    then, so no motor ever acts on a stale setpoint;
  - Ctrl+C, any error, or a fault reported by a motor disables all motors (with -g the
    arm then drops: support it); if this process dies, the motors' watchdog disables
    them within 100 ms.
"""
from __future__ import annotations

import contextlib
import io
import signal
import sys
import time
from typing import Optional

import numpy as np

from motorbridge import CallError

from ..actuator import RebotArm, load_cfg, load_gravity_compensation_config
from ._common import describe_status, external_urls, fmt_deg, parser

VIS_RATE_HZ = 30.0


class ArmViewer:
    """MeshCat view of the arm's URDF. The web page listens on all interfaces."""

    def __init__(self, urdf_path: Optional[str] = None):
        import meshcat
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        from ..kinematics import get_arm_joint_count, pad_q_for_model
        from ..kinematics.robot_model import _resolve_urdf

        urdf, package_dirs = _resolve_urdf(urdf_path)
        self.model = pin.buildModelFromUrdf(urdf)
        visual = pin.buildGeomFromUrdf(self.model, urdf, pin.GeometryType.VISUAL,
                                       package_dirs=package_dirs)
        # meshcat prints a 127.0.0.1 URL; main() prints the addresses other devices can use.
        with contextlib.redirect_stdout(io.StringIO()):
            self.meshcat = meshcat.Visualizer()
        self._viz = MeshcatVisualizer(self.model, collision_model=None, visual_model=visual,
                                      data=self.model.createData(), visual_data=visual.createData())
        self._viz.initViewer(self.meshcat, loadModel=False)
        self._viz.loadViewerModel()
        self._n_arm = get_arm_joint_count(self.model)
        self._pad = pad_q_for_model

    @property
    def port(self) -> int:
        # web_url looks like http://127.0.0.1:7000/static/
        return int(self.meshcat.url().rsplit(":", 1)[1].split("/", 1)[0])

    def update(self, positions) -> None:
        self._viz.display(self._pad(self.model, np.asarray(positions), self._n_arm))


class GravityTorque:
    """tau = g(q) for the arm joints, as the package's GravityCompensation computes it
    (same tau_scale / joint_direction config), clipped to ``limit`` per joint."""

    def __init__(self, urdf_path: Optional[str], gc_cfg: dict, n_arm: int, limit: float):
        from ..kinematics import load_robot_model
        from ..kinematics.robot_model import pad_q_for_model

        self.model = load_robot_model(urdf_path)
        self._data = self.model.createData()
        self._pad = pad_q_for_model
        self.n = n_arm
        self.scale = self._vector(gc_cfg.get("tau_scale", 1.0), "tau_scale")
        self.direction = self._vector(gc_cfg.get("joint_direction", 1.0), "joint_direction")
        self.limit = float(limit)
        self.ramp_s = max(0.0, float(gc_cfg.get("transition_duration", 0.5)))

    def _vector(self, value, label: str) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if arr.size == 1:
            return np.full(self.n, float(arr[0]))
        if arr.size != self.n:
            raise ValueError(f"gravity_compensation.{label} must be a scalar or {self.n} values")
        return arr

    def __call__(self, q_arm: np.ndarray) -> np.ndarray:
        from ..dynamics import compute_generalized_gravity

        q = self._pad(self.model, np.asarray(q_arm) * self.direction, self.n)
        tau = compute_generalized_gravity(self.model, q, self._data)[: self.n]
        return np.clip(tau * self.direction * self.scale, -self.limit, self.limit)


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _controller(arm: RebotArm, gravity: Optional[GravityTorque], enabled_at: dict):
    """Zero gains on every joint; with ``gravity``, tau = g(q) on the arm joints.

    ``enabled_at["t"]`` is set by main() once the motors are enabled; until then the
    torque is zero, and afterwards it fades in over ``gravity.ramp_s``.
    """
    n_arm, n_grip = arm.arm.num_joints, arm.gripper.num_joints
    za, zg = np.zeros(n_arm), np.zeros(n_grip)
    arm_names = arm.arm.joint_names if gravity else []

    def arm_tau(r: RebotArm) -> np.ndarray:
        if gravity is None or enabled_at.get("t") is None:
            return za
        # Positions from the feedback each MIT command brings back: no extra bus traffic.
        states = [r._motor_map[n].get_state() for n in arm_names]
        if any(st is None for st in states):
            return za
        tau = gravity(np.array([st.pos for st in states], dtype=np.float64))
        if not np.all(np.isfinite(tau)):
            return za
        elapsed = time.monotonic() - enabled_at["t"]
        return tau * (_smoothstep(elapsed / gravity.ramp_s) if gravity.ramp_s else 1.0)

    def controller(r: RebotArm, dt: float) -> None:
        tau = arm_tau(r)
        enabled_at["tau"] = tau
        r.arm.send_mit(pos=za, vel=za, kp=za, kd=za, tau=tau)
        if n_grip:
            r.gripper.send_mit(pos=zg, vel=zg, kp=zg, kd=zg, tau=zg)

    return controller


def main(argv=None) -> int:
    p = parser("All joints slack (MIT, zero gains) with the live pose in MeshCat.")
    p.add_argument("--source", choices=("output", "feedback"), default="output",
                   help="position shown: output-shaft encoder (default) or motor feedback")
    p.add_argument("--rate", type=float, default=100.0,
                   help="rate of the command stream in Hz (default 100)")
    p.add_argument("-g", "--gravity", action="store_true",
                   help="add gravity-compensation torque on the arm joints (still zero gains)")
    p.add_argument("--tau-limit", type=float, default=12.0,
                   help="with -g, clip each joint's torque to this many N·m (default 12)")
    p.add_argument("--profile", default="default",
                   help="with -g, gravity_compensation profile in the config (if it has any)")
    args = p.parse_args(argv)

    arm = RebotArm(args.hardware_yaml, channel=args.channel)
    print(f"Config: {arm.hardware_yaml} on {arm.channel}")
    urdf_path = load_cfg(args.hardware_yaml).get("urdf_path") or None
    gravity = None
    if args.gravity:
        gc_cfg = load_gravity_compensation_config(args.profile, args.hardware_yaml)
        gravity = GravityTorque(urdf_path, gc_cfg, arm.arm.num_joints, args.tau_limit)
        print(f"Gravity compensation: tau_scale {gravity.scale.tolist()}, "
              f"limit ±{gravity.limit:g} N·m, fade-in {gravity.ramp_s:g} s")
    # Start the viewer before touching the motors, so a failure here never leaves them enabled.
    print("Starting MeshCat...")
    viewer = ArmViewer(urdf_path)

    stop = {"now": False}
    arm.connect()  # verifies the CAN watchdog and the joint limits; raises if either fails
    faulted = None
    try:
        # Until here Ctrl+C raises KeyboardInterrupt (and lands in the finally below).
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.__setitem__("now", True))
        for name in arm.joint_names:
            arm._motor_map[name].clear_error()  # faults left by a previous run block enabling
        if not (arm.arm.mode_mit() and arm.gripper.mode_mit()):
            print("\nNot every motor confirmed MIT mode; nothing enabled.")
            return 1
        enabled_at: dict = {"t": None}
        arm.start_control_loop(_controller(arm, gravity, enabled_at), rate=args.rate)
        time.sleep(0.05)  # zero-torque frames are flowing before anything is enabled
        arm.enable_all()
        enabled_at["t"] = time.monotonic()

        if gravity:
            print("\nGravity compensated, zero stiffness: the arm floats but holds no pose."
                  "\nIt drops when this stops: support it before Ctrl+C.")
        else:
            print("\nAll joints are slack: support the arm, it will not hold itself up.")
        print("MeshCat (open from any device on this network):")
        for url in external_urls(viewer.port):
            print(f"  {url}")
        print("\nCtrl+C to stop (disables all motors).\n")
        print("  " + "  ".join(f"{n:>8}" for n in arm.joint_names))

        last_status: dict = {}
        period = 1.0 / VIS_RATE_HZ
        while not stop["now"]:
            t0 = time.monotonic()
            # The command stream's MIT frames already bring back feedback; no extra requests.
            try:
                pos = arm.get_positions(position_source=args.source, request_feedback=False)
            except CallError as e:
                print(f"\n  read skipped: {e}", flush=True)
            else:
                viewer.update(pos)
                line = f"\r  {fmt_deg(pos)}"
                if gravity is not None:
                    line += "   τ " + " ".join(f"{t:+5.2f}" for t in enabled_at.get("tau", ()))
                print(line, end="", flush=True)

            for jc in arm._all_joints:
                st = arm._motor_map[jc.name].get_state()
                if st is None:
                    continue
                code = int(st.status_code)
                if code != last_status.get(jc.name):
                    if jc.name in last_status:
                        print(f"\n  {jc.name}: {describe_status(jc.vendor, code)}", flush=True)
                    last_status[jc.name] = code
                if jc.vendor == "damiao" and code >= 0x8:
                    faulted = f"{jc.name}: {describe_status(jc.vendor, code)}"
            if faulted:
                print(f"\n  Motor fault ({faulted}); stopping.", flush=True)
                break
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        print("\nDisabling all motors.")
        arm.disconnect()
    return 1 if faulted else 0


if __name__ == "__main__":
    sys.exit(main())
