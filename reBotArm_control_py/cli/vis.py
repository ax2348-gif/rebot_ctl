"""rebot_vis: all joints slack (MIT, zero gains) with the live pose in MeshCat.

Every joint is commanded MIT with kp = kd = tau = 0, so the arm can be moved freely by
hand (support it: it does not hold itself up), and MeshCat mirrors the pose. The MeshCat
page is served on all network interfaces, so open it from any device on the same network;
the addresses are printed at startup.

Safety:
  - connect() verifies the 100 ms motor CAN watchdog and the joint limits first;
  - the zero-gain command stream starts *before* the motors are enabled, so no motor ever
    acts on a stale setpoint;
  - Ctrl+C, any error, or a fault reported by a motor disables all motors; if this process
    dies, the motors' watchdog disables them within 100 ms.
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

from ..actuator import RebotArm, load_cfg
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


def _slack_controller(arm: RebotArm):
    n_arm, n_grip = arm.arm.num_joints, arm.gripper.num_joints
    za, zg = np.zeros(n_arm), np.zeros(n_grip)

    def controller(r: RebotArm, dt: float) -> None:
        r.arm.send_mit(pos=za, vel=za, kp=za, kd=za, tau=za)
        if n_grip:
            r.gripper.send_mit(pos=zg, vel=zg, kp=zg, kd=zg, tau=zg)

    return controller


def main(argv=None) -> int:
    p = parser("All joints slack (MIT, zero gains) with the live pose in MeshCat.")
    p.add_argument("--source", choices=("output", "feedback"), default="output",
                   help="position shown: output-shaft encoder (default) or motor feedback")
    p.add_argument("--rate", type=float, default=100.0,
                   help="rate of the zero-gain command stream in Hz (default 100)")
    args = p.parse_args(argv)

    arm = RebotArm(args.hardware_yaml, channel=args.channel)
    print(f"Config: {arm.hardware_yaml} on {arm.channel}")
    # Start the viewer before touching the motors, so a failure here never leaves them enabled.
    print("Starting MeshCat...")
    viewer = ArmViewer(load_cfg(args.hardware_yaml).get("urdf_path") or None)

    stop = {"now": False}
    arm.connect()  # verifies the CAN watchdog and the joint limits; raises if either fails
    faulted = None
    try:
        # Until here Ctrl+C raises KeyboardInterrupt (and lands in the finally below).
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.__setitem__("now", True))
        for name in arm.joint_names:
            arm._motor_map[name].clear_error()  # faults left by a previous run block enabling
        arm.arm.mode_mit()
        arm.gripper.mode_mit()
        arm.start_control_loop(_slack_controller(arm), rate=args.rate)
        time.sleep(0.05)  # zero-gain frames are flowing before anything is enabled
        arm.enable_all()

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
                print(f"\r  {fmt_deg(pos)}", end="", flush=True)

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
