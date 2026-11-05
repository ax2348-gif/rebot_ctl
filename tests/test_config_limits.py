"""Optional q_min_deg / q_max_deg: joint-space limits on q, alongside the URDF's."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from reBotArm_control_py.actuator import RebotArm, joint_limits, load_cfg
from reBotArm_control_py.actuator.rebotarm import JointCfg, JointLimitError, load_joint_limits

DM_CONFIGS = ("rebotarm_dm.yaml", "rebotarm_dm_soft.yaml", "rebotarm_dm_safe.yaml")


def cfg_with(**limits):
    """A config whose joint1 / gripper carry the given q_min / q_max (rad)."""
    cfg = load_cfg("rebotarm_dm_safe.yaml")
    joints = []
    for jc in cfg["joints"]:
        extra = limits.get(jc.name, {})
        joints.append(JointCfg(**{**jc.__dict__, "q_min": None, "q_max": None, **extra}))
    return {**cfg, "joints": joints}


class ConfigLimitTests(unittest.TestCase):
    def test_yaw_bounded_at_minus_39_in_every_damiao_config(self):
        for name in DM_CONFIGS:
            cfg = load_cfg(name)
            urdf = load_joint_limits(cfg["urdf_path"])
            lim = joint_limits(cfg)
            self.assertAlmostEqual(math.degrees(lim["joint1"][0]), -39.0, msg=name)
            self.assertEqual(lim["joint1"][1], urdf["joint1"][1], name)
            for j in ("joint2", "joint3", "joint4", "joint5", "joint6"):
                self.assertEqual(lim[j], urdf[j], f"{name}: {j} unchanged")

    def test_no_fields_means_urdf_limits(self):
        cfg = cfg_with()
        self.assertEqual(joint_limits(cfg), load_joint_limits(cfg["urdf_path"]))

    def test_urdf_limit_still_applies_when_config_is_wider(self):
        cfg = cfg_with(joint1={"q_min": math.radians(-400), "q_max": math.radians(400)})
        self.assertEqual(joint_limits(cfg)["joint1"], load_joint_limits(cfg["urdf_path"])["joint1"])

    def test_limit_on_a_joint_without_urdf_limits(self):
        cfg = cfg_with(gripper={"q_max": 1.0})
        self.assertEqual(joint_limits(cfg)["gripper"], (-math.inf, 1.0))

    def test_contradictory_limits_rejected(self):
        cfg = cfg_with(joint1={"q_min": 1.0, "q_max": 0.5})
        with self.assertRaises(ValueError):
            joint_limits(cfg)

    def test_connect_check_enforces_config_limit(self):
        robot = RebotArm("rebotarm_dm_safe.yaml")
        self.assertAlmostEqual(math.degrees(robot._joint_limits["joint1"][0]), -39.0)
        # -39° - 0.2 rad margin is about -50.5°: -60° must be refused, -45° allowed
        for deg, ok in ((-45.0, True), (-60.0, False)):
            state = {n: SimpleNamespace(pos=0.0) for n in robot.joint_names}
            state["joint2"] = state["joint3"] = SimpleNamespace(pos=-0.5)
            state["joint1"] = SimpleNamespace(pos=math.radians(deg))
            robot._motor_map = {n: SimpleNamespace(request_feedback=lambda: None,
                                                   get_state=lambda n=n: state[n])
                                for n in robot.joint_names}
            robot._ctrl_map = {}
            if ok:
                robot.check_joint_limits(timeout=0.0)
            else:
                with self.assertRaises(JointLimitError) as e:
                    robot.check_joint_limits(timeout=0.0)
                self.assertIn("joint1", str(e.exception))


if __name__ == "__main__":
    unittest.main()
