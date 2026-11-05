"""Exercise tools/persist_can_timeout.py against fake motors (no hardware)."""
import contextlib
import io
import runpy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "persist_can_timeout.py"


class FakeMotor:
    def __init__(self, status=0, sticky_write=True):
        self.status = status
        self.sticky_write = sticky_write  # False: writes don't take effect
        self.regs = {9: 0}
        self.writes = []
        self.stores = 0

    def request_feedback(self):
        pass

    def get_state(self):
        return SimpleNamespace(status_code=self.status)

    def get_register_u32(self, rid, timeout_ms):
        return self.regs.get(rid, 1)

    def get_register_f32(self, rid, timeout_ms):
        return 1.0

    def write_register_u32(self, rid, value):
        self.writes.append((rid, value))
        if self.sticky_write:
            self.regs[rid] = value

    def store_parameters(self):
        self.stores += 1

    def close(self):
        pass


class PersistCanTimeoutTests(unittest.TestCase):
    def run_tool(self, motors, answer="STORE"):
        it = iter(motors)
        ctrl = Mock()
        ctrl.add_damiao_motor.side_effect = lambda *a: next(it)
        out = io.StringIO()
        # The config names the arm by serial_by_id; pretend it is attached.
        with patch("motorbridge.Controller.from_dm_serial", return_value=ctrl), \
                patch("reBotArm_control_py.cli.setup.resolve_channel", return_value="/dev/ttyACM1"), \
                patch("builtins.input", return_value=answer), \
                patch("sys.argv", [str(SCRIPT), "rebotarm_dm.yaml"]), \
                contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                runpy.run_path(str(SCRIPT), run_name="__main__")
        ctrl.enable_all.assert_not_called()
        ctrl.shutdown.assert_not_called()
        return cm.exception.code, out.getvalue()

    def test_stores_100ms_on_every_motor(self):
        motors = [FakeMotor() for _ in range(7)]
        code, out = self.run_tool(motors)
        self.assertEqual(code, 0)
        for m in motors:
            self.assertEqual(m.writes, [(9, 2000)])
            self.assertEqual(m.stores, 1)
        self.assertIn("Power-cycle the arm", out)

    def test_nothing_written_without_confirmation(self):
        motors = [FakeMotor() for _ in range(7)]
        code, out = self.run_tool(motors, answer="yes")
        self.assertEqual(code, 1)
        for m in motors:
            self.assertEqual((m.writes, m.stores), ([], 0))

    def test_aborts_if_any_motor_is_enabled(self):
        motors = [FakeMotor() for _ in range(7)]
        motors[3].status = 1
        code, out = self.run_tool(motors)
        self.assertEqual(code, 1)
        self.assertIn("joint4: status 0x1", out)
        for m in motors:
            self.assertEqual((m.writes, m.stores), ([], 0))

    def test_no_store_if_a_write_does_not_read_back(self):
        motors = [FakeMotor() for _ in range(7)]
        motors[5].sticky_write = False
        code, out = self.run_tool(motors)
        self.assertEqual(code, 1)
        self.assertIn("joint6: TIMEOUT write did not read back", out)
        for m in motors:
            self.assertEqual(m.stores, 0)


if __name__ == "__main__":
    unittest.main()
