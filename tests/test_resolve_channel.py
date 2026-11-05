"""resolve_channel: --channel override, Linux by-id path, macOS serial-number match."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reBotArm_control_py.actuator import rebotarm
from reBotArm_control_py.actuator.rebotarm import SerialPortNotFoundError, resolve_channel

BY_ID = "usb-HDSC_CDC_Device_00000000050C-if00"
CFG = {"channel": "/dev/tty.usbmodem00000000050C1", "serial_by_id": BY_ID}


def ports(*entries):
    return patch("serial.tools.list_ports.comports",
                 return_value=[SimpleNamespace(device=d, serial_number=s) for d, s in entries])


class ResolveChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # No by-id directory unless a test creates it
        self.by_id_dir = patch.object(rebotarm, "SERIAL_BY_ID_DIR", self.root / "by-id")
        self.by_id_dir.start()

    def tearDown(self):
        self.by_id_dir.stop()
        self.tmp.cleanup()

    def test_override_wins(self):
        self.assertEqual(resolve_channel(CFG, "/dev/ttyACM9"), "/dev/ttyACM9")

    def test_plain_channel_without_by_id(self):
        self.assertEqual(resolve_channel({"channel": "can0", "serial_by_id": None}), "can0")

    def test_linux_by_id_path(self):
        (self.root / "by-id").mkdir()
        (self.root / "by-id" / BY_ID).touch()
        with ports(("/dev/ttyACM0", "OTHER")):
            self.assertEqual(resolve_channel(CFG), str(self.root / "by-id" / BY_ID))

    def test_macos_serial_number_match_prefers_tty(self):
        cu = self.root / "cu.usbmodem00000000050C1"
        tty = self.root / "tty.usbmodem00000000050C1"
        tty.touch()
        with patch.object(rebotarm, "Path", side_effect=lambda p: Path(str(p).replace("/dev/", f"{self.root}/"))):
            with ports(("/dev/cu.usbmodem00000000050C1", "00000000050C"), ("/dev/cu.other", "X")):
                self.assertEqual(resolve_channel(CFG), "/dev/tty.usbmodem00000000050C1")

    def test_macos_match_without_tty_twin_keeps_cu(self):
        with ports(("/dev/cu.usbmodem00000000050C1", "00000000050C")):
            self.assertEqual(resolve_channel(CFG), "/dev/cu.usbmodem00000000050C1")

    def test_missing_device_raises_instead_of_falling_back(self):
        with ports(("/dev/ttyACM0", "3667335B3036")):  # a CANable, not the arm
            with self.assertRaises(SerialPortNotFoundError) as e:
                resolve_channel(CFG)
        self.assertIn("not attached", str(e.exception))

    def test_ambiguous_serial_number_raises(self):
        with ports(("/dev/ttyACM1", "00000000050C"), ("/dev/ttyACM2", "00000000050C")):
            with self.assertRaises(SerialPortNotFoundError):
                resolve_channel(CFG)

    def test_rebotarm_resolves_on_connect_not_construct(self):
        arm = rebotarm.RebotArm("rebotarm_dm.yaml")  # must not touch devices
        with ports():
            if arm._channel_cfg["serial_by_id"]:
                self.assertIn("not attached", arm.channel)
            with patch.object(rebotarm.Controller, "from_dm_serial") as opened:
                if arm._channel_cfg["serial_by_id"]:
                    with self.assertRaises(SerialPortNotFoundError):
                        arm._make_controller("damiao")
                    opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
