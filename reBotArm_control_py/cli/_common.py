"""Helpers shared by the rebot_* command-line tools."""
from __future__ import annotations

import argparse
import socket
import time
from typing import Dict, Iterable, List, Optional

from motorbridge import Controller

# Damiao feedback-frame ERR nibble (raw status_code)
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


def describe_status(vendor: str, code: int) -> str:
    if vendor == "damiao":
        return f"{DM_STATUS.get(code, 'unknown')} (0x{code:X})"
    return f"status 0x{code:X}"


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("hardware_yaml", nargs="?", default=None,
                   help="hardware config file in config/ (default: hardware_yaml in "
                        "config/rebotarm.yaml)")
    p.add_argument("--channel", default=None,
                   help="serial port or CAN interface, overriding the config's channel "
                        "(e.g. /dev/ttyACM0 on the robot)")
    return p


def confirm(prompt: str, word: str) -> bool:
    """Ask the user to type ``word``; anything else (or EOF) declines."""
    try:
        return input(f"{prompt} Type {word} to continue: ").strip() == word
    except EOFError:
        return False


def make_controller(channel: str) -> Controller:
    # Same transport selection as RebotArm._make_controller
    if channel.startswith("/dev/"):  # any device path is the serial bridge; CAN interfaces are bare names (can0)
        return Controller.from_dm_serial(channel, 921600)
    return Controller(channel)


def wait_for_feedback(ctrl, motors: Dict[str, object], timeout: float = 2.0) -> Dict[str, object]:
    """Request feedback from every motor until all have answered or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for m in motors.values():
            m.request_feedback()
        ctrl.poll_feedback_once()
        states = {n: m.get_state() for n, m in motors.items()}
        if all(st is not None for st in states.values()):
            return states
        time.sleep(0.02)
    return {n: m.get_state() for n, m in motors.items()}


def lan_addresses() -> List[str]:
    """This machine's non-loopback IPv4 addresses (best effort, no packets sent)."""
    found: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    try:
        # Connecting a UDP socket only selects the outgoing interface; nothing is sent.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    return list(dict.fromkeys(a for a in found if not a.startswith("127.")))


def external_urls(port: int, path: str = "/static/") -> List[str]:
    hosts: Iterable[str] = [*lan_addresses(), f"{socket.gethostname()}.local"]
    return [f"http://{h}:{port}{path}" for h in hosts]


def fmt_deg(values) -> str:
    return "  ".join(f"{v * 57.29577951308232:+8.2f}" for v in values)


def print_joint_table(names: List[str], rows: Dict[str, List[str]]) -> None:
    w = 2 + max(7, *(len(n) for n in names), *(len(c) for cells in rows.values() for c in cells))
    print(f"{'':<14}" + "".join(f"{n:>{w}}" for n in names))
    for label, cells in rows.items():
        print(f"{label:<14}" + "".join(f"{c:>{w}}" for c in cells))


def optional(value: Optional[float], fmt: str) -> str:
    return "—" if value is None else format(value, fmt)
