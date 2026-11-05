#!/usr/bin/env python3
"""Store the CAN watchdog (CAN_TIMEOUT_MS, 100 ms) in every Damiao motor's flash.

Kept for existing instructions; the implementation is `rebot_setup --store-watchdog`
(reBotArm_control_py/cli/setup.py), which documents the procedure.

Usage::

    python tools/persist_can_timeout.py [hardware.yaml]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.cli.setup import store_watchdog

if __name__ == "__main__":
    sys.exit(store_watchdog(sys.argv[1] if len(sys.argv) > 1 else None))
