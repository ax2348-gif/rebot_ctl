"""Signal-driven emergency stop that does not depend on the main thread.

The safety model: every motor has a 100 ms CAN watchdog (checked on connect), so the
fastest reliable stop is simply to stop sending commands. A control loop that checks
``EStop.tripped`` on every tick and sends nothing once it is set lets the watchdog disable
the motors within 100 ms, whatever else is happening.

Python runs signal handlers only in the main thread, between bytecodes, so a plain handler
would be delayed while the main thread is blocked in a native call (a register read, the
native shutdown). Instead the C-level handler writes the signal number to a pipe
(signal.set_wakeup_fd) and a watcher thread trips the stop as soon as it arrives. The
Python-level handler then raises KeyboardInterrupt, so blocking calls such as input()
unwind and the main thread runs its cleanup.

Escalation: a second signal, or a shutdown still running ``grace_s`` after the stop
tripped, ends the process with os._exit (after the registered ``on_hard_exit`` callbacks,
e.g. restoring the terminal); nothing is commanded by then, so the watchdog completes
the stop.
"""
from __future__ import annotations

import os
import signal
import threading
from typing import Callable, List, Optional

STOP_SIGNALS = tuple(s for s in (getattr(signal, n, None) for n in
                                 ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGTSTP"))
                     if s is not None)


class EStop:
    def __init__(self, grace_s: float = 3.0):
        self.grace_s = grace_s
        self.reason: Optional[str] = None
        self._tripped = threading.Event()
        self._signals = 0
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._hard_exit_hooks: List[Callable[[], None]] = []
        self._previous: dict = {}
        self._old_wakeup_fd: Optional[int] = None
        self._rfd = self._wfd = -1
        self._installed = False
        self._raise = True

    # -- state -------------------------------------------------------------

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    def wait(self, timeout: float) -> bool:
        return self._tripped.wait(timeout)

    def trip(self, reason: str) -> None:
        """Stop commanding the motors (idempotent; the first reason is kept)."""
        with self._lock:
            if self._tripped.is_set():
                return
            self.reason = reason
            self._tripped.set()
            if self.grace_s is not None:
                self._timer = threading.Timer(self.grace_s, self._hard_exit,
                                              args=(1, f"shutdown took over {self.grace_s:g} s"))
                self._timer.daemon = True
                self._timer.start()

    def on_hard_exit(self, fn: Callable[[], None]) -> None:
        self._hard_exit_hooks.append(fn)

    def _hard_exit(self, code: int, why: str) -> None:
        for fn in self._hard_exit_hooks:
            try:
                fn()
            except Exception:
                pass
        try:
            os.write(2, f"\n[e-stop] {why}; exiting now (motor watchdog stops the arm).\n".encode())
        finally:
            os._exit(code)

    # -- signals -----------------------------------------------------------

    def install(self) -> "EStop":
        """Install handlers for STOP_SIGNALS. Must be called from the main thread."""
        self._rfd, self._wfd = os.pipe()
        os.set_blocking(self._wfd, False)
        self._old_wakeup_fd = signal.set_wakeup_fd(self._wfd, warn_on_full_buffer=False)
        for sig in STOP_SIGNALS:
            self._previous[sig] = signal.signal(sig, self._python_handler)
        threading.Thread(target=self._watch, name="estop-watcher", daemon=True).start()
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        for sig, prev in self._previous.items():
            signal.signal(sig, prev)
        signal.set_wakeup_fd(self._old_wakeup_fd if self._old_wakeup_fd is not None else -1)
        if self._timer is not None:
            self._timer.cancel()
        os.close(self._wfd)  # ends the watcher's read

    def _watch(self) -> None:
        while True:
            try:
                data = os.read(self._rfd, 64)
            except OSError:
                return
            if not data:
                return
            for signum in data:
                self._signals += 1
                name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
                self.trip(f"{name} received")
                if self._signals >= 2:
                    self._hard_exit(130, f"second signal ({name})")

    def quiet(self) -> None:
        """Stop raising KeyboardInterrupt (e.g. during cleanup); signals still trip and escalate."""
        self._raise = False

    def _python_handler(self, signum, frame) -> None:
        # The watcher has already tripped the stop; unwind the main thread.
        if self._raise:
            raise KeyboardInterrupt
