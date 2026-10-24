"""Coordination between the bots that share one exchange account.

TradeLock: an exclusive file lock (bot/state/trade.lock). A bot holds it while it sends orders and cleans up
(cancels), so (1) the ETF-arb bot can measure its fills from the change in positions and (2) one bot's
cancel-all never cancels the other bot's live orders.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parent / "state" / "trade.lock"


class TradeLock:
    def __init__(self, timeout: float = 0.5, path: Path = LOCK):
        self.timeout, self.path, self.fd = timeout, path, None

    def acquire(self) -> bool:
        self.path.parent.mkdir(exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        end = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= end:
                    os.close(self.fd)
                    self.fd = None
                    return False
                time.sleep(0.01)

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
