"""Faults put down by hand, each brought back up by a dead-man timer.

A fault is held for a number of seconds; then the timer puts it up, also
when nobody asks for it any more. The held faults are written to a state
file at every change, so a server that was killed (SIGKILL, OOM) puts
them up when it starts again.

A restore builds the fault again from its spec, with fresh connections:
a cluster or switch session may have timed out while the fault was held,
and the spec needs no lookup of what the fault took away.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..faults import Fault

# Seconds until a failed restore is tried again
RETRY = 30.0
# Events kept for status
HISTORY = 50

OpenFault = Callable[[dict[str, Any]], AbstractContextManager[Fault]]


@dataclass
class Hold:
    key: str
    raw: dict[str, Any]
    since: float  # wall clock
    until: float  # wall clock: put it up then
    error: str = ""  # why the last restore failed

    def json(self, now: float) -> dict[str, Any]:
        return {
            "key": self.key,
            "since": iso(self.since),
            "restore_at": iso(self.until),
            "seconds_left": round(max(0.0, self.until - now), 1),
            "error": self.error,
        }


@dataclass
class Holds:
    """The held faults; every method may be called from any thread."""

    open_fault: OpenFault
    state: Path
    clock: Callable[[], float] = time.time
    holds: dict[str, Hold] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Event()

    # --- actions ----------------------------------------------------------

    def down(self, raw: dict[str, Any], seconds: float) -> dict[str, Any]:
        """Put the fault down for @seconds; again: re-arm its timer."""
        with self._lock, self.open_fault(raw) as fault:
            start = self.clock()
            held = self.holds.get(fault.key)
            # remember it before: a half-applied fault must be undone too
            # restored from its spec: what it was found by may be gone then
            self.holds[fault.key] = Hold(
                fault.key,
                fault.spec(),
                held.since if held else start,
                start + seconds,
            )
            self._save()
            try:
                fault.down()
            except Exception:
                self._restore(self.holds[fault.key], "down failed")
                raise
            # the hold starts when the fault is in place: putting an lpfc
            # link down takes seconds
            done = self.clock()
            self.holds[fault.key].until = done + seconds
            self._save()
            self._event(f"down {fault.key} for {seconds:g}s")
            self._wake.set()
            return {
                "key": fault.key,
                "action": "down",
                "applied_at": iso(start),
                "completed_at": iso(done),
                "took_seconds": round(done - start, 3),
                "restore_at": iso(done + seconds),
            }

    def up(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Put the fault up now, held or not."""
        with self._lock, self.open_fault(raw) as fault:
            start = self.clock()
            fault.up()
            done = self.clock()
            if self.holds.pop(fault.key, None):
                self._save()
            self._event(f"up {fault.key}")
            return {
                "key": fault.key,
                "action": "up",
                "applied_at": iso(start),
                "completed_at": iso(done),
                "took_seconds": round(done - start, 3),
            }

    def bounce(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self.open_fault(raw) as fault:
            start = self.clock()
            fault.bounce()
            done = self.clock()
            self._event(f"bounce {fault.key}")
            return {
                "key": fault.key,
                "action": "bounce",
                "applied_at": iso(start),
                "completed_at": iso(done),
                "took_seconds": round(done - start, 3),
            }

    def restore_all(self, why: str = "restore_all") -> list[str]:
        """Put up every held fault; the keys of those that failed."""
        with self._lock:
            for hold in list(self.holds.values()):
                self._restore(hold, why)
            return [h.key for h in self.holds.values()]

    def tick(self) -> float | None:
        """Put up the faults whose time is up; when the next one is due."""
        with self._lock:
            now = self.clock()
            for hold in list(self.holds.values()):
                if hold.until <= now:
                    self._restore(hold, f"held {now - hold.since:.0f}s")
            return min((h.until for h in self.holds.values()), default=None)

    def recover(self) -> list[str]:
        """Put up what a previous server left down; the keys that failed."""
        try:
            data = json.loads(self.state.read_text())
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as err:
            self._event(f"state file {self.state} unreadable: {err}")
            return []
        with self._lock:
            for item in data if isinstance(data, list) else []:
                hold = Hold(**item)
                self.holds[hold.key] = hold
            return self.restore_all("left down by a previous server")

    # --- state ------------------------------------------------------------

    def active(self) -> list[str]:
        with self._lock:
            return sorted(self.holds)

    def json(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            return {
                "held": [h.json(now) for h in self.holds.values()],
                "history": list(self.history),
            }

    def run_timer(self, stop: threading.Event) -> None:
        """The dead-man timer: tick until @stop is set."""
        while not stop.is_set():
            due = self.tick()
            self._wake.clear()
            wait = 1.0 if due is None else max(0.0, min(1.0, due - self.clock()))
            self._wake.wait(wait)

    def _restore(self, hold: Hold, why: str) -> None:
        try:
            with self.open_fault(hold.raw) as fault:
                fault.up()
        except Exception as err:  # noqa: BLE001 -- keep it, try again later
            hold.error = str(err)
            hold.until = self.clock() + RETRY
            self._event(f"restore {hold.key} ({why}) failed: {err}")
        else:
            del self.holds[hold.key]
            self._event(f"restored {hold.key} ({why})")
        self._save()

    def _event(self, text: str) -> None:
        self.history.append({"time": iso(self.clock()), "text": text})
        del self.history[:-HISTORY]

    def _save(self) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(h) for h in self.holds.values()]))
        with contextlib.suppress(OSError), tmp.open() as f:
            os.fsync(f.fileno())
        tmp.replace(self.state)


def iso(t: float) -> str:
    """Wall clock time, ISO 8601 with milliseconds and the UTC offset, as
    nvme-mcp's nvme_watch reports its start: for lining up logs."""
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")
