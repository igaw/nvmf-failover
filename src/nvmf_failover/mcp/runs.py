"""Scenario runs in their own systemd units, watched through their logs.

A run is "nvmf-failover run --run-dir DIR", started with systemd-run. It
doesn't depend on the server: a night run goes on when the ssh session of
the client drops, and the server reads its state from DIR and the unit
when asked. The run restores its own faults (also on systemctl stop, which
sends SIGTERM) and writes summary.json when it ends.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..system import CommandError, System
from .holds import iso

RUNNING = ("active", "activating", "deactivating", "reloading")
_ROUND = re.compile(r"\bround (\d+)(?:/| of )(\d+)")


class RunError(RuntimeError):
    """A run can't be started, or there is no such run."""


@dataclass
class Runs:
    """The runs in @out; @command is the argv of a run, given its dir."""

    system: System
    out: Path
    command: Callable[[Path, list[str]], list[str]]
    clock: Callable[[], float] = time.time

    def start(self, scenario: Path, options: list[str]) -> dict[str, Any]:
        active = self.active()
        if active:
            raise RunError(f"run {active[0]} is still running")
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.clock()))
        run_id = f"{stamp}-{re.sub(r'[^A-Za-z0-9_-]', '_', scenario.stem)}"
        rundir = self.out.resolve() / run_id
        rundir.mkdir(parents=True)
        unit = f"nvmf-failover-{run_id}"
        meta = {
            "id": run_id,
            "unit": unit,
            "scenario": str(scenario),
            "options": options,
            "started": iso(self.clock()),
            "started_epoch": self.clock(),
        }
        (rundir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        argv = [
            "systemd-run",
            f"--unit={unit}",
            "--collect",
            "--quiet",
            # SIGTERM to the run only; it stops fio itself
            "--property=KillMode=mixed",
            "--property=TimeoutStopSec=900",
            f"--property=StandardOutput=file:{rundir}/console.log",
            "--property=StandardError=inherit",
            "--",
            *self.command(rundir, options),
        ]
        try:
            self.system.run(argv)
        except CommandError:
            (rundir / "run.json").unlink()
            rundir.rmdir()
            raise
        return {"id": run_id, "unit": unit, "dir": str(rundir)}

    def ids(self) -> list[str]:
        """The runs started through the server, newest first."""
        if not self.out.is_dir():
            return []
        return sorted(
            (d.name for d in self.out.iterdir() if (d / "run.json").is_file()),
            reverse=True,
        )

    def active(self) -> list[str]:
        return [i for i in self.ids() if self.state(i) == "running"]

    def state(self, run_id: str) -> str:
        """running, passed, failed, stopped or crashed."""
        rundir, meta = self._meta(run_id)
        unit = self.system.run(
            ["systemctl", "show", "--property=ActiveState", "--value", meta["unit"]],
            check=False,
        ).strip()
        summary = self._summary(rundir)
        if unit in RUNNING:
            return "running"
        if summary is None:
            return "crashed"
        if summary.get("interrupted"):
            return "stopped"
        return "passed" if summary.get("passed") else "failed"

    def status(self, run_id: str, lines: int = 20) -> dict[str, Any]:
        rundir, meta = self._meta(run_id)
        state = self.state(run_id)
        events = _tail(rundir / "events.log", 10_000)
        rounds = [m for e in events if (m := _ROUND.search(e))]
        out: dict[str, Any] = {
            "id": run_id,
            "state": state,
            "scenario": meta["scenario"],
            "started": meta["started"],
            "elapsed_seconds": round(self._end(rundir) - meta["started_epoch"]),
            "round": (
                {"current": int(rounds[-1][1]), "of": int(rounds[-1][2])}
                if rounds
                else None
            ),
            "events": events[-lines:],
        }
        if state == "crashed":
            out["console"] = _tail(rundir / "console.log", lines)
        return out

    def stop(self, run_id: str) -> dict[str, Any]:
        _, meta = self._meta(run_id)
        if self.state(run_id) != "running":
            raise RunError(f"run {run_id} is not running")
        # the run restores its faults on SIGTERM; don't wait for it here
        self.system.run(["systemctl", "stop", "--no-block", meta["unit"]])
        return {"id": run_id, "state": "stopping"}

    def report(self, run_id: str) -> dict[str, Any]:
        rundir, meta = self._meta(run_id)
        summary = self._summary(rundir)
        state = self.state(run_id)
        if summary is None:
            raise RunError(
                f"run {run_id} is {state}"
                + (", ask run_status for the console" if state == "crashed" else "")
            )
        return {
            "id": run_id,
            "state": state,
            "scenario": meta["scenario"],
            "options": meta["options"],
            "started": meta["started"],
            **summary,
        }

    def _meta(self, run_id: str) -> tuple[Path, dict[str, Any]]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise RunError(f"{run_id!r}: not a run id")
        rundir = self.out / run_id
        try:
            meta: dict[str, Any] = json.loads((rundir / "run.json").read_text())
        except (OSError, ValueError) as err:
            raise RunError(f"no run {run_id}") from err
        return rundir, meta

    def _summary(self, rundir: Path) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = json.loads((rundir / "summary.json").read_text())
        except (OSError, ValueError):
            return None
        return data

    def _end(self, rundir: Path) -> float:
        summary = rundir / "summary.json"
        return summary.stat().st_mtime if summary.exists() else self.clock()


def _tail(path: Path, lines: int) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []
