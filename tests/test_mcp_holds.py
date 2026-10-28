import contextlib
import json
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from nvmf_failover.faults import Fault, FaultError
from nvmf_failover.mcp.holds import RETRY, Holds, iso


class FakeFault(Fault):
    kind = "fake"

    def __init__(self, name: str, log: list[str], broken: set[str]) -> None:
        self.name, self.log, self.broken = name, log, broken

    @property
    def target(self) -> str:
        return self.name

    def spec(self) -> dict[str, Any]:
        return {"fault": self.kind, "host": self.name}

    def down(self) -> None:
        self.log.append(f"down {self.name}")
        if f"down {self.name}" in self.broken:
            raise FaultError("down broke")

    def up(self) -> None:
        self.log.append(f"up {self.name}")
        if f"up {self.name}" in self.broken:
            raise FaultError("up broke")

    def bounce(self) -> None:
        self.log.append(f"bounce {self.name}")


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def world(tmp_path: Path) -> tuple[Holds, list[str], set[str], Clock]:
    log: list[str] = []
    broken: set[str] = set()
    clock = Clock()

    @contextlib.contextmanager
    def open_fault(raw: dict[str, Any]) -> Iterator[Fault]:
        yield FakeFault(raw["host"], log, broken)

    return Holds(open_fault, tmp_path / "state.json", clock), log, broken, clock


def raw(host: str) -> dict[str, Any]:
    return {"fault": "fake", "host": host, "action": "down"}


def saved(holds: Holds) -> list[str]:
    return [h["key"] for h in json.loads(holds.state.read_text())]


def test_down_holds_until_the_timer(world: Any) -> None:
    holds, log, _, clock = world

    out = holds.down(raw("host3"), 20)

    assert log == ["down host3"]
    assert out["key"] == "fake:host3" and out["action"] == "down"
    assert saved(holds) == ["fake:host3"]
    clock.now += 19
    assert holds.tick() == clock.now + 1
    assert log == ["down host3"]
    clock.now += 1
    assert holds.tick() is None
    assert log == ["down host3", "up host3"]
    assert holds.active() == [] and saved(holds) == []
    assert "restored fake:host3 (held 20s)" in holds.history[-1]["text"]


def test_down_again_rearms_the_timer(world: Any) -> None:
    holds, _, _, clock = world
    holds.down(raw("host3"), 20)
    since = holds.holds["fake:host3"].since
    clock.now += 15

    holds.down(raw("host3"), 20)

    assert holds.holds["fake:host3"].until == clock.now + 20
    assert holds.holds["fake:host3"].since == since


def test_up_ends_the_hold(world: Any) -> None:
    holds, log, _, clock = world
    holds.down(raw("host3"), 20)

    holds.up(raw("host3"))
    clock.now += 30
    holds.tick()

    assert log == ["down host3", "up host3"]
    assert saved(holds) == []


def test_a_failed_down_is_undone(world: Any) -> None:
    holds, log, broken, _ = world
    broken.add("down host3")

    with pytest.raises(FaultError, match="down broke"):
        holds.down(raw("host3"), 20)

    assert log == ["down host3", "up host3"]
    assert holds.active() == []


def test_a_failed_restore_is_kept_and_retried(world: Any) -> None:
    holds, log, broken, clock = world
    holds.down(raw("host3"), 20)
    broken.add("up host3")
    clock.now += 20

    assert holds.tick() == clock.now + RETRY

    hold = holds.json()["held"][0]
    assert hold["error"] == "up broke"
    assert saved(holds) == ["fake:host3"]
    broken.clear()
    clock.now += RETRY
    holds.tick()
    assert holds.active() == []
    assert log.count("up host3") == 2


def test_restore_all(world: Any) -> None:
    holds, log, broken, _ = world
    holds.down(raw("host3"), 20)
    holds.down(raw("host4"), 20)
    broken.add("up host4")

    assert holds.restore_all() == ["fake:host4"]
    assert log[-2:] == ["up host4", "up host3"] or log[-2:] == [
        "up host3",
        "up host4",
    ]


def test_recover_puts_up_what_a_killed_server_left(world: Any) -> None:
    holds, log, _, clock = world
    holds.down(raw("host3"), 600)
    # the server is killed: a new one finds the state file
    fresh = Holds(holds.open_fault, holds.state, clock)

    assert fresh.recover() == []

    assert log == ["down host3", "up host3"]
    assert saved(fresh) == []
    assert "previous server" in fresh.history[-1]["text"]


def test_recover_without_state(world: Any) -> None:
    holds, log, _, _ = world
    assert holds.recover() == [] and log == []


def test_the_timer_thread_restores(world: Any, tmp_path: Path) -> None:
    log: list[str] = []

    @contextlib.contextmanager
    def open_fault(raw: dict[str, Any]) -> Iterator[Fault]:
        yield FakeFault(raw["host"], log, set())

    holds = Holds(open_fault, tmp_path / "real.json")
    stop = threading.Event()
    thread = threading.Thread(target=holds.run_timer, args=(stop,))
    thread.start()
    try:
        holds.down(raw("host3"), 0.2)
        for _ in range(50):
            if not holds.active():
                break
            threading.Event().wait(0.05)
    finally:
        stop.set()
        thread.join()
    assert log == ["down host3", "up host3"]


def test_the_hold_starts_when_the_fault_is_in_place(tmp_path: Path) -> None:
    clock = Clock()

    class SlowFault(FakeFault):
        def down(self) -> None:
            clock.now += 5  # an lpfc link takes seconds to go down

    @contextlib.contextmanager
    def open_fault(raw: dict[str, Any]) -> Iterator[Fault]:
        yield SlowFault(raw["host"], [], set())

    holds = Holds(open_fault, tmp_path / "state.json", clock)
    start = clock.now

    out = holds.down(raw("host3"), 20)

    assert out["took_seconds"] == 5
    assert holds.holds["fake:host3"].until == start + 25
    assert json.loads(holds.state.read_text())[0]["until"] == start + 25


def test_iso_is_iso_8601_with_the_offset() -> None:
    text = iso(1_800_000_000.25)
    # the form nvme-mcp's nvme_watch reports: 2026-10-28T13:02:56.384+01:00
    assert text[-3] == ":" and text[19:23] == ".250"
    parsed = datetime.fromisoformat(iso(1_800_000_000.25))
    assert parsed.tzinfo is not None
    assert parsed.timestamp() == 1_800_000_000.25
