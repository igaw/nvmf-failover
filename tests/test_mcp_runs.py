import json
from pathlib import Path

import pytest

from nvmf_failover.mcp.runs import RunError, Runs
from nvmf_failover.system import CommandError

from conftest import FakeSystem


def command(rundir: Path, options: list[str]) -> list[str]:
    return ["nvmf-failover", "run", "--run-dir", str(rundir), *options]


@pytest.fixture
def runs(system: FakeSystem, tmp_path: Path) -> Runs:
    return Runs(system, tmp_path / "runs", command, clock=lambda: 1_800_000_000.0)


def unit_state(system: FakeSystem, unit: str, state: str) -> None:
    system.answers[f"systemctl show --property=ActiveState --value {unit}"] = (
        state + "\n"
    )


def test_start_spawns_a_unit(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("/s/flap-short.toml"), ["--io", "auto"])

    run_id = out["id"]
    assert run_id.endswith("-flap-short")
    assert out["unit"] == f"nvmf-failover-{run_id}"
    argv = system.commands[-1]
    assert argv[0] == "systemd-run" and f"--unit={out['unit']}" in argv
    assert "--collect" in argv and "--property=KillMode=mixed" in argv
    rundir = runs.out.resolve() / run_id
    assert argv[argv.index("--") + 1 :] == command(rundir, ["--io", "auto"])
    meta = json.loads((rundir / "run.json").read_text())
    assert meta["scenario"] == "/s/flap-short.toml"
    assert meta["options"] == ["--io", "auto"]


def test_one_run_at_a_time(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("a.toml"), [])
    unit_state(system, out["unit"], "active")

    with pytest.raises(RunError, match="still running"):
        runs.start(Path("b.toml"), [])


def test_a_failed_spawn_leaves_no_run(runs: Runs, system: FakeSystem) -> None:
    def fail(argv: list[str], check: bool = True) -> str:
        raise CommandError(argv, 1, "no systemd")

    system.run = fail  # type: ignore[method-assign]

    with pytest.raises(CommandError):
        runs.start(Path("a.toml"), [])
    assert runs.ids() == []


@pytest.mark.parametrize(
    ("unit", "summary", "state"),
    [
        ("active", None, "running"),
        ("deactivating", {"passed": True}, "running"),
        ("inactive", {"passed": True, "interrupted": False}, "passed"),
        ("inactive", {"passed": False, "interrupted": False}, "failed"),
        ("inactive", {"passed": False, "interrupted": True}, "stopped"),
        ("inactive", None, "crashed"),
    ],
)
def test_state(
    runs: Runs,
    system: FakeSystem,
    unit: str,
    summary: dict[str, object] | None,
    state: str,
) -> None:
    out = runs.start(Path("a.toml"), [])
    unit_state(system, out["unit"], unit)
    if summary is not None:
        (Path(out["dir"]) / "summary.json").write_text(json.dumps(summary))

    assert runs.state(out["id"]) == state


def test_status_shows_the_round_and_events(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("a.toml"), [])
    unit_state(system, out["unit"], "active")
    (Path(out["dir"]) / "events.log").write_text(
        "      0.0s step     round 1/50: host3\n"
        "     10.0s path     ctrl x: live -> resetting\n"
        "     90.0s step     round 2/50: host4\n"
    )

    status = runs.status(out["id"], lines=2)

    assert status["state"] == "running"
    assert status["round"] == {"current": 2, "of": 50}
    assert status["events"][-1].endswith("round 2/50: host4")
    assert len(status["events"]) == 2


def test_status_of_a_crash_has_the_console(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("a.toml"), [])
    (Path(out["dir"]) / "console.log").write_text("Traceback\nBoom\n")

    status = runs.status(out["id"])

    assert status["state"] == "crashed"
    assert status["console"] == ["Traceback", "Boom"]


def test_stop(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("a.toml"), [])
    unit_state(system, out["unit"], "active")

    assert runs.stop(out["id"])["state"] == "stopping"
    assert system.commands[-1] == ["systemctl", "stop", "--no-block", out["unit"]]


def test_stop_a_finished_run(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("a.toml"), [])
    with pytest.raises(RunError, match="not running"):
        runs.stop(out["id"])


def test_report(runs: Runs, system: FakeSystem) -> None:
    out = runs.start(Path("a.toml"), [])
    with pytest.raises(RunError, match="crashed"):
        runs.report(out["id"])
    (Path(out["dir"]) / "summary.json").write_text(
        json.dumps({"passed": True, "interrupted": False, "lines": ["result: PASS"]})
    )

    report = runs.report(out["id"])

    assert report["state"] == "passed" and report["lines"] == ["result: PASS"]


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x y"])
def test_bad_ids(runs: Runs, bad: str) -> None:
    with pytest.raises(RunError):
        runs.status(bad)
