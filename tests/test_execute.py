import json
import time
from pathlib import Path

import pytest

from nvmf_failover.cli import main
from nvmf_failover.execute import Observer, execute, expectation_failures
from nvmf_failover.faults import FaultError
from nvmf_failover.load import LoadResult
from nvmf_failover.resolve import Expect, Plan, Round
from nvmf_failover.scenario import Runner
from nvmf_failover.topology import Identity, controllers
from test_model import NETAPP, fc, inventory, world
from test_scenario import Recorder

from conftest import FakeSystem

A = Identity("fc", NETAPP, "t1", "h1")
B = Identity("fc", NETAPP, "t2", "h1")


def test_observer_counts_live_paths_and_replaced(system: FakeSystem) -> None:
    world(system)
    ctrls = {c.name: c for c in controllers(system)}
    obs = Observer({n: ctrls[n].identity for n in ("nvme1", "nvme2", "nvme3")})
    assert obs.min_live == {NETAPP: 3}

    # nvme2 reconnecting: identity hidden, still the same device
    system.path("/sys/class/nvme/nvme2/state").write_text("connecting\n")
    system.path("/sys/class/nvme/nvme2/subsysnqn").write_text("(efault)\n")
    obs(controllers(system))
    assert obs.min_live == {NETAPP: 2} and not obs.replaced

    # nvme3 deleted, then a new device for the same path
    import shutil

    shutil.rmtree(system.path("/sys/class/nvme/nvme3"))
    obs(controllers(system))
    system.add_ctrl("nvme7", fc("0x201800a09890f5bf", "0x21000024ff7fa448"), nqn=NETAPP)
    obs(controllers(system))
    assert obs.min_live == {NETAPP: 1}
    assert obs.replaced == {ctrls["nvme3"].identity}


def plan(rounds: list[Round]) -> Plan:
    return Plan("t", [], rounds, {}, 60.0, Expect())


def test_execute_rounds_and_gaps() -> None:
    log: list[str] = []
    a, b = Recorder("a", log), Recorder("b", log)
    rounds = [
        Round(1, "one", (a, b), ("nvme1",), 10, 5),
        Round(1, "two", (a,), ("nvme1",), 20, None),
    ]
    waits: list[float] = []

    def wait(t: float) -> bool:
        waits.append(t)
        return False

    runner = Runner(sleep=lambda s: log.append(f"sleep {s:g}"), log=lambda t: None)
    unrecovered = execute(
        plan(rounds), runner, lambda s: log.append(f"sleep {s:g}"), wait, lambda t: None
    )
    # the faults of a round are applied at once, in no particular order
    assert sorted(log[:2]) == ["down a", "down b"]
    assert sorted(log[3:5]) == ["up a", "up b"]
    assert log[2:3] + log[5:] == [
        "sleep 10",
        "sleep 5",
        "down a",
        "sleep 20",
        "up a",
    ]
    assert waits == [60.0]
    assert unrecovered == ["round 2 (two)"]


def test_execute_restores_on_failure() -> None:
    log: list[str] = []
    a, boom = Recorder("a", log), Recorder("boom", log, fail_down=True)
    runner = Runner(sleep=lambda s: None, log=lambda t: None)
    with pytest.raises(FaultError):
        execute(
            plan([Round(1, "x", (a, boom), (), 10, 0)]),
            runner,
            lambda s: None,
            lambda t: True,
            lambda t: None,
        )
    assert sorted(log[-2:]) == ["up a", "up boom"]


def test_expectation_failures() -> None:
    obs = Observer({"nvme1": A, "nvme2": B})
    obs.min_live = {NETAPP: 0}
    obs.replaced = {A}
    out = expectation_failures(
        Expect(io_errors=0, paths_during=1, reconnected_by="kernel"),
        obs,
        LoadResult(2, 0, 0, 0),
        ["round 1 (x)"],
        {NETAPP: "dolin-lpfc"},
    )
    assert out == [
        "not recovered after round 1 (x)",
        "dolin-lpfc: down to 0 live paths, expected at least 1",
        "2 I/O errors, expected at most 0",
        "1 controller(s) were deleted and reconnected, "
        "expected the kernel to keep them",
    ]
    obs.replaced = set()
    out = expectation_failures(Expect(reconnected_by="discoverd"), obs, None, [], {})
    assert "didn't exercise" in out[0]


SCENARIO = """\
name = "host port loss"
[paths]
subsystem = "*dolin-lpfc*"
[[phase]]
fail = "host-port"
pick = "each"
duration = "dev_loss_tmo + 1s"
gap = "until-recovered"
[expect]
recovered_within = 5
reconnected_by = "kernel"
"""


def _setup(
    system: FakeSystem, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    world(system)
    rport = system.path("/sys/class/fc_remote_ports/rport-3:0-1")
    rport.mkdir(parents=True)
    (rport / "roles").write_text("NVMe Target\n")
    (rport / "dev_loss_tmo").write_text("0\n")
    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    cfg.chmod(0o600)
    scen = tmp_path / "s.toml"
    scen.write_text(SCENARIO)
    # the tests use no ONTAP: discovery comes from a fixed inventory
    monkeypatch.setattr(
        "nvmf_failover.cli.inventory", lambda cfg, shell, nqn, system: inventory()
    )
    return cfg, scen


def test_plan_prints_rounds(
    system: FakeSystem,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg, scen = _setup(system, tmp_path, monkeypatch)
    assert main(["-c", str(cfg), "plan", str(scen)], system) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "scenario host port loss: 3 paths selected"
    assert out[1] == "timers: dev_loss_tmo 0s"
    assert out[2].startswith(
        "round 1 (phase 1) host-port host2: 1s down, then "
        "until recovered; 1 paths: nvme3"
    )
    assert out[3] == "    pci-unbind:host2"
    assert out[4].startswith("round 2 (phase 1) host-port host3")
    assert out[5] == "    fc-link:host3"


def test_run_described_passes_and_restores(
    system: FakeSystem, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, scen = _setup(system, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = tmp_path / "runs"
    rc = main(
        ["-c", str(cfg), "run", str(scen), "--out", str(out), "--no-journal"], system
    )
    assert system.path("/sys/class/scsi_host/host3/link_state").read_text() == "up"
    (run,) = out.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert rc == 0, summary
    texts = [
        json.loads(line)["text"]
        for line in (run / "events.jsonl").read_text().splitlines()
    ]
    assert "round 2/2: host-port host3" in texts
    assert "down fc-link:host3" in texts and "up fc-link:host3" in texts


class Slow(Recorder):
    def down(self) -> None:
        time.sleep(0.3)  # an lpfc link takes seconds to go down
        super().down()

    def up(self) -> None:
        time.sleep(0.3)
        super().up()


def test_execute_applies_the_faults_of_a_round_at_once() -> None:
    log: list[str] = []
    faults = tuple(Slow(n, log) for n in "abcd")
    runner = Runner(sleep=lambda s: None, log=lambda t: None)
    start = time.monotonic()

    execute(
        plan([Round(1, "all", faults, (), 0, 0)]),
        runner,
        lambda s: None,
        lambda t: True,
        lambda t: None,
    )

    # four downs and four ups of 0.3 s each, not 2.4 s one after the other
    assert time.monotonic() - start < 1.2
    assert sorted(log) == sorted([f"{a} {n}" for a in ("down", "up") for n in "abcd"])
    assert runner.down == {}
