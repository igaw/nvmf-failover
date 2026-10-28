import json
from pathlib import Path

import pytest

from nvmf_failover.cli import main

from conftest import FakeSystem, fc_addr


def test_status_lists_controllers(
    system: FakeSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    system.add_ctrl(
        "nvme5", fc_addr("0x201900a09890f5bf"), paths={"nvme1c5n1": "optimized"}
    )
    system.add_ctrl("nvme6", fc_addr("0x201800a09890f5bf"))

    assert main(["status", "0x2019"], system) == 0

    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert out[0].startswith("nvme5") and "nvme1c5n1:optimized" in out[0]


def test_fault_by_hand(
    system: FakeSystem, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    system.add_scsi_host("host3", "lpfc")
    cfg = tmp_path / "none.toml"
    cfg.write_text("")
    cfg.chmod(0o600)

    assert main(["-c", str(cfg), "fault", "fc-link", "host3", "down"], system) == 0

    assert system.path("/sys/class/scsi_host/host3/link_state").read_text() == "down"
    assert capsys.readouterr().out == "down fc-link:host3\n"


def test_fault_error_exits_2(
    system: FakeSystem, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    system.add_scsi_host("host2", "qla2xxx")
    cfg = tmp_path / "none.toml"
    cfg.write_text("")
    cfg.chmod(0o600)

    assert main(["-c", str(cfg), "fault", "fc-link", "host2", "down"], system) == 2
    assert "can't hold its link down" in capsys.readouterr().err


def test_run_restores_and_passes(system: FakeSystem, tmp_path: Path) -> None:
    system.add_scsi_host("host3", "lpfc")
    system.add_ctrl("nvme5", fc_addr("0x201900a09890f5bf"))
    cfg = tmp_path / "none.toml"
    cfg.write_text("")
    cfg.chmod(0o600)
    scen = tmp_path / "bounce.toml"
    scen.write_text(
        'name = "link"\n'
        '[[steps]]\nfault = "fc-link"\nhost = "host3"\n'
        'action = "down"\nhold = 0\n'
    )
    out = tmp_path / "runs"

    rc = main(
        ["-c", str(cfg), "run", str(scen), "--out", str(out), "--no-journal"], system
    )

    # the scenario left the link down: the run puts it back up
    assert system.path("/sys/class/scsi_host/host3/link_state").read_text() == "up"
    assert rc == 0
    (run,) = out.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["passed"] is True
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    texts = [e["text"] for e in events]
    assert "down fc-link:host3" in texts and "restore fc-link:host3" in texts


def test_link_fault_opens_no_cluster_shell(
    system: FakeSystem, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(
        "nvmf_failover.context.OntapShell", lambda cfg: opened.append(cfg)
    )
    system.add_scsi_host("host3", "lpfc")
    cfg = tmp_path / "ontap.toml"
    cfg.write_text('[ontap]\nhost = "cluster"\nuser = "admin"\npassword = "pw"\n')
    cfg.chmod(0o600)

    assert main(["-c", str(cfg), "fault", "fc-link", "host3", "down"], system) == 0

    assert opened == []
