import fcntl
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import nvmf_failover
from nvmf_failover import mcp
from nvmf_failover.config import Config, OntapConfig
from nvmf_failover.discover import Inventory, Lif, Subsystem
from nvmf_failover.mcp.server import Options, Server

from conftest import FakeSystem, fc_addr

LINK = "/sys/class/scsi_host/host3/link_state"


@pytest.fixture
def opts(tmp_path: Path) -> Options:
    scenarios = tmp_path / "scenarios"
    (scenarios / "steps").mkdir(parents=True)
    (scenarios / "flap.toml").write_text(
        "# One port flaps.\n"
        'name = "flap one port"\n'
        "[[phase]]\n"
        'fail = "host-port"\npick = "one"\nduration = "10s"\n'
    )
    (scenarios / "steps" / "bounce.toml").write_text(
        '[[steps]]\nfault = "fc-lip"\nhost = "host3"\naction = "bounce"\n'
    )
    return Options(
        config=Config(),
        config_path=None,
        scenarios=scenarios,
        out=tmp_path / "runs",
        state=tmp_path / "state" / "held.json",
    )


@pytest.fixture
def host(system: FakeSystem) -> FakeSystem:
    system.add_scsi_host("host3", "lpfc", wwpn="0x100000109b579ef5")
    system.add_scsi_host("host4", "qla2xxx", wwpn="0x100000109b579ef6")
    system.add_pci("host4", "0000:81:00.1", "qla2xxx")
    system.add_ctrl("nvme1", fc_addr("0x201900a09890f5bf"))
    system.add_ctrl("nvme2", fc_addr("0x201800a09890f5bf"))
    system.path("/sys/class/nvme/nvme1/ctrl_loss_tmo").write_text("600\n")
    system.path("/sys/class/nvme/nvme1/kato").write_text("5\n")
    return system


@pytest.fixture
def server(opts: Options, host: FakeSystem) -> Server:
    return Server(opts, host)


def call(server: Server, name: str, **args: Any) -> dict[str, Any]:
    resp = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
    )
    assert resp is not None and "result" in resp, resp
    result: dict[str, Any] = resp["result"]
    return result


def ok(server: Server, name: str, **args: Any) -> dict[str, Any]:
    result = call(server, name, **args)
    assert result["isError"] is False, result["content"][0]["text"]
    out: dict[str, Any] = result["structuredContent"]
    return out


def error(server: Server, name: str, **args: Any) -> str:
    result = call(server, name, **args)
    assert result["isError"] is True, result
    text: str = result["content"][0]["text"]
    return text


def test_initialize_and_list(server: Server) -> None:
    resp = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    assert resp is not None
    assert resp["result"]["protocolVersion"] == "2025-03-26"
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp is not None
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    assert set(tools) == {
        "status",
        "discover",
        "list_scenarios",
        "plan",
        "fault",
        "restore_all",
        "run_start",
        "run_status",
        "run_stop",
        "run_report",
        "list_runs",
    }
    assert tools["fault"]["annotations"]["destructiveHint"] is True
    assert tools["status"]["annotations"]["readOnlyHint"] is True


def test_unknown_argument(server: Server) -> None:
    resp = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "status", "arguments": {"password": "x"}},
        }
    )
    assert resp is not None and resp["error"]["code"] == -32602


def test_fault_down_is_held_and_restored(server: Server, host: FakeSystem) -> None:
    clock = [1_000_000.0]
    server.holds.clock = lambda: clock[0]

    out = ok(
        server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=20
    )

    assert out["key"] == "fc-link:host3"
    assert host.path(LINK).read_text() == "down"
    status = ok(server, "status")
    assert [h["key"] for h in status["held"]] == ["fc-link:host3"]
    assert status["held"][0]["seconds_left"] == 20
    clock[0] += 20
    server.holds.tick()
    assert host.path(LINK).read_text() == "up"
    assert ok(server, "status")["held"] == []


def test_hold_takes_kernel_timers(server: Server, host: FakeSystem) -> None:
    out = ok(
        server,
        "fault",
        kind="fc-link",
        target="host3",
        action="down",
        hold_seconds="kato + 10s",
    )
    held = ok(server, "status")["held"][0]
    assert held["seconds_left"] == pytest.approx(15, abs=1)
    assert out["restore_at"] == held["restore_at"]


def test_max_hold_comes_from_the_kernel(server: Server) -> None:
    status = ok(server, "status")
    assert status["kernel_timers"] == {"ctrl_loss_tmo": 600.0, "kato": 5.0}
    assert status["max_hold_seconds"] == 720

    text = error(
        server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=721
    )
    assert "at most 720" in text


def test_max_hold_option(opts: Options, host: FakeSystem) -> None:
    opts.max_hold = 30
    server = Server(opts, host)
    assert "at most 30" in error(
        server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=31
    )


def test_max_hold_without_timers(opts: Options, system: FakeSystem) -> None:
    assert ok(Server(opts, system), "status")["max_hold_seconds"] == 600


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"kind": "fc-link", "target": "host3", "action": "down"}, "hold_seconds"),
        (
            {"kind": "fc-link", "target": "host9", "action": "down", "hold_seconds": 5},
            "not a host port",
        ),
        (
            {"kind": "fc-link", "target": "host4", "action": "down", "hold_seconds": 5},
            "can't hold its link down",
        ),
        (
            {"kind": "fc-link", "target": "host3", "action": "up", "hold_seconds": 5},
            "takes no hold_seconds",
        ),
        (
            {"kind": "brocade-port", "target": "x", "action": "down", "force": True},
            "--allow-force",
        ),
        (
            {"kind": "fc-link", "target": "host3", "action": "down", "hold_seconds": 0},
            "above 0",
        ),
        (
            {
                "kind": "fc-link",
                "target": "host3",
                "action": "down",
                "hold_seconds": "nope_tmo",
            },
            "unknown timer",
        ),
        (
            {
                "kind": "tcp-port",
                "target": "10.0.0.9",
                "action": "down",
                "hold_seconds": 5,
            },
            "no path of this host",
        ),
        (
            {"kind": "ontap-lif", "target": "svm/l", "action": "up"},
            "not a LIF of this host",
        ),
    ],
)
def test_fault_refusals(
    server: Server, host: FakeSystem, args: dict[str, Any], message: str
) -> None:
    assert message in error(server, "fault", **args)
    assert host.path(LINK).read_text() == "Link Up - Ready\n"
    assert ok(server, "status")["held"] == []


def test_bounce_and_up(server: Server, host: FakeSystem) -> None:
    ok(server, "fault", kind="fc-lip", target="host3", action="bounce")
    assert host.path("/sys/class/fc_host/host3/issue_lip").read_text() == "1"
    ok(server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=9)

    ok(server, "fault", kind="fc-link", target="host3", action="up")

    assert host.path(LINK).read_text() == "up"
    assert ok(server, "status")["held"] == []


def test_discover_without_arrays(server: Server) -> None:
    out = ok(server, "discover")
    ports = {p["name"]: p for p in out["host_ports"]}
    assert ports["host3"]["kinds"] == ["fc-link", "fc-lip"]
    assert ports["host4"]["kinds"] == ["fc-lip", "pci-unbind"]
    assert ports["host3"]["wwpn"] == "10:00:00:10:9b:57:9e:f5"
    assert {p["controller"] for p in out["paths"]} == {"nvme1", "nvme2"}
    assert out["target_ports"] == []


def test_list_scenarios(server: Server) -> None:
    out = ok(server, "list_scenarios")["scenarios"]
    assert out == [
        {
            "name": "flap",
            "title": "flap one port",
            "format": "phases",
            "description": "One port flaps.",
        },
        {
            "name": "steps/bounce",
            "title": "steps/bounce",
            "format": "steps",
            "description": "",
        },
    ]


@pytest.mark.parametrize("name", ["../etc/passwd", "/etc/x", "nope", "a/b/c"])
def test_run_start_takes_listed_scenarios_only(server: Server, name: str) -> None:
    assert "scenario" in error(server, "run_start", scenario=name)


def test_run_start(server: Server, host: FakeSystem, opts: Options) -> None:
    host.path("/dev").mkdir()
    host.path("/dev/nvme1n1").write_text("")

    out = ok(server, "run_start", scenario="flap", io=["/dev/nvme1n1", "auto"])

    argv = host.commands[-1]
    assert argv[0] == "systemd-run"
    child = argv[argv.index("--") + 1 :]
    package = Path(nvmf_failover.__file__).resolve().parent.parent
    assert child[:2] == ["env", f"PYTHONPATH={package}"]
    assert child[3:5] == ["-m", "nvmf_failover.cli"]
    assert child[5:] == [
        "run",
        str((opts.scenarios / "flap.toml").resolve()),
        "--run-dir",
        out["dir"],
        "--io",
        "/dev/nvme1n1",
        "--io",
        "auto",
    ]


def test_run_start_checks_devices(server: Server) -> None:
    assert "not a namespace device" in error(
        server, "run_start", scenario="flap", io=["/dev/sda"]
    )
    assert "--allow-io-verify" in error(
        server, "run_start", scenario="flap", io_verify=["/dev/nvme1n1"]
    )


def test_no_run_while_a_fault_is_held(server: Server) -> None:
    ok(server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=9)
    assert "put them up first" in error(server, "run_start", scenario="flap")


def test_no_fault_while_a_run_runs(server: Server, host: FakeSystem) -> None:
    out = ok(server, "run_start", scenario="flap")
    host.answers[f"systemctl show --property=ActiveState --value {out['unit']}"] = (
        "active\n"
    )

    assert "is running" in error(
        server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=9
    )
    assert ok(server, "list_runs")["runs"] == [{"id": out["id"], "state": "running"}]
    stopped = ok(server, "restore_all")
    assert stopped == {"restore_failed": [], "runs_stopping": [out["id"]]}


# --- the server process -----------------------------------------------------


def serve(
    opts: Options, host: FakeSystem, monkeypatch: pytest.MonkeyPatch, *msgs: Any
) -> tuple[int, list[dict[str, Any]]]:
    lines = "".join(json.dumps(m) + "\n" for m in msgs)
    out = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))
    monkeypatch.setattr("sys.stdout", out)
    rc = mcp.serve(opts, host)
    return rc, [json.loads(line) for line in out.getvalue().splitlines()]


def fault_down(host: str, seconds: float) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "fault",
            "arguments": {
                "kind": "fc-link",
                "target": host,
                "action": "down",
                "hold_seconds": seconds,
            },
        },
    }


def test_client_gone_restores(
    opts: Options, host: FakeSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, resps = serve(opts, host, monkeypatch, fault_down("host3", 600))

    assert resps[0]["result"]["isError"] is False
    # stdin ended while the fault was held: it is up again
    assert rc == 0
    assert host.path(LINK).read_text() == "up"
    assert json.loads(opts.state.read_text()) == []


def test_a_killed_server_is_cleaned_up_by_the_next(
    opts: Options, host: FakeSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = Server(opts, host)
    ok(server, "fault", kind="fc-link", target="host3", action="down", hold_seconds=600)
    # SIGKILL: no cleanup ran, the state file still has the fault
    assert host.path(LINK).read_text() == "down"

    rc, _ = serve(opts, host, monkeypatch)

    assert rc == 0
    assert host.path(LINK).read_text() == "up"


def test_one_server_at_a_time(
    opts: Options,
    host: FakeSystem,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opts.state.parent.mkdir(parents=True)
    with opts.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert mcp.serve(opts, host) == 2
    assert "another mcp server" in capsys.readouterr().err


def test_plan(server: Server) -> None:
    out = ok(server, "plan", scenario="flap")
    assert out["scenario"] == "flap one port"
    assert any("fc-link:host3" in line for line in out["lines"]), out["lines"]
    assert "step scenario" in error(server, "plan", scenario="steps/bounce")


def test_a_held_pci_unbind_comes_back(server: Server, host: FakeSystem) -> None:
    clock = [1_000_000.0]
    server.holds.clock = lambda: clock[0]
    ok(
        server,
        "fault",
        kind="pci-unbind",
        target="host4",
        action="down",
        hold_seconds=9,
    )
    # unbound: the SCSI host and the driver link are gone
    shutil.rmtree(host.path("/sys/class/scsi_host/host4"))
    host.path("/sys/bus/pci/devices/0000:81:00.1/driver").unlink()

    clock[0] += 9
    server.holds.tick()

    bind = host.path("/sys/bus/pci/drivers/qla2xxx/bind")
    assert bind.read_text() == "0000:81:00.1"
    assert ok(server, "status")["held"] == []


class Cluster:
    def __init__(self, cfg: object) -> None:
        pass

    def run(self, command: str) -> str:
        return ""

    def close(self) -> None:
        pass


def test_only_the_ports_of_this_host(
    opts: Options, host: FakeSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inventory(*args: object) -> Inventory:
        return Inventory(
            hostnqn="nqn.host:mine",
            lifs=[
                Lif(
                    "svm-mine", "lif1", "fc-nvme", "20:19:00:a0:98:90:f5:bf", True, True
                ),
                Lif(
                    "svm-other",
                    "lif1",
                    "fc-nvme",
                    "20:30:00:a0:98:90:f5:bf",
                    True,
                    True,
                ),
            ],
            subsystems=[Subsystem("svm-mine", "sub", "nqn.sub")],
        )

    monkeypatch.setattr("nvmf_failover.cli.inventory", inventory)
    monkeypatch.setattr("nvmf_failover.cli.OntapShell", Cluster)
    opts.config = Config(ontap=OntapConfig("cluster", "admin", "pw"))
    server = Server(opts, host)

    ports = ok(server, "discover")["target_ports"]

    assert [(p["group"], p["kinds"]) for p in ports] == [("svm-mine", ["ontap-lif"])]
    assert "not a LIF of this host" in error(
        server, "fault", kind="ontap-lif", target="svm-other/lif1", action="up"
    )
