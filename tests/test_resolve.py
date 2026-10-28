import pytest

from nvmf_failover.config import BrocadeConfig
from nvmf_failover.model import build
from nvmf_failover.resolve import duration, elements, kernel_timers, parse, resolve
from nvmf_failover.scenario import Context, ScenarioError
from test_model import Switch, inventory, world

from conftest import FakeSystem

TIMERS = {"dev_loss_tmo": 30.0, "ctrl_loss_tmo": 600.0}


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        (90, 90.0),
        ("300s", 300.0),
        ("5m", 300.0),
        ("1h", 3600.0),
        ("dev_loss_tmo + 30s", 60.0),
        ("ctrl_loss_tmo - 1m", 540.0),
        ("2.5s", 2.5),
    ],
)
def test_duration(text: object, seconds: float) -> None:
    assert duration(text, TIMERS) == seconds


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("fast_io_fail_tmo", "unknown timer"),
        ("30 parsecs", "not a duration"),
        ("dev_loss_tmo - 1m", "negative"),
        ("", "not a duration"),
    ],
)
def test_duration_errors(text: str, message: str) -> None:
    with pytest.raises(ScenarioError, match=message):
        duration(text, TIMERS)


def base() -> dict[str, object]:
    return {
        "name": "t",
        "paths": {"subsystem": "*dolin-lpfc*"},
        "phase": [{"fail": "host-port", "pick": "each", "duration": "60s"}],
    }


def test_parse_described() -> None:
    scen = parse(base())
    assert scen.selector == {"subsystem": "*dolin-lpfc*"}
    assert scen.phases[0].pick == "each"
    assert scen.expect.reconnected_by == "any"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"paths": {"host": "x"}}, r"\[paths\] takes"),
        ({"phase": []}, "at least one"),
        ({"phase": [{"fail": "cpu", "duration": 1}]}, "fail must be"),
        ({"phase": [{"fail": "host-port", "pick": "some", "duration": 1}]}, "pick"),
        ({"phase": [{"fail": "host-port"}]}, "duration"),
        (
            {"phase": [{"fail": "host-port", "pick": "all", "duration": 1}]},
            "allow_outage",
        ),
        ({"expect": {"reconnected_by": "magic"}}, "reconnected_by"),
    ],
)
def test_parse_rejects(change: dict[str, object], message: str) -> None:
    data = base() | change
    with pytest.raises(ScenarioError, match=message):
        parse(data)


class NullShell:
    def run(self, command: str) -> str:
        return ""


def context(system: FakeSystem, switches: bool = False) -> Context:
    brc = [BrocadeConfig("brc4", "h", "admin", "p")] if switches else []
    return Context(
        system,
        NullShell(),
        "",
        None,
        switches=brc,
        connect=(lambda cfg: Switch()) if switches else None,
    )


def test_host_port_elements_choose_the_fault_by_driver(system: FakeSystem) -> None:
    world(system)
    model = build(system, inventory())
    elems = elements("host-port", model.select(), context(system))
    assert [(e.label, [f.kind for f in e.faults], e.paths) for e in elems] == [
        ("host-port host2", ["pci-unbind"], ("nvme3",)),
        ("host-port host3", ["fc-link"], ("nvme1", "nvme2", "nvme4")),
    ]


def test_target_ports_netapp_by_lif(system: FakeSystem) -> None:
    world(system)
    model = build(system, inventory())
    elems = elements("target-port", model.select(array="netapp"), context(system))
    assert [(e.label, [f.key for f in e.faults]) for e in elems] == [
        ("target-port svm-lpfc/lif_1", ["ontap-lif:svm-lpfc/lif_1"]),
        ("target-port svm-lpfc/lif_2", ["ontap-lif:svm-lpfc/lif_2"]),
    ]


def test_array_node_groups_target_ports(system: FakeSystem) -> None:
    world(system)
    model = build(system, inventory())
    elems = elements("array-node", model.select(array="netapp"), context(system))
    assert [(e.label, e.paths) for e in elems] == [
        ("array-node netapp:node-1", ("nvme1", "nvme3")),
        ("array-node netapp:node-2", ("nvme2",)),
    ]


def test_host_cable_uses_the_switch_port(system: FakeSystem) -> None:
    world(system)
    ctx = context(system, switches=True)
    model = build(system, inventory(), ctx.switches, ctx.connect)
    paths = [p for p in model.select() if p.host_port and p.host_port.name == "host3"]
    (elem,) = elements("host-cable", paths, ctx)
    assert [f.key for f in elem.faults] == [
        "brocade-port:brc4/1(10:00:00:10:9b:57:9e:f5)"
    ]


def test_resolve_picks_and_repeats(system: FakeSystem) -> None:
    world(system)
    model = build(system, inventory())
    data = base() | {
        "paths": {"subsystem": "*dolin-lpfc*"},
        "repeat": 2,
        "phase": [
            {
                "fail": "host-port",
                "pick": "each",
                "duration": "dev_loss_tmo + 30s",
                "gap": "until-recovered",
            },
            {"fail": "array-node", "pick": "all-but-one", "duration": 10, "gap": 5},
        ],
    }
    plan = resolve(parse(data), model, context(system), TIMERS)
    labels = [(r.phase, r.label, r.duration, r.gap) for r in plan.rounds]
    assert (
        labels
        == [
            (1, "host-port host2", 60.0, None),
            (1, "host-port host3", 60.0, None),
            (2, "array-node netapp:node-1", 10.0, 5.0),
        ]
        * 2
    )
    assert plan.lines()[0] == "scenario t: 3 paths selected"


def test_resolve_without_matching_paths(system: FakeSystem) -> None:
    world(system)
    model = build(system, inventory())
    data = base() | {"paths": {"subsystem": "*nothing*"}}
    with pytest.raises(ScenarioError, match="no path matches"):
        resolve(parse(data), model, context(system), TIMERS)


def test_kernel_timers(system: FakeSystem) -> None:
    world(system)
    for i, (roles, tmo) in enumerate(
        [("NVMe Target, NVMe Discovery", "30"), ("FCP Target", "60")]
    ):
        rport = system.path(f"/sys/class/fc_remote_ports/rport-3:0-{i}")
        rport.mkdir(parents=True)
        (rport / "roles").write_text(roles + "\n")
        (rport / "dev_loss_tmo").write_text(tmo + "\n")
    system.path("/sys/class/nvme/nvme1/ctrl_loss_tmo").write_text("600\n")
    system.path("/sys/class/nvme/nvme2/ctrl_loss_tmo").write_text("off\n")
    for ctrl, attr, value in (
        ("nvme1", "reconnect_delay", "2"),
        ("nvme2", "reconnect_delay", "10"),
        ("nvme1", "kato", "5"),
        ("nvme1", "io_timeout", "30000"),
        ("nvme2", "admin_timeout", "60000"),
        ("nvme1", "fast_io_fail_tmo", "off"),
    ):
        system.path(f"/sys/class/nvme/{ctrl}/{attr}").write_text(value + "\n")
    model = build(system, inventory())
    assert kernel_timers(system, [p.ctrl.name for p in model.select()]) == {
        "dev_loss_tmo": 30.0,
        "ctrl_loss_tmo": 600.0,
        "reconnect_delay": 10.0,
        "kato": 5.0,
        "io_timeout": 30.0,
        "admin_timeout": 60.0,
    }
