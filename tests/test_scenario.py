import tomllib
from pathlib import Path

import pytest

from nvmf_failover.discover import Inventory, Lif
from nvmf_failover.faults import Fault, FaultError
from nvmf_failover.resolve import is_described
from nvmf_failover.resolve import parse as resolve_parse
from nvmf_failover.scenario import (
    Context,
    Runner,
    ScenarioError,
    Step,
    expand,
    fault_raw,
    load,
    parse,
)

from conftest import FakeSystem


class Recorder(Fault):
    kind = "rec"

    def __init__(
        self, name: str, log: list[str], fail_up: bool = False, fail_down: bool = False
    ) -> None:
        self.name, self.log = name, log
        self.fail_up, self.fail_down = fail_up, fail_down

    @property
    def target(self) -> str:
        return self.name

    def down(self) -> None:
        self.log.append(f"down {self.name}")
        if self.fail_down:
            raise FaultError("half applied")

    def up(self) -> None:
        self.log.append(f"up {self.name}")
        if self.fail_up:
            raise FaultError("stuck")

    def bounce(self) -> None:
        self.log.append(f"bounce {self.name}")


def runner(log: list[str]) -> Runner:
    return Runner(sleep=lambda s: log.append(f"sleep {s:g}"), log=lambda t: None)


def test_parse_valid(tmp_path: Path) -> None:
    path = tmp_path / "s.toml"
    path.write_text(
        'name = "x"\nrepeat = 2\n[[steps]]\nfault = "fc-link"\n'
        'host = "host3"\naction = "down"\nhold = 5\n'
    )
    scen = load(path)
    assert (scen.name, scen.repeat, scen.recover_timeout) == ("x", 2, 300.0)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "at least one"),
        ({"steps": [{"fault": "nope", "action": "down"}]}, "fault must be"),
        ({"steps": [{"fault": "fc-link", "host": "h", "action": "off"}]}, "action"),
        ({"steps": [{"fault": "fc-link", "action": "down"}]}, "needs host"),
        (
            {
                "steps": [
                    {"fault": "fc-lip", "host": "h", "action": "bounce", "hold": -1}
                ]
            },
            "hold",
        ),
        (
            {
                "repeat": 0,
                "steps": [{"fault": "fc-lip", "host": "h", "action": "bounce"}],
            },
            "repeat",
        ),
    ],
)
def test_parse_rejects(data: dict[str, object], message: str) -> None:
    with pytest.raises(ScenarioError, match=message):
        parse(data)


def test_runner_holds_and_repeats() -> None:
    log: list[str] = []
    a = Recorder("a", log)
    runner(log).run([], [Step(a, "down", 2), Step(a, "up", 1)], repeat=2)
    assert log == ["down a", "sleep 2", "up a", "sleep 1"] * 2


def test_runner_restores_on_failure_in_reverse_order() -> None:
    log: list[str] = []
    a, b = Recorder("a", log), Recorder("b", log)
    boom = Recorder("boom", log, fail_down=True)
    r = runner(log)
    with pytest.raises(FaultError):
        r.run(
            [],
            [Step(a, "down", 0), Step(b, "down", 0), Step(boom, "down", 0)],
            repeat=1,
        )
    # the half-applied fault is mended too, last first
    assert log[-3:] == ["up boom", "up b", "up a"]
    assert r.down == {}


def test_runner_restores_on_interrupt() -> None:
    log: list[str] = []
    a = Recorder("a", log)

    def sleep(seconds: float) -> None:
        raise KeyboardInterrupt

    r = Runner(sleep=sleep, log=lambda t: None)
    with pytest.raises(KeyboardInterrupt):
        r.run([], [Step(a, "down", 10)], repeat=1)
    assert log == ["down a", "up a"]


def test_restore_keeps_going_and_reports_failures() -> None:
    log: list[str] = []
    a, b = Recorder("a", log), Recorder("b", log, fail_up=True)
    r = runner(log)
    r.apply(Step(a, "down", 0))
    r.apply(Step(b, "down", 0))
    assert r.restore() == ["rec:b"]
    assert log[-2:] == ["up b", "up a"]
    assert list(r.down) == ["rec:b"]


class NullShell:
    def run(self, command: str) -> str:
        return ""


def test_lif_cycle_unrolls_the_lifs_with_paths(system: FakeSystem) -> None:
    inv = Inventory(
        hostnqn="h",
        lifs=[
            Lif("svm", "lif_2", "fc-nvme", "x", True, True),
            Lif("svm", "lif_1", "fc-nvme", "y", True, True),
            Lif("svm", "lif_3", "fc-nvme", "z", True, True),
        ],
        paths={
            ("svm", "lif_1"): ["nvme5"],
            ("svm", "lif_2"): ["nvme6"],
            ("svm", "lif_3"): [],
        },
    )
    inv.subsystems = []
    ctx = Context(system, NullShell(), "svm", inv)

    steps = expand([{"fault": "ontap-lif-cycle", "hold": 20}], ctx)

    assert [(s.fault.target, s.action) for s in steps] == [
        ("svm/lif_1", "down"),
        ("svm/lif_1", "up"),
        ("svm/lif_2", "down"),
        ("svm/lif_2", "up"),
    ]
    assert all(s.hold == 20 for s in steps)


def test_ontap_steps_need_the_config(system: FakeSystem) -> None:
    with pytest.raises(FaultError, match=r"\[ontap\]"):
        expand([{"fault": "ontap-lif", "lif": "l", "action": "down"}], Context(system))


SCENARIOS = sorted((Path(__file__).parent.parent / "scenarios").rglob("*.toml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_shipped_scenarios_parse(path: Path) -> None:
    data = tomllib.loads(path.read_text())
    if is_described(data):
        assert resolve_parse(data, str(path)).phases
    else:
        assert load(path).steps


@pytest.mark.parametrize(
    ("kind", "target", "expected"),
    [
        ("fc-link", "host3", {"host": "host3"}),
        ("ontap-lif", "lif1", {"lif": "lif1", "force": False}),
        ("ontap-lif", "svm/lif1", {"lif": "lif1", "force": False, "vserver": "svm"}),
        ("brocade-port", "20:17", {"wwpn": "20:17", "force": False}),
        ("tcp-port", "10.0.0.1:4421", {"address": "10.0.0.1", "port": "4421"}),
        ("tcp-port", "10.0.0.1", {"address": "10.0.0.1", "port": "4420"}),
        ("tcp-port", "fe80::1", {"address": "fe80::1", "port": "4420"}),
        ("tcp-port", "[fe80::1]:4421", {"address": "fe80::1", "port": "4421"}),
    ],
)
def test_fault_raw(kind: str, target: str, expected: dict[str, object]) -> None:
    assert fault_raw(kind, target, "down") == {
        "fault": kind,
        "action": "down",
        **expected,
    }


def test_fault_raw_refuses_a_cycle() -> None:
    with pytest.raises(FaultError, match="not a single fault"):
        fault_raw("ontap-lif-cycle", "svm", "down")
