import errno
import fcntl
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nvmf_failover.faults import (
    FaultError,
    FcLinkFault,
    FcLipFault,
    OntapLifFault,
    PciUnbindFault,
    TcpPortFault,
    TcpRefuseFault,
)
from nvmf_failover.scenario import Context, make_fault

from conftest import FakeSystem


def test_fc_link_writes_link_state(system: FakeSystem) -> None:
    system.add_scsi_host("host3", "lpfc")
    fault = FcLinkFault(system, "host3")
    state = system.path("/sys/class/scsi_host/host3/link_state")

    fault.down()
    assert state.read_text() == "down"
    fault.up()
    assert state.read_text() == "up"
    assert fault.key == "fc-link:host3"


def test_fc_link_refuses_other_drivers(system: FakeSystem) -> None:
    system.add_scsi_host("host2", "qla2xxx")
    with pytest.raises(FaultError, match="qla2xxx can't hold"):
        FcLinkFault(system, "host2")


def test_fc_link_refuses_bad_host_names(system: FakeSystem) -> None:
    with pytest.raises(FaultError, match="not a SCSI host"):
        FcLinkFault(system, "../../etc")


def test_fc_lip_only_bounces(system: FakeSystem) -> None:
    system.add_scsi_host("host2", "qla2xxx")
    fault = FcLipFault(system, "host2")
    fault.bounce()
    assert system.path("/sys/class/fc_host/host2/issue_lip").read_text() == "1"
    with pytest.raises(FaultError, match="no down state"):
        fault.down()


def _pci_host(system: FakeSystem, host: str, pci: str, driver: str) -> None:
    dev = system.path(f"/sys/devices/pci0000:80/{pci}/{host}")
    dev.mkdir(parents=True)
    drv = system.path(f"/sys/bus/pci/drivers/{driver}")
    drv.mkdir(parents=True)
    devices = system.path("/sys/bus/pci/devices")
    devices.mkdir(parents=True)
    (devices / pci).symlink_to(dev.parent)
    (dev.parent / "driver").symlink_to(drv)
    scsi = system.path(f"/sys/class/scsi_host/{host}")
    scsi.mkdir(parents=True)
    (scsi / "device").symlink_to(dev)


def test_pci_unbind_finds_function_and_driver(system: FakeSystem) -> None:
    _pci_host(system, "host2", "0000:81:00.0", "qla2xxx")
    fault = PciUnbindFault(system, "host2")
    assert (fault.pci, fault.driver) == ("0000:81:00.0", "qla2xxx")

    fault.down()
    unbind = system.path("/sys/bus/pci/drivers/qla2xxx/unbind")
    assert unbind.read_text() == "0000:81:00.0"

    # still bound (the fake doesn't unbind): up() leaves it alone
    fault.up()
    assert not system.path("/sys/bus/pci/drivers/qla2xxx/bind").exists()

    system.path("/sys/devices/pci0000:80/0000:81:00.0/driver").unlink()
    fault.up()
    bind = system.path("/sys/bus/pci/drivers/qla2xxx/bind")
    assert bind.read_text() == "0000:81:00.0"


def _nft_chain(*comments: str) -> str:
    rules = [{"rule": {"handle": i + 7, "comment": c}} for i, c in enumerate(comments)]
    return json.dumps({"nftables": [{"metainfo": {}}, *rules]})


def test_tcp_port_adds_one_rule_in_its_own_table(system: FakeSystem) -> None:
    system.failing.add("nft list table inet nvmf_failover")
    fault = TcpPortFault(system, "10.100.128.29", 4420)

    fault.down()

    adds = [c for c in system.commands if c[:2] == ["nft", "add"]]
    assert adds[0] == ["nft", "add", "table", "inet", "nvmf_failover"]
    assert "ip" in adds[-1] and "10.100.128.29" in adds[-1]
    assert adds[-1][-2:] == ["comment", '"tcp-port:10.100.128.29:4420"']


def test_tcp_port_up_deletes_only_its_rule(system: FakeSystem) -> None:
    fault = TcpPortFault(system, "10.100.128.29", 4421)
    chain = "nft -j -a list chain inet nvmf_failover input"
    system.answers[chain] = _nft_chain(
        "tcp-port:10.100.128.29:4420", "tcp-port:10.100.128.29:4421"
    )
    system.answers["nft -j list table inet nvmf_failover"] = _nft_chain(
        "tcp-port:10.100.128.29:4420"
    )

    fault.up()

    deletes = [c for c in system.commands if c[:2] == ["nft", "delete"]]
    assert deletes == [
        ["nft", "delete", "rule", "inet", "nvmf_failover", "input", "handle", "8"]
    ]


def test_tcp_port_ipv6_uses_ip6(system: FakeSystem, tmp_path: Path) -> None:
    system.failing.add("nft list table inet nvmf_failover")
    TcpPortFault(system, "fd00::5", 4420).down()
    rule = next(c for c in system.commands if c[:3] == ["nft", "add", "rule"])
    assert rule[6:8] == ["ip6", "saddr"]


def test_tcp_refuse_rejects_in_the_output_hook(system: FakeSystem) -> None:
    system.failing.add("nft list table inet nvmf_failover")
    fault = TcpRefuseFault(system, "10.100.128.29", 4421)

    fault.down()

    adds = [c for c in system.commands if c[:2] == ["nft", "add"]]
    assert adds[1][-1] == "{ type filter hook output priority -10 ; }"
    assert adds[-1][:6] == ["nft", "add", "rule", "inet", "nvmf_failover", "output"]
    assert adds[-1][6:] == [
        "ip",
        "daddr",
        "10.100.128.29",
        "tcp",
        "dport",
        "4421",
        "reject",
        "with",
        "tcp",
        "reset",
        "comment",
        '"tcp-refuse:10.100.128.29:4421"',
    ]


def test_tcp_refuse_up_without_its_chain(system: FakeSystem) -> None:
    """A tcp-port fault made the table, but there is no output chain."""
    system.failing.add("nft list chain inet nvmf_failover output")
    system.answers["nft -j list table inet nvmf_failover"] = _nft_chain(
        "tcp-port:10.100.128.29:4420"
    )

    TcpRefuseFault(system, "10.100.128.29", 4420).up()

    assert not [c for c in system.commands if c[:2] == ["nft", "delete"]]


@dataclass
class LockCheckingSystem(FakeSystem):
    """Records whether another process could take the fault lock while
    each nft command ran."""

    unlocked: list[list[str]] = field(default_factory=list)

    def _check(self, argv: list[str]) -> None:
        with self.path(TcpPortFault.lock_path).open("w") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            fcntl.flock(other, fcntl.LOCK_UN)
            self.unlocked.append(argv)

    def run(self, argv: list[str], check: bool = True) -> str:
        self._check(argv)
        return super().run(argv, check)

    def ok(self, argv: list[str]) -> bool:
        self._check(argv)
        return super().ok(argv)


@pytest.mark.parametrize("cls", [TcpPortFault, TcpRefuseFault])
def test_tcp_fault_holds_the_lock(tmp_path: Path, cls: type) -> None:
    """Parallel runs share the table, and up() deletes it with its last
    rule: no nft command may run without the lock, or one run deletes
    the table while another adds its rule."""
    system = LockCheckingSystem(root=tmp_path)
    system.answers["nft -j list table inet nvmf_failover"] = _nft_chain()
    for chain in ("input", "output"):
        system.answers[f"nft -j -a list chain inet nvmf_failover {chain}"] = (
            _nft_chain()
        )
    fault = cls(system, "10.100.128.29", 4420)

    fault.down()
    fault.up()

    assert system.commands
    assert system.unlocked == []


def test_tcp_refuse_from_the_command_line() -> None:
    from nvmf_failover.scenario import fault_raw

    assert fault_raw("tcp-refuse", "10.0.0.1:4422", "down") == {
        "fault": "tcp-refuse",
        "action": "down",
        "address": "10.0.0.1",
        "port": "4422",
    }


def test_pci_unbind_comes_back_from_its_spec(system: FakeSystem) -> None:
    _pci_host(system, "host4", "0000:81:00.1", "qla2xxx")
    fault = PciUnbindFault(system, "host4")
    spec = fault.spec()
    fault.down()
    # unbound: the SCSI host and the driver link are gone
    shutil.rmtree(system.path("/sys/class/scsi_host/host4"))
    system.path("/sys/bus/pci/devices/0000:81:00.1/driver").unlink()

    again = make_fault(spec, Context(system))
    again.up()

    assert again.key == fault.key
    bind = system.path("/sys/bus/pci/drivers/qla2xxx/bind")
    assert bind.read_text() == "0000:81:00.1"


def test_pci_unbind_spec_is_checked(system: FakeSystem) -> None:
    with pytest.raises(FaultError, match="bad PCI function"):
        PciUnbindFault(system, "host4", "../../x", "qla2xxx")


def test_specs_build_the_same_fault(system: FakeSystem) -> None:
    system.add_scsi_host("host3", "lpfc")
    for fault in (
        FcLinkFault(system, "host3"),
        FcLipFault(system, "host3"),
        TcpPortFault(system, "10.0.0.1", 4421),
        TcpRefuseFault(system, "10.0.0.1", 4422),
    ):
        assert make_fault(fault.spec(), Context(system)).key == fault.key


HOSTS = """\
vserver   subsystem host-nqn
--------- --------- -------------
svm-mine  sub-a     nqn.host:mine
svm-mixed sub-b     nqn.host:mine
svm-mixed sub-c     nqn.host:other
3 entries were displayed.
"""


class Cluster:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> str:
        self.commands.append(command)
        if command.startswith("vserver nvme subsystem host show"):
            vserver = command.split()[6]
            rows = [r for r in HOSTS.splitlines()[2:5] if r.startswith(vserver + " ")]
            return "\n".join([*HOSTS.splitlines()[:2], *rows]) + "\n"
        return ""


def test_lif_down_in_a_vserver_of_this_host_only() -> None:
    cluster = Cluster()
    OntapLifFault(cluster, "svm-mine", "lif1", "nqn.host:mine").down()
    assert cluster.commands[-1] == (
        "network interface modify -vserver svm-mine -lif lif1 -status-admin down"
    )


def test_lif_down_refused_in_a_shared_vserver() -> None:
    cluster = Cluster()
    fault = OntapLifFault(cluster, "svm-mixed", "lif1", "nqn.host:mine")
    with pytest.raises(FaultError, match=r"svm-mixed also serves 1 other host"):
        fault.down()
    assert not any("modify" in c for c in cluster.commands)
    # putting it up is never refused
    fault.up()
    assert "-status-admin up" in cluster.commands[-1]
    OntapLifFault(cluster, "svm-mixed", "lif1", "nqn.host:mine", force=True).down()
    assert "-status-admin down" in cluster.commands[-1]


def test_lif_down_without_a_host_nqn_is_refused() -> None:
    with pytest.raises(FaultError, match="other host"):
        OntapLifFault(Cluster(), "svm-mine", "lif1", "").down()


@dataclass
class LpfcSystem(FakeSystem):
    """lpfc's link_state: EINVAL for down unless up, for up unless down.

    @coming: the states a transition still passes, one per read.
    """

    coming: list[str] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)

    def read(self, path: str) -> str:
        if path.endswith("/link_state") and self.coming:
            self.path(path).write_text(self.coming.pop(0))
        return super().read(path)

    def write(self, path: str, value: str) -> None:
        if path.endswith("/link_state"):
            state = self.path(path).read_text()
            ready = "Link Up" if value == "down" else "Link Down"
            if not state.startswith(ready) or self.coming:
                raise OSError(errno.EINVAL, "Invalid argument")
            self.accepted.append(value)
            value = "Link Down" if value == "down" else "Link Up - Ready"
        super().write(path, value)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def lpfc_fault(tmp_path: Path) -> tuple[LpfcSystem, FcLinkFault, Path, Clock]:
    system = LpfcSystem(root=tmp_path)
    system.add_scsi_host("host3", "lpfc")
    clock = Clock()
    fault = FcLinkFault(system, "host3", clock=clock, sleep=clock.sleep)
    return system, fault, system.path("/sys/class/scsi_host/host3/link_state"), clock


def test_fc_link_down_and_up_twice(tmp_path: Path) -> None:
    _, fault, state, _ = lpfc_fault(tmp_path)

    fault.down()
    fault.down()  # a held fault put down again
    assert state.read_text() == "Link Down"
    fault.up()
    fault.up()  # a fault someone else put up already
    assert state.read_text() == "Link Up - Ready"


def test_fc_link_up_while_it_still_goes_down(tmp_path: Path) -> None:
    system, fault, state, clock = lpfc_fault(tmp_path)
    # down was asked for, the link still shows up for a while
    system.coming = ["Link Up - Ready"] * 3 + ["Link Down"]

    fault.up()

    # not taken for up: asked again once the link was down
    assert system.coming == [] and system.accepted == ["up"]
    assert state.read_text() == "Link Up - Ready"
    assert clock.now < 5


def test_fc_link_down_while_it_still_comes_up(tmp_path: Path) -> None:
    system, fault, state, _ = lpfc_fault(tmp_path)
    state.write_text("Link Down")
    system.coming = ["Link Down"] * 3 + ["Link Up - Discovery", "Link Up - Ready"]

    fault.down()

    assert system.coming == [] and system.accepted == ["down"]
    assert state.read_text() == "Link Down"


def test_fc_link_that_never_gets_there_is_an_error(tmp_path: Path) -> None:
    _, fault, state, clock = lpfc_fault(tmp_path)
    state.write_text("HBA ERROR")
    with pytest.raises(FaultError, match="stays 'HBA ERROR'"):
        fault.up()
    assert clock.now >= 30
