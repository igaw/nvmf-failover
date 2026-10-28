import pytest

from nvmf_failover.brocade import (
    BrocadeError,
    find_port,
    locate,
    normalize_wwpn,
    parse_nsshow,
    parse_switchshow,
    port_users,
    set_port,
)
from nvmf_failover.config import BrocadeConfig
from nvmf_failover.faults import BrocadePortFault, FaultError
from nvmf_failover.scenario import Context, make_fault
from nvmf_failover.system import System

# trimmed from a Brocade G620, Fabric OS v9.2.0
SWITCHSHOW = """\
switchName:\tRD-LABS-BRC2
switchState:\tOnline
Index Port Address  Media Speed   State       Proto
==================================================
   0   0   600000   id    N32\t  Online      FC  E-Port  segmented,10:00:d8:1f:cc:47:65:a8 (zone conflict)
   1   1   600100   id    N32\t  No_Light    FC
   3   3   600300   id    N16\t  Online      FC  F-Port  1 N Port + 20 NPIV public
   6   6   600600   id    N16\t  Online      FC  F-Port  10:00:00:10:9b:e5:d0:be
  19  19   601300   --    N32\t  No_Module   FC  Disabled (ESC Enhanced Zone Object Name Conflict)
  22  22   601600   id    32G\t  No_Sync     FC
"""

NSSHOW = """\
{
 Type Pid    COS     PortName                NodeName                 SCR
 N    600300;      3;10:00:00:10:9b:ee:0b:ab;20:00:00:10:9b:ee:0b:ab; 0x00000003
    SCR: Fabric-Detected Nx-Port-Detected
    Fabric Port Name: 20:03:88:94:71:58:86:b0
    Permanent Port Name: 10:00:00:10:9b:ee:0b:ab
    Device type: Physical Unknown(initiator/target)
    Port Index: 3
    Device link speed: 16G
 N    600301;      3;c0:50:76:0c:89:e2:00:02;c0:50:76:0c:89:e2:00:02; 0x00000003
    SCR: Fabric-Detected Nx-Port-Detected
    Permanent Port Name: 10:00:00:10:9b:ee:0b:ab
    Device type: NPIV Unknown(initiator/target)
    Port Index: 3
 N    600600;      3;10:00:00:10:9b:e5:d0:be;20:00:00:10:9b:e5:d0:be; 0x00000003
    Device type: Physical Initiator
    Port Index: 6
}
The Local Name Server has 3 entries }
"""


class FakeSwitch:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> str:
        self.commands.append(command)
        return NSSHOW if command == "nsshow" else ""


@pytest.mark.parametrize(
    "text",
    ["0x100000109b579ef5", "pn-0x100000109b579ef5", "10:00:00:10:9B:57:9E:F5"],
)
def test_normalize_wwpn(text: str) -> None:
    assert normalize_wwpn(text) == "10:00:00:10:9b:57:9e:f5"


def test_normalize_rejects_garbage() -> None:
    with pytest.raises(BrocadeError, match="not a WWPN"):
        normalize_wwpn("0x1234")


def test_parse_switchshow_ports() -> None:
    ports = {p.index: p for p in parse_switchshow(SWITCHSHOW)}
    assert sorted(ports) == [0, 1, 3, 6, 19, 22]
    assert ports[3].state == "Online"
    assert ports[3].comment == "F-Port  1 N Port + 20 NPIV public"
    assert ports[1].comment == ""
    assert ports[19].disabled and not ports[3].disabled


def test_parse_nsshow_entries() -> None:
    entries = parse_nsshow(NSSHOW)
    assert [(e.pid, e.wwpn, e.port_index, e.npiv) for e in entries] == [
        ("600300", "10:00:00:10:9b:ee:0b:ab", 3, False),
        ("600301", "c0:50:76:0c:89:e2:00:02", 3, True),
        ("600600", "10:00:00:10:9b:e5:d0:be", 6, False),
    ]


def test_locate_and_port_users() -> None:
    switch = FakeSwitch()
    entry = locate(switch, "0x100000109be5d0be")
    assert entry.port_index == 6
    # port 3 carries a physical port and an NPIV login: both are hit
    assert len(port_users(switch, 3)) == 2


def test_locate_unknown_wwpn() -> None:
    with pytest.raises(BrocadeError, match="not logged in"):
        locate(FakeSwitch(), "0x100000109b579ef5")


def test_set_port_commands() -> None:
    switch = FakeSwitch()
    set_port(switch, 6, enable=False)
    set_port(switch, 6, enable=True)
    assert switch.commands == ["portdisable 6", "portenable 6"]


def _switches() -> list[BrocadeConfig]:
    return [
        BrocadeConfig("empty", "h1", "admin", "p"),
        BrocadeConfig("brc", "h2", "admin", "p"),
    ]


class Fabric:
    """Two switches: only "brc" has logins."""

    def __init__(self) -> None:
        self.shells = {"empty": EmptySwitch(), "brc": FakeSwitch()}

    def __call__(self, cfg: BrocadeConfig) -> FakeSwitch:
        return self.shells[cfg.name]


class EmptySwitch(FakeSwitch):
    def run(self, command: str) -> str:
        self.commands.append(command)
        return "{\n}\nThe Local Name Server has 0 entries }\n"


def test_find_port_walks_the_switches() -> None:
    cfg, entry, users = find_port(_switches(), Fabric(), "0x100000109be5d0be")
    assert (cfg.name, entry.port_index, len(users)) == ("brc", 6, 1)


def test_find_port_names_the_switches_searched() -> None:
    with pytest.raises(BrocadeError, match="not logged in to empty, brc"):
        find_port(_switches(), Fabric(), "0x100000109b579ef5")


def test_port_fault_on_a_single_login_port() -> None:
    fabric = Fabric()
    fault = BrocadePortFault(_switches(), fabric, "10:00:00:10:9b:e5:d0:be")
    assert fault.target == "brc/6(10:00:00:10:9b:e5:d0:be)"
    fault.down()
    fault.up()
    assert fabric.shells["brc"].commands[-2:] == ["portdisable 6", "portenable 6"]


def test_port_fault_refuses_a_shared_port() -> None:
    with pytest.raises(FaultError, match=r"port 3 also carries 1 other login"):
        BrocadePortFault(_switches(), Fabric(), "10:00:00:10:9b:ee:0b:ab")
    forced = BrocadePortFault(
        _switches(), Fabric(), "10:00:00:10:9b:ee:0b:ab", force=True
    )
    assert forced.others == ["c0:50:76:0c:89:e2:00:02"]


def test_port_fault_from_its_spec_finds_a_disabled_port() -> None:
    fabric = Fabric()
    fault = BrocadePortFault(_switches(), fabric, "10:00:00:10:9b:e5:d0:be")
    spec = fault.spec()
    assert spec == {
        "fault": "brocade-port",
        "wwpn": "10:00:00:10:9b:e5:d0:be",
        "switch": "brc",
        "port": 6,
        "force": False,
    }
    fault.down()
    # disabled: the WWPN logged out, the name server doesn't know it
    fabric.shells["brc"] = EmptySwitch()

    again = make_fault(spec, Context(System(), connect=fabric, switches=_switches()))
    again.up()

    assert again.key == fault.key
    assert fabric.shells["brc"].commands[-1] == "portenable 6"


def test_port_fault_spec_still_refuses_a_shared_port() -> None:
    with pytest.raises(FaultError, match="other login"):
        BrocadePortFault(
            _switches(), Fabric(), "10:00:00:10:9b:ee:0b:ab", switch="brc", port=3
        )


def test_port_fault_spec_names_a_known_switch() -> None:
    with pytest.raises(FaultError, match="not a configured switch"):
        BrocadePortFault(_switches(), Fabric(), "10:00", switch="nope", port=1)
