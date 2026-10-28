import threading

from nvmf_failover.discover import Lif, discover, lif_matches
from nvmf_failover.ontap import OntapError, OntapShell, lif_admin, parse_fields
from nvmf_failover.topology import controllers

from conftest import FakeSystem, fc_addr

HOSTNQN = "nqn.2014-08.org.nvmexpress:uuid:3133"

LIFS = """\
vserver             lif                     wwpn                    address status-admin status-oper curr-node curr-port
------------------- ----------------------- ----------------------- ------- ------------ ----------- --------- ---------
nvme-svm-dolin-lpfc suse-a700-136-231_lif_1 20:19:00:a0:98:90:f5:bf -       up           up          node-1    1a
nvme-svm-dolin-lpfc suse-a700-136-231_lif_2 20:18:00:a0:98:90:f5:bf -       down         down        node-1    1b
other-svm           other_lif               20:30:00:a0:98:90:f5:bf -       up           up          node-2    2a
3 entries were displayed.
"""

HOSTS = """\
vserver             subsystem             host-nqn
------------------- --------------------- ------------------------------------
nvme-svm-dolin-lpfc nvme-svm-dolin-lpfc-s nqn.2014-08.org.nvmexpress:uuid:3133
other-svm           other-s               nqn.2014-08.org.nvmexpress:uuid:9999
2 entries were displayed.
"""

MAPS = """\
vserver             subsystem             path
------------------- --------------------- -------------------
nvme-svm-dolin-lpfc nvme-svm-dolin-lpfc-s /vol/v1/ns1
nvme-svm-dolin-lpfc nvme-svm-dolin-lpfc-s /vol/v1/ns2
2 entries were displayed.
"""

SUBS = """\
vserver             subsystem             target-nqn
------------------- --------------------- ---------------------------------
nvme-svm-dolin-lpfc nvme-svm-dolin-lpfc-s nqn.1992-08.com.netapp:sn.1:sub.s
1 entry is displayed.
"""


class FakeShell:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> str:
        self.commands.append(command)
        if command.startswith("network interface show -data-protocol fc-nvme"):
            return LIFS
        if command.startswith("vserver nvme subsystem host show"):
            return HOSTS
        if command.startswith("vserver nvme subsystem map show"):
            return MAPS
        if command.startswith("vserver nvme subsystem show"):
            return SUBS
        return "There are no entries matching your query.\n"


def test_parse_fields_reads_rows_until_the_count() -> None:
    rows = parse_fields(LIFS)
    assert len(rows) == 3
    assert rows[1]["status-admin"] == "down"
    assert rows[0]["curr-port"] == "1a"


def test_parse_fields_without_table_is_empty() -> None:
    assert parse_fields("There are no entries matching your query.") == []


def test_discover_matches_lifs_to_controllers(system: FakeSystem) -> None:
    system.add_ctrl("nvme5", fc_addr("0x201900a09890f5bf"))
    system.add_ctrl(
        "nvme6", fc_addr("0x201900a09890f5bf", host="nn-0x2:pn-0x100000109b579ef6")
    )

    inv = discover(FakeShell(), HOSTNQN, controllers(system))

    assert [s.name for s in inv.subsystems] == ["nvme-svm-dolin-lpfc-s"]
    assert inv.subsystems[0].namespaces == ("/vol/v1/ns1", "/vol/v1/ns2")
    assert inv.subsystems[0].nqn == "nqn.1992-08.com.netapp:sn.1:sub.s"
    # only the vservers serving this host
    assert [lif.name for lif in inv.host_lifs()] == [
        "suse-a700-136-231_lif_1",
        "suse-a700-136-231_lif_2",
    ]
    assert inv.paths[("nvme-svm-dolin-lpfc", "suse-a700-136-231_lif_1")] == [
        "nvme5",
        "nvme6",
    ]
    assert inv.paths[("nvme-svm-dolin-lpfc", "suse-a700-136-231_lif_2")] == []


def test_tcp_lif_matches_by_address(system: FakeSystem) -> None:
    system.add_ctrl("nvme1", "traddr=10.0.0.5,trsvcid=4420", transport="tcp")
    lif = Lif("svm", "lif", "nvme-tcp", "10.0.0.5", True, True)
    assert lif_matches(lif, controllers(system)[0])


class ErrorShell:
    def run(self, command: str) -> str:
        return 'Error: invalid argument "-lif"'


def test_lif_admin_sends_the_state() -> None:
    shell = FakeShell()
    lif_admin(shell, "svm", "lif_1", up=False)
    assert shell.commands == [
        "network interface modify -vserver svm -lif lif_1 -status-admin down"
    ]


def test_ontap_errors_on_stdout_raise() -> None:
    # OntapShell.run checks the output; exercise the check without ssh
    shell = OntapShell.__new__(OntapShell)
    shell._lock = threading.Lock()

    class Stream:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def read(self) -> bytes:
            return self.data

    class Client:
        def exec_command(
            self, command: str, timeout: int
        ) -> tuple[None, Stream, Stream]:
            return None, Stream(b"Error: entry doesn't exist.\n"), Stream(b"")

    shell._client = Client()
    try:
        shell.run("network interface modify ...")
    except OntapError as err:
        assert "doesn't exist" in str(err)
    else:
        raise AssertionError("no OntapError")
