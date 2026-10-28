"""The ways to break a path, and to mend it.

A fault has a target and two states. down() breaks the path, up() mends
it; up() of a fault that isn't down does nothing harmful. A bounce (a
link reset) has no down state: it breaks and mends in one go.
"""

from __future__ import annotations

import errno
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from .brocade import BrocadeError, Connect, find_port, set_port
from .config import BrocadeConfig
from .ontap import Shell, lif_admin
from .system import System


class FaultError(RuntimeError):
    """A fault can't be applied to this target."""


class Fault(ABC):
    kind: str = ""

    @property
    @abstractmethod
    def target(self) -> str: ...

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.target}"

    def down(self) -> None:
        raise FaultError(f"{self.key}: has no down state, use bounce")

    def up(self) -> None:
        raise FaultError(f"{self.key}: has no up state, use bounce")

    def bounce(self) -> None:
        raise FaultError(f"{self.key}: can't bounce, use down and up")

    def __repr__(self) -> str:
        return f"<{self.key}>"


def _scsi_host(name: str) -> str:
    if not re.fullmatch(r"host\d+", name):
        raise FaultError(f"{name}: not a SCSI host name (hostN)")
    return name


# Seconds a link state must hold to count, and to wait for one at most
LINK_SETTLE = 5.0
LINK_TIMEOUT = 30.0


class FcLinkFault(Fault):
    """Take the link of an lpfc port down and up (scsi_host link_state)."""

    kind = "fc-link"

    def __init__(
        self,
        system: System,
        host: str,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.system, self.host = system, _scsi_host(host)
        self.clock, self.sleep = clock, sleep
        driver = system.read(f"/sys/class/scsi_host/{host}/proc_name")
        if driver != "lpfc":
            raise FaultError(
                f"{host}: driver {driver or 'unknown'} can't hold "
                "its link down, use fc-lip or pci-unbind"
            )

    @property
    def target(self) -> str:
        return self.host

    def down(self) -> None:
        self._set("down", "Link Down")

    def up(self) -> None:
        self._set("up", "Link Up")

    def _set(self, value: str, done: str) -> None:
        """lpfc refuses (EINVAL) down unless the link is up, and up unless
        it is down; while it changes, it still shows the old state. A link
        that stays where it should be is no error; one that is still on its
        way is asked again."""
        attr = f"/sys/class/scsi_host/{self.host}/link_state"
        deadline = self.clock() + LINK_TIMEOUT
        while True:
            try:
                self.system.write(attr, value)
                return
            except OSError as err:
                if err.errno != errno.EINVAL:
                    raise
            if self._stays(attr, done):
                return
            if self.clock() >= deadline:
                raise FaultError(
                    f"{self.key}: link stays {self.system.read(attr)!r}, "
                    f"can't set it {value}"
                )
            self.sleep(0.5)

    def _stays(self, attr: str, done: str) -> bool:
        """Whether the link shows @done for LINK_SETTLE seconds."""
        end = self.clock() + LINK_SETTLE
        while self.system.read(attr).startswith(done):
            if self.clock() >= end:
                return True
            self.sleep(0.5)
        return False


class FcLipFault(Fault):
    """Reset the link of an FC port (fc_host issue_lip), any driver."""

    kind = "fc-lip"

    def __init__(self, system: System, host: str) -> None:
        self.system, self.host = system, _scsi_host(host)
        if not system.path(f"/sys/class/fc_host/{host}").is_dir():
            raise FaultError(f"{host}: not an FC host")

    @property
    def target(self) -> str:
        return self.host

    def bounce(self) -> None:
        self.system.write(f"/sys/class/fc_host/{self.host}/issue_lip", "1")


class PciUnbindFault(Fault):
    """Unbind the PCI function of an HBA port from its driver, and rebind.

    For drivers without a link control (qla2xxx). The port, and every
    controller through it, disappears until the rebind.
    """

    kind = "pci-unbind"

    def __init__(self, system: System, host: str) -> None:
        self.system, self.host = system, _scsi_host(host)
        dev = system.path(f"/sys/class/scsi_host/{host}/device")
        pci = next(
            (
                p
                for p in [dev.resolve(), *dev.resolve().parents]
                if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d", p.name)
            ),
            None,
        )
        if pci is None:
            raise FaultError(f"{host}: no PCI device")
        self.pci = pci.name
        driver = system.path(f"/sys/bus/pci/devices/{self.pci}/driver")
        self.driver = Path(driver.resolve()).name if driver.exists() else ""
        if not self.driver:
            raise FaultError(f"{host}: {self.pci} has no driver")

    @property
    def target(self) -> str:
        return self.host

    def down(self) -> None:
        self.system.write(f"/sys/bus/pci/drivers/{self.driver}/unbind", self.pci)

    def up(self) -> None:
        if self.system.path(f"/sys/bus/pci/devices/{self.pci}/driver").exists():
            return
        self.system.write(f"/sys/bus/pci/drivers/{self.driver}/bind", self.pci)


class OntapLifFault(Fault):
    """Set a LIF of the array administratively down and up."""

    kind = "ontap-lif"

    def __init__(self, shell: Shell, vserver: str, lif: str) -> None:
        if not vserver:
            raise FaultError(f"{lif}: no vserver given or configured")
        self.shell, self.vserver, self.lif = shell, vserver, lif

    @property
    def target(self) -> str:
        return f"{self.vserver}/{self.lif}"

    def down(self) -> None:
        lif_admin(self.shell, self.vserver, self.lif, up=False)

    def up(self) -> None:
        lif_admin(self.shell, self.vserver, self.lif, up=True)


class TcpPortFault(Fault):
    """Drop what a TCP target port sends to the host (nftables input hook).

    The rules live in their own table, inet nvmf_failover, never in the
    host's ruleset. Each carries the fault's key as its comment.
    """

    kind = "tcp-port"
    table = ("inet", "nvmf_failover")
    chain = "input"

    def __init__(self, system: System, address: str, port: int) -> None:
        self.system, self.address, self.port = system, address, port
        self.family = "ip6" if ":" in address else "ip"

    @property
    def target(self) -> str:
        return f"{self.address}:{self.port}"

    def _handles(self) -> list[int]:
        """The handles of this fault's rules."""
        family, name = self.table
        if not self.system.ok(["nft", "list", "table", family, name]):
            return []
        data = json.loads(
            self.system.run(
                ["nft", "-j", "-a", "list", "chain", family, name, self.chain]
            )
        )
        return [
            item["rule"]["handle"]
            for item in data.get("nftables", [])
            if "rule" in item and item["rule"].get("comment") == self.key
        ]

    def down(self) -> None:
        if self._handles():
            return
        family, name = self.table
        self.system.run(["nft", "add", "table", family, name])
        self.system.run(
            [
                "nft",
                "add",
                "chain",
                family,
                name,
                self.chain,
                "{ type filter hook input priority -10 ; }",
            ]
        )
        self.system.run(
            [
                "nft",
                "add",
                "rule",
                family,
                name,
                self.chain,
                self.family,
                "saddr",
                self.address,
                "tcp",
                "sport",
                str(self.port),
                "drop",
                "comment",
                f'"{self.key}"',
            ]
        )

    def up(self) -> None:
        family, name = self.table
        for handle in self._handles():
            self.system.run(
                [
                    "nft",
                    "delete",
                    "rule",
                    family,
                    name,
                    self.chain,
                    "handle",
                    str(handle),
                ]
            )
        if (
            self.system.ok(["nft", "list", "table", family, name])
            and not self._rules_left()
        ):
            self.system.run(["nft", "delete", "table", family, name])

    def _rules_left(self) -> bool:
        family, name = self.table
        data = json.loads(self.system.run(["nft", "-j", "list", "table", family, name]))
        return any("rule" in item for item in data.get("nftables", []))


class BrocadePortFault(Fault):
    """Disable and enable the switch port a WWPN is logged in through.

    The port is looked up in the name servers of the configured switches.
    A port that other WWPNs are logged in through as well (an array port
    with NPIV LIFs of other hosts) is refused unless @force: disabling it
    breaks their paths too.
    """

    kind = "brocade-port"

    def __init__(
        self,
        switches: list[BrocadeConfig],
        connect: Connect,
        wwpn: str,
        force: bool = False,
    ) -> None:
        try:
            self.switch, entry, users = find_port(switches, connect, wwpn)
        except BrocadeError as err:
            raise FaultError(str(err)) from err
        self.connect, self.wwpn, self.port = connect, entry.wwpn, entry.port_index
        others = [u.wwpn for u in users if u.wwpn != entry.wwpn]
        if others and not force:
            raise FaultError(
                f"{self.switch.name} port {self.port} also carries "
                f"{len(others)} other login(s) ({', '.join(others[:4])}"
                f"{', ...' if len(others) > 4 else ''}): set force to "
                "disable it anyway"
            )
        self.others = others

    @property
    def target(self) -> str:
        return f"{self.switch.name}/{self.port}({self.wwpn})"

    def down(self) -> None:
        set_port(self.connect(self.switch), self.port, enable=False)

    def up(self) -> None:
        set_port(self.connect(self.switch), self.port, enable=True)
