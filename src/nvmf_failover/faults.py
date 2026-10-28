"""The ways to break a path, and to mend it.

A fault has a target and two states. down() breaks the path, up() mends
it; up() of a fault that isn't down does nothing harmful. A bounce (a
link reset) has no down state: it breaks and mends in one go.

spec() gives the step that builds the same fault again without looking
anything up: while a fault is down, what it was found by may be gone (a
disabled switch port's logins, an unbound HBA's SCSI host).
"""

from __future__ import annotations

import errno
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .brocade import (
    BrocadeError,
    Connect,
    find_port,
    named_port_users,
    normalize_wwpn,
    set_port,
)
from .config import BrocadeConfig
from .ontap import Shell, lif_admin, vserver_hosts
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

    def spec(self) -> dict[str, Any]:
        """The step that builds this very fault again, without lookups."""
        raise FaultError(f"{self.key}: can't be built again")

    def down(self) -> None:
        raise FaultError(f"{self.key}: has no down state, use bounce")

    def up(self) -> None:
        raise FaultError(f"{self.key}: has no up state, use bounce")

    def bounce(self) -> None:
        raise FaultError(f"{self.key}: can't bounce, use down and up")

    def __repr__(self) -> str:
        return f"<{self.key}>"


_PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d")


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

    def spec(self) -> dict[str, Any]:
        return {"fault": self.kind, "host": self.host}

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

    def spec(self) -> dict[str, Any]:
        return {"fault": self.kind, "host": self.host}

    def bounce(self) -> None:
        self.system.write(f"/sys/class/fc_host/{self.host}/issue_lip", "1")


class PciUnbindFault(Fault):
    """Unbind the PCI function of an HBA port from its driver, and rebind.

    For drivers without a link control (qla2xxx). The port, and every
    controller through it, disappears until the rebind.
    """

    kind = "pci-unbind"

    def __init__(
        self, system: System, host: str, pci: str = "", driver: str = ""
    ) -> None:
        self.system, self.host = system, _scsi_host(host)
        if pci or driver:
            # from spec(): the SCSI host is gone while the function is unbound
            if not _PCI.fullmatch(pci) or not re.fullmatch(r"[\w-]+", driver):
                raise FaultError(f"{host}: bad PCI function {pci!r} or driver")
            self.pci, self.driver = pci, driver
            return
        dev = system.path(f"/sys/class/scsi_host/{host}/device")
        function = next(
            (
                p
                for p in [dev.resolve(), *dev.resolve().parents]
                if _PCI.fullmatch(p.name)
            ),
            None,
        )
        if function is None:
            raise FaultError(f"{host}: no PCI device")
        self.pci = function.name
        link = system.path(f"/sys/bus/pci/devices/{self.pci}/driver")
        self.driver = Path(link.resolve()).name if link.exists() else ""
        if not self.driver:
            raise FaultError(f"{host}: {self.pci} has no driver")

    @property
    def target(self) -> str:
        return self.host

    def spec(self) -> dict[str, Any]:
        return {
            "fault": self.kind,
            "host": self.host,
            "pci": self.pci,
            "driver": self.driver,
        }

    def down(self) -> None:
        self.system.write(f"/sys/bus/pci/drivers/{self.driver}/unbind", self.pci)

    def up(self) -> None:
        if self.system.path(f"/sys/bus/pci/devices/{self.pci}/driver").exists():
            return
        self.system.write(f"/sys/bus/pci/drivers/{self.driver}/bind", self.pci)


class OntapLifFault(Fault):
    """Set a LIF of the array administratively down and up.

    A LIF serves every host of its vserver. down() refuses a vserver whose
    subsystems list other hosts than @hostnqn unless @force: their paths
    would break too.
    """

    kind = "ontap-lif"

    def __init__(
        self,
        shell: Shell,
        vserver: str,
        lif: str,
        hostnqn: str = "",
        force: bool = False,
    ) -> None:
        if not vserver:
            raise FaultError(f"{lif}: no vserver given or configured")
        self.shell, self.vserver, self.lif = shell, vserver, lif
        self.hostnqn, self.force = hostnqn, force

    @property
    def target(self) -> str:
        return f"{self.vserver}/{self.lif}"

    def spec(self) -> dict[str, Any]:
        return {
            "fault": self.kind,
            "vserver": self.vserver,
            "lif": self.lif,
            "force": self.force,
        }

    def down(self) -> None:
        if not self.force:
            others = sorted(
                set(vserver_hosts(self.shell, self.vserver)) - {self.hostnqn}
            )
            if others:
                raise FaultError(
                    f"{self.vserver} also serves {len(others)} other host(s) "
                    f"({', '.join(others[:4])}{', ...' if len(others) > 4 else ''})"
                    ": set force to take its LIF down anyway"
                )
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

    def spec(self) -> dict[str, Any]:
        return {"fault": self.kind, "address": self.address, "port": self.port}

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
        switch: str = "",
        port: int | None = None,
    ) -> None:
        try:
            if switch and port is not None:
                # from spec(): a disabled port has no logins to find it by
                self.switch, users = named_port_users(switches, connect, switch, port)
                self.wwpn, self.port = normalize_wwpn(wwpn), port
            else:
                self.switch, entry, users = find_port(switches, connect, wwpn)
                self.wwpn, self.port = entry.wwpn, entry.port_index
        except BrocadeError as err:
            raise FaultError(str(err)) from err
        self.connect, self.force = connect, force
        others = [u.wwpn for u in users if u.wwpn != self.wwpn]
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

    def spec(self) -> dict[str, Any]:
        return {
            "fault": self.kind,
            "wwpn": self.wwpn,
            "switch": self.switch.name,
            "port": self.port,
            "force": self.force,
        }

    def down(self) -> None:
        set_port(self.connect(self.switch), self.port, enable=False)

    def up(self) -> None:
        set_port(self.connect(self.switch), self.port, enable=True)
