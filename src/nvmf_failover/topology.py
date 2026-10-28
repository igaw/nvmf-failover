"""The NVMe-oF controllers and paths of the host, read from sysfs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .system import System

_PATH_RE = re.compile(r"^nvme\d+c\d+n\d+$")

# What sysfs shows for the subsystem NQN and the controller type before a
# controller has connected (and after it lost its subsystem)
_UNKNOWN = ("", "(efault)")


@dataclass(frozen=True, order=True)
class Identity:
    """What makes a controller the same one after a reconnect."""

    transport: str
    subsysnqn: str
    traddr: str
    host_traddr: str
    # TCP and RDMA: a target serves several ports on one address
    trsvcid: str = ""

    def short(self) -> str:
        """A compact form for logs: subsystem name, target, host port."""
        nqn = self.subsysnqn.rsplit(":", 1)[-1]
        target = _port(self.traddr) + (f":{self.trsvcid}" if self.trsvcid else "")
        return f"{nqn} {target} <- {_port(self.host_traddr)}"


@dataclass(frozen=True)
class Controller:
    name: str
    identity: Identity
    state: str
    cntrltype: str
    paths: dict[str, str] = field(default_factory=dict)  # path -> ana_state

    @property
    def identified(self) -> bool:
        """Whether the controller knows its subsystem yet."""
        return self.identity.subsysnqn != ""


def _port(addr: str) -> str:
    """The port part of an FC address (pn-0x...), else the address."""
    return addr.rsplit("pn-", 1)[-1] if "pn-" in addr else addr


def _address(text: str) -> dict[str, str]:
    """The key=value pairs of a controller's address attribute."""
    pairs = (item.split("=", 1) for item in text.split(",") if "=" in item)
    return {key.strip(): value.strip() for key, value in pairs}


def controllers(
    system: System, transports: Iterable[str] = ("fc", "tcp", "rdma")
) -> list[Controller]:
    """The fabrics controllers of the host, sorted by name."""
    wanted = set(transports)
    out = []
    base = system.path("/sys/class/nvme")
    if not base.is_dir():
        return []
    for entry in sorted(base.iterdir(), key=lambda p: _natural(p.name)):
        sysfs = f"/sys/class/nvme/{entry.name}"
        transport = system.read(f"{sysfs}/transport")
        if transport not in wanted:
            continue
        addr = _address(system.read(f"{sysfs}/address"))
        nqn = system.read(f"{sysfs}/subsysnqn")
        identity = Identity(
            transport=transport,
            subsysnqn="" if nqn in _UNKNOWN else nqn,
            traddr=addr.get("traddr", ""),
            host_traddr=addr.get("host_traddr", ""),
            trsvcid=addr.get("trsvcid", ""),
        )
        try:
            children = sorted(entry.iterdir())
        except OSError:
            # deleted while we looked at it: a failover in progress
            continue
        paths = {
            child.name: system.read(f"{sysfs}/{child.name}/ana_state")
            for child in children
            if _PATH_RE.match(child.name)
        }
        out.append(
            Controller(
                name=entry.name,
                identity=identity,
                state=system.read(f"{sysfs}/state"),
                cntrltype=_cntrltype(system.read(f"{sysfs}/cntrltype")),
                paths=paths,
            )
        )
    return out


def _cntrltype(text: str) -> str:
    """io, discovery or admin; "" before the controller has connected."""
    return text if text in ("io", "discovery", "admin") else ""


def by_identity(ctrls: Iterable[Controller]) -> dict[Identity, list[Controller]]:
    """The controllers grouped by identity; more than one is a duplicate."""
    out: dict[Identity, list[Controller]] = {}
    for ctrl in ctrls:
        out.setdefault(ctrl.identity, []).append(ctrl)
    return out


def _natural(name: str) -> tuple[str, int]:
    """Sort key: nvme2 before nvme10."""
    match = re.match(r"^(\D*)(\d+)$", name)
    return (match.group(1), int(match.group(2))) if match else (name, -1)
