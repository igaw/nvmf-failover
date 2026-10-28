"""Brocade Fabric OS switches, over ssh: where a WWPN is plugged in, and
taking that switch port down and up.

The switches are shared: a fault only ever touches the port that carries a
WWPN the caller names, found in the name server (nsshow), never a port
number from a file.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from . import ssh
from .config import BrocadeConfig


class BrocadeError(RuntimeError):
    """The switch rejected a command, or a WWPN isn't logged in."""


class Shell(Protocol):
    def run(self, command: str) -> str: ...


@dataclass(frozen=True)
class NameServerEntry:
    pid: str  # 24 bit FC address, e.g. 600300
    wwpn: str  # lower case, colons
    port_index: int
    npiv: bool


@dataclass(frozen=True)
class SwitchPort:
    index: int
    state: str  # Online, No_Light, ...
    comment: str  # F-Port ..., E-Port ..., Disabled ...

    @property
    def disabled(self) -> bool:
        return "Disabled" in self.comment


def normalize_wwpn(wwpn: str) -> str:
    """aa:bb:...: lower case with colons, from 0xaabb..., pn-0x..., or aa:bb."""
    digits = re.sub(r"^(pn-)?0x", "", wwpn.strip().lower()).replace(":", "")
    if not re.fullmatch(r"[0-9a-f]{16}", digits):
        raise BrocadeError(f"{wwpn}: not a WWPN")
    return ":".join(digits[i : i + 2] for i in range(0, 16, 2))


_NS_ENTRY = re.compile(
    r"^\s*N\s+(?P<pid>[0-9a-f]{6});\s*\d+;(?P<wwpn>[0-9a-f:]{23});", re.M
)


def parse_nsshow(text: str) -> list[NameServerEntry]:
    """The local N_Ports of the name server."""
    out = []
    matches = list(_NS_ENTRY.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        index = re.search(r"Port Index:\s*(\d+)", block)
        if index is None:
            continue
        out.append(
            NameServerEntry(
                pid=match["pid"],
                wwpn=match["wwpn"],
                port_index=int(index.group(1)),
                npiv="NPIV" in block.split("Port Index:")[0],
            )
        )
    return out


_PORT_LINE = re.compile(
    r"^\s*(?P<index>\d+)\s+\d+\s+[0-9a-f]{6}\s+\S+\s+\S+\s+(?P<state>\S+)\s+FC\s*"
    r"(?P<comment>.*)$"
)


def parse_switchshow(text: str) -> list[SwitchPort]:
    out = []
    for line in text.splitlines():
        match = _PORT_LINE.match(line)
        if match:
            out.append(
                SwitchPort(
                    int(match["index"]), match["state"], match["comment"].strip()
                )
            )
    return out


class BrocadeShell:
    """One ssh connection to a switch, as the read user or the admin."""

    def __init__(self, cfg: BrocadeConfig, admin: bool = False) -> None:
        import paramiko

        self.name = cfg.name
        # the faults of a round run at once: one command at a time per switch
        self._lock = threading.Lock()
        accounts = [(cfg.user, cfg.password)]
        if not admin and cfg.read_user:
            accounts.insert(0, cfg.reader())
        for i, (user, password) in enumerate(accounts):
            try:
                self._client = ssh.connect(cfg.host, user, password)
                break
            except paramiko.AuthenticationException:
                if i == len(accounts) - 1:
                    raise
                # the read user is a convenience: fall back to the admin
                print(f"{cfg.name}: login as {user} failed, using {cfg.user}")

    def run(self, command: str) -> str:
        with self._lock:
            out, err = ssh.run(self._client, command)
        err = err.strip()
        if err and ("rbash" in err or "not permitted" in err.lower()):
            raise BrocadeError(f"{self.name}: {command}: {err}")
        return out

    @property
    def alive(self) -> bool:
        transport = self._client.get_transport()
        return bool(transport and transport.is_active())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BrocadeShell:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def locate(shell: Shell, wwpn: str) -> NameServerEntry:
    """Where @wwpn is logged in on this switch."""
    wanted = normalize_wwpn(wwpn)
    for entry in parse_nsshow(shell.run("nsshow")):
        if entry.wwpn == wanted:
            return entry
    raise BrocadeError(f"{wanted}: not logged in to this switch")


def port_users(shell: Shell, port_index: int) -> list[NameServerEntry]:
    """Everything logged in through a switch port: what disabling it hits."""
    return [e for e in parse_nsshow(shell.run("nsshow")) if e.port_index == port_index]


def set_port(shell: Shell, port_index: int, enable: bool) -> None:
    shell.run(f"{'portenable' if enable else 'portdisable'} {port_index}")


Connect = Callable[[BrocadeConfig], Shell]


def find_port(
    switches: list[BrocadeConfig], connect: Connect, wwpn: str
) -> tuple[BrocadeConfig, NameServerEntry, list[NameServerEntry]]:
    """The switch and port of @wwpn, and everything logged in through it."""
    wanted = normalize_wwpn(wwpn)
    for cfg in switches:
        shell = connect(cfg)
        entries = parse_nsshow(shell.run("nsshow"))
        for entry in entries:
            if entry.wwpn == wanted:
                users = [e for e in entries if e.port_index == entry.port_index]
                return cfg, entry, users
    names = ", ".join(cfg.name for cfg in switches) or "no switch configured"
    raise BrocadeError(f"{wanted}: not logged in to {names}")


def named_port_users(
    switches: list[BrocadeConfig], connect: Connect, switch: str, port: int
) -> tuple[BrocadeConfig, list[NameServerEntry]]:
    """The switch named @switch, and what is logged in through its @port."""
    cfg = next((c for c in switches if c.name == switch), None)
    if cfg is None:
        raise BrocadeError(f"{switch}: not a configured switch")
    return cfg, port_users(connect(cfg), port)


class SwitchPool:
    """One admin connection per switch, reopened when it dropped.

    A fault may hold a port down longer than the switch's idle timeout;
    the restore must still get in.
    """

    def __init__(self) -> None:
        self._shells: dict[str, BrocadeShell] = {}
        self._lock = threading.Lock()

    def __call__(self, cfg: BrocadeConfig) -> Shell:
        with self._lock:
            shell = self._shells.get(cfg.name)
            if shell is None or not shell.alive:
                shell = BrocadeShell(cfg, admin=True)
                self._shells[cfg.name] = shell
            return shell

    def close(self) -> None:
        for shell in self._shells.values():
            shell.close()
        self._shells.clear()
