"""The ONTAP cluster shell, over ssh."""

from __future__ import annotations

import threading
from typing import Protocol

from . import ssh
from .config import OntapConfig


class OntapError(RuntimeError):
    """ONTAP rejected a command."""


class Shell(Protocol):
    def run(self, command: str) -> str: ...


# ONTAP reports most errors on stdout with exit status 0
_ERROR_MARKERS = ("Error:", "error:", "is not a recognized command", "Invalid choice")


class OntapShell:
    """Runs cluster shell commands; one ssh connection for its lifetime."""

    def __init__(self, cfg: OntapConfig) -> None:
        self._client = ssh.connect(cfg.host, cfg.user, cfg.password)
        # the faults of a round run at once: one command at a time
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        with self._lock:
            out, err = ssh.run(self._client, command)
        text = out + err
        if any(marker in text for marker in _ERROR_MARKERS):
            raise OntapError(f"{command}: {text.strip()}")
        return text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OntapShell:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def lif_admin(shell: Shell, vserver: str, lif: str, up: bool) -> None:
    """Set the administrative status of a LIF."""
    state = "up" if up else "down"
    shell.run(
        f"network interface modify -vserver {vserver} -lif {lif} -status-admin {state}"
    )


def vserver_hosts(shell: Shell, vserver: str) -> list[str]:
    """The host NQNs the subsystems of @vserver list."""
    rows = parse_fields(
        shell.run(
            f"vserver nvme subsystem host show -vserver {vserver} "
            "-fields vserver,subsystem,host-nqn"
        )
    )
    return sorted({row["host-nqn"] for row in rows if row.get("host-nqn")})


def parse_fields(text: str) -> list[dict[str, str]]:
    """Rows of a 'show -fields ...' table: the header, a dashes line, rows.

    ONTAP separates the columns by blanks; values have none except the
    last column, which takes the rest of the line.
    """
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    try:
        start = next(
            i for i, line in enumerate(lines) if set(line.replace(" ", "")) == {"-"}
        )
    except StopIteration:
        return []
    keys = lines[start - 1].split()
    rows = []
    for line in lines[start + 1 :]:
        if line.endswith("entries were displayed.") or line.endswith(
            "entry is displayed."
        ):
            break
        values = line.split(None, len(keys) - 1)
        if len(values) == len(keys):
            rows.append(dict(zip(keys, values, strict=True)))
    return rows
