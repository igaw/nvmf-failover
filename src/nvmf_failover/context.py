"""The connections a fault needs, opened only for the faults that need them.

A link fault on the host writes to sysfs; logging into the cluster shell
for it costs seconds and shifts the timing of the test.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from .brocade import SwitchPool
from .config import Config
from .ontap import OntapShell
from .scenario import Context
from .system import System

ONTAP_KINDS = ("ontap-lif", "ontap-lif-cycle")


@contextlib.contextmanager
def fault_context(
    cfg: Config, system: System, kind: str, pool: SwitchPool
) -> Iterator[Context]:
    """A context for faults of @kind; the cluster shell is closed after.

    The switch connections stay in @pool, which reopens them when they
    dropped.
    """
    shell = OntapShell(cfg.ontap) if cfg.ontap and kind in ONTAP_KINDS else None
    try:
        yield Context(
            system,
            shell,
            cfg.ontap.vserver if cfg.ontap else "",
            switches=cfg.brocade,
            connect=pool,
        )
    finally:
        if shell:
            shell.close()
