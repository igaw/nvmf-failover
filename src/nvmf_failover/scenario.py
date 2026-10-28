"""Failover scenarios: what to break, in which order, for how long.

    name = "lpfc LIF failover"
    repeat = 3                 # run the steps this often
    recover_timeout = 300      # seconds for every path to come back

    [[steps]]
    fault = "fc-link"          # see FAULT_KINDS
    host = "host3"
    action = "down"            # down, up or bounce
    hold = 60                  # seconds to wait after the step

    [[steps]]
    fault = "ontap-lif-cycle"  # every LIF of the vserver serving this host,
    vserver = "svm"            # one after the other: down, hold, up, hold
    hold = 20

Every fault a run put down is put up again at the end, also when the run
fails or is interrupted.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .brocade import Connect
from .config import BrocadeConfig
from .discover import Inventory
from .faults import (
    BrocadePortFault,
    Fault,
    FaultError,
    FcLinkFault,
    FcLipFault,
    OntapLifFault,
    PciUnbindFault,
    TcpPortFault,
)
from .ontap import Shell
from .system import System

ACTIONS = ("down", "up", "bounce")
FAULT_KINDS = (
    "fc-link",
    "fc-lip",
    "pci-unbind",
    "ontap-lif",
    "ontap-lif-cycle",
    "tcp-port",
    "brocade-port",
)


class ScenarioError(ValueError):
    """The scenario file is invalid."""


@dataclass(frozen=True)
class Step:
    fault: Fault
    action: str
    hold: float


@dataclass
class Scenario:
    name: str
    steps: list[dict[str, Any]]
    setup: list[dict[str, Any]] = field(default_factory=list)
    repeat: int = 1
    recover_timeout: float = 300.0


@dataclass
class Context:
    """What building the faults needs; the shell only for ontap faults."""

    system: System
    shell: Shell | None = None
    vserver: str = ""
    inventory: Inventory | None = None
    switches: list[BrocadeConfig] = field(default_factory=list)
    connect: Connect | None = None

    def hostnqn(self) -> str:
        """This host's NQN: what a LIF fault may break without force."""
        if self.inventory and self.inventory.hostnqn:
            return self.inventory.hostnqn
        return self.system.read("/etc/nvme/hostnqn")


def load(path: Path) -> Scenario:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise ScenarioError(f"{path}: {err}") from err
    return parse(data, str(path))


def parse(data: dict[str, Any], where: str = "scenario") -> Scenario:
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ScenarioError(f"{where}: needs at least one [[steps]]")
    setup = data.get("setup", [])
    for i, raw in enumerate([*setup, *steps]):
        _check_step(raw, f"{where}: step {i + 1}")
    repeat = data.get("repeat", 1)
    timeout = data.get("recover_timeout", 300)
    if not isinstance(repeat, int) or repeat < 1:
        raise ScenarioError(f"{where}: repeat must be a positive integer")
    if not isinstance(timeout, int | float) or timeout <= 0:
        raise ScenarioError(f"{where}: recover_timeout must be positive")
    return Scenario(
        name=str(data.get("name", where)),
        steps=steps,
        setup=setup,
        repeat=repeat,
        recover_timeout=float(timeout),
    )


def _check_step(raw: object, where: str) -> None:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: must be a table")
    kind = raw.get("fault")
    if kind not in FAULT_KINDS:
        raise ScenarioError(f"{where}: fault must be one of {', '.join(FAULT_KINDS)}")
    if kind != "ontap-lif-cycle" and raw.get("action") not in ACTIONS:
        raise ScenarioError(f"{where}: action must be one of {', '.join(ACTIONS)}")
    hold = raw.get("hold", 0)
    if not isinstance(hold, int | float) or hold < 0:
        raise ScenarioError(f"{where}: hold must be a number >= 0")
    needs = TARGETS.get(str(kind))
    if needs and needs not in raw:
        raise ScenarioError(f"{where}: {kind} needs {needs}")


# What the target of a single fault names, per kind
TARGETS = {
    "fc-link": "host",
    "fc-lip": "host",
    "pci-unbind": "host",
    "ontap-lif": "lif",
    "tcp-port": "address",
    "brocade-port": "wwpn",
}


def fault_raw(
    kind: str, target: str, action: str, vserver: str = "", force: bool = False
) -> dict[str, Any]:
    """The step of one fault given by hand: hostN, [VSERVER/]LIF, WWPN or
    ADDRESS[:PORT] as @target."""
    raw: dict[str, Any] = {"fault": kind, "action": action}
    if kind not in TARGETS:
        raise FaultError(f"{kind}: not a single fault")
    if kind == "ontap-lif":
        if "/" in target and not vserver:
            vserver, _, target = target.partition("/")
        raw |= {"lif": target, "force": force}
        if vserver:
            raw["vserver"] = vserver
    elif kind == "brocade-port":
        raw |= {"wwpn": target, "force": force}
    elif kind == "tcp-port":
        address, sep, port = target.rpartition(":")
        if (
            not sep
            or not port.isdigit()
            or (
                ":" in address
                and not (address.startswith("[") and address.endswith("]"))
            )
        ):
            # a bare address, IPv6 too
            address, port = target, "4420"
        raw |= {"address": address.strip("[]"), "port": port}
    else:
        raw["host"] = target
    return raw


def make_fault(raw: dict[str, Any], ctx: Context) -> Fault:
    """The fault a step names."""
    kind = raw["fault"]
    if kind == "fc-link":
        return FcLinkFault(ctx.system, raw["host"])
    if kind == "fc-lip":
        return FcLipFault(ctx.system, raw["host"])
    if kind == "pci-unbind":
        return PciUnbindFault(
            ctx.system, raw["host"], raw.get("pci", ""), raw.get("driver", "")
        )
    if kind == "tcp-port":
        return TcpPortFault(ctx.system, raw["address"], int(raw.get("port", 4420)))
    if kind == "brocade-port":
        if ctx.connect is None or not ctx.switches:
            raise FaultError("brocade-port needs [[brocade]] in the configuration")
        port = raw.get("port")
        return BrocadePortFault(
            ctx.switches,
            ctx.connect,
            raw["wwpn"],
            bool(raw.get("force", False)),
            switch=str(raw.get("switch", "")),
            port=int(port) if port is not None else None,
        )
    if kind == "ontap-lif":
        if ctx.shell is None:
            raise FaultError("ontap-lif needs [ontap] in the configuration")
        return OntapLifFault(
            ctx.shell,
            raw.get("vserver", ctx.vserver),
            raw["lif"],
            ctx.hostnqn(),
            bool(raw.get("force", False)),
        )
    raise FaultError(f"{kind}: not a single fault")


def expand(raws: list[dict[str, Any]], ctx: Context) -> list[Step]:
    """The steps, with every ontap-lif-cycle unrolled into its LIFs."""
    steps = []
    for raw in raws:
        hold = float(raw.get("hold", 0))
        if raw["fault"] != "ontap-lif-cycle":
            steps.append(Step(make_fault(raw, ctx), raw["action"], hold))
            continue
        if ctx.shell is None or ctx.inventory is None:
            raise FaultError("ontap-lif-cycle needs [ontap] in the configuration")
        vserver = raw.get("vserver", ctx.vserver)
        lifs = [
            lif
            for lif in ctx.inventory.host_lifs(vserver)
            if ctx.inventory.paths.get((lif.vserver, lif.name))
        ]
        if not lifs:
            raise FaultError(f"{vserver}: no LIF of it carries a path of this host")
        for lif in lifs:
            fault = OntapLifFault(
                ctx.shell,
                lif.vserver,
                lif.name,
                ctx.hostnqn(),
                bool(raw.get("force", False)),
            )
            steps += [Step(fault, "down", hold), Step(fault, "up", hold)]
    return steps


Log = Callable[[str], None]


class Runner:
    """Runs steps and always puts back up what it put down."""

    def __init__(self, sleep: Callable[[float], None], log: Log) -> None:
        self.sleep, self.log = sleep, log
        self.down: dict[str, Fault] = {}  # key -> fault, in order put down

    def apply(self, step: Step) -> None:
        self.log(f"{step.action} {step.fault.key}")
        if step.action == "down":
            # remember it before: a half-applied fault must be undone too
            self.down[step.fault.key] = step.fault
            step.fault.down()
        elif step.action == "up":
            step.fault.up()
            self.down.pop(step.fault.key, None)
        else:
            step.fault.bounce()
        if step.hold:
            self.sleep(step.hold)

    def run(self, setup: list[Step], steps: list[Step], repeat: int) -> None:
        try:
            for step in setup:
                self.apply(step)
            for i in range(repeat):
                self.log(f"round {i + 1} of {repeat}")
                for step in steps:
                    self.apply(step)
        finally:
            self.restore()

    def restore(self) -> list[str]:
        """Put up every fault still down; the keys of those that failed."""
        failed = []
        for key, fault in reversed(list(self.down.items())):
            self.log(f"restore {key}")
            try:
                fault.up()
            except Exception as err:  # noqa: BLE001 -- restore the others too
                self.log(f"restore {key} failed: {err}")
                failed.append(key)
            else:
                del self.down[key]
        return failed
