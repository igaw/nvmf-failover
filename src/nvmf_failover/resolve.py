"""Descriptive scenarios: what to break, said by its place in the paths.

    name = "lose each host port for 5 minutes"
    repeat = 3                       # the phases this often

    [paths]                          # the paths the test is about
    subsystem = "*"                 # glob: name on the array or NQN
    array = "netapp"                 # optional: netapp, powerstore
    node = "*"                       # optional: array node

    [[phase]]
    fail = "host-port"               # see LAYERS
    pick = "each"                    # one, each, all-but-one, all
    duration = "300s"                # also "dev_loss_tmo + 30s"
    gap = "300s"                     # or "until-recovered"

    [expect]
    io_errors = 0                    # with --io
    paths_during = 1                 # never fewer live paths per subsystem
    recovered_within = "dev_loss_tmo + 120s"
    reconnected_by = "any"           # kernel, discoverd or any

The resolver picks how to break each element: an lpfc port through its
link, a qla2xxx port by unbinding it, a cable or a PowerStore port at
its switch port, a NetApp port by its LIF. A switch port that other
hosts use too is refused unless the phase sets allow_shared.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .faults import (
    BrocadePortFault,
    Fault,
    FaultError,
    FcLinkFault,
    OntapLifFault,
    PciUnbindFault,
)
from .model import HostPort, Model, Path, TargetPort
from .scenario import Context, ScenarioError

LAYERS = ("host-port", "host-cable", "target-port", "array-node", "fabric")
PICKS = ("one", "each", "all-but-one", "all")
RECONNECTORS = ("any", "kernel", "discoverd")
UNTIL_RECOVERED = "until-recovered"


# --- durations -------------------------------------------------------------

_TERM = re.compile(r"\s*([+-]?)\s*([a-z_]+|\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*")
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, None: 1.0}


def duration(text: object, timers: dict[str, float]) -> float:
    """Seconds of "300s", "5m", 90 or "dev_loss_tmo + 30s"."""
    if isinstance(text, int | float) and not isinstance(text, bool):
        return float(text)
    if not isinstance(text, str) or not text.strip():
        raise ScenarioError(f"{text!r}: not a duration")
    total, pos = 0.0, 0
    while pos < len(text):
        match = _TERM.match(text, pos)
        if not match or match.end() == pos:
            raise ScenarioError(f"{text!r}: not a duration")
        sign, value, unit = match.groups()
        if pos and not sign:
            # terms are joined by + or -
            raise ScenarioError(f"{text!r}: not a duration")
        if value[0].isdigit():
            seconds = float(value) * _UNITS[unit]
        elif value in timers and unit is None:
            seconds = timers[value]
        else:
            known = ", ".join(sorted(timers)) or "none"
            raise ScenarioError(f"{text!r}: unknown timer {value} (known: {known})")
        total += -seconds if sign == "-" else seconds
        pos = match.end()
    if total < 0:
        raise ScenarioError(f"{text!r}: negative duration")
    return total


# --- the scenario ----------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    fail: str
    pick: str
    duration: str | float
    gap: str | float = 0
    repeat: int = 1
    allow_shared: bool = False


@dataclass(frozen=True)
class Expect:
    io_errors: int | None = 0
    paths_during: int | None = None
    recovered_within: str | float = 300
    reconnected_by: str = "any"


@dataclass(frozen=True)
class Described:
    name: str
    selector: dict[str, str]
    phases: list[Phase]
    expect: Expect = field(default_factory=Expect)
    repeat: int = 1


def is_described(data: dict[str, Any]) -> bool:
    return "phase" in data


def parse(data: dict[str, Any], where: str = "scenario") -> Described:
    selector = data.get("paths", {})
    if not isinstance(selector, dict) or not set(selector) <= {
        "subsystem",
        "array",
        "node",
    }:
        raise ScenarioError(f"{where}: [paths] takes subsystem, array and node")
    phases = []
    raws = data.get("phase")
    if not isinstance(raws, list) or not raws:
        raise ScenarioError(f"{where}: needs at least one [[phase]]")
    for i, raw in enumerate(raws):
        at = f"{where}: phase {i + 1}"
        if raw.get("fail") not in LAYERS:
            raise ScenarioError(f"{at}: fail must be one of {', '.join(LAYERS)}")
        if raw.get("pick", "one") not in PICKS:
            raise ScenarioError(f"{at}: pick must be one of {', '.join(PICKS)}")
        if "duration" not in raw:
            raise ScenarioError(f"{at}: needs a duration")
        if raw.get("pick") == "all" and not raw.get("allow_outage"):
            raise ScenarioError(
                f"{at}: pick = all breaks every path, set allow_outage = true"
            )
        phases.append(
            Phase(
                fail=raw["fail"],
                pick=raw.get("pick", "one"),
                duration=raw["duration"],
                gap=raw.get("gap", 0),
                repeat=int(raw.get("repeat", 1)),
                allow_shared=bool(raw.get("allow_shared", False)),
            )
        )
    exp = data.get("expect", {})
    if exp.get("reconnected_by", "any") not in RECONNECTORS:
        raise ScenarioError(
            f"{where}: reconnected_by must be one of {', '.join(RECONNECTORS)}"
        )
    expect = Expect(
        io_errors=exp.get("io_errors", 0),
        paths_during=exp.get("paths_during"),
        recovered_within=exp.get("recovered_within", 300),
        reconnected_by=exp.get("reconnected_by", "any"),
    )
    return Described(
        str(data.get("name", where)),
        {k: str(v) for k, v in selector.items()},
        phases,
        expect,
        int(data.get("repeat", 1)),
    )


# --- resolving ---------------------------------------------------------------


@dataclass(frozen=True)
class Element:
    """One thing of a layer, e.g. a host port, and the faults that break it."""

    label: str
    faults: tuple[Fault, ...]
    paths: tuple[str, ...]  # controller names that go through it


@dataclass(frozen=True)
class Round:
    phase: int
    label: str
    faults: tuple[Fault, ...]
    paths: tuple[str, ...]
    duration: float
    gap: float | None  # None: until recovered


@dataclass
class Plan:
    name: str
    paths: list[Path]
    rounds: list[Round]
    timers: dict[str, float]
    recover_timeout: float
    expect: Expect
    names: dict[str, str] = field(default_factory=dict)  # NQN -> array name

    def lines(self) -> list[str]:
        out = [f"scenario {self.name}: {len(self.paths)} paths selected"]
        if self.timers:
            out.append(
                "timers: "
                + ", ".join(f"{k} {v:g}s" for k, v in sorted(self.timers.items()))
            )
        for i, rnd in enumerate(self.rounds):
            gap = "until recovered" if rnd.gap is None else f"{rnd.gap:g}s"
            out.append(
                f"round {i + 1} (phase {rnd.phase}) {rnd.label}: "
                f"{rnd.duration:g}s down, then {gap}; "
                f"{len(rnd.paths)} paths: {' '.join(rnd.paths)}"
            )
            out += [f"    {f.key}" for f in rnd.faults]
        out.append(f"recovery within {self.recover_timeout:g}s")
        return out


def _host_port_fault(ctx: Context, hp: HostPort) -> Fault:
    if hp.driver == "lpfc":
        return FcLinkFault(ctx.system, hp.name)
    return PciUnbindFault(ctx.system, hp.name)


def _switch_fault(ctx: Context, wwpn: str, allow_shared: bool) -> Fault:
    if ctx.connect is None or not ctx.switches:
        raise FaultError("needs [[brocade]] switches in the configuration")
    return BrocadePortFault(ctx.switches, ctx.connect, wwpn, force=allow_shared)


def _target_fault(ctx: Context, tp: TargetPort, allow_shared: bool) -> Fault:
    if tp.array == "netapp":
        if ctx.shell is None:
            raise FaultError("a NetApp port needs [ontap] in the configuration")
        return OntapLifFault(ctx.shell, tp.group, tp.name)
    return _switch_fault(ctx, tp.address, allow_shared)


def elements(
    layer: str, paths: list[Path], ctx: Context, allow_shared: bool = False
) -> list[Element]:
    """The elements of @layer that the paths go through, sorted by label."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        hp, tp = path.host_port, path.target_port
        if layer in ("host-port", "host-cable") and hp:
            key = hp.name
        elif layer == "target-port" and tp:
            key = tp.key
        elif layer == "array-node" and tp:
            key = f"{tp.array}:{tp.node}"
        elif layer == "fabric" and hp and hp.switch_port:
            key = hp.switch_port.switch
        else:
            continue
        groups.setdefault(key, []).append(path)
    out = []
    for key in sorted(groups):
        members = groups[key]
        names = tuple(p.ctrl.name for p in members)
        if layer == "host-port":
            hp0 = members[0].host_port
            if hp0 is None:  # grouped by host port: never
                continue
            faults: tuple[Fault, ...] = (_host_port_fault(ctx, hp0),)
        elif layer in ("host-cable", "fabric"):
            hps = sorted({p.host_port.wwpn for p in members if p.host_port})
            faults = tuple(_switch_fault(ctx, w, allow_shared) for w in hps)
        else:
            tps = {p.target_port.key: p.target_port for p in members if p.target_port}
            faults = tuple(
                _target_fault(ctx, tps[k], allow_shared) for k in sorted(tps)
            )
        out.append(Element(f"{layer} {key}", faults, names))
    return out


def _pick(pick: str, elems: list[Element]) -> list[list[Element]]:
    """The elements broken together, per round."""
    if pick == "one":
        return [elems[:1]]
    if pick == "each":
        return [[e] for e in elems]
    if pick == "all-but-one":
        return [elems[:-1]] if len(elems) > 1 else []
    return [elems]


Timers = Callable[[list[Path]], dict[str, float]]


def resolve(
    scen: Described, model: Model, ctx: Context, timers: dict[str, float]
) -> Plan:
    paths = model.select(**scen.selector)
    if not paths:
        raise ScenarioError(f"{scen.name}: no path matches {scen.selector}")
    rounds = []
    for _ in range(scen.repeat):
        for i, phase in enumerate(scen.phases, start=1):
            elems = elements(phase.fail, paths, ctx, phase.allow_shared)
            if not elems:
                raise ScenarioError(
                    f"{scen.name}: phase {i}: the selected paths "
                    f"have no {phase.fail} to break"
                )
            for _ in range(phase.repeat):
                for group in _pick(phase.pick, elems):
                    gap = (
                        None
                        if phase.gap == UNTIL_RECOVERED
                        else duration(phase.gap, timers)
                    )
                    rounds.append(
                        Round(
                            phase=i,
                            label=", ".join(e.label for e in group),
                            faults=tuple(f for e in group for f in e.faults),
                            paths=tuple(n for e in group for n in e.paths),
                            duration=duration(phase.duration, timers),
                            gap=gap,
                        )
                    )
    return Plan(
        scen.name,
        paths,
        rounds,
        timers,
        duration(scen.expect.recovered_within, timers),
        scen.expect,
    )


# Controller attributes (sysfs name, timer name, unit in seconds); the
# per-controller timeouts are in milliseconds
CTRL_TIMERS = (
    ("ctrl_loss_tmo", "ctrl_loss_tmo", 1.0),
    ("fast_io_fail_tmo", "fast_io_fail_tmo", 1.0),
    ("reconnect_delay", "reconnect_delay", 1.0),
    ("kato", "kato", 1.0),
    ("io_timeout", "io_timeout", 0.001),
    ("admin_timeout", "admin_timeout", 0.001),
)


def kernel_timers(system: Any, ctrls: Iterable[str]) -> dict[str, float]:
    """The timers durations may name, in seconds, the largest of each:
    dev_loss_tmo of the NVMe target remote ports, and the fabrics timers of
    the controllers named @ctrls. A timer that is off ("off", -1) or that
    this kernel doesn't show is left out."""
    ctrls = list(ctrls)
    out: dict[str, float] = {}
    base = system.path("/sys/class/fc_remote_ports")
    if base.is_dir():
        values = []
        for rport in base.iterdir():
            prefix = f"/sys/class/fc_remote_ports/{rport.name}"
            if "NVMe Target" in system.read(f"{prefix}/roles"):
                value = system.read(f"{prefix}/dev_loss_tmo")
                if value.isdigit():
                    values.append(float(value))
        if values:
            out["dev_loss_tmo"] = max(values)
    for attr, name, unit in CTRL_TIMERS:
        values = []
        for ctrl in ctrls:
            value = system.read(f"/sys/class/nvme/{ctrl}/{attr}")
            if value.lstrip("-").isdigit() and int(value) >= 0:
                values.append(float(value) * unit)
        if values:
            out[name] = max(values)
    return out
