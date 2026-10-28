"""The MCP server: JSON-RPC 2.0 over stdio, one message per line."""

from __future__ import annotations

import contextlib
import json
import re
import sys
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .. import __version__, cli, model, resolve
from ..brocade import BrocadeError, SwitchPool, normalize_wwpn
from ..config import Config
from ..context import fault_context
from ..faults import Fault
from ..scenario import fault_raw, make_fault
from ..system import System
from ..topology import controllers
from .holds import Holds
from .runs import Runs

# The protocol versions this server speaks, newest first
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

# Added to the longest kernel timer for the default --max-hold
MAX_HOLD_MARGIN = 120.0
# --max-hold when no kernel timer can be read
MAX_HOLD_FALLBACK = 600.0

INSTRUCTIONS = """\
Breaks NVMe-oF paths of this host on purpose (faults) and runs failover
scenarios. It does not watch the host: use nvme-mcp's nvme_watch for that,
started in parallel with the fault.

Call discover first: faults only take the ports it lists. A fault put
down is held for hold_seconds and then put up by the server, also when
the client is gone. hold_seconds takes kernel timer names, e.g.
"dev_loss_tmo + 10s"; status lists the timers of this host.

Scenario runs take minutes to hours: run_start returns at once, the run
goes on in its own systemd unit when the client disconnects. Follow it
with run_status, read the verdict with run_report."""

_NAME = re.compile(r"[a-z0-9_-]+(/[a-z0-9_-]+)?")
_DEVICE = re.compile(r"/dev/nvme\d+n\d+")
_WWPN = re.compile(r"([0-9a-f]{2}:){7}[0-9a-f]{2}")
_SINGLE_KINDS = ("fc-link", "fc-lip", "pci-unbind", "ontap-lif", "tcp-port")


class ToolError(Exception):
    """A tool call failed: the result reports it to the agent."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    read_only: bool = True
    destructive: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                **self.schema,
            },
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "idempotentHint": self.read_only,
                "openWorldHint": False,
            },
        }


@dataclass
class Options:
    """What the one who starts the server decides, never the agent."""

    config: Config
    config_path: Path | None
    scenarios: Path
    out: Path
    state: Path
    max_hold: float | None = None  # None: from the kernel timers
    allow_force: bool = False
    allow_io_verify: bool = False


@dataclass
class Server:
    opts: Options
    system: System
    holds: Holds = field(init=False)
    runs: Runs = field(init=False)
    run_command: Callable[[Path, list[str]], list[str]] | None = None

    def __post_init__(self) -> None:
        self.pool = SwitchPool()
        self.holds = Holds(self._open_fault, self.opts.state)
        self.runs = Runs(
            self.system, self.opts.out, self.run_command or self._run_command
        )
        self.model: model.Model | None = None
        self.vservers: set[str] = set()  # of the host's subsystems
        self.tools = {t.name: t for t in self._tools()}

    def close(self) -> None:
        self.pool.close()

    # --- tool definitions ---------------------------------------------------

    def _tools(self) -> list[Tool]:
        run_id = {
            "type": "string",
            "description": "the id run_start returned (see list_runs)",
        }
        return [
            Tool(
                "status",
                "The fabrics controllers of this host with their state and "
                "the ANA state of each path; the faults this server holds "
                "down, with the seconds left; the running scenario; the "
                "kernel timers (seconds) and the longest hold allowed.",
                {
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "only controllers whose identity "
                            "contains this text",
                        }
                    }
                },
                self._status,
            ),
            Tool(
                "discover",
                "The ports the paths of this host go through: host ports "
                "(SCSI host, WWPN, driver, switch port), target ports (array, "
                "group, node, switch port), the paths over them, and the "
                "fault kinds each port can take. Logs into the arrays and "
                "switches, so it takes seconds; the result is cached.",
                {
                    "properties": {
                        "refresh": {
                            "type": "boolean",
                            "default": False,
                            "description": "read the arrays and switches again",
                        }
                    }
                },
                self._discover,
            ),
            Tool(
                "list_scenarios",
                "The scenarios run_start accepts, with their descriptions.",
                {},
                self._list_scenarios,
            ),
            Tool(
                "plan",
                "What a scenario would do on this host: its rounds, faults "
                "and durations, resolved against the ports and timers here.",
                {
                    "properties": {"scenario": {"type": "string"}},
                    "required": ["scenario"],
                },
                self._plan,
            ),
            Tool(
                "fault",
                "Puts a port down, up, or bounces its link. down holds the "
                "fault for hold_seconds, then the server puts it up (again: "
                "re-arms the timer); up ends a hold early. The result has "
                "the wall clock time it was applied, to line it up with "
                "nvme_watch. Refused while a scenario runs.",
                {
                    "properties": {
                        "kind": {"enum": [*_SINGLE_KINDS, "brocade-port"]},
                        "target": {
                            "type": "string",
                            "description": "a port from discover: hostN, "
                            "VSERVER/LIF, a WWPN (brocade-port), "
                            "ADDRESS[:PORT] (tcp-port)",
                        },
                        "action": {"enum": ["down", "up", "bounce"]},
                        "hold_seconds": {
                            "type": ["number", "string"],
                            "description": "for down: seconds, or an "
                            'expression like "dev_loss_tmo + 10s"',
                        },
                        "force": {
                            "type": "boolean",
                            "default": False,
                            "description": "brocade-port: disable a port "
                            "other WWPNs use too; ontap-lif: a LIF of a "
                            "vserver other hosts use too (if the server "
                            "allows it)",
                        },
                    },
                    "required": ["kind", "target", "action"],
                },
                self._fault,
                read_only=False,
                destructive=True,
            ),
            Tool(
                "restore_all",
                "Puts up every fault this server holds and stops the running "
                "scenario (which restores its own faults).",
                {},
                self._restore_all,
                read_only=False,
            ),
            Tool(
                "run_start",
                "Starts a scenario in its own systemd unit and returns its "
                "id at once; the run goes on without the client.",
                {
                    "properties": {
                        "scenario": {
                            "type": "string",
                            "description": "a name from list_scenarios",
                        },
                        "io": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "read with fio during the run: "
                            "/dev/nvmeXnY, auto (the first namespace of the "
                            "selected paths) or all",
                        },
                        "io_verify": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "write and verify /dev/nvmeXnY: "
                            "DESTROYS its data (if the server allows it)",
                        },
                    },
                    "required": ["scenario"],
                },
                self._run_start,
                read_only=False,
                destructive=True,
            ),
            Tool(
                "run_status",
                "The state of a run (running, passed, failed, stopped, "
                "crashed), its current round and its latest events.",
                {
                    "properties": {
                        "id": run_id,
                        "lines": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "default": 20,
                        },
                    },
                    "required": ["id"],
                },
                self._run_status,
            ),
            Tool(
                "run_stop",
                "Stops a run; it restores its faults and writes its report.",
                {"properties": {"id": run_id}, "required": ["id"]},
                self._run_stop,
                read_only=False,
            ),
            Tool(
                "run_report",
                "The verdict of a finished run: PASS/FAIL, recovery time, "
                "paths not back, failed expectations, I/O errors.",
                {"properties": {"id": run_id}, "required": ["id"]},
                self._run_report,
            ),
            Tool(
                "list_runs",
                "The runs started through this server, newest first.",
                {},
                self._list_runs,
            ),
        ]

    # --- tools ------------------------------------------------------------

    def _status(self, args: dict[str, Any]) -> dict[str, Any]:
        flt = _opt_str(args, "filter")
        ctrls = [
            {
                "name": c.name,
                "state": c.state,
                "type": c.cntrltype,
                "identity": c.identity.short(),
                "paths": c.paths,
            }
            for c in controllers(self.system)
            if not flt or flt in c.identity.short()
        ]
        timers = self._timers()
        return {
            "controllers": ctrls,
            **self.holds.json(),
            "running": self.runs.active(),
            "kernel_timers": timers,
            "max_hold_seconds": self._max_hold(timers),
        }

    def _discover(self, args: dict[str, Any]) -> dict[str, Any]:
        mdl = self._model(refresh=bool(args.get("refresh", False)))
        pci = {hp.name for hp in mdl.host_ports if self._has_pci(hp.name)}
        return {
            "host_ports": [
                {
                    "name": hp.name,
                    "wwpn": hp.wwpn,
                    "driver": hp.driver,
                    "switch_port": _switch_port(hp.switch_port),
                    "kinds": [
                        k
                        for k, ok in (
                            ("fc-link", hp.driver == "lpfc"),
                            ("fc-lip", True),
                            ("pci-unbind", hp.name in pci),
                            ("brocade-port", hp.switch_port is not None),
                        )
                        if ok
                    ],
                }
                for hp in mdl.host_ports
            ],
            "target_ports": [
                {
                    "array": tp.array,
                    "group": tp.group,
                    "name": tp.name,
                    "address": tp.address,
                    "node": tp.node,
                    "switch_port": _switch_port(tp.switch_port),
                    "kinds": self._target_kinds(tp),
                }
                for tp in self._target_ports(mdl)
            ],
            "paths": [
                {
                    "controller": p.ctrl.name,
                    "state": p.ctrl.state,
                    "subsystem": mdl.subsystem_names.get(p.subsysnqn, p.subsysnqn),
                    "host_port": p.host_port.name if p.host_port else None,
                    "target_port": p.target_port.key if p.target_port else None,
                }
                for p in mdl.paths
            ],
        }

    def _list_scenarios(self, args: dict[str, Any]) -> dict[str, Any]:
        out = []
        for path in sorted(self.opts.scenarios.rglob("*.toml")):
            name = path.relative_to(self.opts.scenarios).with_suffix("").as_posix()
            text = path.read_text()
            try:
                data = tomllib.loads(text)
            except tomllib.TOMLDecodeError as err:
                out.append({"name": name, "error": str(err)})
                continue
            comment = [
                line.lstrip("# ").rstrip()
                for line in text.splitlines()
                if line.startswith("#")
            ]
            out.append(
                {
                    "name": name,
                    "title": str(data.get("name", name)),
                    "format": "phases" if resolve.is_described(data) else "steps",
                    "description": " ".join(comment),
                }
            )
        return {"scenarios": out}

    def _plan(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._scenario(_str(args, "scenario"))
        data = cli.load_scenario(path)
        if not resolve.is_described(data):
            raise ToolError(f"{path.stem}: a step scenario, it has no plan")
        with cli.open_context(self.opts.config, self.system, True) as ctx:
            plan = cli.plan_for(data, str(path), ctx, self.system)
        return {"scenario": plan.name, "lines": plan.lines()}

    def _fault(self, args: dict[str, Any]) -> dict[str, Any]:
        kind, target = _str(args, "kind"), _str(args, "target")
        action = _str(args, "action")
        force = bool(args.get("force", False))
        if force and not self.opts.allow_force:
            raise ToolError("force needs the server started with --allow-force")
        if action == "down" and self.runs.active():
            raise ToolError(f"run {self.runs.active()[0]} is running")
        self._check_target(kind, target)
        raw = fault_raw(kind, target, action, force=force)
        if action == "down":
            if "hold_seconds" not in args:
                raise ToolError("down needs hold_seconds")
            timers = self._timers()
            seconds = resolve.duration(args["hold_seconds"], timers)
            limit = self._max_hold(timers)
            if not 0 < seconds <= limit:
                raise ToolError(
                    f"hold_seconds must be above 0 and at most {limit:g} "
                    "(the server's --max-hold)"
                )
            return self.holds.down(raw, seconds)
        if "hold_seconds" in args:
            raise ToolError(f"{action} takes no hold_seconds")
        if action == "up":
            return self.holds.up(raw)
        return self.holds.bounce(raw)

    def _restore_all(self, args: dict[str, Any]) -> dict[str, Any]:
        failed = self.holds.restore_all()
        stopping = [self.runs.stop(i)["id"] for i in self.runs.active()]
        return {"restore_failed": failed, "runs_stopping": stopping}

    def _run_start(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._scenario(_str(args, "scenario"))
        if self.holds.active():
            raise ToolError(
                f"faults held by hand: {', '.join(self.holds.active())}; "
                "put them up first"
            )
        options = []
        for device in _str_list(args, "io"):
            if device not in ("auto", "all"):
                self._check_device(device)
            options += ["--io", device]
        verify = _str_list(args, "io_verify")
        if verify and not self.opts.allow_io_verify:
            raise ToolError("io_verify needs the server started with --allow-io-verify")
        for device in verify:
            self._check_device(device)
            options += ["--io-verify", device]
        return self.runs.start(path, options)

    def _run_status(self, args: dict[str, Any]) -> dict[str, Any]:
        lines = args.get("lines", 20)
        if not isinstance(lines, int) or not 1 <= lines <= 500:
            raise ToolError("lines must be from 1 to 500")
        return self.runs.status(_str(args, "id"), lines)

    def _run_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.runs.stop(_str(args, "id"))

    def _run_report(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.runs.report(_str(args, "id"))

    def _list_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "runs": [{"id": i, "state": self.runs.state(i)} for i in self.runs.ids()]
        }

    # --- helpers ----------------------------------------------------------

    @contextlib.contextmanager
    def _open_fault(self, raw: dict[str, Any]) -> Iterator[Fault]:
        with fault_context(
            self.opts.config, self.system, raw["fault"], self.pool
        ) as ctx:
            yield make_fault(raw, ctx)

    def _model(self, refresh: bool = False) -> model.Model:
        if self.model is None or refresh:
            with cli.open_context(self.opts.config, self.system, True) as ctx:
                if ctx.inventory is None:
                    raise ToolError("no inventory")
                self.vservers = set(ctx.inventory.vservers())
                self.model = model.build(
                    self.system,
                    ctx.inventory,
                    ctx.switches,
                    ctx.connect if ctx.switches else None,
                )
        return self.model

    def _timers(self) -> dict[str, float]:
        # I/O controllers only: a discovery controller has a kato of its own
        ctrls = controllers(self.system)
        return resolve.kernel_timers(
            self.system, [c.name for c in ctrls if c.cntrltype == "io"]
        )

    def _max_hold(self, timers: dict[str, float]) -> float:
        if self.opts.max_hold is not None:
            return self.opts.max_hold
        if not timers:
            return MAX_HOLD_FALLBACK
        return max(timers.values()) + MAX_HOLD_MARGIN

    def _target_ports(self, mdl: model.Model) -> list[model.TargetPort]:
        """The ports of the vservers (clusters) with a subsystem of this host.

        The cluster shows the LIFs of every vserver: those of other hosts
        are none of this server's business.
        """
        return [tp for tp in mdl.target_ports if tp.group in self.vservers]

    def _has_pci(self, host: str) -> bool:
        dev = self.system.path(f"/sys/class/scsi_host/{host}/device")
        return any(
            re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d", p.name)
            for p in dev.resolve().parents
        )

    def _target_kinds(self, tp: model.TargetPort) -> list[str]:
        out = []
        if tp.array == "netapp" and self.opts.config.ontap:
            out.append("ontap-lif")
        if not _WWPN.fullmatch(tp.address):
            out.append("tcp-port")  # an IP address
        if tp.switch_port is not None:
            out.append("brocade-port")
        return out

    def _check_target(self, kind: str, target: str) -> None:
        """Refuse a target that discover doesn't list for @kind."""
        if kind in ("fc-link", "fc-lip", "pci-unbind"):
            hosts = [hp.name for hp in model.host_ports(self.system)]
            if target not in hosts:
                raise ToolError(
                    f"{target}: not a host port (known: {', '.join(hosts) or 'none'})"
                )
            return
        mdl = self._model()
        if kind == "ontap-lif":
            names = {
                tp.key
                for tp in self._target_ports(mdl)
                if "ontap-lif" in self._target_kinds(tp)
            }
            if target not in names:
                raise ToolError(f"{target}: not a LIF of this host, use VSERVER/LIF")
        elif kind == "brocade-port":
            try:
                wwpn = normalize_wwpn(target)
            except BrocadeError as err:
                raise ToolError(str(err)) from err
            known = {hp.wwpn for hp in mdl.host_ports if hp.switch_port} | {
                tp.address for tp in self._target_ports(mdl) if tp.switch_port
            }
            if wwpn not in known:
                raise ToolError(f"{target}: not a port of this host's paths")
        elif kind == "tcp-port":
            address = fault_raw(kind, target, "down")["address"]
            known = {c.identity.traddr for c in controllers(self.system, ("tcp",))} | {
                tp.address for tp in self._target_ports(mdl)
            }
            if address not in known:
                raise ToolError(f"{address}: no path of this host goes there")

    def _check_device(self, device: str) -> None:
        if not _DEVICE.fullmatch(device) or not self.system.path(device).exists():
            raise ToolError(f"{device}: not a namespace device (/dev/nvmeXnY)")

    def _scenario(self, name: str) -> Path:
        if not _NAME.fullmatch(name):
            raise ToolError(f"{name!r}: not a scenario name, see list_scenarios")
        path = self.opts.scenarios / f"{name}.toml"
        if not path.is_file():
            raise ToolError(f"no scenario {name}, see list_scenarios")
        return path

    def _run_command(self, rundir: Path, options: list[str]) -> list[str]:
        """nvmf-failover run, the same code as this server.

        systemd-run passes no environment on: a checkout run through
        PYTHONPATH needs it in the command.
        """
        package = Path(__file__).resolve().parents[2]
        argv = [
            "env",
            f"PYTHONPATH={package}",
            sys.executable,
            "-m",
            "nvmf_failover.cli",
        ]
        if self.opts.config_path:
            argv += ["-c", str(self.opts.config_path)]
        scenario = Path(json.loads((rundir / "run.json").read_text())["scenario"])
        return [
            *argv,
            "run",
            str(scenario.resolve()),
            "--run-dir",
            str(rundir),
            *options,
        ]

    # --- protocol ---------------------------------------------------------

    def handle(self, msg: Any) -> dict[str, Any] | None:
        """The response to @msg, None for a notification."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        method, mid = msg.get("method"), msg.get("id")
        if "id" not in msg:
            return None  # notifications/initialized and the like
        if not isinstance(method, str):
            return _error(mid, INVALID_REQUEST, "no method")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return _error(mid, INVALID_PARAMS, "params must be an object")
        if method == "initialize":
            return _result(mid, self._initialize(params))
        if method == "ping":
            return _result(mid, {})
        if method == "tools/list":
            return _result(mid, {"tools": [t.describe() for t in self.tools.values()]})
        if method == "tools/call":
            return self._call(mid, params)
        return _error(mid, METHOD_NOT_FOUND, f"unknown method {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "nvmf-failover", "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def _call(self, mid: Any, params: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(params.get("name", ""))
        if tool is None:
            return _error(mid, INVALID_PARAMS, f"unknown tool {params.get('name')}")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return _error(mid, INVALID_PARAMS, "arguments must be an object")
        unknown = set(args) - set(tool.schema.get("properties", {}))
        if unknown:
            return _error(mid, INVALID_PARAMS, f"unknown arguments: {sorted(unknown)}")
        try:
            out = tool.run(args)
        except Exception as err:  # noqa: BLE001 -- a failed tool, not a failed server
            return _result(
                mid,
                {"content": [{"type": "text", "text": str(err)}], "isError": True},
            )
        return _result(
            mid,
            {
                "content": [{"type": "text", "text": json.dumps(out)}],
                "structuredContent": out,
                "isError": False,
            },
        )

    def serve(self, rd: IO[str], wr: IO[str]) -> None:
        """Answer the messages from @rd on @wr until @rd ends."""
        for line in rd:
            if not line.strip():
                continue
            resp: dict[str, Any] | None
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as err:
                resp = _error(None, PARSE_ERROR, str(err))
            else:
                resp = self.handle(msg)
            if resp is not None:
                wr.write(json.dumps(resp) + "\n")
                wr.flush()


def _switch_port(sp: model.SwitchPort | None) -> dict[str, Any] | None:
    if sp is None:
        return None
    return {"switch": sp.switch, "index": sp.index, "shared": sp.shared}


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"{key} is required")
    return value


def _opt_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key, "")
    if not isinstance(value, str):
        raise ToolError(f"{key} must be a string")
    return value


def _str_list(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ToolError(f"{key} must be a list of strings")
    return value


def _result(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}
