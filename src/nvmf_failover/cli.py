"""nvmf-failover: break NVMe-oF paths on purpose and check the host recovers."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import time
import tomllib
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import FrameType

from . import __version__, model, powerstore, resolve
from . import config as config_mod
from . import scenario as scenario_mod
from .brocade import SwitchPool
from .context import fault_context
from .discover import Inventory, discover
from .execute import Observer, execute, expectation_failures
from .faults import FaultError
from .load import Job, Load, LoadResult
from .monitor import Event, Monitor, live_by_identity
from .ontap import OntapShell, Shell
from .system import System
from .topology import Identity, controllers
from .verify import failed_units, verdict, wait_recovered


def _hostnqn(system: System) -> str:
    return system.read("/etc/nvme/hostnqn")


def inventory(
    cfg: config_mod.Config, shell: Shell | None, hostnqn: str, system: System
) -> Inventory:
    """What every configured array knows about this host."""
    ctrls = controllers(system)
    inv = discover(shell, hostnqn, ctrls) if shell else Inventory(hostnqn=hostnqn)
    if cfg.powerstore:
        sub, ports, paths = powerstore.discover(
            powerstore.PowerStoreApi(cfg.powerstore), hostnqn, ctrls
        )
        if sub:
            inv.subsystems.append(sub)
            inv.lifs += ports
            inv.paths |= paths
    return inv


@contextlib.contextmanager
def open_context(
    cfg: config_mod.Config, system: System, with_inventory: bool = False
) -> Iterator[scenario_mod.Context]:
    """What building faults needs; the connections are closed afterwards."""
    pool = SwitchPool()
    shell: OntapShell | None = OntapShell(cfg.ontap) if cfg.ontap else None
    try:
        inv = (
            inventory(cfg, shell, _hostnqn(system), system) if with_inventory else None
        )
        yield scenario_mod.Context(
            system,
            shell,
            cfg.ontap.vserver if cfg.ontap else "",
            inv,
            switches=cfg.brocade,
            connect=pool,
        )
    finally:
        pool.close()
        if shell:
            shell.close()


def cmd_status(args: argparse.Namespace, system: System) -> int:
    for ctrl in controllers(system):
        if args.filter and args.filter not in ctrl.identity.short():
            continue
        paths = " ".join(f"{p}:{a}" for p, a in ctrl.paths.items())
        print(
            f"{ctrl.name:8} {ctrl.state:10} {ctrl.cntrltype:9} "
            f"{ctrl.identity.short()}  {paths}"
        )
    return 0


def cmd_discover(args: argparse.Namespace, system: System) -> int:
    cfg = config_mod.load(args.config)
    if cfg.ontap is None and cfg.powerstore is None:
        print("no [ontap] or [powerstore] in the configuration", file=sys.stderr)
        return 2
    hostnqn = args.hostnqn or _hostnqn(system)
    shell = OntapShell(cfg.ontap) if cfg.ontap else None
    try:
        inv = inventory(cfg, shell, hostnqn, system)
    finally:
        if shell:
            shell.close()
    if args.json:
        print(json.dumps(_inventory_json(inv), indent=2))
        return 0
    print(f"host NQN: {inv.hostnqn}")
    for sub in inv.subsystems:
        print(f"subsystem {sub.vserver}/{sub.name} {sub.nqn}")
        for path in sub.namespaces:
            print(f"  namespace {path}")
    for lif in inv.host_lifs():
        state = f"{'up' if lif.admin_up else 'down'}/{'up' if lif.oper_up else 'down'}"
        ctrls = " ".join(inv.paths.get((lif.vserver, lif.name), [])) or "-"
        print(
            f"lif {lif.vserver}/{lif.name} {lif.protocol} {lif.address} "
            f"{state} {lif.node}:{lif.port} -> {ctrls}"
        )
    return 0


def _inventory_json(inv: Inventory) -> dict[str, object]:
    return {
        "hostnqn": inv.hostnqn,
        "subsystems": [asdict(sub) for sub in inv.subsystems],
        "lifs": [
            asdict(lif) | {"controllers": inv.paths.get((lif.vserver, lif.name), [])}
            for lif in inv.host_lifs()
        ],
    }


def cmd_fault(args: argparse.Namespace, system: System) -> int:
    cfg = config_mod.load(args.config)
    raw = scenario_mod.fault_raw(
        args.kind, args.target, args.action, args.vserver or "", args.force
    )
    pool = SwitchPool()
    try:
        with fault_context(cfg, system, args.kind, pool) as ctx:
            fault = scenario_mod.make_fault(raw, ctx)
            getattr(fault, args.action)()
    finally:
        pool.close()
    print(f"{args.action} {fault.key}")
    return 0


def cmd_monitor(args: argparse.Namespace, system: System) -> int:
    start = time.time()
    monitor = Monitor(
        system,
        journal_units=_journal_units(args),
        sink=lambda e: print(e.line(start), flush=True),
    )
    monitor.start()
    try:
        time.sleep(args.seconds) if args.seconds else signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
    return 0


def _journal_units(args: argparse.Namespace) -> tuple[str, ...]:
    return () if args.no_journal else ("nvme-discoverd*",)


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def load_scenario(path: Path) -> dict[str, object]:
    try:
        data: dict[str, object] = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise scenario_mod.ScenarioError(f"{path}: {err}") from err
    return data


def plan_for(
    data: dict[str, object], where: str, ctx: scenario_mod.Context, system: System
) -> resolve.Plan:
    """The rounds of a descriptive scenario, on this host."""
    described = resolve.parse(data, where)
    inv = ctx.inventory or Inventory(hostnqn=_hostnqn(system))
    mdl = model.build(system, inv, ctx.switches, ctx.connect if ctx.switches else None)
    timers = resolve.kernel_timers(
        system, [p.ctrl.name for p in mdl.select(**described.selector)]
    )
    return resolve.resolve(described, mdl, ctx, timers)


def cmd_plan(args: argparse.Namespace, system: System) -> int:
    cfg = config_mod.load(args.config)
    data = load_scenario(args.scenario)
    if not resolve.is_described(data):
        print("not a descriptive scenario (no [[phase]])", file=sys.stderr)
        return 2
    with open_context(cfg, system, with_inventory=True) as ctx:
        print("\n".join(plan_for(data, str(args.scenario), ctx, system).lines()))
    return 0


def _load_jobs(args: argparse.Namespace, system: System, nqns: list[str]) -> list[Job]:
    """The fio jobs of --io (read) and --io-verify (write and verify)."""
    jobs = []
    for device in args.io or []:
        if device == "all":
            # every namespace of the selected subsystems
            jobs += [Job(d) for nqn in nqns for d in model.namespaces(system, nqn)]
        elif device == "auto":
            first = next(
                (d for nqn in nqns for d in model.namespaces(system, nqn)), None
            )
            if first is None:
                raise scenario_mod.ScenarioError(
                    "--io auto: no namespace of the selected paths"
                )
            jobs.append(Job(first))
        else:
            jobs.append(Job(device))
    jobs += [Job(device, "verify") for device in args.io_verify or []]
    return jobs


def cmd_run(args: argparse.Namespace, system: System) -> int:
    cfg = config_mod.load(args.config)
    data = load_scenario(args.scenario)
    described = resolve.is_described(data)
    out = args.run_dir or args.out / time.strftime(
        f"%Y%m%d-%H%M%S-{args.scenario.stem}"
    )
    if (out / "events.log").exists():
        print(f"{out}: holds a run already", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log = (out / "events.log").open("w")

    def sink(event: Event) -> None:
        line = event.line(start)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    signal.signal(signal.SIGTERM, _interrupt)
    failures: list[str] = []
    with open_context(cfg, system, with_inventory=True) as ctx:
        if described:
            plan = plan_for(data, str(args.scenario), ctx, system)
            name, recover_timeout = plan.name, plan.recover_timeout
            baseline = {
                p.ctrl.identity: [p.ctrl.name]
                for p in plan.paths
                if p.ctrl.state == "live"
            }
            observer = Observer({p.ctrl.name: p.ctrl.identity for p in plan.paths})
            nqns = sorted({p.subsysnqn for p in plan.paths})
        else:
            scen = scenario_mod.parse(data, str(args.scenario))
            setup = scenario_mod.expand(scen.setup, ctx)
            steps = scenario_mod.expand(scen.steps, ctx)
            name, recover_timeout = scen.name, scen.recover_timeout
            baseline = live_by_identity(system)
            observer = None
            nqns = sorted({ident.subsysnqn for ident in baseline})

        def live() -> dict[Identity, list[str]]:
            return {k: v for k, v in live_by_identity(system).items() if k in baseline}

        failed_before = failed_units(system)
        monitor = Monitor(
            system, journal_units=_journal_units(args), sink=sink, sampler=observer
        )
        jobs = _load_jobs(args, system, nqns)
        load = Load(jobs, out / "fio.json") if jobs else None
        monitor.add("step", f"scenario {name}: {len(baseline)} live paths")
        if described:
            for line in plan.lines():
                monitor.add("step", line)
        if load:
            monitor.add("step", f"I/O on {', '.join(load.devices)}")
        monitor.start()
        if load:
            load.start()
        runner = scenario_mod.Runner(time.sleep, lambda t: monitor.add("step", t))
        interrupted = False
        unrecovered: list[str] = []
        try:
            if described:
                unrecovered = execute(
                    plan,
                    runner,
                    time.sleep,
                    lambda t: wait_recovered(baseline, live, t)[0],
                    lambda t: monitor.add("step", t),
                )
            else:
                runner.run(setup, steps, scen.repeat)
        except KeyboardInterrupt:
            interrupted = True
            monitor.add("step", "interrupted, faults restored")
        monitor.add("step", "waiting for every path to come back")
        recovered, seconds, now = wait_recovered(baseline, live, recover_timeout)
        result: LoadResult | None = load.stop() if load else None
        monitor.stop()
        if described and observer is not None:
            failures = expectation_failures(
                plan.expect,
                observer,
                result,
                unrecovered,
                plan.names,
            )

    result_v = verdict(
        baseline,
        recovered,
        seconds,
        now,
        failed_before,
        failed_units(system),
        list(runner.down),
        result,
    )
    result_v.failures = failures
    with (out / "events.jsonl").open("w") as f:
        monitor.dump(f)
    summary = {
        "scenario": name,
        "passed": result_v.passed,
        "interrupted": interrupted,
        "lines": result_v.lines(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.close()
    print("\n".join(result_v.lines()))
    print(f"logs in {out}")
    return 0 if result_v.passed and not interrupted else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nvmf-failover", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        help=f"configuration file (default {config_mod.DEFAULT_PATH})",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="the fabrics controllers and their paths")
    s.add_argument(
        "filter", nargs="?", default="", help="only identities containing this text"
    )
    s.set_defaults(func=cmd_status)

    s = sub.add_parser(
        "discover",
        help="read the array configuration for "
        "this host and match it to the host's controllers",
    )
    s.add_argument("--json", action="store_true")
    s.add_argument("--hostnqn", help="the host's NQN (default: from /etc/nvme/hostnqn)")
    s.set_defaults(func=cmd_discover)

    s = sub.add_parser("fault", help="apply one fault by hand")
    s.add_argument("kind", choices=list(scenario_mod.TARGETS))
    s.add_argument("target", help="hostN, [VSERVER/]LIF, WWPN, or ADDRESS[:PORT]")
    s.add_argument("action", choices=scenario_mod.ACTIONS)
    s.add_argument("--vserver", help="for ontap-lif, default from the config")
    s.add_argument(
        "--force",
        action="store_true",
        help="for brocade-port: disable a port other WWPNs use too; for "
        "ontap-lif: a LIF of a vserver that serves other hosts too",
    )
    s.set_defaults(func=cmd_fault)

    s = sub.add_parser(
        "monitor", help="print controller, path and daemon events until interrupted"
    )
    s.add_argument("--seconds", type=float, default=0)
    s.add_argument(
        "--no-journal",
        action="store_true",
        help="don't follow the nvme-discoverd journal",
    )
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser(
        "plan", help="resolve a descriptive scenario and print what it would do"
    )
    s.add_argument("scenario", type=Path)
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("run", help="run a scenario and check the recovery")
    s.add_argument("scenario", type=Path)
    s.add_argument(
        "--io",
        action="append",
        metavar="DEVICE",
        help="read from DEVICE with fio during the run (repeatable); auto: the "
        "first namespace of the selected paths, all: every one of them",
    )
    s.add_argument(
        "--io-verify",
        action="append",
        metavar="DEVICE",
        help="write to DEVICE and verify every block (repeatable; DESTROYS "
        "the data on DEVICE)",
    )
    s.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="directory for the run logs (default ./runs)",
    )
    s.add_argument(
        "--run-dir",
        type=Path,
        help="the logs of this run exactly here, instead of a new directory in --out",
    )
    s.add_argument(
        "--no-journal",
        action="store_true",
        help="don't follow the nvme-discoverd journal",
    )
    s.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None, system: System | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args, system or System()))
    except (config_mod.ConfigError, scenario_mod.ScenarioError, FaultError) as err:
        print(f"nvmf-failover: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
