"""Running a resolved plan and checking its [expect]."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .faults import Fault
from .load import LoadResult
from .resolve import Expect, Plan
from .scenario import Runner, Step
from .topology import Controller, Identity


@dataclass
class Observer:
    """Sampled by the monitor: live paths per subsystem, replaced controllers.

    A controller the kernel reconnects keeps its device; one it deleted
    comes back (if at all) as a new device. The baseline controllers are
    tracked by device name, because a controller that is reconnecting
    doesn't show its identity in sysfs.
    """

    baseline: dict[str, Identity]  # device name -> identity
    min_live: dict[str, int] = field(default_factory=dict)  # per subsystem NQN
    replaced: set[Identity] = field(default_factory=set)

    def __post_init__(self) -> None:
        counts = Counter(ident.subsysnqn for ident in self.baseline.values())
        self.min_live = dict(counts)

    def __call__(self, ctrls: list[Controller]) -> None:
        by_name = {c.name: c for c in ctrls}
        selected = set(self.baseline.values())
        live = Counter(
            c.identity.subsysnqn
            for c in ctrls
            if c.state == "live" and c.identified and c.identity in selected
        )
        for nqn in self.min_live:
            self.min_live[nqn] = min(self.min_live[nqn], live.get(nqn, 0))
        for name, ident in self.baseline.items():
            ctrl = by_name.get(name)
            if ctrl is None or (ctrl.identified and ctrl.identity != ident):
                self.replaced.add(ident)


Wait = Callable[[float], bool]  # wait up to N seconds for recovery


def apply_all(runner: Runner, faults: Sequence[Fault], action: str) -> None:
    """@action on every fault of a round at once.

    Putting an lpfc link down takes 5 s: one after the other, the first
    of four links would be down 15 s longer than the round says, past
    dev_loss_tmo where the round meant to stay below it.
    """
    if len(faults) == 1:
        runner.apply(Step(faults[0], action, 0))
        return
    with ThreadPoolExecutor(max_workers=len(faults)) as pool:
        done = [pool.submit(runner.apply, Step(f, action, 0)) for f in faults]
    for future in done:
        future.result()  # the first error, after every fault was tried


def execute(
    plan: Plan,
    runner: Runner,
    sleep: Callable[[float], None],
    wait: Wait,
    log: Callable[[str], None],
) -> list[str]:
    """Run the rounds; the rounds after which the paths didn't recover."""
    unrecovered = []
    try:
        for i, rnd in enumerate(plan.rounds, start=1):
            log(f"round {i}/{len(plan.rounds)}: {rnd.label}")
            apply_all(runner, rnd.faults, "down")
            sleep(rnd.duration)
            apply_all(runner, rnd.faults, "up")
            if rnd.gap is None:
                if not wait(plan.recover_timeout):
                    log(
                        f"round {i}: not recovered within "
                        f"{plan.recover_timeout:g}s, going on"
                    )
                    unrecovered.append(f"round {i} ({rnd.label})")
            elif rnd.gap:
                sleep(rnd.gap)
    finally:
        runner.restore()
    return unrecovered


def expectation_failures(
    expect: Expect,
    observer: Observer,
    load: LoadResult | None,
    unrecovered: list[str],
    names: dict[str, str],
) -> list[str]:
    """What [expect] says that didn't hold; @names: NQN -> array name."""
    out = [f"not recovered after {r}" for r in unrecovered]
    if expect.paths_during is not None:
        for nqn, low in sorted(observer.min_live.items()):
            if low < expect.paths_during:
                out.append(
                    f"{names.get(nqn, nqn)}: down to {low} live paths, "
                    f"expected at least {expect.paths_during}"
                )
    if expect.io_errors is not None and load and load.errors > expect.io_errors:
        out.append(f"{load.errors} I/O errors, expected at most {expect.io_errors}")
    if expect.reconnected_by == "kernel" and observer.replaced:
        out.append(
            f"{len(observer.replaced)} controller(s) were deleted and "
            "reconnected, expected the kernel to keep them"
        )
    if expect.reconnected_by == "discoverd" and not observer.replaced:
        out.append(
            "no controller was deleted: the run didn't exercise "
            "nvme-discoverd's reconnect (a longer duration?)"
        )
    return out
