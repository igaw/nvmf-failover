# nvmf-failover

Break NVMe-oF paths on purpose and check that the host recovers: every
path comes back, without duplicates, without failed connection units, and
without I/O errors.

It is written for testing [nvme-discoverd](https://github.com/linux-nvme/nvme-cli/tree/master/discoverd)
and the kernel's multipath against real arrays, but works for any
NVMe-oF host.

## What it does

- **Faults** on either side of a path:
  - `fc-link`: take the link of an lpfc port down and up
    (`/sys/class/scsi_host/hostN/link_state`).
  - `fc-lip`: reset the link of any FC port (`issue_lip`).
  - `pci-unbind`: unbind an HBA port's PCI function and bind it again, for
    drivers without a link control (qla2xxx).
  - `ontap-lif`: set a NetApp ONTAP LIF administratively down and up. A
    LIF serves every host of its vserver: a vserver whose subsystems list
    other hosts too is refused unless `force = true`.
  - `ontap-lif-cycle`: every LIF of a vserver that carries a path of this
    host, one after the other.
  - `tcp-port`: drop what a TCP target port sends (nftables, in its own
    `inet nvmf_failover` table).
  - `brocade-port`: disable and enable the Brocade switch port a WWPN is
    logged in through, found in the switches' name servers. A port that
    other WWPNs use too (an array port with other hosts' NPIV logins) is
    refused unless `force = true`. This is also the way to take a
    PowerStore port down: its REST API has no call for it.
- **Monitoring** during a run: controller states per identity, ANA states
  per path, the nvme-discoverd journal, optionally fio I/O on a namespace.
- **A verdict** at the end, compared with the state before the run.
- **Discovery** of the array's configuration for this host: subsystems,
  namespaces, LIFs, and which of the host's controllers go through which LIF.
  Nothing about the array is kept in files here.

Every fault a run puts down is put up again at the end, also when a step
fails or the run is interrupted (Ctrl-C, SIGTERM).

## Install

Python 3.11 or later. paramiko is needed only for the ONTAP faults.

    pip install '.[ontap]'

On SLE 15 SP7: `zypper in python311-paramiko` and run it with `python3.11`.

## Configuration

`~/.config/nvmf-failover/config.toml`, readable by its owner only:

```toml
[ontap]
host = "10.0.0.1"
user = "admin"
password = "..."        # or password_env = "ONTAP_PASSWORD"
vserver = "svm-name"    # default for ontap-lif
```

See `config.toml.example` for the `[powerstore]` and `[[brocade]]`
sections. ssh host keys are trusted on first use and kept in
`~/.config/nvmf-failover/known_hosts`.

## Use

    nvmf-failover status                  # controllers, states, paths
    nvmf-failover discover                # the array's view, matched to the host
    nvmf-failover monitor                 # watch events until Ctrl-C
    nvmf-failover fault fc-link host3 down
    nvmf-failover fault fc-link host3 up
    nvmf-failover plan scenarios/host-port-long.toml
    nvmf-failover run scenarios/host-port-long.toml --io auto

A run writes `events.log`, `events.jsonl`, `summary.json` and, with
`--io`, `fio.json` to `runs/<time>-<scenario>/`. The exit status is 0 if
the run passed.

`--io DEV` reads (repeatable; `auto`: the first namespace of the selected
paths, `all`: every one of them). `--io-verify DEV` writes at random and
verifies every block, and destroys the data on the
namespace.

## Scenarios

A scenario says what to break by its place in the paths, not by interface
names; `plan` shows what that means on the host at hand:

```toml
name = "each host port offline 300 s, 3 rounds"
repeat = 3

[paths]                         # the paths the test is about
subsystem = "*"                # glob on the name on the array or the NQN
array = "*"                     # optional: netapp, powerstore
node = "*"                      # optional: array node

[[phase]]
fail = "host-port"              # host-port, host-cable, target-port,
                                # array-node, fabric
pick = "each"                   # one, each, all-but-one, all
duration = "dev_loss_tmo + 30s" # kernel timers, see below
gap = "until-recovered"

[expect]
io_errors = 0
paths_during = 1                # live paths per subsystem, at least
recovered_within = "300s"
reconnected_by = "discoverd"    # kernel, discoverd or any
```

Durations may name the kernel timers of the selected paths, read from
sysfs, in seconds (the largest value if the paths differ):
`dev_loss_tmo` (FC remote ports), `ctrl_loss_tmo`, `fast_io_fail_tmo`,
`reconnect_delay`, `kato` (keep alive), `io_timeout` and `admin_timeout`.
A timer that is off or that the kernel doesn't show can't be used.

The tool picks how to break each element: an lpfc port through its link,
another driver's port by unbinding its PCI function, a cable or a PowerStore
port at its switch port, a NetApp port by its LIF. `pick = "all"` needs
`allow_outage = true`, a switch port that other hosts use too needs
`allow_shared = true` in the phase.

    nvmf-failover plan scenarios/port-offline-rounds.toml
    nvmf-failover run scenarios/port-offline-rounds.toml --io auto

See `scenarios/` for examples, and `src/nvmf_failover/resolve.py`.

`scenarios/steps/` has scenarios in the low-level format: a list of steps,
each a fault on a named target (`host = "host3"`, `lif = "..."`), an action
(down, up, bounce) and a hold time. It is still supported, for what the
descriptive form can't say.

## Development

    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
    .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
    .venv/bin/mypy src tests
    .venv/bin/pytest --cov

The tests need no hardware: they run against a fake sysfs tree and fake
command and ONTAP shells. `pre-commit install` runs the checks on commit.
