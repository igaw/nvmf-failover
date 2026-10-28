"""nvmf-failover mcp: faults and scenario runs for an agent, over MCP.

The server runs on the test host, as root, and talks JSON-RPC on stdin
and stdout: ssh root@host nvmf-failover mcp. It holds no fault down
longer than asked, and puts every fault up when the client is gone.
"""

from __future__ import annotations

import fcntl
import signal
import sys
import threading
from types import FrameType

from ..system import System
from .server import Options, Server


class _Quit(Exception):
    pass


def _quit(signum: int, frame: FrameType | None) -> None:
    raise _Quit(f"signal {signum}")


def serve(opts: Options, system: System) -> int:
    """Serve on stdin/stdout until the client is gone or a signal comes."""
    opts.state.parent.mkdir(parents=True, exist_ok=True)
    lock = opts.state.with_suffix(".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"nvmf-failover: another mcp server holds {lock.name}", file=sys.stderr)
        return 2
    server = Server(opts, system)
    stop = threading.Event()
    timer = threading.Thread(
        target=server.holds.run_timer, args=(stop,), name="dead-man", daemon=True
    )
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _quit)
    try:
        failed = server.holds.recover()
        if failed:
            print(
                f"nvmf-failover: could not restore {', '.join(failed)}",
                file=sys.stderr,
            )
        timer.start()
        server.serve(sys.stdin, sys.stdout)
    except (_Quit, BrokenPipeError):
        pass
    finally:
        # the client is gone: nothing stays down
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        stop.set()
        failed = server.holds.restore_all("server exits")
        if failed:
            print(
                f"nvmf-failover: could not restore {', '.join(failed)}, "
                f"left in {opts.state}",
                file=sys.stderr,
            )
        server.close()
        lock.close()
    return 1 if failed else 0
