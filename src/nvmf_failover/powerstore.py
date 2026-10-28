"""Dell PowerStore, over its REST API: the host's volumes and the FC ports.

PowerStore has no call to take a port down: its REST API (3.0) modifies
only the requested speed of an FC or Ethernet port, and the usages of an
IP port. Break its paths with the brocade-port fault on the switch port
of a wwn_nvme instead.
"""

from __future__ import annotations

import base64
import json
import ssl
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any, Protocol

from .config import PowerStoreConfig
from .discover import Lif, Subsystem, match
from .topology import Controller


class PowerStoreError(RuntimeError):
    """The REST API refused a request."""


class Api(Protocol):
    def get(self, resource: str, **query: str) -> Any: ...


class PowerStoreApi:
    """GET requests against https://HOST/api/rest/."""

    def __init__(self, cfg: PowerStoreConfig) -> None:
        self.base = f"https://{cfg.host}/api/rest/"
        token = base64.b64encode(f"{cfg.user}:{cfg.password}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Accept": "application/json"}
        self.context = ssl.create_default_context()
        if not cfg.verify_tls:
            self.context.check_hostname = False
            self.context.verify_mode = ssl.CERT_NONE

    def get(self, resource: str, **query: str) -> Any:
        url = self.base + resource
        if query:
            url += "?" + urllib.parse.urlencode(query, safe="=,*")
        request = urllib.request.Request(url, headers=self.headers)  # noqa: S310
        try:
            with urllib.request.urlopen(  # noqa: S310 -- https only
                request, context=self.context, timeout=30
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            raise PowerStoreError(f"GET {resource}: {err}") from err


def host_subsystem(api: Api, hostnqn: str) -> Subsystem | None:
    """The host with @hostnqn as an initiator, as a subsystem of its volumes."""
    hosts = api.get("host", select="id,name,host_initiators")
    host = next(
        (
            h
            for h in hosts
            if any(i.get("port_name") == hostnqn for i in h.get("host_initiators", []))
        ),
        None,
    )
    if host is None:
        return None
    maps = api.get(
        "host_volume_mapping", host_id=f"eq.{host['id']}", select="volume_id"
    )
    names = sorted(
        api.get(f"volume/{m['volume_id']}", select="name")["name"] for m in maps
    )
    cluster = api.get("cluster", select="name,nvm_subsystem_nqn")
    nqn = cluster[0].get("nvm_subsystem_nqn", "") if cluster else ""
    return Subsystem(
        vserver=f"powerstore:{cluster[0]['name']}" if cluster else "powerstore",
        name=host["name"],
        nqn=nqn,
        namespaces=tuple(names),
    )


def fc_ports(api: Api, vserver: str) -> list[Lif]:
    """The FC ports that serve NVMe, as target ports (Lif) by their wwn_nvme."""
    out = []
    for port in api.get(
        "fc_port", select="name,wwn_nvme,is_link_up,node_id,port_index,protocols"
    ):
        if "NVMe" not in port.get("protocols", []) or not port.get("wwn_nvme"):
            continue
        out.append(
            Lif(
                vserver=vserver,
                name=port["name"],
                protocol="fc-nvme",
                address=port["wwn_nvme"].lower(),
                admin_up=True,  # PowerStore has no administrative port state
                oper_up=bool(port.get("is_link_up")),
                node=str(port.get("node_id", ""))[:8],
                port=str(port.get("port_index", "")),
            )
        )
    return sorted(out, key=lambda lif: lif.name)


def discover(
    api: Api, hostnqn: str, ctrls: Iterable[Controller]
) -> tuple[Subsystem | None, list[Lif], dict[tuple[str, str], list[str]]]:
    sub = host_subsystem(api, hostnqn)
    if sub is None:
        return None, [], {}
    ports = fc_ports(api, sub.vserver)
    return sub, ports, match(ports, ctrls)
