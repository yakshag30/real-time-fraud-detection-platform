"""Entity graph analytics for fraud-ring discovery.

Customers, devices, IP addresses and merchants become nodes; a transaction
links them. Rings surface as dense customer communities that share
identifiers, then risk is propagated through the graph so an account with
one suspicious neighbour is pulled upward too.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

MAX_MEMBERS_PER_ENTITY = 30


def _add_customer(G: nx.Graph, customer_id: str, score: float, amount: float) -> str:
    node = f"customer:{customer_id}"
    if node in G:
        data = G.nodes[node]
        data["risk"] = max(float(data["risk"]), score)
        data["txns"] += 1
        data["amount"] += amount
        data["flagged"] += int(score >= 0.5)
    else:
        G.add_node(
            node,
            kind="customer",
            label=customer_id,
            risk=float(score),
            txns=1,
            amount=float(amount),
            flagged=int(score >= 0.5),
        )
    return node


def _add_entity(G: nx.Graph, kind: str, value: str, customer_node: str, amount: float) -> None:
    node = f"{kind}:{value}"
    if node not in G:
        G.add_node(node, kind=kind, label=value, risk=0.0, txns=0, amount=0.0, flagged=0)
    G.nodes[node]["txns"] += 1
    G.nodes[node]["amount"] += amount
    if G.has_edge(customer_node, node):
        G[customer_node][node]["txns"] += 1
    else:
        G.add_edge(customer_node, node, txns=1)


def build_entity_graph(transactions: Sequence[Mapping[str, Any]]) -> nx.Graph:
    """Build the customer/device/IP/merchant graph from scored transactions."""

    graph = nx.Graph()
    for row in transactions:
        customer_id = str(row.get("customer_id") or "")
        if not customer_id:
            continue
        score = float(row.get("combined_score") or 0.0)
        amount = float(row.get("amount") or 0.0)
        customer_node = _add_customer(graph, customer_id, score, amount)
        for kind, value in (
            ("device", row.get("device_id")),
            ("ip", row.get("ip_address")),
            ("merchant", row.get("merchant_id")),
        ):
            if value:
                _add_entity(graph, kind, str(value), customer_node, amount)

    # Entity baseline risk = average user risk touching it, so propagation has signal.
    for node, data in graph.nodes(data=True):
        if data["kind"] == "customer":
            continue
        neighbour_risk = [
            float(graph.nodes[n]["risk"])
            for n in graph.neighbors(node)
            if graph.nodes[n]["kind"] == "customer"
        ]
        data["risk"] = float(np.mean(neighbour_risk)) if neighbour_risk else 0.0
    return graph


def propagate_risk(
    graph: nx.Graph, alpha: float = 0.4, iterations: int = 4
) -> dict[str, float]:
    """Label-spreading style propagation of fraud risk across shared entities."""

    if graph.number_of_nodes() == 0:
        return {}
    base = {node: float(data.get("risk", 0.0)) for node, data in graph.nodes(data=True)}
    scores = dict(base)
    for _ in range(iterations):
        updated: dict[str, float] = {}
        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            if not neighbours:
                updated[node] = base[node]
                continue
            neighbour_mean = sum(scores[n] for n in neighbours) / len(neighbours)
            updated[node] = (1 - alpha) * base[node] + alpha * neighbour_mean
        scores = updated
    nx.set_node_attributes(graph, scores, "propagated_risk")
    return scores


def _customer_projection(graph: nx.Graph) -> nx.Graph:
    """Project the entity graph onto customers linked by shared entities."""

    projection = nx.Graph()
    for node, data in graph.nodes(data=True):
        if data["kind"] == "customer":
            projection.add_node(node)

    for node, data in graph.nodes(data=True):
        if data["kind"] == "customer":
            continue
        members = [
            n
            for n in graph.neighbors(node)
            if graph.nodes[n]["kind"] == "customer"
        ][:MAX_MEMBERS_PER_ENTITY]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if projection.has_edge(a, b):
                    projection[a][b]["shared"] += 1
                    projection[a][b]["entities"].add(node)
                else:
                    projection.add_edge(a, b, shared=1, entities={node})
    return projection


def _shared_entities(graph: nx.Graph, customers: Iterable[str], kind: str) -> list[str]:
    member_set = set(customers)
    values: set[str] = set()
    for node, data in graph.nodes(data=True):
        if data["kind"] != kind:
            continue
        if sum(1 for n in graph.neighbors(node) if n in member_set) >= 2:
            values.add(str(data["label"]))
    return sorted(values)


def _customer_id(node: str) -> str:
    """Customer nodes are stored as ``customer:C00001``; rings report plain ids."""

    return node.split(":", 1)[1] if node.startswith("customer:") else node


def _community_members(projection: nx.Graph) -> list[set[str]]:
    if projection.number_of_nodes() == 0:
        return []
    try:
        return [set(c) for c in nx.community.louvain_communities(projection, weight="shared", seed=7)]
    except Exception:  # networkx < 3 or unexpected graph shape
        try:
            return [set(c) for c in nx.community.greedy_modularity_communities(projection, weight="shared")]
        except Exception:
            return [set(c) for c in nx.connected_components(projection)]


def detect_rings(
    graph: nx.Graph,
    min_members: int = 2,
    min_risk: float = 0.45,
    max_rings: int = 15,
) -> list[dict[str, Any]]:
    """Group customers that share devices/IPs/merchants into suspicious rings."""

    if graph.number_of_nodes() == 0:
        return []
    propagate_risk(graph)
    projection = _customer_projection(graph)
    rings: list[dict[str, Any]] = []

    for community in _community_members(projection):
        member_nodes = sorted(community)
        if len(member_nodes) < min_members:
            continue
        risks = {
            m: max(
                float(graph.nodes[m].get("propagated_risk", 0.0)),
                float(graph.nodes[m].get("risk", 0.0)),
            )
            for m in member_nodes
        }
        ring_risk = float(np.mean(sorted(risks.values(), reverse=True)[:3]))
        devices = _shared_entities(graph, member_nodes, "device")
        ips = _shared_entities(graph, member_nodes, "ip")
        merchants = _shared_entities(graph, member_nodes, "merchant")
        # Public payloads carry customer ids, not graph node ids, so callers can
        # pass them straight back to /graph/customer/{id} and report builders.
        members = [_customer_id(m) for m in member_nodes]
        flagged = [_customer_id(m) for m in member_nodes if risks[m] >= 0.5]
        high_risk_customers = [
            m
            for m in member_nodes
            if graph.nodes[m].get("flagged", 0) > 0
        ]

        if not (devices or ips or merchants):
            continue
        if ring_risk < min_risk and not flagged:
            continue
        if len(members) < 3 and ring_risk < 0.6:
            continue

        txns = sum(int(graph.nodes[m].get("txns", 0)) for m in member_nodes)
        amount = sum(float(graph.nodes[m].get("amount", 0.0)) for m in member_nodes)
        evidence: list[str] = []
        if devices:
            evidence.append(f"{len(devices)} device(s) shared across members: {', '.join(devices[:3])}")
        if ips:
            evidence.append(f"{len(ips)} IP address(es) reused across members")
        if merchants:
            evidence.append(f"Concentrated spend at {len(merchants)} merchant(s)")

        rings.append(
            {
                "ring_id": "",
                "members": members,
                "size": len(members),
                "risk_score": round(ring_risk, 4),
                "flagged_members": flagged,
                "shared_devices": devices,
                "shared_ips": ips,
                "shared_merchants": merchants,
                "merchants_detail": [
                    {
                        "merchant_id": m,
                        "category": str(
                            graph.nodes.get(f"merchant:{m}", {}).get("label", "")
                        ),
                    }
                    for m in merchants[:5]
                ],
                "transactions": txns,
                "amount": round(amount, 2),
                "evidence": evidence,
                "high_risk_customer_count": len(high_risk_customers),
            }
        )

    rings.sort(key=lambda r: (r["risk_score"], r["amount"]), reverse=True)
    rings = rings[:max_rings]
    for index, ring in enumerate(rings, start=1):
        ring["ring_id"] = f"RING-{index:02d}"
    return rings


def customer_subgraph(
    graph: nx.Graph, customer_id: str, radius: int = 1, limit: int = 60
) -> dict[str, Any]:
    """Neighbourhood around one customer for visual investigation."""

    start = f"customer:{customer_id}"
    if start not in graph:
        return {"nodes": [], "edges": [], "found": False}

    distances = nx.single_source_shortest_path_length(graph, start, cutoff=radius)
    ordered = sorted(distances, key=lambda node: (distances[node], node != start))
    chosen = set(ordered[:limit])
    nodes = []
    for node in chosen:
        data = graph.nodes[node]
        nodes.append(
            {
                "id": node,
                "label": str(data.get("label", node)),
                "kind": data.get("kind", "unknown"),
                "risk": round(float(data.get("propagated_risk", data.get("risk", 0.0))), 4),
                "transactions": int(data.get("txns", 0)),
                "amount": round(float(data.get("amount", 0.0)), 2),
            }
        )
    edges = [
        {
            "source": u,
            "target": v,
            "transactions": int(attrs.get("txns", 1)),
        }
        for u, v, attrs in graph.edges(data=True)
        if u in chosen and v in chosen
    ]
    return {"nodes": nodes, "edges": edges, "found": True}
