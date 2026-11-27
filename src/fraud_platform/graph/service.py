"""Cached facade over graph construction and ring detection."""

from __future__ import annotations

import time
from typing import Any

from ..db.repository import SQLRepository
from .network import build_entity_graph, customer_subgraph, detect_rings


class GraphService:
    def __init__(self, repository: SQLRepository, ttl_seconds: float = 45.0) -> None:
        self.repository = repository
        self.ttl_seconds = ttl_seconds
        self._built_at = 0.0
        self._graph = None
        self._rings: list[dict[str, Any]] = []

    def _ensure_fresh(self) -> None:
        if self._graph is not None and time.time() - self._built_at < self.ttl_seconds:
            return
        rows = self.repository.graph_transactions(hours=24 * 30, limit=5000)
        self._graph = build_entity_graph(rows)
        self._rings = detect_rings(self._graph)
        self._built_at = time.time()

    def invalidate(self) -> None:
        self._graph = None
        self._built_at = 0.0

    def rings(self) -> dict[str, Any]:
        self._ensure_fresh()
        return {
            "rings": self._rings,
            "graph": {
                "nodes": self._graph.number_of_nodes() if self._graph is not None else 0,
                "edges": self._graph.number_of_edges() if self._graph is not None else 0,
            },
            "method": "customer projection + Louvain communities + risk propagation",
        }

    def ring_for_customer(self, customer_id: str) -> dict[str, Any] | None:
        self._ensure_fresh()
        for ring in self._rings:
            if customer_id in ring["members"]:
                return ring
        return None

    def network(self, customer_id: str, radius: int = 1) -> dict[str, Any]:
        self._ensure_fresh()
        return customer_subgraph(self._graph, customer_id, radius=radius)
