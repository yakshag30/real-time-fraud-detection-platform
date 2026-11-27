"""Graph-based fraud detection (shared devices, IPs and merchants)."""

from .network import build_entity_graph, customer_subgraph, detect_rings, propagate_risk
from .service import GraphService

__all__ = [
    "GraphService",
    "build_entity_graph",
    "customer_subgraph",
    "detect_rings",
    "propagate_risk",
]
