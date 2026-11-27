"""Load and operate a persisted model artifact."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Everything needed to score a transaction offline or online."""

    model: Any
    model_name: str
    feature_names: list[str]
    isolation_forest: Any = None
    feature_stats: dict[str, list[float]] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""
    anomaly_center: float = 0.0
    anomaly_scale: float = 1.0

    # ------------------------------------------------------------------ scoring
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        ordered = features[self.feature_names]
        return np.asarray(self.model.predict_proba(ordered), dtype=float)[:, 1]

    def anomaly_risk(self, features: pd.DataFrame) -> np.ndarray:
        if self.isolation_forest is None:
            return np.zeros(len(features), dtype=float)
        decision = np.asarray(
            self.isolation_forest.decision_function(features[self.feature_names]), dtype=float
        )
        # decision_function is higher for normal points -> invert into a risk score
        return 1.0 / (1.0 + np.exp((decision - self.anomaly_center) / max(self.anomaly_scale, 1e-6)))

    def combine(self, probability: np.ndarray, anomaly: np.ndarray, weight: float) -> np.ndarray:
        weight = min(max(float(weight), 0.0), 1.0)
        return (1.0 - weight) * probability + weight * anomaly

    # ------------------------------------------------------------- presentation
    def feature_importance(self) -> list[dict[str, Any]]:
        importance = self.metrics.get("feature_importance")
        if importance:
            return importance
        return [{"feature": name, "importance": 1.0 / len(self.feature_names)} for name in self.feature_names]

    def describe(self) -> dict[str, Any]:
        selected = self.metrics.get("models", {}).get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "trained_at": self.trained_at,
            "features": len(self.feature_names),
            "pr_auc": selected.get("pr_auc"),
            "roc_auc": selected.get("roc_auc"),
            "precision": selected.get("precision"),
            "recall": selected.get("recall"),
            "isolation_forest": self.isolation_forest is not None,
            "thresholds": self.metrics.get("thresholds", {}),
        }


def load_bundle(path: Path) -> ModelBundle | None:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception as exc:  # corrupt artifact -> retrain instead of crashing
        logger.warning("Could not load model artifact %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Unexpected model artifact format in %s", path)
        return None
    return ModelBundle(**payload)
