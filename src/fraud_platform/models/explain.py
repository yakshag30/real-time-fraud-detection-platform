"""Explainable AI layer: SHAP attributions plus analyst-friendly reason text."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .registry import ModelBundle

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier

    TREE_TYPES: tuple[type, ...] = (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        DecisionTreeClassifier,
    )
except Exception:  # pragma: no cover
    TREE_TYPES = ()

try:
    from xgboost import XGBClassifier

    TREE_TYPES = TREE_TYPES + (XGBClassifier,)
except Exception:  # pragma: no cover
    pass


FEATURE_LABELS: dict[str, str] = {
    "amount": "Transaction amount",
    "log_amount": "Transaction amount (log scale)",
    "amount_to_customer_avg": "Amount vs. customer's average spend",
    "amount_zscore": "Amount deviation from history",
    "amount_to_24h_max": "Amount vs. 24h maximum",
    "hour_of_day": "Hour of day",
    "is_night": "Night-time transaction",
    "is_weekend": "Weekend transaction",
    "txn_count_1h": "Transactions in last hour",
    "txn_count_24h": "Transactions in last 24 hours",
    "txn_count_7d": "Transactions in last 7 days",
    "amount_sum_1h": "Value moved in last hour",
    "amount_sum_24h": "Value moved in last 24 hours",
    "distinct_merchants_24h": "Distinct merchants in 24 hours",
    "distinct_countries_24h": "Distinct countries in 24 hours",
    "geo_distance_home_km": "Distance from home city",
    "geo_velocity_kmh": "Implied travel speed",
    "is_foreign": "Cross-border transaction",
    "is_new_device": "New device",
    "device_age_minutes": "Device age",
    "device_customer_count": "Customers sharing device",
    "is_new_merchant": "New merchant",
    "merchant_risk": "Merchant risk rating",
    "category_risk": "Merchant category risk",
    "channel_risk": "Channel risk",
    "customer_avg_amount": "Customer average spend",
    "account_age_days": "Account age",
}


def format_inr(value: float) -> str:
    """Indian digit grouping, e.g. 200000 -> ₹2,00,000."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "₹0"
    sign = "-" if number < 0 else ""
    whole = f"{abs(number):.0f}"
    if len(whole) > 3:
        last_three, rest = whole[-3:], whole[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        whole = ",".join(groups) + "," + last_three
    return f"{sign}₹{whole}"


def _to_probability_scale(
    values: np.ndarray, base: float, probability: float, anchor: float = 0.5
) -> tuple[np.ndarray, float]:
    """Rescale log-odds attributions so they add up to ``probability - 0.5``.

    XGBoost explains the raw margin, which yields numbers like -3.5 that are
    meaningless to an analyst. Rescaling preserves the ranking and the signs
    while putting every contribution on the probability scale.
    """

    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    target = float(probability) - anchor
    if abs(total) < 1e-9 or abs(target) < 1e-9:
        return np.zeros_like(values), anchor
    return values * (target / total), anchor


def _fmt_hour(txn: Mapping[str, Any]) -> str:
    raw = str(txn.get("timestamp") or "")
    for candidate in (raw[11:16], ""):
        if len(candidate) == 5 and candidate[2] == ":":
            return candidate
    return "--:--"


class FraudExplainer:
    """Produces per-feature attributions for a single scored transaction."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._explainer: Any | None = None
        self._shap_failed = False
        self.method = "importance_heuristic"

    def _ensure_explainer(self) -> Any | None:
        if self._explainer is not None or self._shap_failed:
            return self._explainer
        if not isinstance(self.bundle.model, TREE_TYPES):
            self._shap_failed = True
            return None
        try:
            import shap

            self._explainer = shap.TreeExplainer(self.bundle.model)
            self.method = "shap_tree"
        except Exception as exc:  # pragma: no cover - depends on shap version
            logger.warning("SHAP TreeExplainer unavailable, using heuristic attributions: %s", exc)
            self._shap_failed = True
        return self._explainer

    def warmup(self) -> None:
        """Build the explainer and run one throwaway attribution.

        SHAP's TreeExplainer is expensive to construct, so doing it at process
        start keeps the first live transaction response fast.
        """

        explainer = self._ensure_explainer()
        if explainer is None:
            return
        stats = self.bundle.feature_stats or {}
        means = stats.get("mean")
        row = pd.DataFrame(
            [
                {
                    name: float(means[index])
                    if means is not None and index < len(means)
                    else 0.0
                    for index, name in enumerate(self.bundle.feature_names)
                }
            ]
        )
        try:
            values, base = self._shap_values(explainer, row)
            _to_probability_scale(values, base, 0.5)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SHAP warmup failed: %s", exc)

    def explain(
        self, features: Mapping[str, float], probability: float | None = None
    ) -> dict[str, Any]:
        row = pd.DataFrame([{name: float(features[name]) for name in self.bundle.feature_names}])
        explainer = self._ensure_explainer()

        if explainer is not None:
            try:
                values, base = self._shap_values(explainer, row)
                if probability is not None:
                    values, base = _to_probability_scale(values, base, float(probability))
                return self._result(values, base, self.method)
            except Exception as exc:  # pragma: no cover
                logger.warning("SHAP explanation failed at inference: %s", exc)
                self._shap_failed = True

        return self._heuristic(features, probability)

    def _shap_values(self, explainer: Any, row: pd.DataFrame) -> tuple[np.ndarray, float]:
        try:
            explanation = explainer(row)
            values = np.asarray(explanation.values, dtype=float)
            base = np.ravel(np.asarray(explanation.base_values, dtype=float))
        except Exception:
            raw = explainer.shap_values(row)
            values = np.asarray(raw[-1] if isinstance(raw, list) else raw, dtype=float)
            base = np.ravel(np.asarray(explainer.expected_value, dtype=float))

        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        base_value = float(base[-1]) if base.size > 1 else float(base[0])
        return values, base_value

    def _heuristic(
        self, features: Mapping[str, float], probability: float | None
    ) -> dict[str, Any]:
        names = self.bundle.feature_names
        stats = self.bundle.feature_stats or {}
        means = np.asarray(stats.get("mean", [0.0] * len(names)), dtype=float)
        stds = np.asarray(stats.get("std", [1.0] * len(names)), dtype=float)
        stds = np.where(stds == 0, 1.0, stds)
        values = np.asarray([float(features[name]) for name in names], dtype=float)

        weights = getattr(self.bundle.model, "feature_importances_", None)
        if weights is None and hasattr(self.bundle.model, "named_steps"):
            estimator = list(self.bundle.model.named_steps.values())[-1]
            coef = getattr(estimator, "coef_", None)
            weights = np.abs(np.ravel(coef)) if coef is not None else None
        weights = (
            np.asarray(weights, dtype=float)
            if weights is not None
            else np.ones(len(names), dtype=float)
        )

        deviations = np.clip((values - means) / stds, -4.0, 4.0) / 4.0
        contributions = weights * deviations
        total = float(np.abs(contributions).sum())
        base = 0.5
        target = (float(probability) - base) if probability is not None else 0.0
        if total > 1e-9 and abs(target) > 1e-9:
            contributions = contributions / total * target
        else:
            contributions = np.zeros(len(names))
        self.method = "importance_heuristic"
        return self._result(contributions, base, "importance_heuristic")

    def _result(self, values: np.ndarray, base: float, method: str) -> dict[str, Any]:
        contributions = {
            name: float(value) for name, value in zip(self.bundle.feature_names, values)
        }
        ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        self.method = method
        return {
            "method": method,
            "base_value": round(float(base), 5),
            "contributions": {k: round(v, 6) for k, v in contributions.items()},
            "top": [
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "contribution": round(value, 6),
                }
                for name, value in ranked[:8]
            ],
        }


def _reason_candidates(
    features: Mapping[str, float],
    txn: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    contributions: Mapping[str, float],
) -> list[tuple[float, str]]:
    amount = float(features.get("amount", 0))
    avg = float(features.get("customer_avg_amount", amount) or amount)
    # The model sees a saturated ratio; analysts should see the real multiple.
    ratio = amount / avg if avg > 0 else float(features.get("amount_to_customer_avg", 1.0))
    candidates: list[tuple[float, str]] = []

    def add(condition: bool, strength: float, text: str) -> None:
        if condition:
            candidates.append((strength, text))

    add(
        ratio >= 3,
        contributions.get("amount_to_customer_avg", 0) + ratio,
        f"{ratio:.1f}× the customer's usual spend "
        f"({format_inr(amount)} against an average of {format_inr(avg)})",
    )
    add(
        features.get("is_new_device", 0) >= 1,
        contributions.get("is_new_device", 0) + 1.0,
        "first payment from a device never seen on this account",
    )
    add(
        features.get("is_foreign", 0) >= 1,
        contributions.get("is_foreign", 0) + 1.2,
        f"cross-border transaction from {txn.get('country') or 'an unfamiliar country'}",
    )
    add(
        features.get("geo_distance_home_km", 0) >= 300,
        contributions.get("geo_distance_home_km", 0) + 1.0,
        f"{features.get('geo_distance_home_km', 0):,.0f} km from the home city "
        f"({txn.get('city') or 'unknown location'})",
    )
    add(
        features.get("is_night", 0) >= 1 and float(features.get("hour_of_day", 12)) <= 5,
        contributions.get("is_night", 0) + 0.8,
        f"initiated at {_fmt_hour(txn)} when the customer is normally inactive",
    )
    add(
        features.get("txn_count_1h", 0) >= 4,
        contributions.get("txn_count_1h", 0) + 1.0,
        f"{int(features.get('txn_count_1h', 0))} transactions in the last hour (card-testing pattern)",
    )
    add(
        features.get("device_customer_count", 1) >= 3,
        contributions.get("device_customer_count", 0) + 1.4,
        f"device shared by {int(features.get('device_customer_count', 1))} different customers",
    )
    add(
        features.get("geo_velocity_kmh", 0) >= 900,
        contributions.get("geo_velocity_kmh", 0) + 1.2,
        f"implausible travel of {features.get('geo_velocity_kmh', 0):,.0f} km/h since the previous transaction",
    )
    add(
        features.get("amount_sum_24h", 0) >= 3 * max(avg, 1),
        contributions.get("amount_sum_24h", 0) + 0.6,
        f"{format_inr(features.get('amount_sum_24h', 0))} moved in the last 24 hours",
    )
    add(
        features.get("amount_to_24h_max", 1) >= 5,
        contributions.get("amount_to_24h_max", 0) + 0.5,
        f"{features.get('amount_to_24h_max', 1):.1f}× larger than any transaction in the previous 24 hours",
    )
    add(
        features.get("amount_zscore", 0) >= 3,
        contributions.get("amount_zscore", 0) + 0.5,
        f"{features.get('amount_zscore', 0):.1f}σ above this customer's normal transaction size",
    )
    add(
        float(features.get("merchant_risk", 0)) >= 0.6,
        contributions.get("merchant_risk", 0) + 0.7,
        "merchant is in a high-risk category "
        f"({str(txn.get('merchant_category') or 'unknown').replace('_', ' ')})",
    )
    add(
        features.get("is_new_merchant", 0) >= 1 and ratio >= 2,
        contributions.get("is_new_merchant", 0) + 0.4,
        "first payment to this merchant",
    )
    add(
        float(features.get("account_age_days", 365)) <= 30,
        contributions.get("account_age_days", 0) + 0.3,
        "recently opened account",
    )
    return candidates


def build_reasons(
    features: Mapping[str, float],
    txn: Mapping[str, Any],
    contributions: Mapping[str, float] | None = None,
    profile: Mapping[str, Any] | None = None,
    limit: int = 4,
) -> list[str]:
    """Rank domain rules by SHAP contribution to produce analyst-facing reasons."""

    candidates = _reason_candidates(features, txn, profile, contributions or {})
    candidates.sort(key=lambda item: item[0], reverse=True)
    reasons: list[str] = []
    for _, text in candidates:
        if text not in reasons:
            reasons.append(text)
        if len(reasons) >= limit:
            break
    return reasons
