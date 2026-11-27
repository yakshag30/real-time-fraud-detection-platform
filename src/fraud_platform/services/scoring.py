"""Real-time scoring: features -> models -> explanation -> decision -> alert."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ..config import Settings
from ..data.features import FEATURE_NAMES, FeatureBuilder, to_datetime
from ..data.reference import CITY_INDEX, category_risk, channel_risk
from ..db.repository import SQLRepository, iso_utc, utc_now
from ..models.explain import FraudExplainer, build_reasons, format_inr
from ..models.registry import ModelBundle
from .policy import assess_policy

logger = logging.getLogger(__name__)

DEVICE_CACHE_TTL = 30.0


def risk_level(score: float, settings: Settings) -> str:
    if score >= 0.9:
        return "critical"
    if score >= settings.block_threshold:
        return "high"
    if score >= settings.review_threshold:
        return "medium"
    return "low"


def decide(score: float, settings: Settings) -> str:
    if score >= settings.block_threshold:
        return "block"
    if score >= settings.review_threshold:
        return "review"
    return "allow"


def _severity_for(level: str) -> str:
    return {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }.get(level, "low")


class FraudScoringService:
    """Stateful service; keep one instance per process."""

    def __init__(
        self,
        repository: SQLRepository,
        bundle: ModelBundle,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.bundle = bundle
        self.settings = settings
        self.explainer = FraudExplainer(bundle)
        # Pay the SHAP construction cost at startup, not on the first request.
        self.explainer.warmup()
        self._device_cache: tuple[float, dict[str, list[str]]] | None = None

    # ------------------------------------------------------------------ helpers
    def _device_map(self) -> dict[str, list[str]]:
        now = time.time()
        if self._device_cache and now - self._device_cache[0] < DEVICE_CACHE_TTL:
            return self._device_cache[1]
        mapping = self.repository.device_customers()
        self._device_cache = (now, mapping)
        return mapping

    def _profile_for(self, payload: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(payload["customer_id"])
        profile = self.repository.get_customer(customer_id)
        if profile:
            return profile

        city_name = str(payload.get("city") or "Mumbai")
        city = CITY_INDEX.get(city_name, CITY_INDEX["Mumbai"])
        profile = {
            "customer_id": customer_id,
            "name": payload.get("customer_name") or f"Customer {customer_id}",
            "home_city": city_name,
            "home_country": str(payload.get("country") or city["country"]),
            "home_latitude": float(payload.get("latitude") or city["lat"]),
            "home_longitude": float(payload.get("longitude") or city["lon"]),
            "avg_amount": float(payload.get("amount") or 1000.0),
            "account_age_days": 0,
            "segment": "retail",
            "created_at": utc_now(),
        }
        self.repository.upsert_customers([profile])
        logger.info("Registered new customer profile %s from first transaction", customer_id)
        return profile

    def _merchant_for(self, payload: dict[str, Any]) -> dict[str, Any]:
        merchant_id = str(payload.get("merchant_id") or "M-UNKNOWN")
        merchant = self.repository.get_merchant(merchant_id)
        if merchant:
            return merchant
        category = str(payload.get("merchant_category") or "unknown")
        return {
            "merchant_id": merchant_id,
            "name": payload.get("merchant_name") or "Unregistered merchant",
            "category": category,
            "risk_score": category_risk(category),
        }

    def _normalise(self, payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        timestamp = payload.get("timestamp")
        if not timestamp:
            timestamp = datetime.now(timezone.utc)
        ts = to_datetime(timestamp)
        city_name = str(payload.get("city") or profile.get("home_city") or "Mumbai")
        city = CITY_INDEX.get(city_name, {})
        device_id = payload.get("device_id") or f"DEV-UNSEEN-{uuid4().hex[:8]}"
        return {
            "transaction_id": payload.get("transaction_id") or f"T-{uuid4().hex[:12]}",
            "timestamp": ts,
            "customer_id": str(payload["customer_id"]),
            "amount": float(payload["amount"]),
            "currency": str(payload.get("currency") or "INR"),
            "merchant_id": str(payload.get("merchant_id") or "M-UNKNOWN"),
            "merchant_category": None,
            "channel": str(payload.get("channel") or "ecom").lower(),
            "city": city_name,
            "country": str(payload.get("country") or profile.get("home_country") or city.get("country") or "IN"),
            "latitude": float(payload.get("latitude") or city.get("lat") or profile.get("home_latitude") or 0.0),
            "longitude": float(payload.get("longitude") or city.get("lon") or profile.get("home_longitude") or 0.0),
            "device_id": str(device_id),
            "ip_address": str(payload.get("ip_address") or ""),
        }

    # -------------------------------------------------------------------- scoring
    def score(
        self,
        payload: dict[str, Any],
        persist: bool = True,
        explain: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        profile = self._profile_for(payload)
        merchant = self._merchant_for(payload)
        txn = self._normalise(payload, profile)
        txn["merchant_category"] = merchant.get("category")

        device_map = self._device_map()
        device_id = txn["device_id"]
        device_customers = set(device_map.get(device_id, []))
        device_customers.add(txn["customer_id"])
        device_count = len(device_customers)

        history = self.repository.customer_history(
            txn["customer_id"], before=iso_utc(txn["timestamp"]), limit=200
        )
        builder = FeatureBuilder({txn["merchant_id"]: merchant}, {txn["customer_id"]: profile})
        features = builder.build(txn, history, device_customer_count=device_count)

        frame = pd.DataFrame([features], columns=FEATURE_NAMES)
        probability = float(self.bundle.predict_proba(frame)[0])
        anomaly = float(self.bundle.anomaly_risk(frame)[0])
        combined = float(
            self.bundle.combine(
                np.array([probability]), np.array([anomaly]), self.settings.anomaly_weight
            )[0]
        )
        policy = assess_policy(features)
        combined = max(combined, policy.floor)
        level = risk_level(combined, self.settings)
        decision = decide(combined, self.settings)

        if explain:
            explanation = self.explainer.explain(features, probability)
        else:
            explanation = {
                "method": "rules_only",
                "base_value": 0.5,
                "contributions": {},
                "top": [],
            }

        reasons = build_reasons(
            features,
            txn,
            contributions=explanation.get("contributions"),
            profile=profile,
        )
        graph_flags = self._graph_flags(features, device_id, device_count)
        summary = self._summary(level, combined, reasons)
        anomaly_note = (
            f"Isolation Forest anomaly score {anomaly:.2f} "
            "(higher = further from the customer's normal pattern)."
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        result: dict[str, Any] = {
            "transaction_id": txn["transaction_id"],
            "timestamp": txn["timestamp"].astimezone(timezone.utc).isoformat(timespec="seconds"),
            "customer_id": txn["customer_id"],
            "customer_name": profile.get("name"),
            "amount": round(txn["amount"], 2),
            "currency": txn["currency"],
            "merchant_id": txn["merchant_id"],
            "merchant_name": merchant.get("name"),
            "merchant_category": txn["merchant_category"],
            "channel": txn["channel"],
            "city": txn["city"],
            "country": txn["country"],
            "device_id": device_id,
            "fraud_probability": round(probability, 6),
            "anomaly_score": round(anomaly, 6),
            "combined_score": round(combined, 6),
            "risk_level": level,
            "decision": decision,
            "model_name": self.bundle.model_name,
            "review_threshold": self.settings.review_threshold,
            "block_threshold": self.settings.block_threshold,
            "reasons": reasons,
            "summary": summary,
            "anomaly_note": anomaly_note,
            "explanation": [
                {
                    "feature": item["feature"],
                    "label": item["label"],
                    "value": round(float(features.get(item["feature"], 0.0)), 4),
                    "contribution": item["contribution"],
                    "direction": "increases risk" if item["contribution"] > 0 else "reduces risk",
                }
                for item in explanation.get("top", [])
            ],
            "explanation_method": explanation.get("method", "rules_only"),
            "base_value": float(explanation.get("base_value", 0.5)),
            "graph_flags": graph_flags,
            "policy_floor": policy.floor,
            "policy_flags": policy.indicators if policy.applied else [],
            "alert_id": None,
            "latency_ms": latency_ms,
        }

        if persist:
            alert_id = self._persist(txn, features, explanation, reasons, graph_flags, result, payload)
            result["alert_id"] = alert_id
            self._device_cache = None
        return result

    def _graph_flags(
        self, features: dict[str, float], device_id: str, device_count: int
    ) -> list[str]:
        flags: list[str] = []
        if device_count >= 3:
            flags.append(f"Device {device_id} is shared by {device_count} customers")
        if features.get("txn_count_1h", 0) >= 4:
            flags.append(
                f"Velocity burst: {int(features['txn_count_1h'])} transactions within 60 minutes"
            )
        if features.get("geo_velocity_kmh", 0) >= 900:
            flags.append("Impossible-travel pattern detected")
        if features.get("distinct_countries_24h", 0) >= 2:
            flags.append(
                f"Activity in {int(features['distinct_countries_24h'])} countries within 24 hours"
            )
        return flags

    def _summary(self, level: str, score: float, reasons: list[str]) -> str:
        head = f"{level.title()} risk ({score:.0%})"
        if reasons:
            return f"{head} — " + "; ".join(reasons[:3]).rstrip(".") + "."
        return f"{head} — unusual combination of transaction attributes."

    def _persist(
        self,
        txn: dict[str, Any],
        features: dict[str, float],
        explanation: dict[str, Any],
        reasons: list[str],
        graph_flags: list[str],
        result: dict[str, Any],
        payload: dict[str, Any],
    ) -> str | None:
        record = {
            **txn,
            "timestamp": iso_utc(txn["timestamp"]),
            "merchant_category": txn["merchant_category"],
            "fraud_probability": result["fraud_probability"],
            "anomaly_score": result["anomaly_score"],
            "combined_score": result["combined_score"],
            "decision": result["decision"],
            "risk_level": result["risk_level"],
            "model_name": self.bundle.model_name,
            "features": features,
            "explanation": explanation,
            "reasons": reasons,
            "graph_flags": graph_flags,
            "ground_truth": payload.get("ground_truth"),
            "created_at": utc_now(),
        }
        self.repository.insert_transaction(record)

        if result["combined_score"] < self.settings.alert_min_score:
            return None

        alert_id = f"AL-{uuid4().hex[:12]}"
        self.repository.insert_alert(
            {
                "alert_id": alert_id,
                "transaction_id": txn["transaction_id"],
                "customer_id": txn["customer_id"],
                "severity": _severity_for(result["risk_level"]),
                "score": result["combined_score"],
                "status": "open",
                "title": (
                    f"{result['risk_level'].title()} risk · {format_inr(txn['amount'])} · "
                    f"{txn['channel'].upper()} · {txn['customer_id']}"
                ),
                "summary": reasons,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
        )
        return alert_id
