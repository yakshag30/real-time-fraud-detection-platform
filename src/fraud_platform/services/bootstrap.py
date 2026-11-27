"""Cold-start helpers so the platform is useful on first launch."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..config import Settings
from ..data import build_training_matrix, ensure_dataset
from ..db.repository import SQLRepository, iso_utc, utc_now
from ..graph.service import GraphService
from ..models.explain import FraudExplainer, build_reasons
from ..models.registry import ModelBundle, load_bundle
from ..models.training import train_models
from .scoring import FraudScoringService, decide, risk_level

logger = logging.getLogger(__name__)


def ensure_model(
    repository: SQLRepository,
    settings: Settings,
    force: bool = False,
) -> tuple[ModelBundle, dict[str, Any] | None]:
    """Load the saved model or train a new one from the synthetic corpus."""

    settings.ensure_dirs()
    if not force:
        bundle = load_bundle(settings.model_path)
        if bundle is not None:
            logger.info("Loaded model artifact %s (%s)", settings.model_path, bundle.model_name)
            return bundle, None

    dataset = ensure_dataset(settings.data_dir)
    repository.upsert_customers(dataset.customers.to_dict("records"))
    repository.upsert_merchants(dataset.merchants.to_dict("records"))
    metrics = train_models(dataset, settings)
    bundle = load_bundle(settings.model_path)
    if bundle is None:
        raise RuntimeError("Model training finished but the artifact could not be loaded")
    return bundle, metrics


def seed_demo_data(
    repository: SQLRepository,
    service: FraudScoringService,
    settings: Settings,
    rows: int | None = None,
) -> int:
    """Seed recent, pre-scored transactions so dashboards and graphs have data.

    Features are computed against each customer's full generated history (the
    same engineered features used in production scoring) and only suspicious
    rows pay the cost of a SHAP explanation.
    """

    limit = int(rows or settings.bootstrap_rows)
    dataset = ensure_dataset(settings.data_dir)
    repository.upsert_customers(dataset.customers.to_dict("records"))
    repository.upsert_merchants(dataset.merchants.to_dict("records"))

    features, labels = build_training_matrix(
        dataset.transactions, dataset.customers, dataset.merchants, tail=limit
    )
    transactions = dataset.transactions.sort_values("timestamp").reset_index(drop=True)
    limit = min(max(limit, 0), len(transactions))
    if limit == 0:
        return 0

    frame = features
    batch = transactions.tail(limit).reset_index(drop=True)

    probabilities = service.bundle.predict_proba(frame)
    anomalies = service.bundle.anomaly_risk(frame)
    combined = service.bundle.combine(probabilities, anomalies, settings.anomaly_weight)
    explainer = FraudExplainer(service.bundle)
    feature_names = list(service.bundle.feature_names)
    feature_matrix = frame[feature_names].to_numpy(dtype=float)
    seed_tag = utc_now()

    inserted = 0
    alerts = 0
    for offset, (_, row) in enumerate(batch.iterrows()):
        probability = float(probabilities[offset])
        anomaly = float(anomalies[offset])
        score = float(combined[offset])
        level = risk_level(score, settings)
        decision = decide(score, settings)
        record_features = {
            name: float(value) for name, value in zip(feature_names, feature_matrix[offset])
        }
        txn = row.to_dict()
        txn["timestamp"] = pd_timestamp(txn["timestamp"])

        explanation: dict[str, Any] = {"method": "rules_only", "base_value": 0.5, "contributions": {}, "top": []}
        if score >= settings.alert_min_score:
            explanation = explainer.explain(record_features, probability)
        reasons = build_reasons(
            record_features,
            {
                **txn,
                "merchant_category": txn.get("merchant_category"),
            },
            contributions=explanation.get("contributions"),
        )

        transaction_id = str(txn.get("transaction_id") or f"T-SEED-{inserted:06d}")
        repository.insert_transaction(
            {
                "transaction_id": transaction_id,
                "timestamp": iso_utc(txn["timestamp"]),
                "customer_id": txn["customer_id"],
                "amount": round(float(txn["amount"]), 2),
                "currency": txn.get("currency", "INR"),
                "merchant_id": txn.get("merchant_id"),
                "merchant_category": txn.get("merchant_category"),
                "channel": txn.get("channel"),
                "city": txn.get("city"),
                "country": txn.get("country"),
                "latitude": float(txn.get("latitude") or 0.0),
                "longitude": float(txn.get("longitude") or 0.0),
                "device_id": txn.get("device_id"),
                "ip_address": txn.get("ip_address"),
                "fraud_probability": round(probability, 6),
                "anomaly_score": round(anomaly, 6),
                "combined_score": round(score, 6),
                "decision": decision,
                "risk_level": level,
                "model_name": service.bundle.model_name,
                "features": record_features,
                "explanation": explanation,
                "reasons": reasons,
                "graph_flags": [],
                "ground_truth": int(txn.get("is_fraud") or 0),
                "created_at": seed_tag,
            }
        )
        inserted += 1

        if score >= settings.alert_min_score:
            repository.insert_alert(
                {
                    "alert_id": f"AL-SEED-{inserted:06d}",
                    "transaction_id": transaction_id,
                    "customer_id": txn["customer_id"],
                    "severity": level,
                    "score": round(score, 6),
                    "status": "open",
                    "title": (
                        f"{level.title()} risk · ₹{float(txn['amount']):,.0f} · "
                        f"{str(txn.get('channel', '')).upper()} · {txn['customer_id']}"
                    ),
                    "summary": reasons,
                    "created_at": seed_tag,
                    "updated_at": seed_tag,
                }
            )
            alerts += 1

    logger.info(
        "Seeded %s transactions (%s alerts, window starts %s)",
        inserted,
        alerts,
        batch["timestamp"].min(),
    )
    return inserted


def pd_timestamp(value: Any) -> datetime:
    from ..data.features import to_datetime

    return to_datetime(value)


def bootstrap_runtime(
    settings: Settings, seed: bool | None = None
) -> tuple[SQLRepository, FraudScoringService, GraphService, dict[str, Any]]:
    """Wire the runtime used by the API and CLI."""

    from ..db.repository import build_repository

    settings.ensure_dirs()
    repository = build_repository(settings.database_url)
    bundle, metrics = ensure_model(repository, settings)
    service = FraudScoringService(repository, bundle, settings)
    graph = GraphService(repository)

    seeded = 0
    should_seed = settings.bootstrap_on_startup if seed is None else seed
    if should_seed and repository.count("transactions") == 0:
        seeded = seed_demo_data(repository, service, settings)
        graph.invalidate()

    info = {
        "model": bundle.describe(),
        "metrics": metrics,
        "seeded": seeded,
        "database": repository.dialect,
        "transactions": repository.count("transactions"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return repository, service, graph, info
