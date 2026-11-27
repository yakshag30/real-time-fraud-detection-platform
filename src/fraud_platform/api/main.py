"""HTTP API for the real-time fraud platform."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, get_settings
from ..data import ensure_dataset
from ..data.reference import CHANNEL_RISK, CITIES
from ..db import build_repository
from ..db.repository import SQLRepository
from ..graph import GraphService
from ..models import load_bundle, train_models
from ..models.explain import FEATURE_LABELS
from ..services import (
    FraudScoringService,
    build_investigation_report,
    ensure_model,
    seed_demo_data,
)
from .schemas import (
    AlertRecord,
    AlertUpdate,
    BatchRequest,
    BatchResponse,
    ScoreResponse,
    TransactionIn,
    TransactionRecord,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_platform.api")


@dataclass
class Runtime:
    settings: Settings
    repository: SQLRepository | None = None
    service: FraudScoringService | None = None
    graph: GraphService | None = None
    bootstrap: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.service is not None


runtime = Runtime(settings=get_settings())

# Built React console (``cd web && bun run build``). Served at /app when present.
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = runtime.settings
    settings.ensure_dirs()
    logger.info("Starting fraud platform (database=%s)", settings.database_url.split("@")[-1])
    repository = build_repository(settings.database_url)
    bundle, metrics = ensure_model(repository, settings)
    service = FraudScoringService(repository, bundle, settings)
    graph = GraphService(repository)

    seeded = 0
    if settings.bootstrap_on_startup and repository.count("transactions") == 0:
        logger.info("Empty transaction store - seeding %s demo transactions", settings.bootstrap_rows)
        seeded = seed_demo_data(repository, service, settings)
        graph.invalidate()

    runtime.repository = repository
    runtime.service = service
    runtime.graph = graph
    runtime.bootstrap = {
        "model": bundle.describe(),
        "seeded": seeded,
        "database": repository.dialect,
        "transactions": repository.count("transactions"),
        "training_metrics": metrics,
    }
    logger.info("Fraud platform ready: %s", runtime.bootstrap["model"])
    yield
    logger.info("Shutting down fraud platform")


app = FastAPI(
    title="Real-Time Fraud Detection Platform",
    description=(
        "Scores banking transactions in real time with Random Forest / XGBoost, "
        "an Isolation Forest anomaly detector, SHAP explanations and graph-based "
        "fraud-ring analytics."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def service() -> FraudScoringService:
    if runtime.service is None:
        raise HTTPException(status_code=503, detail="Scoring service is still starting up")
    return runtime.service


def repository() -> SQLRepository:
    if runtime.repository is None:
        raise HTTPException(status_code=503, detail="Database is still starting up")
    return runtime.repository


def graph_service() -> GraphService:
    if runtime.graph is None:
        raise HTTPException(status_code=503, detail="Graph service is still starting up")
    return runtime.graph


# --------------------------------------------------------------------- basics
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/" if (WEB_DIST / "index.html").is_file() else "/docs")


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    repo = runtime.repository
    bundle = runtime.service.bundle if runtime.service else None
    return {
        "status": "ok" if runtime.ready else "starting",
        "version": app.version,
        "database": repo.dialect if repo else None,
        "database_ok": repo.ping() if repo else False,
        "transactions": repo.count("transactions") if repo else 0,
        "alerts": repo.count("alerts") if repo else 0,
        "model": bundle.describe() if bundle else None,
        "thresholds": {
            "review": runtime.settings.review_threshold,
            "block": runtime.settings.block_threshold,
            "alert": runtime.settings.alert_min_score,
            "anomaly_weight": runtime.settings.anomaly_weight,
        },
        "bootstrap": {k: v for k, v in runtime.bootstrap.items() if k != "training_metrics"},
    }


# ------------------------------------------------------------------- scoring
@app.post("/transactions/score", response_model=ScoreResponse, tags=["scoring"])
def score_preview(payload: TransactionIn) -> dict[str, Any]:
    """Score a transaction without storing it (what-if analysis)."""

    return service().score(payload.model_dump(), persist=False)


@app.post("/transactions", response_model=ScoreResponse, status_code=201, tags=["scoring"])
def ingest_transaction(payload: TransactionIn) -> dict[str, Any]:
    """Score, explain and store a transaction, raising an alert when risky."""

    repo = repository()
    if payload.transaction_id and repo.get_transaction(payload.transaction_id):
        raise HTTPException(status_code=409, detail="transaction_id already processed")
    return service().score(payload.model_dump(), persist=True)


@app.post("/transactions/batch", response_model=BatchResponse, tags=["scoring"])
def ingest_batch(batch: BatchRequest) -> dict[str, Any]:
    scorer = service()
    results = [
        scorer.score(txn.model_dump(), persist=batch.persist) for txn in batch.transactions
    ]
    return {
        "processed": len(results),
        "flagged": sum(1 for r in results if r["decision"] != "allow"),
        "results": results,
    }


@app.get("/transactions", tags=["scoring"])
def list_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    customer_id: str | None = None,
    min_score: float | None = Query(default=None, ge=0, le=1),
    decision: str | None = None,
    since_hours: int | None = Query(default=None, ge=1, le=24 * 30),
) -> dict[str, Any]:
    rows = repository().list_transactions(
        limit=limit,
        offset=offset,
        customer_id=customer_id,
        min_score=min_score,
        decision=decision,
        since_hours=since_hours,
    )
    names = {
        customer["customer_id"]: customer.get("name")
        for customer in repository().list_customers(limit=1000)
    }
    for row in rows:
        row["customer_name"] = names.get(row.get("customer_id"))
    return {"count": len(rows), "transactions": rows}


@app.get("/transactions/{transaction_id}", response_model=TransactionRecord, tags=["scoring"])
def get_transaction(transaction_id: str) -> dict[str, Any]:
    row = repository().get_transaction(transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return row


# -------------------------------------------------------------------- alerts
@app.get("/alerts", response_model=list[AlertRecord], tags=["alerts"])
def list_alerts(
    status: str | None = Query(default=None, pattern="^(open|acknowledged|investigating|closed|escalated)$"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return repository().list_alerts(status=status, limit=limit)


@app.get("/alerts/{alert_id}", response_model=AlertRecord, tags=["alerts"])
def get_alert(alert_id: str) -> dict[str, Any]:
    alert = repository().get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.patch("/alerts/{alert_id}", response_model=AlertRecord, tags=["alerts"])
def update_alert(alert_id: str, update: AlertUpdate) -> dict[str, Any]:
    repo = repository()
    if repo.get_alert(alert_id) is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    updated = repo.update_alert(alert_id, update.status, update.analyst_note)
    return updated or {}


# ----------------------------------------------------------------- analytics
@app.get("/stats/overview", tags=["analytics"])
def stats_overview() -> dict[str, Any]:
    return repository().overview_stats(alert_threshold=runtime.settings.alert_min_score)


@app.get("/stats/timeseries", tags=["analytics"])
def stats_timeseries(hours: int = Query(default=48, ge=1, le=24 * 30)) -> list[dict[str, Any]]:
    return repository().timeseries(hours=hours, alert_threshold=runtime.settings.alert_min_score)


@app.get("/stats/breakdown", tags=["analytics"])
def stats_breakdown(
    field: str = Query(default="merchant_category"),
    hours: int = Query(default=168, ge=1, le=24 * 30),
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    try:
        return repository().breakdown(
            field, hours=hours, alert_threshold=runtime.settings.alert_min_score, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/stats/top-customers", tags=["analytics"])
def stats_top_customers(limit: int = Query(default=10, ge=1, le=50)) -> list[dict[str, Any]]:
    return repository().top_risky_customers(limit=limit)


# -------------------------------------------------------------------- models
@app.get("/models/info", tags=["models"])
def model_info() -> dict[str, Any]:
    bundle = service().bundle
    return {
        "model": bundle.describe(),
        "metrics": bundle.metrics,
        "features": [
            {"name": name, "label": FEATURE_LABELS.get(name, name)}
            for name in bundle.feature_names
        ],
        "explainer": service().explainer.method,
        "anomaly_weight": runtime.settings.anomaly_weight,
    }


@app.post("/models/retrain", tags=["models"])
def retrain() -> dict[str, Any]:
    settings = runtime.settings
    dataset = ensure_dataset(settings.data_dir)
    metrics = train_models(dataset, settings)
    bundle = load_bundle(settings.model_path)
    if bundle is None:
        raise HTTPException(status_code=500, detail="Retraining finished but artifact is missing")
    runtime.service = FraudScoringService(repository(), bundle, settings)
    graph_service().invalidate()
    return {"model": bundle.describe(), "metrics": metrics}


# ----------------------------------------------------------------- customers
@app.get("/customers", tags=["customers"])
def list_customers(limit: int = Query(default=500, ge=1, le=2000)) -> dict[str, Any]:
    customers = repository().list_customers(limit=limit)
    return {"count": len(customers), "customers": customers}


@app.get("/customers/{customer_id}", tags=["customers"])
def get_customer(customer_id: str) -> dict[str, Any]:
    customer = repository().get_customer(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    stats = repository().overview_stats(alert_threshold=runtime.settings.alert_min_score)
    return {"customer": customer, "platform": stats}


@app.get("/customers/{customer_id}/transactions", tags=["customers"])
def customer_transactions(
    customer_id: str, limit: int = Query(default=50, ge=1, le=500)
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "transactions": repository().customer_transactions(customer_id, limit=limit),
    }


@app.get("/customers/{customer_id}/investigation", tags=["investigations"])
def investigation(customer_id: str, save: bool = False) -> dict[str, Any]:
    try:
        report = build_investigation_report(
            repository(), graph_service(), runtime.settings, customer_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Customer not found") from exc
    if save:
        investigation_id = repository().save_report(
            {
                "customer_id": customer_id,
                "status": "draft",
                "summary": report["summary"],
                "report_markdown": report["markdown"],
            }
        )
        report["investigation_id"] = investigation_id
    return report


@app.get("/reports", tags=["investigations"])
def list_reports(
    customer_id: str | None = None, limit: int = Query(default=50, ge=1, le=200)
) -> dict[str, Any]:
    return {"reports": repository().list_reports(customer_id=customer_id, limit=limit)}


@app.get("/reports/{investigation_id}/markdown", response_class=PlainTextResponse, tags=["investigations"])
def report_markdown(investigation_id: str) -> str:
    report = repository().get_report(investigation_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return str(report.get("report_markdown") or "")


# --------------------------------------------------------------------- graph
@app.get("/graph/rings", tags=["graph"])
def graph_rings() -> dict[str, Any]:
    return graph_service().rings()


@app.get("/graph/customer/{customer_id}", tags=["graph"])
def graph_customer(customer_id: str, radius: int = Query(default=1, ge=1, le=2)) -> dict[str, Any]:
    network = graph_service().network(customer_id, radius=radius)
    if not network.get("found"):
        raise HTTPException(status_code=404, detail="Customer not present in the transaction graph")
    network["ring"] = graph_service().ring_for_customer(customer_id)
    return network


# ---------------------------------------------------------------------- meta
@app.get("/meta/options", tags=["meta"])
def options() -> dict[str, Any]:
    repo = repository()
    customers = repo.list_customers(limit=500)
    merchants = repo.list_merchants(limit=500)
    customer = customers[0] if customers else {"customer_id": "C00001", "name": "Demo customer"}
    risky = next(
        (m for m in merchants if m.get("category") in {"gift_cards", "crypto_exchange", "gaming"}),
        merchants[0] if merchants else {"merchant_id": "M0001", "name": "Demo merchant"},
    )
    suspicious_time = datetime.now(timezone.utc).replace(
        hour=3, minute=12, second=0, microsecond=0
    ) - timedelta(days=0)
    return {
        "customers": [
            {
                "customer_id": c["customer_id"],
                "name": c.get("name"),
                "home_city": c.get("home_city"),
                "avg_amount": c.get("avg_amount"),
            }
            for c in customers
        ],
        "merchants": [
            {
                "merchant_id": m["merchant_id"],
                "name": m.get("name"),
                "category": m.get("category"),
                "risk_score": m.get("risk_score"),
            }
            for m in merchants
        ],
        "cities": [
            {"name": c["name"], "country": c["country"], "lat": c["lat"], "lon": c["lon"]}
            for c in CITIES
        ],
        "channels": sorted(CHANNEL_RISK),
        "demo_scenario": {
            "customer_id": customer["customer_id"],
            "customer_name": customer.get("name"),
            "amount": 200000,
            "timestamp": suspicious_time.isoformat(timespec="seconds"),
            "merchant_id": risky.get("merchant_id"),
            "merchant_name": risky.get("name"),
            "merchant_category": risky.get("category"),
            "channel": "ecom",
            "city": "Moscow",
            "country": "RU",
            "latitude": 55.7558,
            "longitude": 37.6173,
            "device_id": "DEV-NEVER-SEEN-01",
        },
    }


@app.get("/meta/features", tags=["meta"])
def feature_catalog() -> dict[str, Any]:
    return {
        "features": [
            {"name": name, "label": FEATURE_LABELS.get(name, name)}
            for name in service().bundle.feature_names
        ]
    }


# ---------------------------------------------------------------- web console
# Mounted last so it never shadows an API route. Rebuild with `cd web && bun run build`.
if (WEB_DIST / "index.html").is_file():
    app.mount("/app", StaticFiles(directory=str(WEB_DIST), html=True), name="console")
    logger.info("Serving React console from %s at /app", WEB_DIST)
else:
    logger.info("Web console not built; run `cd web && bun install && bun run build`")
