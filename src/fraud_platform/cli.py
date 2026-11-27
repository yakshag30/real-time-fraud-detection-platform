"""Command line interface for the fraud platform."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .data import ensure_dataset, generate_and_save
from .data.features import to_datetime
from .data.reference import CITY_INDEX
from .db import build_repository
from .models import load_bundle, train_models
from .services import FraudScoringService, GraphService, seed_demo_data

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_platform.cli")


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _train(args: argparse.Namespace) -> dict:
    settings = get_settings()
    settings.ensure_dirs()
    dataset = ensure_dataset(
        settings.data_dir,
        force=args.regenerate,
        n_customers=args.customers,
        n_days=args.days,
    )
    metrics = train_models(dataset, settings)
    print(
        f"Selected {metrics['selected_model']} | PR-AUC "
        f"{metrics['models'][metrics['selected_model']]['pr_auc']:.3f} | ROC-AUC "
        f"{metrics['models'][metrics['selected_model']]['roc_auc']:.3f}"
    )
    print(f"Artifacts written to {settings.artifacts_dir}")
    return metrics


def _seed(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = build_repository(settings.database_url)
    bundle = load_bundle(settings.model_path)
    if bundle is None:
        raise SystemExit("No trained model found. Run `python -m fraud_platform.cli train` first.")
    service = FraudScoringService(repository, bundle, settings)
    if args.reset:
        repository.clear_transactions()
    seeded = seed_demo_data(repository, service, settings, rows=args.rows)
    print(f"Seeded {seeded} transactions into {settings.database_url}")
    return seeded


def _score(args: argparse.Namespace) -> dict:
    settings = get_settings()
    repository = build_repository(settings.database_url)
    bundle = load_bundle(settings.model_path)
    if bundle is None:
        raise SystemExit("No trained model found. Run `python -m fraud_platform.cli train` first.")
    service = FraudScoringService(repository, bundle, settings)
    GraphService(repository)

    city = CITY_INDEX.get(args.city, CITY_INDEX["Mumbai"])
    timestamp = datetime.now(timezone.utc).replace(
        hour=args.hour, minute=12, second=0, microsecond=0
    )
    payload = {
        "customer_id": args.customer,
        "amount": args.amount,
        "merchant_id": args.merchant,
        "channel": args.channel,
        "city": args.city,
        "country": city["country"],
        "latitude": city["lat"],
        "longitude": city["lon"],
        "device_id": args.device,
        "timestamp": timestamp,
    }
    result = service.score(payload, persist=not args.no_persist)
    console = {
        "transaction_id": result["transaction_id"],
        "decision": result["decision"],
        "risk_level": result["risk_level"],
        "fraud_probability": result["fraud_probability"],
        "anomaly_score": result["anomaly_score"],
        "combined_score": result["combined_score"],
        "summary": result["summary"],
        "reasons": result["reasons"],
        "top_features": [
            {
                "feature": item["feature"],
                "contribution": item["contribution"],
                "value": item["value"],
            }
            for item in result["explanation"][:5]
        ],
        "explanation_method": result["explanation_method"],
        "alert_id": result["alert_id"],
        "latency_ms": result["latency_ms"],
    }
    _print(console)
    return result


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "fraud_platform.api.main:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )


def _dashboard(args: argparse.Namespace) -> None:
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        "127.0.0.1",
    ]
    print("Starting dashboard:", " ".join(command))
    subprocess.run(command, check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-platform",
        description="Real-time fraud detection platform: data, models, API and dashboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="Generate the synthetic banking dataset")
    p.add_argument("--customers", type=int, default=500)
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", help="Train and evaluate candidate models")
    p.add_argument("--regenerate", action="store_true", help="Regenerate the dataset first")
    p.add_argument("--customers", type=int, default=500)
    p.add_argument("--days", type=int, default=45)

    p = sub.add_parser("seed", help="Seed the database with scored transactions")
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--reset", action="store_true", help="Delete existing transactions first")

    p = sub.add_parser("setup", help="Generate data, train models and seed the database")
    p.add_argument("--rows", type=int, default=3000)

    p = sub.add_parser("score", help="Score a single transaction from the terminal")
    p.add_argument("--customer", default="C00001")
    p.add_argument("--amount", type=float, default=200000)
    p.add_argument("--city", default="Moscow")
    p.add_argument("--hour", type=int, default=3)
    p.add_argument("--channel", default="ecom")
    p.add_argument("--merchant", default="M0001")
    p.add_argument("--device", default="DEV-NEVER-SEEN-01")
    p.add_argument("--no-persist", action="store_true")

    p = sub.add_parser("serve", help="Run the FastAPI service")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")

    p = sub.add_parser("dashboard", help="Run the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501)

    sub.add_parser("info", help="Show runtime configuration and dataset summary")
    return parser


def _info() -> None:
    settings = get_settings()
    dataset = ensure_dataset(settings.data_dir)
    bundle = load_bundle(settings.model_path)
    _print(
        {
            "settings": {
                "database_url": settings.database_url,
                "data_dir": str(settings.data_dir),
                "artifacts_dir": str(settings.artifacts_dir),
                "review_threshold": settings.review_threshold,
                "block_threshold": settings.block_threshold,
                "alert_min_score": settings.alert_min_score,
                "anomaly_weight": settings.anomaly_weight,
            },
            "dataset": dataset.summary(),
            "model": bundle.describe() if bundle else "not trained yet",
        }
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate-data":
        settings = get_settings()
        dataset = generate_and_save(
            settings.data_dir, n_customers=args.customers, n_days=args.days, seed=args.seed
        )
        _print(dataset.summary())
    elif args.command == "train":
        _train(args)
    elif args.command == "seed":
        _seed(args)
    elif args.command == "setup":
        settings = get_settings()
        settings.ensure_dirs()
        dataset = ensure_dataset(settings.data_dir)
        logger.info("Dataset: %s", dataset.summary())
        metrics = train_models(dataset, settings)
        print(f"Model trained: {metrics['selected_model']}")
        repository = build_repository(settings.database_url)
        bundle = load_bundle(settings.model_path)
        assert bundle is not None
        service = FraudScoringService(repository, bundle, settings)
        seeded = seed_demo_data(repository, service, settings, rows=args.rows)
        print(f"Seeded {seeded} transactions. Start the API with `python -m fraud_platform.cli serve`.")
    elif args.command == "score":
        _score(args)
    elif args.command == "serve":
        _serve(args)
    elif args.command == "dashboard":
        _dashboard(args)
    elif args.command == "info":
        _info()


if __name__ == "__main__":
    main()
