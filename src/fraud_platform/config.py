"""Central configuration for the fraud platform.

All runtime knobs are environment driven so the same code runs locally,
in Docker Compose (PostgreSQL) and in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only hit in minimal installs

    def load_dotenv(*_args, **_kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw: str, default: str) -> Path:
    path = Path(_env_str(raw, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    database_url: str
    review_threshold: float
    block_threshold: float
    alert_min_score: float
    anomaly_weight: float
    bootstrap_on_startup: bool
    bootstrap_rows: int
    api_host: str
    api_port: int
    api_url: str

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=_resolve_path("FRAUD_DATA_DIR", "data"),
        artifacts_dir=_resolve_path("FRAUD_ARTIFACTS_DIR", "artifacts"),
        database_url=_env_str("FRAUD_DATABASE_URL", "sqlite:///data/fraud.db"),
        review_threshold=_env_float("FRAUD_REVIEW_THRESHOLD", 0.35),
        block_threshold=_env_float("FRAUD_BLOCK_THRESHOLD", 0.75),
        alert_min_score=_env_float("FRAUD_ALERT_MIN_SCORE", 0.50),
        anomaly_weight=min(max(_env_float("FRAUD_ANOMALY_WEIGHT", 0.25), 0.0), 1.0),
        bootstrap_on_startup=_env_bool("FRAUD_BOOTSTRAP", True),
        bootstrap_rows=_env_int("FRAUD_BOOTSTRAP_ROWS", 1200),
        api_host=_env_str("FRAUD_API_HOST", "127.0.0.1"),
        api_port=_env_int("FRAUD_API_PORT", 8000),
        api_url=_env_str("API_URL", "http://127.0.0.1:8000"),
    )
