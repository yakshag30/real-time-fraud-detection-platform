"""Train supervised fraud classifiers and an Isolation Forest anomaly detector."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..data import Dataset, FEATURE_NAMES, build_training_matrix

logger = logging.getLogger(__name__)

try:  # XGBoost is optional so the platform still runs on minimal hosts
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on the environment
    XGBClassifier = None  # type: ignore[assignment]
    HAS_XGBOOST = False


def _safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return float("nan")


def _curve_points(xs: np.ndarray, ys: np.ndarray, max_points: int = 60) -> list[list[float]]:
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(xs, ys)]


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Classification quality for a continuous risk score."""

    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    precision, recall, f1 = precision_score(y_true, predictions, zero_division=0), recall_score(
        y_true, predictions, zero_division=0
    ), f1_score(y_true, predictions, zero_division=0)
    return {
        "threshold": float(threshold),
        "roc_auc": _safe_metric(roc_auc_score, y_true, scores),
        "pr_auc": _safe_metric(average_precision_score, y_true, scores),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier": _safe_metric(brier_score_loss, y_true, scores),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "support": {
            "rows": int(len(y_true)),
            "positives": int(y_true.sum()),
            "negatives": int((y_true == 0).sum()),
        },
    }


def _curves(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[list[float]]]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "roc": _curve_points(fpr, tpr),
        "pr": _curve_points(recall, precision),
    }


def _feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    importance = getattr(model, "feature_importances_", None)
    if importance is None and hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            importance = np.abs(np.ravel(coef))
    if importance is None:
        importance = np.ones(len(feature_names))
    pairs = sorted(
        zip(feature_names, np.asarray(importance, dtype=float)),
        key=lambda item: item[1],
        reverse=True,
    )
    total = sum(value for _, value in pairs) or 1.0
    return [
        {"feature": name, "importance": round(float(value) / total, 5)}
        for name, value in pairs
    ]


def candidate_models(scale_pos_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=0.5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=320,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            reg_lambda=1.2,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    return models


def train_models(dataset: Dataset, settings: Settings) -> dict[str, Any]:
    """Fit candidate models, evaluate them on a time-ordered holdout and persist the best."""

    settings.ensure_dirs()
    started = time.perf_counter()
    features, labels = build_training_matrix(
        dataset.transactions, dataset.customers, dataset.merchants
    )

    split_at = max(int(len(features) * 0.8), 1)
    X_train, X_test = features.iloc[:split_at], features.iloc[split_at:]
    y_train, y_test = labels[:split_at], labels[split_at:]
    if y_test.sum() == 0:
        raise ValueError("Holdout window contains no fraudulent transactions; enlarge the dataset.")

    positives = max(int(y_train.sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    model_results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}

    for name, model in candidate_models(negatives / positives).items():
        start = time.perf_counter()
        try:
            model.fit(X_train, y_train)
        except Exception as exc:  # keep training even if one library misbehaves
            logger.warning("Skipping model %s: %s", name, exc)
            continue
        probability = np.asarray(model.predict_proba(X_test))[:, 1]
        result = evaluate_scores(y_test, probability, threshold=0.5)
        result["at_policy_threshold"] = evaluate_scores(
            y_test, probability, threshold=settings.alert_min_score
        )
        result["curves"] = _curves(y_test, probability)
        result["fit_seconds"] = round(time.perf_counter() - start, 2)
        model_results[name] = result
        fitted[name] = model
        logger.info(
            "Trained %-20s PR-AUC %.3f ROC-AUC %.3f in %.1fs",
            name,
            result["pr_auc"],
            result["roc_auc"],
            result["fit_seconds"],
        )

    if not fitted:
        raise RuntimeError("No model could be trained; check the installed dependencies.")

    # ---------------------------------------------------------- anomaly detector
    contamination = float(np.clip(y_train.mean() * 1.5, 0.01, 0.1))
    isolation = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    # Semi-supervised: learn the shape of normal spend, then flag deviations.
    isolation.fit(X_train)
    train_decision = isolation.decision_function(X_train)
    center = float(train_decision.mean())
    scale = float(train_decision.std() or 1.0)
    anomaly_test = 1.0 / (1.0 + np.exp((isolation.decision_function(X_test) - center) / max(scale, 1e-6)))
    anomaly_metrics = evaluate_scores(y_test, anomaly_test, threshold=settings.alert_min_score)
    anomaly_metrics["curves"] = _curves(y_test, anomaly_test)
    anomaly_metrics["contamination"] = contamination

    # ------------------------------------------------------------ pick best model
    best_name = max(
        (name for name in model_results if np.isfinite(model_results[name]["pr_auc"])),
        key=lambda name: model_results[name]["pr_auc"],
        default=next(iter(model_results)),
    )
    best_model = fitted[best_name]
    probability = np.asarray(best_model.predict_proba(X_test))[:, 1]
    combined = (1 - settings.anomaly_weight) * probability + settings.anomaly_weight * anomaly_test
    model_results[best_name]["combined"] = evaluate_scores(
        y_test, combined, threshold=settings.alert_min_score
    )

    metrics: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best_name,
        "dataset": {
            "rows": int(len(features)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "fraud_rate": round(float(labels.mean()), 5),
            "features": len(FEATURE_NAMES),
            "split": "chronological 80/20",
            "scenarios": {
                str(k): int(v)
                for k, v in dataset.transactions["fraud_type"].value_counts().items()
            },
        },
        "models": model_results,
        "anomaly": anomaly_metrics,
        "feature_importance": _feature_importance(best_model, FEATURE_NAMES),
        "thresholds": {
            "review": settings.review_threshold,
            "block": settings.block_threshold,
            "alert": settings.alert_min_score,
        },
        "training_seconds": round(time.perf_counter() - started, 2),
        "library_versions": {
            "xgboost": __import__("xgboost").__version__ if HAS_XGBOOST else None,
        },
    }

    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": list(FEATURE_NAMES),
        "isolation_forest": isolation,
        "feature_stats": {
            "mean": [float(v) for v in X_train.mean().to_numpy()],
            "std": [float(v if v else 1.0) for v in X_train.std(ddof=0).to_numpy()],
        },
        "metrics": metrics,
        "trained_at": metrics["trained_at"],
        "anomaly_center": center,
        "anomaly_scale": scale,
    }
    joblib.dump(payload, settings.model_path)
    settings.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(
        "Saved %s model to %s (PR-AUC %.3f) in %.1fs",
        best_name,
        settings.model_path,
        model_results[best_name]["pr_auc"],
        metrics["training_seconds"],
    )
    return metrics
