"""Feature engineering shared by offline training and online scoring.

Every feature is derived from the incoming transaction plus that customer's
prior history, so the exact same code path produces training rows and
real-time inference rows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .reference import category_risk, channel_risk

MAX_HISTORY = 100

# Tree models cannot extrapolate beyond the values they were trained on, so the
# unbounded amount ratios are saturated: anything above this is already extreme.
RATIO_CAP = 50.0

FEATURE_NAMES: list[str] = [
    "amount",
    "log_amount",
    "amount_to_customer_avg",
    "amount_zscore",
    "amount_to_24h_max",
    "hour_of_day",
    "is_night",
    "is_weekend",
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    "distinct_merchants_24h",
    "distinct_countries_24h",
    "geo_distance_home_km",
    "geo_velocity_kmh",
    "is_foreign",
    "is_new_device",
    "device_age_minutes",
    "device_customer_count",
    "is_new_merchant",
    "merchant_risk",
    "category_risk",
    "channel_risk",
    "customer_avg_amount",
    "account_age_days",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""

    try:
        lat1, lon1, lat2, lon2 = (float(v) for v in (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError):
        return 0.0
    if any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class FeatureBuilder:
    """Stateless transformer: (transaction, history) -> model features."""

    def __init__(
        self,
        merchants: Mapping[str, Mapping[str, Any]] | None = None,
        customer_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.merchants = dict(merchants or {})
        self.customer_profiles = dict(customer_profiles or {})

    def build(
        self,
        txn: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] = (),
        device_customer_count: int = 1,
    ) -> dict[str, float]:
        ts = to_datetime(txn.get("timestamp") or datetime.now(timezone.utc))
        amount = max(_safe_float(txn.get("amount")), 0.0)
        customer_id = str(txn.get("customer_id") or "")
        profile = self.customer_profiles.get(customer_id, {})

        home_lat = _safe_float(profile.get("home_latitude"), _safe_float(txn.get("latitude")))
        home_lon = _safe_float(profile.get("home_longitude"), _safe_float(txn.get("longitude")))
        home_country = str(profile.get("home_country") or txn.get("country") or "IN")
        avg_amount = max(_safe_float(profile.get("avg_amount"), amount or 1.0), 1.0)
        account_age_days = _safe_float(profile.get("account_age_days"), 0.0)

        merchant_id = str(txn.get("merchant_id") or "")
        merchant = self.merchants.get(merchant_id, {})
        category = str(merchant.get("category") or txn.get("merchant_category") or "unknown")
        merchant_score = _safe_float(merchant.get("risk_score"), category_risk(category))
        channel = str(txn.get("channel") or "ecom").lower()
        country = str(txn.get("country") or home_country)
        latitude = _safe_float(txn.get("latitude"), home_lat)
        longitude = _safe_float(txn.get("longitude"), home_lon)
        device_id = str(txn.get("device_id") or "")

        hist = list(history)[-MAX_HISTORY:]
        epochs = np.fromiter(
            (
                float(h["_epoch"])
                if h.get("_epoch") is not None
                else to_datetime(h.get("timestamp")).timestamp()
                for h in hist
            ),
            dtype=float,
            count=len(hist),
        )
        amounts = np.fromiter((_safe_float(h.get("amount")) for h in hist), dtype=float, count=len(hist))
        now = ts.timestamp()

        if len(hist):
            delta = now - epochs
            mask_1h = (delta >= 0) & (delta <= 3600)
            mask_24h = (delta >= 0) & (delta <= 86400)
            mask_7d = (delta >= 0) & (delta <= 604800)
            count_1h = int(mask_1h.sum())
            count_24h = int(mask_24h.sum())
            count_7d = int(mask_7d.sum())
            sum_1h = float(amounts[mask_1h].sum())
            sum_24h = float(amounts[mask_24h].sum())
            max_24h = float(amounts[mask_24h].max()) if count_24h else 0.0
            merchants_24h = {
                str(h.get("merchant_id"))
                for h, flag in zip(hist, mask_24h)
                if flag
            }
            countries_24h = {
                str(h.get("country"))
                for h, flag in zip(hist, mask_24h)
                if flag
            }
            hist_mean = float(amounts.mean())
            hist_std = float(amounts.std(ddof=0))
            seen_devices = {str(h.get("device_id")) for h in hist}
            seen_merchants = {str(h.get("merchant_id")) for h in hist}
            prev = hist[-1]
            prev_ts = to_datetime(prev.get("timestamp"))
            hours = max((ts - prev_ts).total_seconds() / 3600, 0.0)
            step_km = haversine_km(
                latitude,
                longitude,
                _safe_float(prev.get("latitude"), latitude),
                _safe_float(prev.get("longitude"), longitude),
            )
            velocity = step_km / hours if hours > 0 else 0.0
            device_epochs = [
                float(epoch)
                for epoch, row in zip(epochs, hist)
                if str(row.get("device_id")) == device_id
            ]
            device_age_minutes = (
                max(0.0, (now - min(device_epochs)) / 60.0) if device_epochs else 0.0
            )
        else:
            count_1h = count_24h = count_7d = 0
            sum_1h = sum_24h = 0.0
            max_24h = 0.0
            merchants_24h = set()
            countries_24h = set()
            hist_mean = avg_amount
            hist_std = avg_amount * 0.5
            seen_devices = set()
            seen_merchants = set()
            velocity = 0.0
            device_age_minutes = 0.0

        distance_home = haversine_km(latitude, longitude, home_lat, home_lon)
        zscore = (amount - hist_mean) / max(hist_std, 1.0)

        features = {
            "amount": round(amount, 2),
            "log_amount": round(math.log1p(amount), 6),
            "amount_to_customer_avg": round(min(amount / avg_amount, RATIO_CAP), 6),
            "amount_zscore": round(max(min(zscore, 12.0), -12.0), 6),
            "amount_to_24h_max": round(min(amount / max(max_24h, 1.0), RATIO_CAP), 6),
            "hour_of_day": float(ts.hour),
            "is_night": float(1 if ts.hour <= 5 else 0),
            "is_weekend": float(1 if ts.weekday() >= 5 else 0),
            "txn_count_1h": float(count_1h),
            "txn_count_24h": float(count_24h),
            "txn_count_7d": float(count_7d),
            "amount_sum_1h": round(sum_1h, 2),
            "amount_sum_24h": round(sum_24h, 2),
            "distinct_merchants_24h": float(len(merchants_24h)),
            "distinct_countries_24h": float(len(countries_24h)),
            "geo_distance_home_km": round(distance_home, 3),
            "geo_velocity_kmh": round(min(velocity, 5000.0), 3),
            "is_foreign": float(1 if country.upper() != home_country.upper() else 0),
            "is_new_device": float(1 if device_id and device_id not in seen_devices else 0),
            "device_age_minutes": round(min(device_age_minutes, 525600.0), 3),
            "device_customer_count": float(max(device_customer_count, 1)),
            "is_new_merchant": float(1 if merchant_id and merchant_id not in seen_merchants else 0),
            "merchant_risk": round(merchant_score, 4),
            "category_risk": round(category_risk(category), 4),
            "channel_risk": round(channel_risk(channel), 4),
            "customer_avg_amount": round(avg_amount, 2),
            "account_age_days": round(account_age_days, 1),
        }
        return {name: float(features[name]) for name in FEATURE_NAMES}


def build_training_matrix(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    tail: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Walk the transaction stream chronologically and build model features.

    ``tail`` limits feature construction to the most recent N rows while still
    replaying full history, which keeps demo seeding fast.
    """

    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
    tx = tx.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    total = len(tx)
    cutoff = 0 if tail is None else max(total - int(tail), 0)

    merchant_index = {
        str(row["merchant_id"]): row for row in merchants.to_dict("records")
    }
    profiles = {str(row["customer_id"]): row for row in customers.to_dict("records")}
    builder = FeatureBuilder(merchant_index, profiles)

    history: dict[str, list[dict]] = defaultdict(list)
    device_customers: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, float]] = []
    labels: list[int] = []

    for position, record in enumerate(tx.to_dict("records")):
        record["_epoch"] = record["timestamp"].timestamp()
        customer_id = str(record["customer_id"])
        device_id = str(record.get("device_id") or "")
        device_customers[device_id].add(customer_id)
        if position >= cutoff:
            rows.append(
                builder.build(
                    record,
                    history[customer_id],
                    device_customer_count=len(device_customers[device_id]),
                )
            )
            labels.append(int(record["is_fraud"]))
        history[customer_id].append(record)
        if len(history[customer_id]) > MAX_HISTORY:
            history[customer_id] = history[customer_id][-MAX_HISTORY:]

    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return features, np.asarray(labels, dtype=int)
