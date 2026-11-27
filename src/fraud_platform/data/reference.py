"""Static reference data shared by the generator, models and API.

Values here behave like a bank's dimension tables: city coordinates,
merchant category risk ratings and channel risk weights.
"""

from __future__ import annotations

from typing import Any

# name, ISO country, latitude, longitude, home-city weight, fraud-city weight
CITIES: list[dict[str, Any]] = [
    {"name": "Mumbai", "country": "IN", "lat": 19.0760, "lon": 72.8777, "home_weight": 18, "fraud_weight": 0},
    {"name": "Delhi", "country": "IN", "lat": 28.6139, "lon": 77.2090, "home_weight": 15, "fraud_weight": 0},
    {"name": "Bengaluru", "country": "IN", "lat": 12.9716, "lon": 77.5946, "home_weight": 14, "fraud_weight": 0},
    {"name": "Hyderabad", "country": "IN", "lat": 17.3850, "lon": 78.4867, "home_weight": 10, "fraud_weight": 0},
    {"name": "Chennai", "country": "IN", "lat": 13.0827, "lon": 80.2707, "home_weight": 9, "fraud_weight": 0},
    {"name": "Pune", "country": "IN", "lat": 18.5204, "lon": 73.8567, "home_weight": 9, "fraud_weight": 0},
    {"name": "Kolkata", "country": "IN", "lat": 22.5726, "lon": 88.3639, "home_weight": 8, "fraud_weight": 0},
    {"name": "Ahmedabad", "country": "IN", "lat": 23.0225, "lon": 72.5714, "home_weight": 6, "fraud_weight": 0},
    {"name": "Jaipur", "country": "IN", "lat": 26.9124, "lon": 75.7873, "home_weight": 5, "fraud_weight": 0},
    {"name": "Kochi", "country": "IN", "lat": 9.9312, "lon": 76.2673, "home_weight": 4, "fraud_weight": 0},
    {"name": "Lucknow", "country": "IN", "lat": 26.8467, "lon": 80.9462, "home_weight": 4, "fraud_weight": 0},
    {"name": "Chandigarh", "country": "IN", "lat": 30.7333, "lon": 76.7794, "home_weight": 3, "fraud_weight": 0},
    {"name": "Indore", "country": "IN", "lat": 22.7196, "lon": 75.8577, "home_weight": 3, "fraud_weight": 0},
    {"name": "Singapore", "country": "SG", "lat": 1.3521, "lon": 103.8198, "home_weight": 0, "fraud_weight": 4},
    {"name": "Dubai", "country": "AE", "lat": 25.2048, "lon": 55.2708, "home_weight": 0, "fraud_weight": 5},
    {"name": "London", "country": "GB", "lat": 51.5074, "lon": -0.1278, "home_weight": 0, "fraud_weight": 4},
    {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060, "home_weight": 0, "fraud_weight": 4},
    {"name": "Bangkok", "country": "TH", "lat": 13.7563, "lon": 100.5018, "home_weight": 0, "fraud_weight": 4},
    {"name": "Lagos", "country": "NG", "lat": 6.5244, "lon": 3.3792, "home_weight": 0, "fraud_weight": 4},
    {"name": "Moscow", "country": "RU", "lat": 55.7558, "lon": 37.6173, "home_weight": 0, "fraud_weight": 4},
    {"name": "Hong Kong", "country": "HK", "lat": 22.3193, "lon": 114.1694, "home_weight": 0, "fraud_weight": 3},
    {"name": "Istanbul", "country": "TR", "lat": 41.0082, "lon": 28.9784, "home_weight": 0, "fraud_weight": 3},
    {"name": "Sao Paulo", "country": "BR", "lat": -23.5505, "lon": -46.6333, "home_weight": 0, "fraud_weight": 3},
]

HOME_CITIES = [c for c in CITIES if c["home_weight"] > 0]
FOREIGN_CITIES = [c for c in CITIES if c["home_weight"] == 0]
FRAUD_CITIES = [c for c in CITIES if c["fraud_weight"] > 0]
CITY_INDEX = {c["name"]: c for c in CITIES}

COUNTRY_RISK: dict[str, float] = {
    "IN": 0.05,
    "SG": 0.10,
    "GB": 0.15,
    "US": 0.15,
    "AE": 0.35,
    "HK": 0.30,
    "TR": 0.45,
    "TH": 0.30,
    "BR": 0.40,
    "NG": 0.55,
    "RU": 0.60,
}

# category -> (risk rating, sampling weight)
CATEGORY_PROFILE: dict[str, tuple[float, float]] = {
    "grocery": (0.08, 0.24),
    "dining": (0.10, 0.18),
    "fuel": (0.12, 0.12),
    "pharmacy": (0.10, 0.08),
    "fashion": (0.22, 0.13),
    "electronics": (0.35, 0.10),
    "travel": (0.30, 0.07),
    "utilities": (0.06, 0.08),
    "telecom": (0.14, 0.06),
    "gaming": (0.55, 0.06),
    "gift_cards": (0.72, 0.04),
    "crypto_exchange": (0.82, 0.03),
    "money_transfer": (0.66, 0.04),
}

CATEGORY_RISK: dict[str, float] = {k: v[0] for k, v in CATEGORY_PROFILE.items()}

CHANNEL_RISK: dict[str, float] = {
    "pos": 0.15,
    "ecom": 0.45,
    "upi": 0.35,
    "atm": 0.30,
    "wallet": 0.40,
}

CHANNEL_WEIGHTS: dict[str, float] = {
    "upi": 0.32,
    "pos": 0.28,
    "ecom": 0.28,
    "atm": 0.05,
    "wallet": 0.07,
}

HIGH_RISK_CATEGORIES = [
    "gaming",
    "gift_cards",
    "crypto_exchange",
    "money_transfer",
    "electronics",
    "travel",
]


def category_risk(category: str | None) -> float:
    if not category:
        return 0.30
    return CATEGORY_RISK.get(category, 0.30)


def channel_risk(channel: str | None) -> float:
    if not channel:
        return 0.30
    return CHANNEL_RISK.get(channel, 0.30)


def country_risk(country: str | None) -> float:
    if not country:
        return 0.20
    return COUNTRY_RISK.get(str(country).upper(), 0.35)
