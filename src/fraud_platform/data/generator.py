"""Synthetic core-banking dataset with labelled fraud scenarios.

The generator mimics a retail bank's transaction stream and injects four
realistic fraud patterns so models and graph analytics have something to
learn from:

* account takeover    - new device, foreign geo, night hours, high amount
* card testing        - bursts of tiny e-commerce payments on one device
* fraud ring          - several customers sharing a device/IP and merchants
* merchant collusion  - repeated identical amounts at one merchant

The ``is_fraud`` / ``fraud_type`` columns are ground truth used for training
and evaluation only; they are never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .reference import (
    CATEGORY_PROFILE,
    CHANNEL_WEIGHTS,
    FOREIGN_CITIES,
    FRAUD_CITIES,
    HIGH_RISK_CATEGORIES,
    HOME_CITIES,
)

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Ishaan", "Kabir", "Meera",
    "Rohan", "Sanya", "Tara", "Vihaan", "Zara", "Neha", "Arjun", "Priya",
    "Karan", "Sneha", "Rahul", "Pooja", "Nikhil", "Aisha", "Dev", "Riya",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Iyer", "Nair", "Reddy", "Gupta", "Mehta",
    "Kulkarni", "Chopra", "Bose", "Menon", "Kapoor", "Joshi", "Rao", "Desai",
    "Bhat", "Malhotra", "Sethi", "Pillai",
]

# Retail spending peaks in the evening; fraud scenarios override this.
_HOUR_WEIGHTS = np.array(
    [0.6, 0.4, 0.3, 0.25, 0.25, 0.35, 0.8, 1.4, 2.2, 2.8, 3.0, 2.9,
     2.7, 2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.0, 2.4, 1.8, 1.2, 0.8],
    dtype=float,
)
_HOUR_PROBS = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

# A minority of customers genuinely transact overnight (shift workers, expats)
_NIGHT_WEIGHTS = np.array(
    [3.0, 2.4, 1.6, 1.0, 0.8, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6,
     1.8, 1.8, 1.9, 2.0, 2.2, 2.4, 2.8, 3.0, 3.4, 3.8, 4.0, 3.6],
    dtype=float,
)
_NIGHT_PROBS = _NIGHT_WEIGHTS / _NIGHT_WEIGHTS.sum()


@dataclass
class Dataset:
    """In-memory dataset bundle."""

    customers: pd.DataFrame
    merchants: pd.DataFrame
    transactions: pd.DataFrame

    def summary(self) -> dict:
        tx = self.transactions
        fraud = tx[tx["is_fraud"] == 1]
        return {
            "customers": int(len(self.customers)),
            "merchants": int(len(self.merchants)),
            "transactions": int(len(tx)),
            "fraud_transactions": int(len(fraud)),
            "fraud_rate": round(float(len(fraud) / max(len(tx), 1)), 5),
            "volume": round(float(tx["amount"].sum()), 2),
            "scenarios": {
                str(k): int(v) for k, v in fraud["fraud_type"].value_counts().items()
            },
            "window": {
                "start": str(tx["timestamp"].min()),
                "end": str(tx["timestamp"].max()),
            },
        }


def _pick_weighted(rng: np.random.Generator, options: list, weights: list[float]):
    probs = np.asarray(weights, dtype=float)
    probs = probs / probs.sum()
    return options[int(rng.choice(len(options), p=probs))]


def _jitter_geo(rng: np.random.Generator, city: dict) -> dict:
    """Accept both the reference-city shape and the customer profile shape."""

    return {
        "city": city.get("name") or city.get("city") or "Unknown",
        "country": city.get("country") or "IN",
        "latitude": round(
            float(city.get("lat", city.get("latitude", 0.0)) + rng.normal(0, 0.04)), 5
        ),
        "longitude": round(
            float(city.get("lon", city.get("longitude", 0.0)) + rng.normal(0, 0.04)), 5
        ),
    }


def _random_ts(
    rng: np.random.Generator, start: datetime, n_days: int, night_owl: bool = False
) -> datetime:
    day = int(rng.integers(0, n_days))
    hour = int(rng.choice(24, p=_NIGHT_PROBS if night_owl else _HOUR_PROBS))
    return start + timedelta(
        days=day,
        hours=hour,
        minutes=int(rng.integers(0, 60)),
        seconds=int(rng.integers(0, 60)),
    )


def _random_ip(rng: np.random.Generator) -> str:
    return (
        f"49.{int(rng.integers(0, 255))}.{int(rng.integers(0, 255))}."
        f"{int(rng.integers(1, 255))}"
    )


def generate_dataset(
    n_customers: int = 500,
    n_merchants: int = 140,
    n_days: int = 45,
    seed: int = 42,
    end: datetime | None = None,
) -> Dataset:
    """Generate customers, merchants and a labelled transaction stream."""

    rng = np.random.default_rng(seed)
    if end is None:
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=n_days)

    # ------------------------------------------------------------------ customers
    customers: list[dict] = []
    for i in range(n_customers):
        cid = f"C{i + 1:05d}"
        city = _pick_weighted(rng, HOME_CITIES, [c["home_weight"] for c in HOME_CITIES])
        avg_amount = float(np.exp(rng.normal(np.log(1900), 0.72)) + 120)
        customers.append(
            {
                "customer_id": cid,
                "name": f"{FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]} "
                        f"{LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]}",
                "home_city": city["name"],
                "home_country": city["country"],
                "home_latitude": city["lat"],
                "home_longitude": city["lon"],
                "avg_amount": round(avg_amount, 2),
                "account_age_days": int(rng.integers(30, 3600)),
                "segment": _pick_weighted(rng, ["retail", "priority", "wealth"], [0.7, 0.22, 0.08]),
            }
        )

    # ------------------------------------------------------------------ merchants
    categories = list(CATEGORY_PROFILE)
    category_weights = [CATEGORY_PROFILE[c][1] for c in categories]
    merchants: list[dict] = []
    for i in range(n_merchants):
        category = _pick_weighted(rng, categories, category_weights)
        city = _pick_weighted(rng, HOME_CITIES, [c["home_weight"] for c in HOME_CITIES])
        risk = float(np.clip(CATEGORY_PROFILE[category][0] + rng.normal(0, 0.08), 0.02, 0.98))
        merchants.append(
            {
                "merchant_id": f"M{i + 1:04d}",
                "name": f"{category.replace('_', ' ').title()} {city['name']} {i + 1:03d}",
                "category": category,
                "city": city["name"],
                "country": city["country"],
                "risk_score": round(risk, 3),
            }
        )

    risky_merchants = [m for m in merchants if m["category"] in HIGH_RISK_CATEGORIES]
    big_ticket_merchants = [
        m for m in merchants if m["category"] in {"electronics", "travel", "fashion"}
    ]
    channel_names = list(CHANNEL_WEIGHTS)
    channel_weights = list(CHANNEL_WEIGHTS.values())

    # Families share a tablet/phone; this is the benign explanation for
    # several customers touching the same device.
    households: dict[int, str] = {}
    shuffled = rng.permutation(n_customers)
    for household in range(n_customers // 14):
        for member in shuffled[household * 2 : household * 2 + 2]:
            households[int(member)] = f"DEV-HH-{household:03d}"
    night_owls = set(
        int(i) for i in rng.choice(n_customers, size=min(40, n_customers), replace=False)
    )

    # Small businesses and shared family plans legitimately reuse one terminal.
    group_devices: dict[int, str] = {}
    group_shuffle = rng.permutation(n_customers)
    for group in range(max(n_customers // 20, 1)):
        for member in group_shuffle[group * 4 : group * 4 + 4]:
            group_devices[int(member)] = f"DEV-GRP-{group:03d}"

    txns: list[dict] = []

    def add_txn(
        customer: dict,
        ts: datetime,
        amount: float,
        merchant: dict,
        channel: str,
        device: str,
        ip: str,
        geo: dict,
        is_fraud: int = 0,
        fraud_type: str = "legit",
    ) -> None:
        txns.append(
            {
                "timestamp": ts,
                "customer_id": customer["customer_id"],
                "card_id": f"CARD-{customer['customer_id']}-01",
                "amount": round(max(float(amount), 1.0), 2),
                "currency": "INR",
                "merchant_id": merchant["merchant_id"],
                "merchant_category": merchant["category"],
                "channel": channel,
                "city": geo["city"],
                "country": geo["country"],
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "device_id": device,
                "ip_address": ip,
                "status": "completed",
                "is_fraud": int(is_fraud),
                "fraud_type": fraud_type,
            }
        )

    # -------------------------------------------------------------- legitimate spend
    for index, customer in enumerate(customers):
        cid = customer["customer_id"]
        home = {
            "city": customer["home_city"],
            "country": customer["home_country"],
            "lat": customer["home_latitude"],
            "lon": customer["home_longitude"],
        }
        primary_device = f"DEV-{cid}-01"
        secondary_device = f"DEV-{cid}-02"
        household_device = households.get(index)
        group_device = group_devices.get(index)
        night_owl = index in night_owls
        n_txns = max(4, int(rng.poisson(0.6 * n_days)))

        for _ in range(n_txns):
            ts = _random_ts(rng, start, n_days, night_owl=night_owl)
            merchant = merchants[int(rng.integers(len(merchants)))]
            channel = _pick_weighted(rng, channel_names, channel_weights)

            roll = rng.random()
            travel = False
            if roll < 0.86:
                geo = _jitter_geo(rng, home)
            elif roll < 0.95:
                geo = _jitter_geo(rng, _pick_weighted(
                    rng, HOME_CITIES, [c["home_weight"] for c in HOME_CITIES]
                ))
            else:
                travel = True
                geo = _jitter_geo(rng, _pick_weighted(
                    rng, FOREIGN_CITIES, [c["fraud_weight"] for c in FOREIGN_CITIES]
                ))

            if channel == "atm":
                amount = float(int(rng.integers(1, 21)) * 500)
            else:
                amount = float(np.exp(rng.normal(np.log(customer["avg_amount"]), 0.62)))
                amount = min(amount, 260000.0)
            if travel:
                amount *= float(rng.uniform(1.4, 4.0))
            if big_ticket_merchants and rng.random() < 0.03:
                merchant = big_ticket_merchants[int(rng.integers(len(big_ticket_merchants)))]
                amount *= float(rng.uniform(2.4, 6.0))
            if channel != "atm":
                # Genuine spending stays within a plausible ceiling for the
                # customer so that amount ratios remain a fraud signal instead
                # of being dominated by routine big-ticket purchases.
                ceiling = max(customer["avg_amount"] * 10.0, 5000.0)
                amount = min(amount, ceiling, 300000.0)

            device_choices = [primary_device, secondary_device]
            device_weights = [0.66, 0.22]
            if household_device:
                device_choices.append(household_device)
                device_weights.append(0.34)
            if group_device:
                device_choices.append(group_device)
                device_weights.append(0.20)
            if travel:
                device_choices.append(f"DEV-{cid}-TRAVEL")
                device_weights.append(0.30)
            device_choices.append(f"DEV-{cid}-NEW-{int(rng.integers(1, 999)):03d}")
            device_weights.append(0.12)
            device = _pick_weighted(rng, device_choices, device_weights)

            add_txn(customer, ts, amount, merchant, channel, device, _random_ip(rng), geo)

    # ------------------------------------------------- legitimate shopping sprees
    for index in rng.choice(n_customers, size=max(int(n_customers * 0.08), 1), replace=False):
        customer = customers[int(index)]
        home = {
            "city": customer["home_city"],
            "country": customer["home_country"],
            "lat": customer["home_latitude"],
            "lon": customer["home_longitude"],
        }
        base = start + timedelta(
            days=int(rng.integers(0, n_days)), hours=int(rng.integers(9, 19))
        )
        pool = rng.choice(len(merchants), size=5, replace=False)
        for k in range(int(rng.integers(4, 7))):
            ts = base + timedelta(minutes=int(k * 24 + rng.integers(0, 12)))
            merchant = merchants[int(pool[k % len(pool)])]
            add_txn(
                customer,
                ts,
                customer["avg_amount"] * float(rng.uniform(0.3, 1.5)),
                merchant,
                _pick_weighted(rng, ["pos", "ecom", "upi"], [0.4, 0.35, 0.25]),
                f"DEV-{customer['customer_id']}-01",
                _random_ip(rng),
                _jitter_geo(rng, home),
            )

    # ------------------------------------------- recurring rent/EMI-style payments
    for index in rng.choice(n_customers, size=max(int(n_customers * 0.14), 1), replace=False):
        customer = customers[int(index)]
        merchant = merchants[int(rng.integers(len(merchants)))]
        amount = round(customer["avg_amount"] * float(rng.uniform(0.8, 2.2)), 2)
        home = {
            "city": customer["home_city"],
            "country": customer["home_country"],
            "lat": customer["home_latitude"],
            "lon": customer["home_longitude"],
        }
        for k in range(int(rng.integers(1, 3))):
            ts = start + timedelta(
                days=28 * k + int(rng.integers(0, 7)),
                hours=int(rng.integers(8, 20)),
                minutes=int(rng.integers(0, 60)),
            )
            add_txn(
                customer,
                ts,
                amount,
                merchant,
                "upi",
                f"DEV-{customer['customer_id']}-01",
                _random_ip(rng),
                _jitter_geo(rng, home),
            )

    # ---------------------------------------------------------- account takeover
    ato_customers = rng.choice(n_customers, size=min(55, n_customers), replace=False)
    for idx in ato_customers:
        customer = customers[int(idx)]
        for k in range(int(rng.integers(1, 3))):
            ts = start + timedelta(
                days=int(rng.integers(0, n_days)),
                hours=int(rng.integers(1, 5)),
                minutes=int(rng.integers(0, 60)),
            )
            local_takeover = rng.random() < 0.35
            if local_takeover:
                city = {
                    "name": customer["home_city"],
                    "country": customer["home_country"],
                    "lat": customer["home_latitude"],
                    "lon": customer["home_longitude"],
                }
                amount = customer["avg_amount"] * float(rng.uniform(1.5, 3.0))
            else:
                city = _pick_weighted(rng, FRAUD_CITIES, [c["fraud_weight"] for c in FRAUD_CITIES])
                amount = customer["avg_amount"] * float(rng.uniform(3.0, 15.0))
            # Takeover spend is not confined to gambling/crypto merchants; most
            # of the time it looks like an ordinary purchase made by a stranger.
            if rng.random() < 0.65:
                merchant = merchants[int(rng.integers(len(merchants)))]
            else:
                merchant = risky_merchants[int(rng.integers(len(risky_merchants)))]
            add_txn(
                customer,
                ts,
                amount,
                merchant,
                "ecom" if k % 2 == 0 else "upi",
                f"DEV-ATO-{int(idx):04d}-{k}",
                f"185.220.{int(rng.integers(1, 250))}.{int(rng.integers(1, 250))}",
                _jitter_geo(rng, city),
                is_fraud=1,
                fraud_type="account_takeover",
            )

    # --------------------------------------------------------------- card testing
    ct_customers = rng.choice(n_customers, size=min(22, n_customers), replace=False)
    for idx in ct_customers:
        customer = customers[int(idx)]
        device = f"DEV-CT-{int(idx):04d}"
        ip = f"103.21.{int(rng.integers(1, 250))}.{int(rng.integers(1, 250))}"
        # Card testing hits whatever merchant the stolen card is tried on, so the
        # merchant category carries no signal for this scenario.
        merchant = merchants[int(rng.integers(len(merchants)))]
        base = start + timedelta(
            days=int(rng.integers(0, n_days)),
            hours=int(rng.integers(9, 23)),
            minutes=int(rng.integers(0, 50)),
        )
        for k in range(int(rng.integers(6, 11))):
            ts = base + timedelta(minutes=int(k * 4 + rng.integers(0, 4)))
            add_txn(
                customer,
                ts,
                float(rng.uniform(20, 300)),
                merchant,
                "ecom",
                device,
                ip,
                _jitter_geo(
                    rng,
                    {
                        "city": customer["home_city"],
                        "country": customer["home_country"],
                        "lat": customer["home_latitude"],
                        "lon": customer["home_longitude"],
                    },
                ),
                is_fraud=1,
                fraud_type="card_testing",
            )

    # ------------------------------------------------------------------ fraud ring
    for ring in range(10):
        size = int(rng.integers(4, 8))
        members = rng.choice(n_customers, size=size, replace=False)
        device = f"DEV-RING-{ring:02d}"
        ip = f"198.51.100.{10 + ring}"
        ring_merchants = [
            risky_merchants[int(i)]
            for i in rng.choice(len(risky_merchants), size=6, replace=False)
        ]
        for member in members:
            customer = customers[int(member)]
            home = {
                "city": customer["home_city"],
                "country": customer["home_country"],
                "lat": customer["home_latitude"],
                "lon": customer["home_longitude"],
            }
            for _ in range(int(rng.integers(2, 5))):
                ts = start + timedelta(
                    days=int(rng.integers(0, n_days)),
                    hours=int(rng.integers(10, 20)),
                    minutes=int(rng.integers(0, 60)),
                )
                if rng.random() < 0.5:
                    merchant = merchants[int(rng.integers(len(merchants)))]
                else:
                    merchant = ring_merchants[int(rng.integers(len(ring_merchants)))]
                add_txn(
                    customer,
                    ts,
                    customer["avg_amount"] * float(rng.uniform(1.3, 3.5)),
                    merchant,
                    "upi" if rng.random() < 0.6 else "ecom",
                    device,
                    ip,
                    _jitter_geo(rng, home),
                    is_fraud=1,
                    fraud_type="fraud_ring",
                )

    # --------------------------------------------------------- merchant collusion
    for idx in rng.choice(n_customers, size=min(12, n_customers), replace=False):
        customer = customers[int(idx)]
        merchant = risky_merchants[int(rng.integers(len(risky_merchants)))]
        home = {
            "city": customer["home_city"],
            "country": customer["home_country"],
            "lat": customer["home_latitude"],
            "lon": customer["home_longitude"],
        }
        base = start + timedelta(
            days=int(rng.integers(0, max(n_days - 3, 1))), hours=int(rng.integers(11, 18))
        )
        amount = customer["avg_amount"] * 1.6
        for k in range(int(rng.integers(3, 6))):
            add_txn(
                customer,
                base + timedelta(hours=k * 6, minutes=int(rng.integers(0, 40))),
                amount,
                merchant,
                "pos",
                f"DEV-{customer['customer_id']}-01",
                _random_ip(rng),
                _jitter_geo(rng, home),
                is_fraud=1,
                fraud_type="merchant_collusion",
            )

    transactions = pd.DataFrame(txns)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True)
    transactions = transactions.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    transactions.insert(0, "transaction_id", [f"T{i + 1:07d}" for i in range(len(transactions))])

    return Dataset(
        customers=pd.DataFrame(customers),
        merchants=pd.DataFrame(merchants),
        transactions=transactions,
    )
