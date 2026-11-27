"""Deterministic risk policy layered on top of the ML score.

A supervised tree model is calibrated on the traffic it has seen. A single
live transaction can combine several textbook takeover indicators while the
model probability stays moderate, because genuine travellers and night-shift
customers produce the same combination one indicator at a time.

Banks keep a transparent rule layer for exactly this case: when several
independent high-severity indicators agree, the transaction gets a minimum
risk score so it reaches a human. The layer only ever raises the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

MIN_INDICATORS = 3
BASE_FLOOR = 0.55
PER_INDICATOR = 0.08
MAX_FLOOR = 0.95


@dataclass
class PolicyAssessment:
    """Outcome of the rule layer for one transaction."""

    floor: float = 0.0
    indicators: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return self.floor > 0.0


def assess_policy(features: Mapping[str, float]) -> PolicyAssessment:
    """Score the independent high-severity indicators of one transaction."""

    indicators: list[str] = []

    ratio = float(features.get("amount_to_customer_avg", 1.0))
    if ratio >= 10:
        indicators.append(f"amount is {ratio:.0f}x the customer's average spend")
    if float(features.get("is_foreign", 0)) >= 1:
        indicators.append("cross-border transaction")
    if (
        float(features.get("is_new_device", 0)) >= 1
        and float(features.get("device_age_minutes", 0)) <= 0
    ):
        indicators.append("brand-new device")
    if float(features.get("hour_of_day", 12)) <= 5:
        indicators.append("overnight hour")
    if float(features.get("geo_velocity_kmh", 0)) >= 900:
        indicators.append("impossible travel speed")
    if float(features.get("device_customer_count", 1)) >= 3:
        indicators.append("device shared across customers")

    if len(indicators) < MIN_INDICATORS:
        return PolicyAssessment(floor=0.0, indicators=indicators)

    floor = min(BASE_FLOOR + PER_INDICATOR * len(indicators), MAX_FLOOR)
    return PolicyAssessment(floor=round(floor, 4), indicators=indicators)
