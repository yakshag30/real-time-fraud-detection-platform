"""Pydantic contracts for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Decision = Literal["allow", "review", "block"]
RiskLevel = Literal["low", "medium", "high", "critical"]
AlertStatus = Literal["open", "acknowledged", "investigating", "closed", "escalated"]


class TransactionIn(BaseModel):
    model_config = {"extra": "forbid"}

    transaction_id: str | None = Field(default=None, description="Client-side idempotency key")
    timestamp: datetime | None = Field(default=None, description="ISO-8601; defaults to now (UTC)")
    customer_id: str = Field(min_length=1, max_length=64)
    customer_name: str | None = None
    amount: float = Field(gt=0, le=100_000_000)
    currency: str = Field(default="INR", max_length=8)
    merchant_id: str = Field(min_length=1, max_length=64)
    merchant_name: str | None = None
    merchant_category: str | None = None
    channel: str = Field(default="ecom", max_length=16)
    city: str | None = None
    country: str | None = Field(default=None, max_length=2)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    device_id: str | None = None
    ip_address: str | None = None
    ground_truth: int | None = Field(
        default=None, ge=0, le=1, description="Labelled outcome, for evaluation and demos only"
    )

    @field_validator("channel")
    @classmethod
    def _lower_channel(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("country")
    @classmethod
    def _upper_country(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else value


class BatchRequest(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=500)
    persist: bool = True


class ExplanationItem(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    transaction_id: str
    timestamp: str
    customer_id: str
    customer_name: str | None = None
    amount: float
    currency: str = "INR"
    merchant_id: str | None = None
    merchant_name: str | None = None
    merchant_category: str | None = None
    channel: str | None = None
    city: str | None = None
    country: str | None = None
    device_id: str | None = None
    fraud_probability: float
    anomaly_score: float
    combined_score: float
    risk_level: RiskLevel
    decision: Decision
    model_name: str
    review_threshold: float
    block_threshold: float
    reasons: list[str]
    summary: str
    anomaly_note: str
    explanation: list[ExplanationItem]
    explanation_method: str
    base_value: float
    graph_flags: list[str]
    policy_floor: float = 0.0
    policy_flags: list[str] = Field(default_factory=list)
    alert_id: str | None = None
    latency_ms: float


class BatchResponse(BaseModel):
    processed: int
    flagged: int
    results: list[ScoreResponse]


class AlertUpdate(BaseModel):
    status: AlertStatus
    analyst_note: str | None = Field(default=None, max_length=2000)


class AlertRecord(BaseModel):
    alert_id: str
    transaction_id: str | None = None
    customer_id: str | None = None
    severity: str | None = None
    score: float | None = None
    status: str | None = None
    title: str | None = None
    summary: list[str] | str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    analyst_note: str | None = None


class TransactionRecord(BaseModel):
    transaction_id: str
    timestamp: str
    customer_id: str
    amount: float
    currency: str | None = None
    merchant_id: str | None = None
    merchant_category: str | None = None
    channel: str | None = None
    city: str | None = None
    country: str | None = None
    device_id: str | None = None
    fraud_probability: float | None = None
    anomaly_score: float | None = None
    combined_score: float | None = None
    decision: str | None = None
    risk_level: str | None = None
    reasons: list[str] | None = None
    summary: str | None = None
    graph_flags: list[str] | None = None
    alert_id: str | None = None
    customer_name: str | None = None
    merchant_name: str | None = None
    anomaly_note: str | None = None
    explanation: dict | None = None
    explanation_method: str | None = None
    base_value: float | None = None
    model_name: str | None = None
    review_threshold: float | None = None
    block_threshold: float | None = None
    email: str | None = None
