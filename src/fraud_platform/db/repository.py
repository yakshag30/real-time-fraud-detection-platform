"""SQL repository for transactions, alerts and investigation reports.

SQLite is the zero-configuration default; PostgreSQL is used by Docker
Compose. Both share the same schema and a single ``?`` placeholder style
that is translated for psycopg.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

TRANSACTION_COLUMNS = [
    "transaction_id",
    "timestamp",
    "customer_id",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "channel",
    "city",
    "country",
    "latitude",
    "longitude",
    "device_id",
    "ip_address",
    "fraud_probability",
    "anomaly_score",
    "combined_score",
    "decision",
    "risk_level",
    "model_name",
    "features",
    "explanation",
    "reasons",
    "graph_flags",
    "ground_truth",
    "created_at",
]

ALERT_COLUMNS = [
    "alert_id",
    "transaction_id",
    "customer_id",
    "severity",
    "score",
    "status",
    "title",
    "summary",
    "created_at",
    "updated_at",
    "analyst_note",
]

CUSTOMER_COLUMNS = [
    "customer_id",
    "name",
    "home_city",
    "home_country",
    "home_latitude",
    "home_longitude",
    "avg_amount",
    "account_age_days",
    "segment",
    "created_at",
]

MERCHANT_COLUMNS = [
    "merchant_id",
    "name",
    "category",
    "city",
    "country",
    "risk_score",
    "created_at",
]

REPORT_COLUMNS = [
    "investigation_id",
    "customer_id",
    "alert_id",
    "status",
    "created_at",
    "summary",
    "report_markdown",
]

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT,
        home_city TEXT,
        home_country TEXT,
        home_latitude REAL,
        home_longitude REAL,
        avg_amount REAL,
        account_age_days INTEGER,
        segment TEXT,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS merchants (
        merchant_id TEXT PRIMARY KEY,
        name TEXT,
        category TEXT,
        city TEXT,
        country TEXT,
        risk_score REAL,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        customer_id TEXT,
        amount REAL,
        currency TEXT,
        merchant_id TEXT,
        merchant_category TEXT,
        channel TEXT,
        city TEXT,
        country TEXT,
        latitude REAL,
        longitude REAL,
        device_id TEXT,
        ip_address TEXT,
        fraud_probability REAL,
        anomaly_score REAL,
        combined_score REAL,
        decision TEXT,
        risk_level TEXT,
        model_name TEXT,
        features TEXT,
        explanation TEXT,
        reasons TEXT,
        graph_flags TEXT,
        ground_truth INTEGER,
        created_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_txn_customer_time ON transactions (customer_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_txn_time ON transactions (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_txn_score ON transactions (combined_score)",
    """
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        transaction_id TEXT,
        customer_id TEXT,
        severity TEXT,
        score REAL,
        status TEXT,
        title TEXT,
        summary TEXT,
        created_at TEXT,
        updated_at TEXT,
        analyst_note TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts (status, created_at)",
    """
    CREATE TABLE IF NOT EXISTS investigations (
        investigation_id TEXT PRIMARY KEY,
        customer_id TEXT,
        alert_id TEXT,
        status TEXT,
        created_at TEXT,
        summary TEXT,
        report_markdown TEXT
    )
    """,
]

JSON_COLUMNS = {
    "features",
    "explanation",
    "reasons",
    "graph_flags",
    "summary",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_utc(value: datetime | str | None) -> str:
    if value is None:
        return utc_now()
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


class SQLRepository:
    """Thin, explicit repository with no ORM dependency."""

    def __init__(self, url: str) -> None:
        self.url = url
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            raw = url[len("sqlite:///") :]
            path = Path(raw)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_path = str(path)
            self._dsn = None
        elif url.startswith("postgresql://") or url.startswith("postgres://"):
            self.dialect = "postgres"
            self._dsn = url
            self._sqlite_path = None
        else:
            raise ValueError(
                "Unsupported FRAUD_DATABASE_URL. Use sqlite:///path.db or postgresql://user:pass@host/db"
            )

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.dialect == "sqlite":
            conn = sqlite3.connect(self._sqlite_path, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
        else:
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self._dsn, row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.dialect == "postgres" else statement

    def _query(self, statement: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(self._sql(statement), tuple(params))
            rows = cursor.fetchall()
            cursor.close()
        return [dict(row) for row in rows]

    def _execute(self, statement: str, params: Sequence[Any] = ()) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(self._sql(statement), tuple(params))
            cursor.close()

    def _execute_many(self, statement: str, params: Sequence[Sequence[Any]]) -> None:
        if not params:
            return
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(self._sql(statement), list(params))
            cursor.close()

    def init_schema(self, retries: int = 1) -> None:
        last_error: Exception | None = None
        for attempt in range(max(retries, 1)):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    for statement in SCHEMA_STATEMENTS:
                        cursor.execute(self._sql(statement))
                    cursor.close()
                logger.info("Database schema ready (%s)", self.dialect)
                return
            except Exception as exc:  # pragma: no cover - retry loop for docker startup
                last_error = exc
                logger.warning("Database not ready (attempt %s): %s", attempt + 1, exc)
                time.sleep(1.5)
        raise RuntimeError(f"Could not initialise database schema: {last_error}")

    def ping(self) -> bool:
        try:
            self._query("SELECT 1 AS ok")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------- seeding
    def count(self, table: str) -> int:
        if table not in {"transactions", "alerts", "customers", "merchants"}:
            raise ValueError(f"Unknown table {table}")
        rows = self._query(f"SELECT COUNT(*) AS n FROM {table}")
        return int(rows[0]["n"]) if rows else 0

    def _upsert(self, table: str, key: str, columns: Sequence[str], rows: Sequence[dict]) -> None:
        if not rows:
            return
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{col}=excluded.{col}" for col in columns if col != key)
        statement = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}"
        )
        self._execute_many(statement, [tuple(row.get(col) for col in columns) for row in rows])

    def upsert_customers(self, customers: Sequence[dict]) -> None:
        now = utc_now()
        rows = []
        for customer in customers:
            record = {col: customer.get(col) for col in CUSTOMER_COLUMNS}
            record["created_at"] = record.get("created_at") or now
            rows.append(record)
        self._upsert("customers", "customer_id", CUSTOMER_COLUMNS, rows)

    def upsert_merchants(self, merchants: Sequence[dict]) -> None:
        now = utc_now()
        rows = []
        for merchant in merchants:
            record = {col: merchant.get(col) for col in MERCHANT_COLUMNS}
            record["created_at"] = record.get("created_at") or now
            rows.append(record)
        self._upsert("merchants", "merchant_id", MERCHANT_COLUMNS, rows)

    def clear_transactions(self) -> None:
        self._execute("DELETE FROM transactions")
        self._execute("DELETE FROM alerts")

    # -------------------------------------------------------------- lookups
    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        return rows[0] if rows else None

    def get_merchant(self, merchant_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM merchants WHERE merchant_id = ?", (merchant_id,))
        return rows[0] if rows else None

    def list_customers(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._query(
            "SELECT * FROM customers ORDER BY customer_id LIMIT ?", (limit,)
        )

    def list_merchants(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM merchants ORDER BY merchant_id LIMIT ?", (limit,))

    # ------------------------------------------------------------ transactions
    def insert_transaction(self, record: dict[str, Any]) -> None:
        payload = {col: record.get(col) for col in TRANSACTION_COLUMNS}
        for key in JSON_COLUMNS & set(payload):
            if payload[key] is not None and not isinstance(payload[key], str):
                payload[key] = json.dumps(payload[key])
        payload["timestamp"] = iso_utc(payload.get("timestamp"))
        payload["created_at"] = payload.get("created_at") or utc_now()
        placeholders = ", ".join("?" for _ in TRANSACTION_COLUMNS)
        self._execute(
            f"INSERT INTO transactions ({', '.join(TRANSACTION_COLUMNS)}) VALUES ({placeholders})",
            tuple(payload[col] for col in TRANSACTION_COLUMNS),
        )

    def _hydrate_transaction(self, row: dict[str, Any]) -> dict[str, Any]:
        for key in JSON_COLUMNS & set(row):
            raw = row.get(key)
            if isinstance(raw, str) and raw:
                try:
                    row[key] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        return row

    def get_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        )
        return self._hydrate_transaction(rows[0]) if rows else None

    def list_transactions(
        self,
        limit: int = 50,
        offset: int = 0,
        customer_id: str | None = None,
        min_score: float | None = None,
        decision: str | None = None,
        since_hours: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if customer_id:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if min_score is not None:
            clauses.append("combined_score >= ?")
            params.append(float(min_score))
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        if since_hours:
            cutoff = iso_utc(datetime.now(timezone.utc) - timedelta(hours=since_hours))
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([int(limit), int(offset)])
        rows = self._query(
            f"SELECT * FROM transactions {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        )
        return [self._hydrate_transaction(row) for row in rows]

    def customer_history(
        self, customer_id: str, before: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        params: list[Any] = [customer_id]
        clause = ""
        if before:
            clause = "AND timestamp < ?"
            params.append(iso_utc(before))
        params.append(int(limit))
        rows = self._query(
            f"""
            SELECT timestamp, amount, merchant_id, country, latitude, longitude, device_id, city
            FROM transactions WHERE customer_id = ? {clause}
            ORDER BY timestamp DESC LIMIT ?
            """,
            params,
        )
        return list(reversed(rows))

    def customer_transactions(self, customer_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM transactions WHERE customer_id = ? ORDER BY timestamp DESC LIMIT ?",
            (customer_id, limit),
        )
        return [self._hydrate_transaction(row) for row in rows]

    def device_customers(self, days: int = 180) -> dict[str, list[str]]:
        cutoff = iso_utc(datetime.now(timezone.utc) - timedelta(days=days))
        rows = self._query(
            """
            SELECT device_id, customer_id FROM transactions
            WHERE device_id IS NOT NULL AND device_id <> '' AND timestamp >= ?
            GROUP BY device_id, customer_id
            """,
            (cutoff,),
        )
        mapping: dict[str, list[str]] = {}
        for row in rows:
            mapping.setdefault(row["device_id"], []).append(row["customer_id"])
        return mapping

    def graph_transactions(self, hours: int = 720, limit: int = 5000) -> list[dict[str, Any]]:
        cutoff = iso_utc(datetime.now(timezone.utc) - timedelta(hours=hours))
        return self._query(
            """
            SELECT transaction_id, timestamp, customer_id, device_id, ip_address, merchant_id,
                   merchant_category, city, country, amount, combined_score, fraud_probability,
                   decision, ground_truth
            FROM transactions WHERE timestamp >= ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (cutoff, int(limit)),
        )

    # ------------------------------------------------------------------ alerts
    def insert_alert(self, alert: dict[str, Any]) -> None:
        payload = {col: alert.get(col) for col in ALERT_COLUMNS}
        # ``setdefault`` would keep an explicit None, which then breaks the
        # API response models; an alert always needs an id.
        payload["alert_id"] = payload.get("alert_id") or f"AL-{uuid4().hex[:12]}"
        payload["created_at"] = payload.get("created_at") or utc_now()
        payload["updated_at"] = payload.get("updated_at") or payload["created_at"]
        payload["status"] = payload.get("status") or "open"
        if not isinstance(payload.get("summary"), str):
            payload["summary"] = json.dumps(payload.get("summary") or [])
        placeholders = ", ".join("?" for _ in ALERT_COLUMNS)
        self._execute(
            f"INSERT INTO alerts ({', '.join(ALERT_COLUMNS)}) VALUES ({placeholders})",
            tuple(payload[col] for col in ALERT_COLUMNS),
        )

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        if not rows:
            return None
        alert = rows[0]
        if isinstance(alert.get("summary"), str) and alert["summary"]:
            try:
                alert["summary"] = json.loads(alert["summary"])
            except json.JSONDecodeError:
                pass
        return alert

    def list_alerts(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if status:
            where = "WHERE status = ?"
            params.append(status)
        params.append(int(limit))
        rows = self._query(
            f"SELECT * FROM alerts {where} ORDER BY created_at DESC LIMIT ?", params
        )
        for alert in rows:
            if isinstance(alert.get("summary"), str) and alert["summary"]:
                try:
                    alert["summary"] = json.loads(alert["summary"])
                except json.JSONDecodeError:
                    pass
        return rows

    def update_alert(
        self, alert_id: str, status: str, analyst_note: str | None = None
    ) -> dict[str, Any] | None:
        self._execute(
            "UPDATE alerts SET status = ?, analyst_note = COALESCE(?, analyst_note), updated_at = ? "
            "WHERE alert_id = ?",
            (status, analyst_note, utc_now(), alert_id),
        )
        return self.get_alert(alert_id)

    # ----------------------------------------------------------- investigations
    def save_report(self, report: dict[str, Any]) -> str:
        payload = {col: report.get(col) for col in REPORT_COLUMNS}
        payload["investigation_id"] = payload.get("investigation_id") or f"INV-{uuid4().hex[:10]}"
        payload["created_at"] = payload.get("created_at") or utc_now()
        payload["status"] = payload.get("status") or "draft"
        placeholders = ", ".join("?" for _ in REPORT_COLUMNS)
        self._execute(
            f"INSERT INTO investigations ({', '.join(REPORT_COLUMNS)}) VALUES ({placeholders})",
            tuple(payload[col] for col in REPORT_COLUMNS),
        )
        return str(payload["investigation_id"])

    def list_reports(self, customer_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if customer_id:
            where = "WHERE customer_id = ?"
            params.append(customer_id)
        params.append(int(limit))
        return self._query(
            f"SELECT * FROM investigations {where} ORDER BY created_at DESC LIMIT ?", params
        )

    def get_report(self, investigation_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM investigations WHERE investigation_id = ?", (investigation_id,)
        )
        return rows[0] if rows else None

    # --------------------------------------------------------------- analytics
    def overview_stats(self, alert_threshold: float = 0.5) -> dict[str, Any]:
        since_24h = iso_utc(datetime.now(timezone.utc) - timedelta(hours=24))
        totals = self._query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS volume, "
            "COALESCE(AVG(combined_score), 0) AS avg_score, MAX(timestamp) AS last_seen "
            "FROM transactions"
        )[0]
        recent = self._query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS volume FROM transactions "
            "WHERE timestamp >= ?",
            (since_24h,),
        )[0]
        flagged = self._query(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS volume FROM transactions "
            "WHERE combined_score >= ?",
            (alert_threshold,),
        )[0]
        alerts = self._query(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END), 0) AS open_alerts, "
            "COALESCE(SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END), 0) AS critical "
            "FROM alerts"
        )[0]
        total = int(totals["n"] or 0)
        return {
            "total_transactions": total,
            "total_volume": round(float(totals["volume"] or 0), 2),
            "transactions_24h": int(recent["n"] or 0),
            "volume_24h": round(float(recent["volume"] or 0), 2),
            "flagged_transactions": int(flagged["n"] or 0),
            "flagged_rate": round(float(flagged["n"] or 0) / total, 5) if total else 0.0,
            "flagged_amount": round(float(flagged["volume"] or 0), 2),
            "avg_score": round(float(totals["avg_score"] or 0), 4),
            "alerts_total": int(alerts["total"] or 0),
            "alerts_open": int(alerts["open_alerts"] or 0),
            "alerts_critical": int(alerts["critical"] or 0),
            "customers": self.count("customers"),
            "last_transaction_at": totals["last_seen"],
        }

    def timeseries(self, hours: int = 48, alert_threshold: float = 0.5) -> list[dict[str, Any]]:
        cutoff = iso_utc(datetime.now(timezone.utc) - timedelta(hours=hours))
        rows = self._query(
            """
            SELECT substr(timestamp, 1, 13) AS bucket, COUNT(*) AS count,
                   COALESCE(SUM(amount), 0) AS volume,
                   COALESCE(SUM(CASE WHEN combined_score >= ? THEN 1 ELSE 0 END), 0) AS flagged,
                   COALESCE(AVG(combined_score), 0) AS avg_score
            FROM transactions WHERE timestamp >= ?
            GROUP BY bucket ORDER BY bucket
            """,
            (alert_threshold, cutoff),
        )
        return [
            {
                "bucket": row["bucket"],
                "count": int(row["count"] or 0),
                "flagged": int(row["flagged"] or 0),
                "volume": round(float(row["volume"] or 0), 2),
                "avg_score": round(float(row["avg_score"] or 0), 4),
            }
            for row in rows
        ]

    def breakdown(
        self, field: str, hours: int = 168, alert_threshold: float = 0.5, limit: int = 10
    ) -> list[dict[str, Any]]:
        allowed = {"merchant_category", "channel", "country", "city", "decision", "merchant_id"}
        if field not in allowed:
            raise ValueError(f"Unsupported breakdown field: {field}")
        cutoff = iso_utc(datetime.now(timezone.utc) - timedelta(hours=hours))
        rows = self._query(
            f"""
            SELECT {field} AS label, COUNT(*) AS count,
                   COALESCE(SUM(amount), 0) AS volume,
                   COALESCE(SUM(CASE WHEN combined_score >= ? THEN 1 ELSE 0 END), 0) AS flagged,
                   COALESCE(AVG(combined_score), 0) AS avg_score
            FROM transactions WHERE timestamp >= ? AND {field} IS NOT NULL
            GROUP BY {field} ORDER BY flagged DESC, count DESC LIMIT ?
            """,
            (alert_threshold, cutoff, int(limit)),
        )
        return [
            {
                "label": str(row["label"]),
                "count": int(row["count"] or 0),
                "flagged": int(row["flagged"] or 0),
                "volume": round(float(row["volume"] or 0), 2),
                "avg_score": round(float(row["avg_score"] or 0), 4),
            }
            for row in rows
        ]

    def top_risky_customers(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._query(
            """
            SELECT customer_id, COUNT(*) AS transactions, MAX(combined_score) AS max_score,
                   COALESCE(AVG(combined_score), 0) AS avg_score,
                   COALESCE(SUM(CASE WHEN decision <> 'allow' THEN 1 ELSE 0 END), 0) AS flagged,
                   COALESCE(SUM(amount), 0) AS volume
            FROM transactions GROUP BY customer_id
            ORDER BY max_score DESC, volume DESC LIMIT ?
            """,
            (int(limit),),
        )
        return [
            {
                "customer_id": row["customer_id"],
                "transactions": int(row["transactions"] or 0),
                "max_score": round(float(row["max_score"] or 0), 4),
                "avg_score": round(float(row["avg_score"] or 0), 4),
                "flagged": int(row["flagged"] or 0),
                "volume": round(float(row["volume"] or 0), 2),
            }
            for row in rows
        ]


def build_repository(url: str) -> SQLRepository:
    repository = SQLRepository(url)
    repository.init_schema(retries=10 if repository.dialect == "postgres" else 1)
    return repository
