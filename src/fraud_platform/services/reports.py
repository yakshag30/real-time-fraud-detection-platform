"""Investigation reports for analysts and case files."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..db.repository import SQLRepository
from ..graph.service import GraphService
from ..models.explain import format_inr


def _fmt_amount(value: Any) -> str:
    try:
        return format_inr(float(value))
    except (TypeError, ValueError):
        return "₹0"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No records._"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_investigation_report(
    repository: SQLRepository,
    graph_service: GraphService,
    settings: Settings,
    customer_id: str,
    alert_id: str | None = None,
) -> dict[str, Any]:
    customer = repository.get_customer(customer_id)
    if customer is None:
        raise KeyError(customer_id)

    transactions = repository.customer_transactions(customer_id, limit=40)
    alerts = [
        alert
        for alert in repository.list_alerts(limit=300)
        if alert.get("customer_id") == customer_id
    ]
    network = graph_service.network(customer_id, radius=1)
    ring = graph_service.ring_for_customer(customer_id)

    scores = [float(t.get("combined_score") or 0.0) for t in transactions]
    top_score = max(scores, default=0.0)
    flagged = [t for t in transactions if (t.get("decision") or "allow") != "allow"]
    volume = sum(float(t.get("amount") or 0.0) for t in transactions)
    devices = {
        str(t.get("device_id"))
        for t in transactions
        if t.get("device_id")
    }
    countries = {
        str(t.get("country"))
        for t in transactions
        if t.get("country")
    }
    top_transaction = max(transactions, key=lambda t: float(t.get("combined_score") or 0), default=None)

    highlights: list[str] = []
    if top_transaction:
        highlights = list(top_transaction.get("reasons") or [])
        if not highlights and top_transaction.get("explanation"):
            explanation = top_transaction["explanation"]
            highlights = [
                item.get("label", "")
                for item in explanation.get("top", [])[:4]
                if item.get("contribution", 0) > 0
            ]

    risk_grade = "Critical" if top_score >= 0.9 else "High" if top_score >= settings.block_threshold else "Elevated" if top_score >= settings.review_threshold else "Low"

    generated_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    lines: list[str] = [
        f"# Investigation report — {customer.get('name') or customer_id}",
        "",
        f"**Customer:** {customer_id} · {customer.get('segment', 'retail')} · "
        f"home city {customer.get('home_city')} ({customer.get('home_country')})  ",
        f"**Generated:** {generated_at}  ",
        f"**Risk grade:** {risk_grade} (highest score {top_score:.0%})  ",
        f"**Alert reference:** {alert_id or 'n/a'}",
        "",
        "## Case summary",
        "",
        f"- {len(transactions)} transactions reviewed, {len(flagged)} outside policy thresholds",
        f"- {_fmt_amount(volume)} total value across the reviewed window",
        f"- {len(alerts)} alerts raised, {sum(1 for a in alerts if a.get('status') == 'open')} still open",
        f"- Devices observed: {', '.join(sorted(devices)[:5]) or 'none recorded'}",
        f"- Countries observed: {', '.join(sorted(countries)[:6]) or 'n/a'}",
        "",
    ]

    if highlights:
        lines += ["## Why the activity is suspicious", ""]
        lines += [f"{index}. {reason}" for index, reason in enumerate(highlights[:6], start=1)]
        lines.append("")

    lines += ["## Most suspicious transactions", ""]
    lines.append(
        _table(
            ["Time", "Transaction", "Amount", "Score", "Decision", "Channel", "Location"],
            [
                [
                    str(t.get("timestamp", ""))[:16].replace("T", " "),
                    t.get("transaction_id"),
                    _fmt_amount(t.get("amount")),
                    f"{float(t.get('combined_score') or 0):.0%}",
                    str(t.get("decision", "")).upper(),
                    str(t.get("channel", "")).upper(),
                    f"{t.get('city')}, {t.get('country')}",
                ]
                for t in sorted(transactions, key=lambda t: float(t.get("combined_score") or 0), reverse=True)[:10]
            ],
        )
    )
    lines.append("")

    linked_devices = [
        node["label"]
        for node in network.get("nodes", [])
        if node.get("kind") == "device" and node.get("id") != f"device:{transactions[0].get('device_id') if transactions else ''}"
    ][:6]
    lines += ["## Network context", ""]
    if ring:
        lines += [
            f"- Linked to **{ring['ring_id']}** ({ring['size']} accounts, "
            f"risk {ring['risk_score']:.0%}, {_fmt_amount(ring['amount'])} moved)",
            f"- Shared devices: {', '.join(ring.get('shared_devices', [])[:4]) or 'none'}",
            f"- Shared IPs: {', '.join(ring.get('shared_ips', [])[:4]) or 'none'}",
            f"- Merchants involved: {', '.join(ring.get('shared_merchants', [])[:4]) or 'none'}",
        ]
    else:
        lines.append("- No fraud ring detected around this customer in the current graph window.")
    if linked_devices:
        lines.append(f"- Other devices in the neighbourhood: {', '.join(linked_devices)}")
    lines.append("")

    lines += [
        "## Recommended next steps",
        "",
        "1. Contact the customer on the registered phone number to confirm the flagged transactions.",
        "2. If unconfirmed, block the card/UPI handle and reset credentials for the shared device.",
        "3. Escalate shared-device or shared-IP clusters to the mule-account team for the linked accounts.",
        "4. File a suspicious transaction report (STR) when the customer denies the activity.",
        "",
        "---",
        "",
        f"_Generated by the real-time fraud platform using {settings.bootstrap_rows and 'the'} "
        f"production scoring model and SHAP-based explanations. Scores are probabilistic and "
        f"must be reviewed by an analyst._",
    ]

    markdown = "\n".join(lines)
    summary = (
        f"{risk_grade} risk customer with {len(flagged)} flagged transactions totalling "
        f"{_fmt_amount(sum(float(t.get('amount') or 0) for t in flagged))}"
    )

    return {
        "customer": customer,
        "risk_grade": risk_grade,
        "summary": summary,
        "markdown": markdown,
        "transactions": transactions,
        "alerts": alerts,
        "network": network,
        "ring": ring,
        "generated_at": generated_at,
        "metrics": {
            "transactions_reviewed": len(transactions),
            "flagged_transactions": len(flagged),
            "volume": round(volume, 2),
            "top_score": round(top_score, 4),
        },
    }
