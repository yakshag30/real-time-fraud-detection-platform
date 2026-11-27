"""Power BI-style fraud operations console (Streamlit).

The dashboard talks to the FastAPI service only, so the same UI works against
a local SQLite deployment or a PostgreSQL-backed container stack.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timezone

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30

RISK_COLORS = {
    "critical": "#ef4444",
    "high": "#f97316",
    "medium": "#facc15",
    "low": "#22c55e",
}
KIND_COLORS = {
    "customer": "#38bdf8",
    "device": "#f59e0b",
    "merchant": "#a78bfa",
    "ip": "#34d399",
}

st.set_page_config(
    page_title="Sentinel · Fraud Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1600px;}
    section[data-testid="stSidebar"] {background: #0a1524; border-right: 1px solid #16263d;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    .sentinel-title {font-size: 1.7rem; font-weight: 700; margin-bottom: 0; color: #f1f5f9;}
    .sentinel-sub {color: #7d8ea6; font-size: 0.86rem; margin-top: 0.1rem;}
    .kpi-card {
        background: linear-gradient(160deg, #101e31 0%, #0b1624 100%);
        border: 1px solid #1b2f49; border-radius: 14px; padding: 0.9rem 1.1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.25); height: 100%;
    }
    .kpi-label {color: #7d8ea6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em;}
    .kpi-value {font-size: 1.55rem; font-weight: 700; color: #f8fafc; margin-top: 0.15rem;}
    .kpi-note {font-size: 0.72rem; color: #7d8ea6; margin-top: 0.2rem;}
    .badge {
        display: inline-block; padding: 0.18rem 0.62rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    }
    .badge-critical {background: rgba(239,68,68,0.16); color: #f87171; border: 1px solid rgba(239,68,68,0.4);}
    .badge-high {background: rgba(249,115,22,0.16); color: #fb923c; border: 1px solid rgba(249,115,22,0.4);}
    .badge-medium {background: rgba(250,204,21,0.14); color: #facc15; border: 1px solid rgba(250,204,21,0.35);}
    .badge-low {background: rgba(34,197,94,0.14); color: #4ade80; border: 1px solid rgba(34,197,94,0.35);}
    .reason-box {
        background: #0c1828; border-left: 3px solid #38bdf8; border-radius: 8px;
        padding: 0.75rem 1rem; margin-bottom: 0.5rem; color: #dbe6f3; font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    .stTabs [data-baseweb="tab"] {font-size: 0.9rem;}
    .dataframe {font-size: 0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- api helpers
def api_get(path: str, params: dict | None = None):
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        if response.status_code >= 400:
            return None, f"{response.status_code}: {response.text[:240]}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_post(path: str, payload: dict, method: str = "post"):
    try:
        response = requests.request(method, f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text) if response.content else ""
            return None, f"{response.status_code}: {detail}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_patch(path: str, payload: dict):
    return api_post(path, payload, method="patch")


def chart_layout(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=40 if title else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c3d2e5", size=12),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def badge(level: str, text: str | None = None) -> str:
    level = (level or "low").lower()
    return f'<span class="badge badge-{level}">{text or level}</span>'


def money(value: float | int | None) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return "₹0"
    if number >= 1_00_00_000:
        return f"₹{number / 1_00_00_000:.2f} Cr"
    if number >= 1_00_000:
        return f"₹{number / 1_00_000:.2f} L"
    return f"₹{number:,.0f}"


def pct(value: float | None) -> str:
    return f"{(value or 0) * 100:.1f}%"


@st.cache_data(ttl=8, show_spinner=False)
def cached(path: str, params: tuple | None = None):
    return api_get(path, dict(params) if params else None)


def live_api() -> dict | None:
    health, error = api_get("/health")
    if error or not health:
        st.error(
            f"Cannot reach the fraud API at **{API_URL}**. Start it with "
            "`python -m fraud_platform.cli serve`.\n\n"
            f"Details: {error}"
        )
        return None
    return health


# ------------------------------------------------------------------- sidebar
health = live_api()
if health is None:
    st.stop()

with st.sidebar:
    st.markdown("### 🛡️ Sentinel")
    st.caption("Real-time fraud operations")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Live monitor",
            "Alert queue",
            "Investigations",
            "Model performance",
            "Fraud network",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    model = health.get("model") or {}
    st.markdown(f"**Model** · `{model.get('model_name', 'n/a')}`")
    st.caption(
        f"PR-AUC {model.get('pr_auc') or 0:.3f} · ROC-AUC {model.get('roc_auc') or 0:.3f}"
    )
    st.caption(
        f"Thresholds — review {health['thresholds']['review']:.2f} · "
        f"block {health['thresholds']['block']:.2f}"
    )
    st.divider()
    st.caption(f"API {API_URL}")
    st.caption(f"DB {health.get('database')} · {health.get('transactions', 0):,} txns")


st.markdown(
    '<div class="sentinel-title">Fraud operations console</div>'
    f'<div class="sentinel-sub">{page} · live view of transaction risk, alerts and networks</div>',
    unsafe_allow_html=True,
)
st.write("")


# -------------------------------------------------------------------- overview
def page_overview() -> None:
    stats, error = api_get("/stats/overview")
    if error or stats is None:
        st.warning(f"Overview unavailable: {error}")
        return
    series, _ = api_get("/stats/timeseries", {"hours": 48})
    categories, _ = api_get("/stats/breakdown", {"field": "merchant_category", "hours": 24 * 30})
    channels, _ = api_get("/stats/breakdown", {"field": "channel", "hours": 24 * 30})
    recent, _ = api_get("/transactions", {"limit": 12})

    columns = st.columns(5)
    with columns[0]:
        kpi("Transactions · 24h", f"{stats['transactions_24h']:,}", f"{stats['total_transactions']:,} total")
    with columns[1]:
        kpi("Flagged · 24h", f"{stats['flagged_transactions']:,}", f"{pct(stats['flagged_rate'])} of all volume")
    with columns[2]:
        kpi("Value at risk", money(stats["flagged_amount"]), "above alert threshold")
    with columns[3]:
        kpi("Open alerts", f"{stats['alerts_open']:,}", f"{stats['alerts_critical']} critical")
    with columns[4]:
        kpi("Avg risk score", f"{stats['avg_score']:.2f}", f"{stats['customers']:,} customers monitored")

    left, right = st.columns([2.1, 1])
    with left:
        st.subheader("Transaction flow vs. flagged volume")
        if series:
            frame = pd.DataFrame(series)
            frame["time"] = pd.to_datetime(frame["bucket"])
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=frame["time"], y=frame["count"], name="Transactions",
                    mode="lines", line=dict(color="#38bdf8", width=2.4),
                    fill="tozeroy", fillcolor="rgba(56,189,248,0.14)",
                )
            )
            figure.add_trace(
                go.Bar(
                    x=frame["time"], y=frame["flagged"], name="Flagged",
                    marker_color="rgba(239,68,68,0.75)",
                )
            )
            figure.update_layout(barmode="overlay", hovermode="x unified")
            st.plotly_chart(chart_layout(figure, 330), use_container_width=True)
        else:
            st.info("No transactions in the last 48 hours yet.")
    with right:
        st.subheader("Channel mix")
        if channels:
            frame = pd.DataFrame(channels)
            figure = go.Figure(
                go.Pie(
                    labels=frame["label"].str.upper(),
                    values=frame["count"],
                    hole=0.62,
                    marker=dict(colors=["#38bdf8", "#818cf8", "#f472b6", "#facc15", "#34d399"]),
                )
            )
            st.plotly_chart(chart_layout(figure, 330), use_container_width=True)
        else:
            st.info("No channel data.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Risk by merchant category")
        if categories:
            frame = pd.DataFrame(categories).sort_values("flagged")
            figure = go.Figure(
                go.Bar(
                    x=frame["flagged"], y=frame["label"].str.replace("_", " ").str.title(),
                    orientation="h", marker=dict(color=frame["flagged"], colorscale="OrRd"),
                    text=frame["flagged"], textposition="outside",
                )
            )
            st.plotly_chart(chart_layout(figure, 360), use_container_width=True)
    with right:
        st.subheader("Recent activity")
        if recent and recent.get("transactions"):
            frame = pd.DataFrame(recent["transactions"])[
                ["timestamp", "customer_id", "amount", "risk_level", "decision", "city"]
            ]
            frame["timestamp"] = frame["timestamp"].astype(str).str[:16].str.replace("T", " ")
            frame["amount"] = frame["amount"].apply(money)
            st.dataframe(frame, use_container_width=True, hide_index=True, height=360)
        else:
            st.info("No scored transactions yet.")


# ---------------------------------------------------------------- live monitor
def _explanation_chart(result: dict) -> None:
    items = [item for item in result.get("explanation", []) if item.get("feature")]
    if not items:
        st.info("No feature attribution available for this transaction.")
        return
    frame = pd.DataFrame(items).sort_values("contribution")
    frame["label"] = frame["label"].str.wrap(28).str.replace("\n", "<br>")
    colors = ["#ef4444" if value > 0 else "#22c55e" for value in frame["contribution"]]
    figure = go.Figure(
        go.Bar(
            x=frame["contribution"], y=frame["label"], orientation="h",
            marker_color=colors, text=frame["contribution"].round(3),
            textposition="outside", hovertemplate="%{y}<br>impact %{x:.3f}<extra></extra>",
        )
    )
    st.plotly_chart(
        chart_layout(figure, 340, "SHAP contribution to fraud probability"),
        use_container_width=True,
    )


def _gauge(score: float, level: str) -> None:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%", "font": {"size": 40, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": RISK_COLORS.get(level, "#38bdf8"), "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(34,197,94,0.18)"},
                    {"range": [35, 75], "color": "rgba(250,204,21,0.16)"},
                    {"range": [75, 100], "color": "rgba(239,68,68,0.18)"},
                ],
            },
        )
    )
    st.plotly_chart(chart_layout(figure, 220), use_container_width=True)


def page_live() -> None:
    options, error = api_get("/meta/options")
    if error or not options:
        st.warning(f"Form options unavailable: {error}")
        return

    customers = options["customers"]
    merchants = options["merchants"]
    cities = options["cities"]
    scenario = options.get("demo_scenario") or {}

    if "txn_defaults" not in st.session_state:
        st.session_state.txn_defaults = {
            "amount": 200000.0,
            "city": "Moscow",
            "device": "DEV-NEVER-SEEN-01",
            "customer": customers[0]["customer_id"] if customers else "C00001",
            "merchant": merchants[0]["merchant_id"] if merchants else "M0001",
            "channel": "ecom",
        }

    left, right = st.columns([1.05, 1.6])
    with left:
        st.subheader("Score a transaction")
        if st.button("⚡ Load the ₹2,00,000 / 3 AM / unfamiliar location scenario", use_container_width=True):
            st.session_state.txn_defaults = {
                "amount": float(scenario.get("amount", 200000)),
                "city": scenario.get("city", "Moscow"),
                "device": scenario.get("device_id", "DEV-NEVER-SEEN-01"),
                "customer": scenario.get("customer_id", st.session_state.txn_defaults["customer"]),
                "merchant": scenario.get("merchant_id", st.session_state.txn_defaults["merchant"]),
                "channel": scenario.get("channel", "ecom"),
            }
            st.session_state.pop("score_result", None)
            st.rerun()

        defaults = st.session_state.txn_defaults
        customer_ids = [c["customer_id"] for c in customers]
        customer = st.selectbox(
            "Customer",
            customer_ids,
            index=customer_ids.index(defaults["customer"]) if defaults["customer"] in customer_ids else 0,
            format_func=lambda cid: next(
                (f"{cid} · {c['name']}" for c in customers if c["customer_id"] == cid), cid
            ),
        )
        amount = st.number_input("Amount (₹)", min_value=1.0, value=float(defaults["amount"]), step=500.0)
        city_names = [c["name"] for c in cities]
        city = st.selectbox(
            "Transaction city",
            city_names,
            index=city_names.index(defaults["city"]) if defaults["city"] in city_names else 0,
        )
        merchant_ids = [m["merchant_id"] for m in merchants]
        merchant = st.selectbox(
            "Merchant",
            merchant_ids,
            index=merchant_ids.index(defaults["merchant"]) if defaults["merchant"] in merchant_ids else 0,
            format_func=lambda mid: next(
                (f"{mid} · {m['name']}" for m in merchants if m["merchant_id"] == mid), mid
            ),
        )
        channel = st.selectbox(
            "Channel",
            options["channels"],
            index=options["channels"].index(defaults["channel"]) if defaults["channel"] in options["channels"] else 0,
        )
        when_date = st.date_input("Date", value=date.today())
        when_time = st.time_input("Time (UTC)", value=time(3, 12), step=60)
        device = st.text_input("Device id", value=defaults["device"])
        persist = st.checkbox("Store transaction and raise alerts", value=True)

        if st.button("Score transaction", type="primary", use_container_width=True):
            city_meta = next((c for c in cities if c["name"] == city), {})
            payload = {
                "customer_id": customer,
                "amount": amount,
                "merchant_id": merchant,
                "channel": channel,
                "city": city,
                "country": city_meta.get("country"),
                "latitude": city_meta.get("lat"),
                "longitude": city_meta.get("lon"),
                "device_id": device or None,
                "timestamp": datetime.combine(when_date, when_time)
                .replace(tzinfo=timezone.utc)
                .isoformat(timespec="seconds"),
            }
            endpoint = "/transactions" if persist else "/transactions/score"
            result, failure = api_post(endpoint, payload)
            if failure:
                st.error(f"Scoring failed: {failure}")
            else:
                st.session_state.score_result = result

    with right:
        result = st.session_state.get("score_result")
        if not result:
            st.info(
                "Submit a transaction to see the fraud probability, decision and the reasons "
                "behind it. The preset scenario mirrors a classic account-takeover pattern."
            )
        else:
            st.subheader("Decision")
            head_left, head_right = st.columns([1, 1])
            with head_left:
                _gauge(result["combined_score"], result["risk_level"])
            with head_right:
                st.markdown(
                    f"{badge(result['risk_level'], result['decision'])} "
                    f"<span style='color:#7d8ea6;font-size:0.8rem'>"
                    f"{money(result['amount'])} · {result['channel'].upper()} · "
                    f"{result['city']}, {result['country']}</span>",
                    unsafe_allow_html=True,
                )
                st.metric("Fraud probability", pct(result["fraud_probability"]))
                st.metric("Anomaly score (Isolation Forest)", f"{result['anomaly_score']:.2f}")
                st.caption(
                    f"{result['model_name']} · explanation via {result['explanation_method']} · "
                    f"{result['latency_ms']:.0f} ms"
                )
                if result.get("alert_id"):
                    st.warning(f"Alert created: {result['alert_id']}")

            st.markdown(
                f"<div class='reason-box'><b>Why this was flagged</b><br>"
                + "<br>".join(f"• {reason}" for reason in result.get("reasons") or [result["summary"]])
                + "</div>",
                unsafe_allow_html=True,
            )
            if result.get("graph_flags"):
                st.markdown(
                    " ".join(badge("high", flag) for flag in result["graph_flags"]),
                    unsafe_allow_html=True,
                )
            if result.get("policy_flags") and result.get("policy_floor"):
                st.caption(
                    f"Policy floor {pct(result['policy_floor'])} — "
                    + "; ".join(result["policy_flags"])
                )

            _explanation_chart(result)
            with st.expander("Full detail"):
                st.json(
                    {
                        "fraud_probability": result["fraud_probability"],
                        "anomaly_score": result["anomaly_score"],
                        "combined_score": result["combined_score"],
                        "policy_floor": result.get("policy_floor"),
                        "base_value": result["base_value"],
                        "anomaly_note": result["anomaly_note"],
                        "transaction_id": result["transaction_id"],
                    }
                )

    st.divider()
    st.subheader("Live transaction feed")
    if hasattr(st, "fragment"):
        @st.fragment(run_every=10)
        def feed() -> None:
            recent, failure = api_get("/transactions", {"limit": 25})
            if failure or not recent:
                st.info("Waiting for transactions…")
                return
            frame = pd.DataFrame(recent["transactions"])
            frame["score"] = (frame["combined_score"].astype(float) * 100).round(1)
            view = frame[
                ["timestamp", "transaction_id", "customer_id", "amount", "score", "risk_level", "decision", "city"]
            ].copy()
            view["timestamp"] = view["timestamp"].astype(str).str[11:19]
            view["amount"] = view["amount"].apply(money)
            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                height=330,
                column_config={"score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100)},
            )

        feed()
    else:  # pragma: no cover - older streamlit
        st.dataframe(pd.DataFrame(), use_container_width=True)


# ------------------------------------------------------------------ alert queue
def page_alerts() -> None:
    alerts, error = api_get("/alerts", {"limit": 200})
    if error:
        st.error(f"Alerts unavailable: {error}")
        return
    if not alerts:
        st.info("No alerts yet. Score a suspicious transaction from the Live monitor page.")
        return

    frame = pd.DataFrame(alerts)
    frame["score_pct"] = (frame["score"].astype(float) * 100).round(1)
    frame["created_at"] = frame["created_at"].astype(str).str[:19].str.replace("T", " ")

    counts = st.columns(4)
    with counts[0]:
        kpi("Open", str(int((frame["status"] == "open").sum())), "awaiting analyst")
    with counts[1]:
        kpi("Critical", str(int((frame["severity"] == "critical").sum())), "score ≥ 90%")
    with counts[2]:
        kpi("Investigating", str(int((frame["status"] == "investigating").sum())), "in review")
    with counts[3]:
        kpi("Closed", str(int((frame["status"] == "closed").sum())), "resolved")

    status_filter = st.radio(
        "Status", ["all", "open", "acknowledged", "investigating", "escalated", "closed"], horizontal=True
    )
    view = frame if status_filter == "all" else frame[frame["status"] == status_filter]
    st.dataframe(
        view[["alert_id", "created_at", "customer_id", "severity", "score_pct", "status", "title"]],
        use_container_width=True,
        hide_index=True,
        height=310,
        column_config={"score_pct": st.column_config.ProgressColumn("Score", min_value=0, max_value=100)},
    )

    st.subheader("Work an alert")
    alert_ids = list(view["alert_id"])
    if not alert_ids:
        st.info("No alerts match this filter.")
        return
    selected = st.selectbox("Alert", alert_ids)
    record = next((alert for alert in alerts if alert["alert_id"] == selected), None)
    if record:
        left, right = st.columns([1.5, 1])
        with left:
            st.markdown(f"{badge(record.get('severity', 'low'))} **{record.get('title')}**")
            reasons = record.get("summary")
            if isinstance(reasons, list):
                for reason in reasons:
                    st.markdown(f"- {reason}")
            elif reasons:
                st.write(reasons)
            st.caption(f"Transaction {record.get('transaction_id')} · customer {record.get('customer_id')}")
        with right:
            note = st.text_area("Analyst note", value=record.get("analyst_note") or "", height=90)
            columns = st.columns(3)
            for column, status in zip(columns, ["acknowledged", "investigating", "closed"]):
                with column:
                    if st.button(status.title(), use_container_width=True, key=f"{selected}-{status}"):
                        _, failure = api_patch(
                            f"/alerts/{selected}", {"status": status, "analyst_note": note or None}
                        )
                        if failure:
                            st.error(failure)
                        else:
                            st.success(f"Alert marked {status}")
                            st.rerun()


# --------------------------------------------------------------- investigations
def page_investigation() -> None:
    customers, error = api_get("/customers", {"limit": 500})
    if error or not customers:
        st.error(f"Customers unavailable: {error}")
        return
    ranked, _ = api_get("/stats/top-customers", {"limit": 25})
    ranked_ids = [row["customer_id"] for row in ranked or []]
    ordered = sorted(
        customers["customers"], key=lambda c: (c["customer_id"] not in ranked_ids, c["customer_id"])
    )

    selected = st.selectbox(
        "Customer",
        [c["customer_id"] for c in ordered],
        format_func=lambda cid: next(
            (f"{cid} · {c['name']} · {c.get('home_city')}" for c in ordered if c["customer_id"] == cid), cid
        ),
    )
    report, error = api_get(f"/customers/{selected}/investigation")
    if error or not report:
        st.error(f"Could not build report: {error}")
        return

    metrics = report["metrics"]
    columns = st.columns(4)
    with columns[0]:
        kpi("Risk grade", report["risk_grade"], f"top score {pct(metrics['top_score'])}")
    with columns[1]:
        kpi("Flagged transactions", f"{metrics['flagged_transactions']}", f"of {metrics['transactions_reviewed']} reviewed")
    with columns[2]:
        kpi("Reviewed volume", money(metrics["volume"]), "last 40 transactions")
    with columns[3]:
        ring = report.get("ring")
        kpi("Fraud ring", ring["ring_id"] if ring else "None", f"{ring['size']} accounts" if ring else "no shared entities")

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Case narrative")
        st.markdown(report["markdown"])
        st.download_button(
            "Download report (Markdown)",
            data=report["markdown"],
            file_name=f"investigation-{selected}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("Save report to case file", use_container_width=True):
            saved, failure = api_post(f"/customers/{selected}/investigation?save=true", {})
            if failure:
                st.error(failure)
            else:
                st.success(f"Saved as {saved.get('investigation_id')}")

    with right:
        st.subheader("Linked entities")
        network = report.get("network") or {}
        nodes = pd.DataFrame(network.get("nodes") or [])
        if not nodes.empty:
            view = nodes[nodes["kind"] != "customer"][["kind", "label", "risk", "transactions", "amount"]].copy()
            view["amount"] = view["amount"].apply(money)
            st.dataframe(view, use_container_width=True, hide_index=True, height=280)
        else:
            st.info("No linked entities in the graph window.")
        st.subheader("Recent alerts")
        alerts = report.get("alerts") or []
        if alerts:
            alert_frame = pd.DataFrame(alerts)[["alert_id", "severity", "status", "score", "created_at"]]
            alert_frame["score"] = (alert_frame["score"].astype(float) * 100).round(1)
            alert_frame["created_at"] = alert_frame["created_at"].astype(str).str[:19]
            st.dataframe(alert_frame, use_container_width=True, hide_index=True, height=220)
        else:
            st.info("No alerts for this customer.")


# ------------------------------------------------------------- model performance
def page_models() -> None:
    info, error = api_get("/models/info")
    if error or not info:
        st.error(f"Model info unavailable: {error}")
        return
    metrics = info.get("metrics") or {}
    models = metrics.get("models") or {}

    columns = st.columns(4)
    selected = models.get(info["model"]["model_name"], {})
    with columns[0]:
        kpi("Selected model", info["model"]["model_name"].replace("_", " ").title(), metrics.get("dataset", {}).get("split", ""))
    with columns[1]:
        kpi("PR-AUC", f"{selected.get('pr_auc', 0):.3f}", "fraud-focused quality")
    with columns[2]:
        kpi("ROC-AUC", f"{selected.get('roc_auc', 0):.3f}", f"{metrics.get('dataset', {}).get('rows', 0):,} rows")
    with columns[3]:
        kpi("Anomaly detector", "Isolation Forest", f"PR-AUC {metrics.get('anomaly', {}).get('pr_auc', 0):.3f}")

    if models:
        st.subheader("Candidate model comparison")
        table = pd.DataFrame(
            [
                {
                    "Model": name.replace("_", " ").title(),
                    "PR-AUC": round(result.get("pr_auc", 0), 3),
                    "ROC-AUC": round(result.get("roc_auc", 0), 3),
                    "Precision": round(result.get("precision", 0), 3),
                    "Recall": round(result.get("recall", 0), 3),
                    "F1": round(result.get("f1", 0), 3),
                    "Fit (s)": result.get("fit_seconds"),
                }
                for name, result in models.items()
            ]
        ).sort_values("PR-AUC", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Precision-recall curve")
        figure = go.Figure()
        for name, result in models.items():
            curve = (result.get("curves") or {}).get("pr") or []
            if curve:
                frame = pd.DataFrame(curve, columns=["recall", "precision"])
                figure.add_trace(
                    go.Scatter(x=frame["recall"], y=frame["precision"], mode="lines", name=name.replace("_", " "))
                )
        st.plotly_chart(chart_layout(figure, 330), use_container_width=True)
    with right:
        st.subheader("ROC curve")
        figure = go.Figure()
        for name, result in models.items():
            curve = (result.get("curves") or {}).get("roc") or []
            if curve:
                frame = pd.DataFrame(curve, columns=["fpr", "tpr"])
                figure.add_trace(
                    go.Scatter(x=frame["fpr"], y=frame["tpr"], mode="lines", name=name.replace("_", " "))
                )
        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#475569"), name="Baseline")
        )
        st.plotly_chart(chart_layout(figure, 330), use_container_width=True)

    st.subheader("Feature importance")
    importance = pd.DataFrame(metrics.get("feature_importance") or [])
    if not importance.empty:
        importance = importance.head(14).sort_values("importance")
        labels = {f["name"]: f["label"] for f in info.get("features", [])}
        importance["label"] = importance["feature"].map(labels).fillna(importance["feature"])
        figure = go.Figure(
            go.Bar(
                x=importance["importance"], y=importance["label"], orientation="h",
                marker_color="#38bdf8",
            )
        )
        st.plotly_chart(chart_layout(figure, 380), use_container_width=True)

    with st.expander("Training details"):
        st.json(
            {
                "trained_at": metrics.get("trained_at"),
                "dataset": metrics.get("dataset"),
                "thresholds": metrics.get("thresholds"),
                "anomaly": {
                    "pr_auc": metrics.get("anomaly", {}).get("pr_auc"),
                    "roc_auc": metrics.get("anomaly", {}).get("roc_auc"),
                    "contamination": metrics.get("anomaly", {}).get("contamination"),
                },
            }
        )
    if st.button("Retrain models", use_container_width=True):
        with st.spinner("Retraining Random Forest, XGBoost and Isolation Forest…"):
            _, failure = api_post("/models/retrain", {})
        if failure:
            st.error(failure)
        else:
            st.success("Models retrained and reloaded")
            st.cache_data.clear()
            st.rerun()


# ------------------------------------------------------------------ graph page
def page_graph() -> None:
    payload, error = api_get("/graph/rings")
    if error or payload is None:
        st.error(f"Graph analytics unavailable: {error}")
        return
    rings = payload.get("rings") or []
    graph_meta = payload.get("graph") or {}

    columns = st.columns(4)
    with columns[0]:
        kpi("Rings detected", f"{len(rings)}", payload.get("method", ""))
    with columns[1]:
        kpi("Graph nodes", f"{graph_meta.get('nodes', 0):,}", "customers, devices, IPs, merchants")
    with columns[2]:
        kpi("Graph edges", f"{graph_meta.get('edges', 0):,}", "shared-entity links")
    with columns[3]:
        total = sum(ring["amount"] for ring in rings)
        kpi("Ring exposure", money(total), "aggregate value")

    if not rings:
        st.info("No rings detected in the current graph window.")
    else:
        st.subheader("Detected fraud rings")
        table = pd.DataFrame(
            [
                {
                    "Ring": ring["ring_id"],
                    "Accounts": ring["size"],
                    "Risk": ring["risk_score"],
                    "Flagged": len(ring["flagged_members"]),
                    "Transactions": ring["transactions"],
                    "Exposure": ring["amount"],
                    "Shared devices": ", ".join(ring.get("shared_devices", [])[:2]) or "—",
                }
                for ring in rings
            ]
        )
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
                "Exposure": st.column_config.NumberColumn("Exposure", format="₹%d"),
            },
        )

        ring_ids = [ring["ring_id"] for ring in rings]
        selected = st.selectbox("Inspect ring", ring_ids)
        ring = next(r for r in rings if r["ring_id"] == selected)
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"#### {ring['ring_id']} · risk {ring['risk_score']:.0%}")
            for evidence in ring.get("evidence", []):
                st.markdown(f"- {evidence}")
            st.markdown("**Members**")
            st.dataframe(
                pd.DataFrame({"customer_id": ring["members"], "flagged": [m in ring["flagged_members"] for m in ring["members"]]}),
                use_container_width=True,
                hide_index=True,
                height=230,
            )
        with right:
            st.markdown("**Shared identifiers**")
            st.code(
                "\n".join(
                    [f"device: {d}" for d in ring.get("shared_devices", [])]
                    + [f"ip: {i}" for i in ring.get("shared_ips", [])]
                    + [f"merchant: {m}" for m in ring.get("shared_merchants", [])]
                )
                or "none"
            )
            st.caption("Rings are found by projecting customers onto shared devices/IPs/merchants, then running Louvain communities and risk propagation.")

        st.subheader("Network view")
        network_customer = st.selectbox("Customer neighbourhood", ring["members"])
        network, failure = api_get(f"/graph/customer/{network_customer}", {"radius": 1})
        if failure or not network:
            st.info(f"No network available: {failure}")
            return
        nodes = network.get("nodes") or []
        edges = network.get("edges") or []
        if not nodes:
            st.info("No nodes in this neighbourhood.")
            return
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node["id"], **node)
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"])
        positions = nx.spring_layout(graph, seed=11, k=0.9 / max(len(nodes) ** 0.5, 1))
        edge_x, edge_y = [], []
        for source, target in graph.edges():
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="rgba(148,163,184,0.3)", width=1), hoverinfo="skip")
        )
        for kind, color in KIND_COLORS.items():
            subset = [node for node in nodes if node["kind"] == kind]
            if not subset:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[positions[n["id"]][0] for n in subset],
                    y=[positions[n["id"]][1] for n in subset],
                    mode="markers+text",
                    text=[n["label"] for n in subset],
                    textposition="top center",
                    textfont=dict(size=9, color="#94a3b8"),
                    name=kind,
                    marker=dict(
                        size=[10 + n["risk"] * 26 for n in subset],
                        color=color,
                        line=dict(color="#0b1220", width=1.5),
                    ),
                    customdata=[[n["risk"], n["transactions"], n["amount"]] for n in subset],
                    hovertemplate="%{text}<br>risk %{customdata[0]:.2f}<br>txns %{customdata[1]}<br>value ₹%{customdata[2]:,.0f}<extra></extra>",
                )
            )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        st.plotly_chart(chart_layout(figure, 520), use_container_width=True)


if page == "Overview":
    page_overview()
elif page == "Live monitor":
    page_live()
elif page == "Alert queue":
    page_alerts()
elif page == "Investigations":
    page_investigation()
elif page == "Model performance":
    page_models()
elif page == "Fraud network":
    page_graph()
