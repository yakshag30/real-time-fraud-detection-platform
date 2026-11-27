# Real-Time Fraud Detection Platform

An end-to-end fraud decisioning stack for retail banking: it scores a card
transaction in **130–210 ms**, explains *why* it looks fraudulent, raises an
alert when policy demands a human, and maps the fraud rings hiding in the
payment graph.

- **Scoring:** XGBoost classifier + Isolation Forest anomaly detector + a
  transparent policy layer, all three visible in the decision.
- **Explainability:** SHAP tree attributions on the probability scale, turned
  into plain-English analyst reasons ("196.4× the customer's usual spend").
- **Real-time:** FastAPI service with single, batch and streaming-style ingest.
- **Web console:** a six-page React + Vite app served by the API at `/app` —
  overview, live monitor, alert queue, investigations, model performance and
  fraud network. It is a normal single-page site, not Streamlit.
- **Streamlit console:** kept as a lightweight alternative (`cli dashboard`).
- **Graph analytics:** NetworkX entity graph over customers, devices, IPs and
  merchants, with risk propagation and Louvain community detection for rings.

Everything runs locally with SQLite out of the box; PostgreSQL is available
through Docker Compose.

---

## Demo: ₹2,00,000 at 3 AM from an unfamiliar location

```bash
python -m fraud_platform.cli seed --rows 2500 --reset
python -m fraud_platform.cli score --amount 200000 --city Moscow --hour 3
```

Measured output on the reference machine (abridged):

```json
{
  "decision": "block",
  "risk_level": "high",
  "fraud_probability": 0.190609,
  "anomaly_score": 0.9892,
  "combined_score": 0.87,
  "policy_floor": 0.87,
  "summary": "High risk (87%) — 196.4× the customer's usual spend (₹2,00,000 against an average of ₹1,019); cross-border transaction from RU; first payment from a device never seen on this account.",
  "reasons": [
    "196.4× the customer's usual spend (₹2,00,000 against an average of ₹1,019)",
    "cross-border transaction from RU",
    "first payment from a device never seen on this account",
    "4,626 km from the home city (Moscow)"
  ],
  "alert_id": "AL-2abe47b6b65f",
  "latency_ms": 209.19
}
```

Three independent layers had to agree, and the response shows each one:

| Layer | Result | Meaning |
| --- | --- | --- |
| Supervised model | `0.19` | The combination is *shaped* like genuine travel plus a night-owl purchase, so the classifier alone is unsure. |
| Isolation Forest | `0.99` | The vector is far outside anything this customer normally does. |
| Policy layer | `0.87` floor | Four textbook takeover indicators agree: 50×+ spend, cross-border, brand-new device, overnight hour. |

The policy floor is what turns "unsure model, extreme behaviour" into a block
instead of a silent allow — see [Risk policy](#risk-policy).

---

## Quickstart

Requires **Python 3.12**.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .

# one command: generate data -> train -> seed the demo database
python -m fraud_platform.cli setup --rows 2500

# build the web console once (uses Bun, https://bun.sh)
cd web && bun install && bun run build && cd ..

# API + web console -> http://127.0.0.1:8000/app/   (API docs at /docs)
python -m fraud_platform.cli serve

# optional: the Streamlit console instead -> http://127.0.0.1:8501
python -m fraud_platform.cli dashboard
```

The API bootstraps itself on first start: if `artifacts/model.joblib` is missing
it trains, and if the transactions table is empty it seeds and scores demo
traffic. Set `FRAUD_BOOTSTRAP=false` to disable that.

### CLI

| Command | Purpose |
| --- | --- |
| `generate-data` | Write `data/*.csv` (customers, merchants, labelled transactions). |
| `train` | Build features, train all candidates + Isolation Forest, save artifacts. |
| `seed` | Score historical transactions into the database so the UI has history. |
| `setup` | `generate-data` + `train` + `seed` in one go. |
| `score` | Score one transaction from the terminal (the demo above). |
| `serve` | Run the FastAPI service. |
| `dashboard` | Run the Streamlit console. |
| `info` | Print settings, dataset summary and model metadata. |

---

## Measured results

Chronological 80/20 split, 14,193 transactions over 45 days, 3.19% fraud.
Trained in 14.1 s on a laptop CPU; numbers come from `artifacts/metrics.json`.

| Model | PR-AUC | ROC-AUC | Precision | Recall | Brier |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.728 | 0.964 | 0.255 | 0.927 | 0.0739 |
| Random forest | 0.980 | 0.999 | 0.953 | 0.844 | 0.0055 |
| **XGBoost (selected)** | **0.990** | **1.000** | **0.920** | **0.958** | **0.0029** |

- Isolation Forest (unsupervised, PR-AUC 0.210, ROC-AUC 0.854) is deliberately
  the weakest single signal, weighted 25% into the combined score — it catches
  novel behaviour the supervised model has never been labelled on.
- Combined model + anomaly score: PR-AUC **0.985**.
- Precision/recall above are at a 0.5 cut-off. In production you would move the
  cut-off to match analyst capacity; `FRAUD_REVIEW_THRESHOLD` and
  `FRAUD_BLOCK_THRESHOLD` exist for exactly that.

Top features after training: `device_age_minutes` (17.2%), `is_new_merchant`
(16.1%), `device_customer_count` (11.6%), `amount_to_customer_avg` (8.5%),
`channel_risk` (8.5%), `category_risk` (5.9%).

> An earlier build scored PR-AUC 1.000 because fraud was only ever committed at
> risky merchants on brand-new devices. Adding benign look-alikes (households
> sharing a tablet, office terminals, night-shift customers, real foreign
> travel, big-ticket purchases, rent/EMI repeat payments) made the task honest.
> The same reasoning is why merchant category is no longer the dominant
> feature: card testing now hits ordinary merchants too.

---

## Risk policy

`src/fraud_platform/services/policy.py` holds a small, deterministic layer that
sits on top of the model and **only ever raises** the score.

It counts independent high-severity indicators — spend ≥ 10× the customer's
average, cross-border, brand-new device, overnight hour, impossible travel,
device shared across customers. With three or more, the transaction gets a
floor of `0.55 + 0.08 × indicators` (capped at 0.95); otherwise the model's
score stands unchanged.

This is the standard "rules around a model" pattern: a tree model cannot
extrapolate to a 196× spend, and a genuinely travelling customer looks like a
takeover one indicator at a time. The rule keeps the extreme cases in front of
a human, while the model keeps doing the ranking work.

---

## How scoring works

```
transaction
   │
   ├─ FeatureBuilder (27 features, identical code for training and inference)
   │     amount ratios/z-score · velocity 1h/24h/7d · geo distance & speed
   │     foreign flag · device age & sharing · merchant/category/channel risk
   │
   ├─ XGBoost ──────────► fraud_probability
   ├─ Isolation Forest ─► anomaly_score
   │        └─ combined = 0.75 × model + 0.25 × anomaly
   ├─ Policy layer ─────► max(combined, floor)
   │
   ├─ SHAP TreeExplainer ─► per-feature contributions (rescaled to probability)
   ├─ build_reasons() ────► ranked analyst sentences
   └─ persist + alert if score ≥ FRAUD_ALERT_MIN_SCORE
```

Features are always computed from the customer's *prior* history only, so the
training matrix and a live request follow exactly the same path — no leakage
and no train/serve skew.

---

## API

`http://127.0.0.1:8000/docs` has the interactive schema.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, database status, model metadata. |
| `POST` | `/transactions/score` | Score without persisting (what-if). |
| `POST` | `/transactions` | Score and persist, creating an alert when needed. |
| `POST` | `/transactions/batch` | Up to 500 transactions in one call. |
| `GET` | `/transactions` | Recent scored transactions (filters: customer, score, decision, hours). |
| `GET` | `/transactions/{id}` | One stored transaction with its explanation. |
| `GET` | `/alerts` · `/alerts/{id}` | Alert queue; `PATCH` to acknowledge/investigate/close. |
| `GET` | `/stats/overview` · `/stats/timeseries` · `/stats/breakdown` · `/stats/top-customers` | Dashboard aggregates. |
| `GET` | `/models/info` · `POST /models/retrain` | Model metadata and retraining. |
| `GET` | `/customers` · `/customers/{id}` · `/customers/{id}/transactions` | Customer views. |
| `GET` | `/customers/{id}/investigation` | Generated markdown case file for an analyst. |
| `GET` | `/reports` · `/reports/{id}/markdown` | Stored investigations. |
| `GET` | `/graph/rings` · `/graph/customer/{id}` | Detected rings and a customer's entity network. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/transactions \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"C00007","amount":200000,"merchant_id":"M0002",
       "channel":"ecom","city":"Moscow","country":"RU",
       "latitude":55.7558,"longitude":37.6173,
       "device_id":"DEV-LIVE-01","timestamp":"2026-11-27T03:12:00Z"}'
```

---

## Web console

The site lives in `web/` and talks to the API over HTTP. `python -m fraud_platform.cli serve`
mounts the built bundle at **http://127.0.0.1:8000/app/**; `/` redirects there when a build
is present. For front-end work run the Vite dev server instead:

```bash
cd web
bun run dev        # http://127.0.0.1:5173, hot reload
bun run build      # type-check + production bundle into web/dist
```

The dev server talks to `http://127.0.0.1:8000` by default; override with
`VITE_API_URL` if the API runs elsewhere.

| Page | What it shows |
| --- | --- |
| Overview | Volume, flagged rate, alert mix, hourly trend, category and channel breakdown. |
| Live monitor | The preset takeover scenario (editable), decision gauge, SHAP waterfall, live feed refreshing every 10 s. |
| Alert queue | Severity/status filters, analyst notes, status transitions. |
| Investigations | Per-customer case file with risk grade, timeline and markdown export. |
| Model performance | Model comparison table, ROC/PR curves, feature importance, confusion matrix. |
| Fraud network | Interactive entity graph, ring detection, risk propagation. |

---

## Data

`generate_dataset` produces a synthetic core-banking stream (no real customer
data) with four labelled fraud scenarios plus benign look-alikes:

| Scenario | Signature |
| --- | --- |
| `account_takeover` | Night-time spend, foreign geo, new device, 3–15× usual amount. |
| `card_testing` | Bursts of small e-commerce payments on one device/IP. |
| `fraud_ring` | Several customers sharing a device and IP across merchants. |
| `merchant_collusion` | Repeated identical amounts at one merchant. |

Reference: 500 customers, 140 merchants, 45 days, 14,193 transactions, 3.19%
fraud. `is_fraud` / `fraud_type` are labels for training and evaluation only and
are never model inputs.

### Storage

SQLite (`sqlite:///data/fraud.db`) by default; set `FRAUD_DATABASE_URL` to a
PostgreSQL DSN for production. One repository class handles both dialects, so
there is no ORM in the hot path. MongoDB is intentionally out of scope — the
repository interface (`src/fraud_platform/db/repository.py`) is the extension
point if you need it.

---

## Configuration

Copy `.env.example` to `.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FRAUD_DATABASE_URL` | `sqlite:///data/fraud.db` | SQLite or PostgreSQL DSN. |
| `FRAUD_REVIEW_THRESHOLD` | `0.35` | Score at which a transaction goes to review. |
| `FRAUD_BLOCK_THRESHOLD` | `0.75` | Score at which it is blocked. |
| `FRAUD_ALERT_MIN_SCORE` | `0.50` | Score at which an alert row is created. |
| `FRAUD_ANOMALY_WEIGHT` | `0.25` | Weight of the Isolation Forest in the combined score. |
| `FRAUD_BOOTSTRAP` | `true` | Train/seed automatically on first start. |
| `FRAUD_BOOTSTRAP_ROWS` | `1200` | Demo transactions to score at bootstrap. |
| `FRAUD_API_HOST` / `FRAUD_API_PORT` | `127.0.0.1` / `8000` | API bind address. |
| `API_URL` | `http://127.0.0.1:8000` | API base URL used by the dashboard. |
| `FRAUD_DATA_DIR` / `FRAUD_ARTIFACTS_DIR` | `data` / `artifacts` | Generated files. |

---

## Public demo

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yakshag30/real-time-fraud-detection-platform)

The repository ships a Render blueprint (`render.yaml`), so the whole stack —
FastAPI plus the built React console at `/app` — can be published as one HTTPS
service with no server of your own:

1. Sign in at <https://dashboard.render.com> with GitHub.
2. **New → Blueprint**, pick this repository; Render reads `render.yaml`.
3. **Apply**. The first image build takes a few minutes; afterwards the URL is
   `https://fraud-detection-platform-<suffix>.onrender.com/app/`.

Two things make the free plan viable here. The image **trains the model and seeds
the demo database during the build** (`python -m fraud_platform.cli setup`), so
start-up only loads them instead of spending minutes of 0.1 CPU on training and
SHAP scoring, and the console calls the API on the same origin it was served
from. A free Render service sleeps after 15 minutes without traffic and takes
about a minute to wake; the demo database lives in the container, so data created
while it runs is lost on the next restart. Set `FRAUD_BOOTSTRAP=true` if you would
rather re-seed on boot than ship the baked database.

The Streamlit console is not part of the blueprint — it is a second process
(`cli dashboard`), intended for local use or a separate service.

## Docker

```bash
docker compose up --build
```

Brings up PostgreSQL 16, the API and the Streamlit console (API on `:8000`,
Streamlit on `:8501`). The image builds the React console in a Bun stage, so
`http://localhost:8000/app/` serves the website as well. The API container seeds
2,000 transactions on first boot and keeps the model in the `artifacts` volume.

> The compose stack is provided but was **not** executed while building this
> repository — Docker is not installed on the development machine. Treat it as
> unverified until you run it once.

---

## Project layout

```
web/                   React + Vite console (built into web/dist, served at /app)
  src/App.tsx          shell, navigation, hash router
  src/api.ts           typed API client, INR/percent formatting
  src/components/      panels, KPIs, gauge, charts, markdown renderer
  src/pages/           overview, live monitor, alerts, investigations, models, network
src/fraud_platform/
  config.py            settings from environment
  cli.py               generate-data / train / seed / setup / score / serve / dashboard / info
  data/
    generator.py       synthetic stream + labelled fraud scenarios
    features.py        27 features, shared by training and inference
    reference.py       cities, merchant categories, risk tables
    dataset.py         CSV cache
  models/
    training.py        candidates, chronological split, Isolation Forest, artifacts
    registry.py        ModelBundle: predict / anomaly / combine
    explain.py         SHAP explainer, probability rescaling, reason generation
  db/repository.py     schema + queries (SQLite & PostgreSQL)
  graph/               entity graph, risk propagation, ring detection
  services/
    scoring.py         score -> explain -> persist -> alert
    policy.py          deterministic risk floor
    bootstrap.py       model/data bootstrap, demo seeding
    reports.py         investigation markdown
  api/                 FastAPI app + pydantic contracts
  dashboard/app.py     Streamlit console
```

---

## Limitations

- The dataset is synthetic. Metrics measure the pipeline, not real-world lift.
- Scores are not probability-calibrated in the statistical sense; treat them as
  risk ranking and use the thresholds to tune the operating point.
- Graph rings are detected from co-used devices and IPs — a fraud ring that
  rotates both will evade it.
- Single-node deployment: the device cache and graph cache live in process
  memory, so running several API replicas needs a shared cache.
