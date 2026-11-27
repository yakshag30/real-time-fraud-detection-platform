import { useEffect, useRef, useState } from "react";
import { api, inr, pct } from "../api";
import {
  Badge,
  Contributions,
  ErrorBox,
  Gauge,
  PageHead,
  Panel,
  useApi,
} from "../components/ui";
import type { Options, ScoreResult } from "../types";

const blank = {
  customer_id: "",
  amount: 200000,
  merchant_id: "",
  channel: "ecom",
  city: "Moscow",
  device_id: "DEV-NEVER-SEEN-01",
  hour: 3,
  date: new Date().toISOString().slice(0, 10),
};

export default function LiveMonitor() {
  const options = useApi<Options>("/meta/options");
  const [form, setForm] = useState(blank);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const primed = useRef(false);

  useEffect(() => {
    const demo = options.data?.demo_scenario;
    if (!demo || primed.current) return;
    primed.current = true;
    setForm((current) => ({
      ...current,
      customer_id: demo.customer_id,
      merchant_id: demo.merchant_id,
      city: demo.city,
    }));
  }, [options.data]);

  const city = options.data?.cities.find((item) => item.name === form.city);

  const submit = async (persist: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const payload = {
        customer_id: form.customer_id,
        amount: Number(form.amount),
        merchant_id: form.merchant_id,
        channel: form.channel,
        city: form.city,
        country: city?.country,
        latitude: city?.lat,
        longitude: city?.lon,
        device_id: form.device_id || undefined,
        timestamp: `${form.date}T${String(form.hour).padStart(2, "0")}:12:00Z`,
      };
      const response = await api.post<ScoreResult>(
        persist ? "/transactions" : "/transactions/score",
        payload,
      );
      setResult(response);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  };

  const resetToTakeover = () => {
    const demo = options.data?.demo_scenario;
    setForm({
      ...blank,
      customer_id: demo?.customer_id ?? "",
      merchant_id: demo?.merchant_id ?? "",
      city: demo?.city ?? "Moscow",
      amount: demo?.amount ?? 200000,
    });
    setResult(null);
  };

  return (
    <div className="stack" style={{ gap: 18 }}>
      <PageHead
        title="Live monitor"
        sub="Score a transaction, inspect the attribution, decide with the policy layer."
        right={
          <button className="btn ghost small" onClick={resetToTakeover}>
            Load takeover scenario
          </button>
        }
      />

      <div className="split">
        <Panel title="Transaction" note="POST /transactions/score">
          <div className="grid cols-2" style={{ gap: 12 }}>
            <div className="field">
              <label>Customer</label>
              <select
                value={form.customer_id}
                onChange={(event) => setForm({ ...form, customer_id: event.target.value })}
              >
                {(options.data?.customers ?? []).map((customer) => (
                  <option key={customer.customer_id} value={customer.customer_id}>
                    {customer.customer_id} · {customer.name} ({customer.home_city})
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Amount (INR)</label>
              <input
                type="number"
                value={form.amount}
                onChange={(event) => setForm({ ...form, amount: Number(event.target.value) })}
              />
            </div>
            <div className="field">
              <label>Merchant</label>
              <select
                value={form.merchant_id}
                onChange={(event) => setForm({ ...form, merchant_id: event.target.value })}
              >
                {(options.data?.merchants ?? []).map((merchant) => (
                  <option key={merchant.merchant_id} value={merchant.merchant_id}>
                    {merchant.merchant_id} · {merchant.category}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Channel</label>
              <select
                value={form.channel}
                onChange={(event) => setForm({ ...form, channel: event.target.value })}
              >
                {(options.data?.channels ?? ["ecom", "pos", "upi", "atm"]).map((channel) => (
                  <option key={channel} value={channel}>
                    {channel.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>City</label>
              <select
                value={form.city}
                onChange={(event) => setForm({ ...form, city: event.target.value })}
              >
                {(options.data?.cities ?? []).map((item) => (
                  <option key={`${item.name}-${item.country}`} value={item.name}>
                    {item.name}, {item.country}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Device</label>
              <input
                value={form.device_id}
                onChange={(event) => setForm({ ...form, device_id: event.target.value })}
              />
            </div>
            <div className="field">
              <label>Date (UTC)</label>
              <input
                type="date"
                value={form.date}
                onChange={(event) => setForm({ ...form, date: event.target.value })}
              />
            </div>
            <div className="field">
              <label>Hour (UTC)</label>
              <input
                type="number"
                min={0}
                max={23}
                value={form.hour}
                onChange={(event) => setForm({ ...form, hour: Number(event.target.value) })}
              />
            </div>
          </div>

          <div className="row" style={{ marginTop: 16 }}>
            <button className="btn" disabled={busy || !form.customer_id} onClick={() => submit(true)}>
              {busy ? "Scoring…" : "Score & store"}
            </button>
            <button
              className="btn ghost"
              disabled={busy || !form.customer_id}
              onClick={() => submit(false)}
            >
              What-if only
            </button>
            <span className="spacer" />
            <span className="dim">
              {form.city}, {city?.country ?? "—"} · {form.hour}:12 UTC
            </span>
          </div>
          <ErrorBox message={error} />
        </Panel>

        <Panel title="Decision">
          {!result ? (
            <div className="empty">
              Submit a transaction to see the model probability, anomaly score, policy floor and
              the reasons behind them.
            </div>
          ) : (
            <div className="stack">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <Gauge score={result.combined_score} />
                <div className="stack" style={{ gap: 6, alignItems: "flex-end" }}>
                  <Badge tone={result.risk_level}>{result.risk_level} risk</Badge>
                  <Badge tone={result.decision}>{result.decision}</Badge>
                  <span className="dim">
                    {inr(result.amount)} · {result.channel?.toUpperCase()}
                  </span>
                  <span className="dim">
                    {result.city}, {result.country}
                  </span>
                </div>
              </div>

              <div className="grid cols-3" style={{ gap: 8 }}>
                <div>
                  <div className="kpi-label">Model</div>
                  <div className="mono">{pct(result.fraud_probability, 1)}</div>
                </div>
                <div>
                  <div className="kpi-label">Anomaly</div>
                  <div className="mono">{result.anomaly_score.toFixed(2)}</div>
                </div>
                <div>
                  <div className="kpi-label">Latency</div>
                  <div className="mono">{result.latency_ms.toFixed(0)} ms</div>
                </div>
              </div>

              {result.policy_flags.length > 0 && (
                <div>
                  <div className="kpi-label" style={{ marginBottom: 6 }}>
                    Policy floor {pct(result.policy_floor)}
                  </div>
                  <div className="row" style={{ gap: 6 }}>
                    {result.policy_flags.map((flag) => (
                      <Badge key={flag} tone="info">
                        {flag}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <div className="kpi-label" style={{ marginBottom: 6 }}>
                  Why this was flagged
                </div>
                <ul className="reasons">
                  {result.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>

              {result.alert_id && (
                <div className="error" style={{ borderColor: "rgba(154,100,0,.35)", background: "#fbf0d8", color: "#6f4b00" }}>
                  Alert created: <span className="mono">{result.alert_id}</span>
                </div>
              )}
            </div>
          )}
        </Panel>
      </div>

      {result && (
        <Panel
          title="Feature attribution"
          note={`${result.explanation_method} · base ${result.base_value.toFixed(2)}`}
        >
          <Contributions items={result.explanation} />
          <details style={{ marginTop: 12 }}>
            <summary className="dim" style={{ cursor: "pointer" }}>
              Full response
            </summary>
            <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </Panel>
      )}
    </div>
  );
}
