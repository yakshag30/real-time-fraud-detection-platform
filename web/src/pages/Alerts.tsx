import { useMemo, useState } from "react";
import { api, pct, shortTime } from "../api";
import { Badge, ErrorBox, PageHead, Panel, Table, useApi } from "../components/ui";
import type { Alert } from "../types";

const FILTERS = ["all", "open", "investigating", "acknowledged", "closed"] as const;

export default function Alerts() {
  const alerts = useApi<Alert[]>("/alerts?limit=300");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("open");
  const [selected, setSelected] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const rows = useMemo(() => {
    const list = alerts.data ?? [];
    return filter === "all" ? list : list.filter((alert) => alert.status === filter);
  }, [alerts.data, filter]);

  const record = (alerts.data ?? []).find((alert) => alert.alert_id === selected) ?? rows[0] ?? null;
  const reasons = Array.isArray(record?.summary)
    ? record?.summary
    : typeof record?.summary === "string"
      ? [record?.summary]
      : [];

  const update = async (status: string) => {
    if (!record) return;
    try {
      await api.patch(`/alerts/${record.alert_id}`, {
        status,
        analyst_note: note.trim() ? note.trim() : null,
      });
      setMessage(`Alert ${record.alert_id} marked ${status}`);
      setNote("");
      alerts.reload();
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const counts = useMemo(() => {
    const list = alerts.data ?? [];
    return {
      open: list.filter((alert) => alert.status === "open").length,
      critical: list.filter((alert) => alert.severity === "critical").length,
      investigating: list.filter((alert) => alert.status === "investigating").length,
      closed: list.filter((alert) => alert.status === "closed").length,
    };
  }, [alerts.data]);

  return (
    <div className="stack" style={{ gap: 18 }}>
      <PageHead
        title="Alert queue"
        sub="Every transaction above the alert threshold, ready for triage."
        right={
          <div className="row">
            <span className="dim">
              {counts.open} open · {counts.critical} critical · {counts.investigating} investigating
            </span>
            <button className="btn ghost small" onClick={alerts.reload}>
              Refresh
            </button>
          </div>
        }
      />

      <ErrorBox message={alerts.error} />

      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="tabs">
          {FILTERS.map((option) => (
            <button
              key={option}
              className={filter === option ? "active" : ""}
              onClick={() => setFilter(option)}
            >
              {option}
            </button>
          ))}
        </div>
        <span className="dim">{rows.length} matching alerts</span>
      </div>

      <div className="split">
        <Panel title="Queue" note="click a row to open the case" flush>
          <div style={{ padding: "12px 18px 16px" }}>
            <Table
              columns={["Alert", "Created", "Customer", "Severity", "Score", "Status"]}
              selectedIndex={record ? rows.findIndex((row) => row.alert_id === record.alert_id) : undefined}
              onRowClick={(index) => {
                setSelected(rows[index].alert_id);
                setMessage(null);
              }}
              rows={rows.map((alert) => [
                <span className="mono">{alert.alert_id}</span>,
                <span className="dim">{shortTime(alert.created_at)}</span>,
                <span className="mono">{alert.customer_id}</span>,
                <Badge tone={alert.severity ?? "neutral"}>{alert.severity}</Badge>,
                <span className="num">{pct(alert.score ?? 0).replace("%", "")}</span>,
                <Badge tone={alert.status === "open" ? "warn" : alert.status === "closed" ? "ok" : "info"}>
                  {alert.status}
                </Badge>,
              ])}
            />
          </div>
        </Panel>

        <Panel title="Case file">
          {!record ? (
            <div className="empty">No alert selected.</div>
          ) : (
            <div className="stack">
              <div>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>{record.title}</div>
                <div className="dim">
                  {record.alert_id} · {shortTime(record.created_at)} · txn{" "}
                  <span className="mono">{record.transaction_id}</span>
                </div>
              </div>

              <div className="row">
                <Badge tone={record.severity ?? "neutral"}>{record.severity}</Badge>
                <Badge tone="info">score {pct(record.score ?? 0)}</Badge>
                <Badge tone={record.status === "open" ? "warn" : "ok"}>{record.status}</Badge>
              </div>

              {reasons.length > 0 && (
                <ul className="reasons">
                  {reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              )}

              {record.analyst_note && (
                <div>
                  <div className="kpi-label">Analyst note</div>
                  <div className="muted">{record.analyst_note}</div>
                </div>
              )}

              <div className="field">
                <label>Add note</label>
                <textarea
                  rows={3}
                  value={note}
                  onChange={(event) => setNote(event.target.value)}
                  placeholder="Called customer, verified travel to Moscow…"
                />
              </div>

              <div className="row">
                <button className="btn small" onClick={() => update("investigating")}>
                  Investigating
                </button>
                <button className="btn ghost small" onClick={() => update("acknowledged")}>
                  Acknowledge
                </button>
                <button className="btn ghost small" onClick={() => update("closed")}>
                  Close
                </button>
                <button className="btn ghost small" onClick={() => update("escalated")}>
                  Escalate
                </button>
              </div>
              {message && <div className="dim">{message}</div>}

              <a className="btn ghost small" href="#/investigations">
                Open customer investigation →
              </a>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
