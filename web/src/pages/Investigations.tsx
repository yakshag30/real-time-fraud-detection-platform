import { useEffect, useMemo, useState } from "react";
import { inr, pct, shortTime } from "../api";
import {
  Badge,
  ErrorBox,
  Markdown,
  PageHead,
  Panel,
  Table,
  useApi,
} from "../components/ui";
import type { Investigation, Transaction } from "../types";

interface CustomerRow {
  customer_id: string;
  name: string;
  home_city: string;
  segment?: string;
  avg_amount?: number;
}

export default function Investigations() {
  const customers = useApi<{ customers: CustomerRow[] }>("/customers?limit=500");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string>("");

  useEffect(() => {
    if (!selected && customers.data?.customers.length) {
      setSelected(customers.data.customers[0].customer_id);
    }
  }, [customers.data, selected]);

  const filtered = useMemo(() => {
    const list = customers.data?.customers ?? [];
    const needle = query.trim().toLowerCase();
    if (!needle) return list.slice(0, 40);
    return list
      .filter(
        (customer) =>
          customer.customer_id.toLowerCase().includes(needle) ||
          (customer.name ?? "").toLowerCase().includes(needle),
      )
      .slice(0, 40);
  }, [customers.data, query]);

  const report = useApi<Investigation>(
    selected ? `/customers/${selected}/investigation` : null,
  );
  const transactions = useApi<{ transactions: Transaction[] }>(
    selected ? `/customers/${selected}/transactions?limit=15` : null,
  );

  return (
    <div className="stack" style={{ gap: 18 }}>
      <PageHead
        title="Investigations"
        sub="Assemble a case file for any customer: history, network context and alerts."
        right={
          <button className="btn ghost small" onClick={() => report.reload()}>
            Regenerate
          </button>
        }
      />

      <div className="split narrow">
        <Panel title="Customers" note={`${customers.data?.customers.length ?? 0} on file`}>
          <div className="field" style={{ marginBottom: 10 }}>
            <input
              placeholder="Search id or name…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <div className="stack" style={{ gap: 4, maxHeight: 520, overflow: "auto" }}>
            {filtered.map((customer) => (
              <button
                key={customer.customer_id}
                className={`nav-item${customer.customer_id === selected ? " active" : ""}`}
                onClick={() => setSelected(customer.customer_id)}
                style={{ justifyContent: "space-between" }}
              >
                <span>
                  <span className="mono">{customer.customer_id}</span>{" "}
                  <span style={{ color: "inherit" }}>{customer.name}</span>
                </span>
                <span className="dim">{customer.home_city}</span>
              </button>
            ))}
          </div>
        </Panel>

        <div className="stack">
          <ErrorBox message={report.error} />
          {report.data && (
            <Panel
              title={report.data.customer.name}
              note={`${report.data.customer.customer_id} · ${report.data.customer.segment} · home ${report.data.customer.home_city}`}
              action={
                <Badge
                  tone={
                    report.data.risk_grade === "Critical"
                      ? "critical"
                      : report.data.risk_grade === "High"
                        ? "high"
                        : report.data.risk_grade === "Elevated"
                          ? "medium"
                          : "low"
                  }
                >
                  {report.data.risk_grade} risk
                </Badge>
              }
            >
              <p className="muted" style={{ marginTop: 0 }}>
                {report.data.summary}
              </p>
              <Markdown text={report.data.markdown} />
            </Panel>
          )}

          <Panel title="Recent transactions" flush>
            <div style={{ padding: "12px 18px 16px" }}>
              <Table
                columns={["Time", "Amount", "Score", "Decision", "Channel", "Location"]}
                rows={(transactions.data?.transactions ?? []).map((row) => [
                  <span className="dim">{shortTime(row.timestamp)}</span>,
                  <span className="num">{inr(row.amount)}</span>,
                  <span className="num">{pct(row.combined_score ?? 0)}</span>,
                  <Badge
                    tone={
                      row.decision === "block"
                        ? "block"
                        : row.decision === "review"
                          ? "review"
                          : "allow"
                    }
                  >
                    {row.decision ?? "allow"}
                  </Badge>,
                  <span className="muted">{row.channel?.toUpperCase()}</span>,
                  <span className="muted">
                    {row.city}, {row.country}
                  </span>,
                ])}
              />
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
