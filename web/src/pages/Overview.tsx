import { AreaChart, BarList, Badge, Empty, ErrorBox, Kpi, PageHead, Panel, Table, useApi } from "../components/ui";
import { compactInr, inr, pct, shortTime } from "../api";
import type { BreakdownRow, Bucket, Overview as OverviewData, Transaction } from "../types";

const decisionTone = (decision?: string) =>
  decision === "block" ? "block" : decision === "review" ? "review" : "allow";

export default function Overview() {
  const overview = useApi<OverviewData>("/stats/overview");
  const series = useApi<Bucket[]>("/stats/timeseries?hours=48");
  const channels = useApi<BreakdownRow[]>("/stats/breakdown?field=channel&hours=168&limit=6");
  const categories = useApi<BreakdownRow[]>(
    "/stats/breakdown?field=merchant_category&hours=168&limit=7",
  );
  const recent = useApi<{ transactions: Transaction[] }>("/transactions?limit=10");
  const top = useApi<{ customer_id: string; transactions: number; max_score: number; flagged: number; volume: number }[]>(
    "/stats/top-customers?limit=5",
  );

  const data = overview.data;

  return (
    <div className="stack" style={{ gap: 18 }}>
      <PageHead
        title="Fraud operations"
        sub="Live view of transaction risk, analyst queue and emerging networks."
        right={
          <div className="row">
            <span className="dim">
              last txn {shortTime(data?.last_transaction_at)}
            </span>
            <button className="btn ghost small" onClick={() => {
              overview.reload();
              series.reload();
              channels.reload();
              categories.reload();
              recent.reload();
              top.reload();
            }}>
              Refresh
            </button>
          </div>
        }
      />

      <ErrorBox message={overview.error} />

      <div className="grid cols-5">
        <Kpi
          label="Transactions · 24h"
          value={(data?.transactions_24h ?? 0).toLocaleString()}
          note={`${(data?.total_transactions ?? 0).toLocaleString()} lifetime`}
        />
        <Kpi
          label="Flagged · 24h"
          value={(data?.flagged_transactions ?? 0).toLocaleString()}
          note={`${pct(data?.flagged_rate ?? 0, 1)} of volume`}
          tone="warn"
        />
        <Kpi
          label="Value at risk"
          value={compactInr(data?.flagged_amount ?? 0)}
          note="above alert threshold"
          tone="danger"
        />
        <Kpi
          label="Open alerts"
          value={(data?.alerts_open ?? 0).toLocaleString()}
          note={`${data?.alerts_critical ?? 0} critical`}
          tone="danger"
        />
        <Kpi
          label="Avg risk score"
          value={(data?.avg_score ?? 0).toFixed(2)}
          note={`${(data?.customers ?? 0).toLocaleString()} customers monitored`}
          tone="ok"
        />
      </div>

      <div className="split">
        <Panel
          title="Transaction flow vs. flagged volume"
          note="hourly buckets, last 48 hours"
        >
          {series.error ? (
            <ErrorBox message={series.error} />
          ) : (
            <AreaChart data={series.data ?? []} />
          )}
        </Panel>

        <Panel title="Channel mix" note="value share, last 7 days">
          {channels.error ? (
            <ErrorBox message={channels.error} />
          ) : (
            <BarList
              items={(channels.data ?? []).map((row) => ({
                label: row.label,
                value: row.volume,
                note: compactInr(row.volume),
              }))}
            />
          )}
        </Panel>
      </div>

      <div className="split">
        <Panel title="Risk by merchant category" note="flagged transactions, last 7 days">
          {categories.error ? (
            <ErrorBox message={categories.error} />
          ) : (
            <BarList
              danger
              items={(categories.data ?? []).map((row) => ({
                label: row.label,
                value: row.flagged,
                note: `${row.flagged}`,
              }))}
            />
          )}
        </Panel>

        <Panel title="Highest-risk customers" note="ranked by peak score">
          {top.error ? (
            <ErrorBox message={top.error} />
          ) : !(top.data ?? []).length ? (
            <Empty>No scored customers yet.</Empty>
          ) : (
            <Table
              columns={["Customer", "Peak", "Flagged", "Volume"]}
              rows={(top.data ?? []).map((row) => [
                <a href={`#/investigations`} className="mono">
                  {row.customer_id}
                </a>,
                <Badge tone={row.max_score >= 0.75 ? "block" : row.max_score >= 0.35 ? "review" : "allow"}>
                  {(row.max_score * 100).toFixed(0)}%
                </Badge>,
                <span className="num">{row.flagged}</span>,
                <span className="num">{compactInr(row.volume)}</span>,
              ])}
            />
          )}
        </Panel>
      </div>

      <Panel title="Recent activity" note="latest scored transactions" flush>
        <div style={{ padding: "0 18px 14px" }}>
          {recent.error ? (
            <ErrorBox message={recent.error} />
          ) : (
            <Table
              columns={["Time", "Customer", "Amount", "Score", "Decision", "Location"]}
              rows={(recent.data?.transactions ?? []).map((row) => [
                <span className="dim">{shortTime(row.timestamp)}</span>,
                <span className="mono">{row.customer_id}</span>,
                <span className="num">{inr(row.amount)}</span>,
                <span className="num">{pct(row.combined_score ?? 0)}</span>,
                <Badge tone={decisionTone(row.decision)}>{row.decision ?? "allow"}</Badge>,
                <span className="muted">
                  {row.city ?? "—"}
                  {row.country ? `, ${row.country}` : ""}
                </span>,
              ])}
            />
          )}
        </div>
      </Panel>
    </div>
  );
}
