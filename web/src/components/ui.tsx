import { useEffect, useState, type ReactNode } from "react";
import { api, inr } from "../api";
import type { Contribution } from "../types";

/* --------------------------------------------------------------- data hook */
export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    setLoading(true);
    api
      .get<T>(path)
      .then((payload) => {
        if (!alive) return;
        setData(payload);
        setError(null);
      })
      .catch((exc: unknown) => {
        if (alive) setError(exc instanceof Error ? exc.message : String(exc));
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [path, nonce]);

  return { data, error, loading, reload: () => setNonce((value) => value + 1) };
}

/* ------------------------------------------------------------------ layout */
export function Panel({
  title,
  note,
  action,
  children,
  flush,
}: {
  title?: string;
  note?: string;
  action?: ReactNode;
  children: ReactNode;
  flush?: boolean;
}) {
  return (
    <section className={`panel${flush ? " flush" : ""}`}>
      {(title || action) && (
        <header className="panel-head" style={flush ? { padding: "16px 18px 0" } : undefined}>
          <div>
            {title && <div className="panel-title">{title}</div>}
            {note && <div className="panel-note">{note}</div>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Kpi({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: ReactNode;
  note?: string;
  tone?: "danger" | "warn" | "ok";
}) {
  return (
    <div className={`kpi ${tone ?? ""}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {note && <div className="kpi-note">{note}</div>}
    </div>
  );
}

export function Badge({ tone, children }: { tone?: string; children: ReactNode }) {
  return <span className={`badge ${tone ?? "neutral"}`}>{children}</span>;
}

export function PageHead({
  title,
  sub,
  right,
}: {
  title: string;
  sub?: string;
  right?: ReactNode;
}) {
  return (
    <div className="page-head">
      <div>
        <h1 className="page-title">{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

export function ErrorBox({ message }: { message: string | null }) {
  if (!message) return null;
  return <div className="error">{message}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Table({
  columns,
  rows,
  onRowClick,
  selectedIndex,
}: {
  columns: string[];
  rows: ReactNode[][];
  onRowClick?: (index: number) => void;
  selectedIndex?: number;
}) {
  if (!rows.length) return <Empty>Nothing to show yet.</Empty>;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={index}
              className={
                onRowClick ? `clickable${selectedIndex === index ? " selected" : ""}` : undefined
              }
              onClick={onRowClick ? () => onRowClick(index) : undefined}
            >
              {row.map((cell, cellIndex) => (
                <td key={cellIndex}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ charts */
export function Gauge({ score, caption }: { score: number; caption?: string }) {
  const value = Math.min(Math.max(score, 0), 1);
  const radius = 78;
  const length = Math.PI * radius;
  const color = value >= 0.75 ? "#a52a20" : value >= 0.35 ? "#9a6400" : "#1c6b4d";
  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 124" className="chart" style={{ maxWidth: 260 }}>
        <defs>
          <linearGradient id="gauge-bg" x1="0" x2="1">
            <stop offset="0%" stopColor="#e6e0d2" />
            <stop offset="100%" stopColor="#f0ebdf" />
          </linearGradient>
        </defs>
        <path
          d={`M 22 100 A ${radius} ${radius} 0 0 1 178 100`}
          fill="none"
          stroke="url(#gauge-bg)"
          strokeWidth="13"
          strokeLinecap="round"
        />
        <path
          d={`M 22 100 A ${radius} ${radius} 0 0 1 178 100`}
          fill="none"
          stroke={color}
          strokeWidth="13"
          strokeLinecap="round"
          strokeDasharray={`${value * length} ${length}`}
          style={{ filter: `drop-shadow(0 0 6px ${color}66)`, transition: "stroke-dasharray 0.5s ease" }}
        />
      </svg>
      <div className="gauge-value" style={{ color }}>
        {Math.round(value * 100)}%
      </div>
      <div className="gauge-label">{caption ?? "combined risk"}</div>
    </div>
  );
}

export function AreaChart({
  data,
  height = 190,
}: {
  data: { bucket: string; count: number; flagged: number }[];
  height?: number;
}) {
  if (!data.length) return <Empty>No traffic in this window.</Empty>;
  const width = 720;
  const padding = { top: 12, right: 8, bottom: 22, left: 30 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;
  const max = Math.max(...data.map((d) => d.count), 1);
  const stepX = data.length > 1 ? innerW / (data.length - 1) : innerW;

  const line = (key: "count" | "flagged") =>
    data
      .map((d, i) => {
        const x = padding.left + i * stepX;
        const y = padding.top + innerH - (d[key] / max) * innerH;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");

  const area = `${line("count")} L ${(padding.left + (data.length - 1) * stepX).toFixed(1)} ${
    padding.top + innerH
  } L ${padding.left} ${padding.top + innerH} Z`;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        <defs>
          <linearGradient id="area-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#14543e" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#14543e" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y = padding.top + innerH - tick * innerH;
          return (
            <g key={tick}>
              <line
                x1={padding.left}
                x2={width - padding.right}
                y1={y}
                y2={y}
                stroke="#e2dccc"
                strokeWidth="1"
              />
              <text x={2} y={y + 3} fill="#8b8677" fontSize="9">
                {Math.round(tick * max)}
              </text>
            </g>
          );
        })}
        <path d={area} fill="url(#area-fill)" />
        <path d={line("count")} fill="none" stroke="#14543e" strokeWidth="2" />
        <path d={line("flagged")} fill="none" stroke="#a52a20" strokeWidth="1.8" strokeDasharray="4 3" />
      </svg>
      <div className="legend" style={{ marginTop: 8 }}>
        <span>
          <i className="swatch" style={{ background: "#14543e" }} /> transactions
        </span>
        <span>
          <i className="swatch" style={{ background: "#a52a20" }} /> flagged
        </span>
        <span className="dim">
          {data[0]?.bucket?.replace("T", " ")} → {data[data.length - 1]?.bucket?.replace("T", " ")}
        </span>
      </div>
    </div>
  );
}

export function BarList({
  items,
  danger,
}: {
  items: { label: string; value: number; note?: string }[];
  danger?: boolean;
}) {
  if (!items.length) return <Empty>No data.</Empty>;
  const max = Math.max(...items.map((item) => item.value), 1);
  return (
    <div className="stack" style={{ gap: 9 }}>
      {items.map((item) => (
        <div className="bar-row" key={item.label}>
          <span className="label" title={item.label}>
            {item.label.replace(/_/g, " ")}
          </span>
          <span className="bar-track">
            <span
              className={`bar-fill${danger ? " danger" : ""}`}
              style={{ width: `${Math.max((item.value / max) * 100, 2)}%`, display: "block" }}
            />
          </span>
          <span className="value">{item.note ?? item.value}</span>
        </div>
      ))}
    </div>
  );
}

export function Contributions({ items }: { items: Contribution[] }) {
  if (!items.length) return <Empty>No attribution returned.</Empty>;
  const max = Math.max(...items.map((item) => Math.abs(item.contribution)), 0.01);
  return (
    <div className="stack" style={{ gap: 8 }}>
      {items.map((item) => {
        const ratio = Math.abs(item.contribution) / max;
        const positive = item.contribution >= 0;
        return (
          <div key={item.feature} style={{ display: "grid", gridTemplateColumns: "1fr 58px", gap: 10 }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                <span style={{ color: "#23272d" }}>{item.label}</span>
                <span className="mono" style={{ color: positive ? "#a52a20" : "#1c6b4d" }}>
                  {item.value.toFixed(2)}
                  {item.feature === "amount_to_customer_avg" ? "×" : ""}
                </span>
              </div>
              <div className="bar-track" style={{ marginTop: 4 }}>
                <span
                  className="bar-fill"
                  style={{
                    width: `${Math.max(ratio * 100, 2)}%`,
                    display: "block",
                    background: positive
                      ? "#a52a20"
                      : "#1c6b4d",
                  }}
                />
              </div>
            </div>
            <span
              className="mono"
              style={{ textAlign: "right", color: positive ? "#a52a20" : "#1c6b4d", paddingTop: 2 }}
            >
              {positive ? "+" : ""}
              {item.contribution.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function CurveChart({
  series,
  xLabel,
  yLabel,
}: {
  series: { name: string; color: string; points: number[][] }[];
  xLabel: string;
  yLabel: string;
}) {
  const width = 320;
  const height = 220;
  const pad = 28;
  const path = (points: number[][]) =>
    points
      .map(([x, y], index) => {
        const px = pad + x * (width - pad * 2);
        const py = height - pad - y * (height - pad * 2);
        return `${index === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)}`;
      })
      .join(" ");
  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        <rect
          x={pad}
          y={pad}
          width={width - pad * 2}
          height={height - pad * 2}
          fill="none"
          stroke="#ded8ca"
        />
        {[0.25, 0.5, 0.75].map((tick) => (
          <line
            key={tick}
            x1={pad}
            x2={width - pad}
            y1={height - pad - tick * (height - pad * 2)}
            y2={height - pad - tick * (height - pad * 2)}
            stroke="#ece7da"
          />
        ))}
        {series.map((line) => (
          <path key={line.name} d={path(line.points)} fill="none" stroke={line.color} strokeWidth="2" />
        ))}
        <text x={width / 2} y={height - 4} fill="#8b8677" fontSize="9" textAnchor="middle">
          {xLabel}
        </text>
        <text x={9} y={height / 2} fill="#8b8677" fontSize="9" textAnchor="middle" transform={`rotate(-90 9 ${height / 2})`}>
          {yLabel}
        </text>
      </svg>
      <div className="legend" style={{ marginTop: 4 }}>
        {series.map((line) => (
          <span key={line.name}>
            <i className="swatch" style={{ background: line.color }} /> {line.name}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- markdown */
export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let table: string[] = [];

  const inline = (value: string) =>
    value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .split(/(\*\*[^*]+\*\*)/g)
      .map((chunk, index) =>
        chunk.startsWith("**") && chunk.endsWith("**") ? (
          <strong key={index}>{chunk.slice(2, -2)}</strong>
        ) : (
          <span key={index}>{chunk}</span>
        ),
      );

  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {list.map((item, index) => (
            <li key={index}>{inline(item)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };
  const flushTable = () => {
    if (table.length) {
      blocks.push(
        <pre key={`tb-${blocks.length}`}>{table.join("\n")}</pre>,
      );
      table = [];
    }
  };

  lines.forEach((raw, index) => {
    const line = raw.trimEnd();
    if (line.startsWith("|")) {
      flushList();
      table.push(line);
      return;
    }
    flushTable();
    if (line.startsWith("- ")) {
      list.push(line.slice(2));
      return;
    }
    flushList();
    if (line.startsWith("# ")) {
      blocks.push(<h1 key={index}>{inline(line.slice(2))}</h1>);
    } else if (line.startsWith("## ")) {
      blocks.push(<h2 key={index}>{inline(line.slice(3))}</h2>);
    } else if (line.trim()) {
      blocks.push(<p key={index}>{inline(line)}</p>);
    }
  });
  flushList();
  flushTable();

  return <div className="md-block">{blocks}</div>;
}

export const money = (value: number | null | undefined) => inr(value);
