import { useEffect, useState } from "react";
import { compactInr, inr, pct } from "../api";
import { Badge, ErrorBox, PageHead, Panel, useApi } from "../components/ui";
import type { GraphNetwork, Ring } from "../types";

const COLORS: Record<string, string> = {
  customer: "#14543e",
  device: "#5b4b8a",
  ip: "#9a6400",
  merchant: "#146b7a",
};

const RADIUS: Record<string, number> = {
  customer: 9,
  device: 6,
  ip: 5,
  merchant: 6,
};

export default function Network() {
  const rings = useApi<{ rings: Ring[]; graph: { nodes: number; edges: number }; method: string }>(
    "/graph/rings",
  );
  const [customer, setCustomer] = useState("");
  const [draft, setDraft] = useState("");

  useEffect(() => {
    if (!customer && rings.data?.rings.length) {
      const first = rings.data.rings[0];
      const target = first.flagged_members[0] ?? first.members[0];
      setCustomer(target);
      setDraft(target);
    }
  }, [rings.data, customer]);

  const network = useApi<GraphNetwork>(
    customer ? `/graph/customer/${customer}?radius=2` : null,
  );

  const nodes = network.data?.nodes ?? [];
  const center = nodes.find((node) => node.id === `customer:${customer}`) ?? nodes[0];
  const others = nodes.filter((node) => node !== center);
  const width = 640;
  const height = 420;
  const positions = new Map<string, { x: number; y: number }>();
  if (center) positions.set(center.id, { x: width / 2, y: height / 2 });
  others.forEach((node, index) => {
    const angle = (index / Math.max(others.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const ring = RADIUS[node.kind] ? 150 : 145;
    positions.set(node.id, {
      x: width / 2 + Math.cos(angle) * ring,
      y: height / 2 + Math.sin(angle) * ring * 0.78,
    });
  });

  return (
    <div className="stack" style={{ gap: 18 }}>
      <PageHead
        title="Fraud network"
        sub="Customers, devices, IPs and merchants linked into communities by shared identifiers."
        right={
          <div className="row">
            <span className="dim">
              {rings.data?.graph.nodes ?? 0} nodes · {rings.data?.graph.edges ?? 0} edges
            </span>
            <button className="btn ghost small" onClick={rings.reload}>
              Rebuild
            </button>
          </div>
        }
      />

      <ErrorBox message={rings.error} />

      <div className="split">
        <Panel
          title="Detected rings"
          note="Louvain communities on the customer projection"
        >
          <div className="stack" style={{ gap: 10, maxHeight: 620, overflow: "auto" }}>
            {(rings.data?.rings ?? []).map((ring) => (
              <button
                key={ring.ring_id}
                className={`panel${ring.members.includes(customer) ? " selected" : ""}`}
                style={{
                  textAlign: "left",
                  cursor: "pointer",
                  background: ring.members.includes(customer)
                    ? "#e4efe8"
                    : undefined,
                  padding: 13,
                }}
                onClick={() => {
                  const target = ring.flagged_members[0] ?? ring.members[0];
                  setCustomer(target);
                  setDraft(target);
                }}
              >
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <strong>{ring.ring_id}</strong>
                  <Badge tone={ring.risk_score >= 0.75 ? "critical" : "high"}>
                    risk {pct(ring.risk_score)}
                  </Badge>
                </div>
                <div className="dim" style={{ margin: "3px 0 6px" }}>
                  {ring.size} accounts · {ring.transactions} txns · {compactInr(ring.amount)}
                </div>
                <div className="row" style={{ gap: 5 }}>
                  {ring.evidence.slice(0, 2).map((item) => (
                    <Badge key={item} tone="neutral">
                      {item.length > 46 ? `${item.slice(0, 46)}…` : item}
                    </Badge>
                  ))}
                </div>
              </button>
            ))}
            {!rings.data?.rings.length && <div className="empty">No rings detected.</div>}
          </div>
        </Panel>

        <div className="stack">
          <Panel
            title="Entity neighbourhood"
            note={customer ? `around ${customer}` : "pick a customer"}
            action={
              <div className="row" style={{ gap: 6 }}>
                <input
                  style={{ width: 120 }}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value.toUpperCase())}
                  placeholder="C00011"
                />
                <button className="btn ghost small" onClick={() => setCustomer(draft.trim())}>
                  Load
                </button>
              </div>
            }
          >
            <ErrorBox message={network.error} />
            {!nodes.length ? (
              <div className="empty">No network loaded.</div>
            ) : (
              <svg viewBox={`0 0 ${width} ${height}`} className="chart">
                {network.data?.edges.map((edge, index) => {
                  const from = positions.get(edge.source);
                  const to = positions.get(edge.target);
                  if (!from || !to) return null;
                  return (
                    <line
                      key={index}
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      stroke="#cfc7b4"
                      strokeWidth={Math.min(1 + edge.transactions / 4, 2.4)}
                    />
                  );
                })}
                {nodes.map((node) => {
                  const position = positions.get(node.id);
                  if (!position) return null;
                  const color = COLORS[node.kind] ?? "#8b8677";
                  return (
                    <g
                      key={node.id}
                      className="net-node"
                      onClick={() => {
                        if (node.kind === "customer") {
                          setCustomer(node.label);
                          setDraft(node.label);
                        }
                      }}
                    >
                      <circle
                        cx={position.x}
                        cy={position.y}
                        r={RADIUS[node.kind] ?? 6}
                        fill={color}
                        fillOpacity={node.kind === "customer" ? 0.9 : 0.65}
                        stroke={node.risk >= 0.5 ? "#a52a20" : "#fffefb"}
                        strokeWidth={node.risk >= 0.5 ? 2 : 1}
                      />
                      {(node.kind === "customer" || node.kind === "device") && (
                        <text
                          x={position.x}
                          y={position.y + (RADIUS[node.kind] ?? 6) + 11}
                          textAnchor="middle"
                        >
                          {node.label.length > 18 ? `${node.label.slice(0, 18)}…` : node.label}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
            <div className="legend" style={{ marginTop: 8 }}>
              {Object.entries(COLORS).map(([kind, color]) => (
                <span key={kind}>
                  <i className="swatch" style={{ background: color }} /> {kind}
                </span>
              ))}
              <span className="dim">red outline = risk ≥ 50%</span>
            </div>
          </Panel>

          {network.data?.ring && (
            <Panel title="Ring evidence" note={network.data.ring.ring_id}>
              <div className="stack" style={{ gap: 6 }}>
                {network.data.ring.evidence.map((item) => (
                  <div key={item} className="muted">
                    • {item}
                  </div>
                ))}
                <div className="dim">
                  {network.data.ring.size} accounts moved {inr(network.data.ring.amount)} across{" "}
                  {network.data.ring.transactions} transactions.
                </div>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
