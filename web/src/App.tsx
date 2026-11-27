import { useEffect, useState } from "react";
import { useApi } from "./components/ui";
import type { Health } from "./types";
import Overview from "./pages/Overview";
import LiveMonitor from "./pages/LiveMonitor";
import Alerts from "./pages/Alerts";
import Investigations from "./pages/Investigations";
import ModelPerformance from "./pages/ModelPerformance";
import Network from "./pages/Network";

const PAGES = [
  { id: "overview", label: "Overview", icon: "◎", render: Overview },
  { id: "monitor", label: "Live monitor", icon: "⚡", render: LiveMonitor },
  { id: "alerts", label: "Alert queue", icon: "▲", render: Alerts },
  { id: "investigations", label: "Investigations", icon: "⌕", render: Investigations },
  { id: "models", label: "Model performance", icon: "◈", render: ModelPerformance },
  { id: "network", label: "Fraud network", icon: "⬡", render: Network },
];

function currentRoute(): string {
  const raw = window.location.hash.replace(/^#\/?/, "");
  return PAGES.some((page) => page.id === raw) ? raw : "overview";
}

export default function App() {
  const [route, setRoute] = useState(currentRoute);
  const health = useApi<Health>("/health");

  useEffect(() => {
    const onHash = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const active = PAGES.find((page) => page.id === route) ?? PAGES[0];
  const Page = active.render;
  const model = health.data?.model;
  const online = Boolean(health.data) && !health.error;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">S</span>
          <div>
            <div className="brand-name">Sentinel</div>
            <div className="brand-sub">Fraud Ops</div>
          </div>
        </div>

        <nav className="nav">
          <div className="nav-label">Workspace</div>
          {PAGES.map((page) => (
            <a
              key={page.id}
              href={`#/${page.id}`}
              className={`nav-item${page.id === active.id ? " active" : ""}`}
            >
              <span className="nav-icon">{page.icon}</span>
              {page.label}
            </a>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div>
            <span className={`dot${online ? "" : " off"}`} />
            {online ? "API connected" : "API unreachable"}
          </div>
          {model && (
            <>
              <div>
                Model <span className="mono">{model.model_name}</span>
              </div>
              <div>
                PR-AUC {model.pr_auc?.toFixed(3)} · ROC {model.roc_auc?.toFixed(3)}
              </div>
            </>
          )}
          <div>
            {health.data
              ? `${health.data.transactions.toLocaleString()} txns · ${health.data.database}`
              : "http://127.0.0.1:8000"}
          </div>
        </div>
      </aside>

      <main className="main">
        {health.error && (
          <div className="error" style={{ marginBottom: 16 }}>
            Cannot reach the scoring API ({health.error}). Start it with{" "}
            <span className="mono">python -m fraud_platform.cli serve</span>.
          </div>
        )}
        <Page />
      </main>
    </div>
  );
}
