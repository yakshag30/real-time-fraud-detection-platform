import { useState } from "react";
import { api } from "../api";
import {
  BarList,
  Badge,
  CurveChart,
  ErrorBox,
  Kpi,
  PageHead,
  Panel,
  Table,
  useApi,
} from "../components/ui";
import type { MetricsResponse } from "../types";

export default function ModelPerformance() {
  const info = useApi<MetricsResponse>("/models/info");
  const [retraining, setRetraining] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const metrics = info.data?.metrics;
  const selected = metrics?.selected_model ?? "xgboost";
  const candidates = Object.entries(metrics?.models ?? {});
  const best = metrics?.models[selected];
  const anomaly = metrics?.anomaly;

  const retrain = async () => {
    setRetraining(true);
    setMessage(null);
    try {
      await api.post("/models/retrain", {});
      setMessage("Retrained. Metrics refreshed.");
      info.reload();
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="stack" style={{ gap: 18 }}>
      <PageHead
        title="Model performance"
        sub="Chronological holdout, measured on the synthetic bank stream."
        right={
          <button className="btn ghost small" onClick={retrain} disabled={retraining}>
            {retraining ? "Retraining…" : "Retrain"}
          </button>
        }
      />

      <ErrorBox message={info.error} />
      {message && <div className="dim">{message}</div>}

      <div className="grid cols-4">
        <Kpi label="PR-AUC" value={best?.pr_auc?.toFixed(3) ?? "—"} note={`${selected} on holdout`} tone="ok" />
        <Kpi label="ROC-AUC" value={best?.roc_auc?.toFixed(3) ?? "—"} note="rank quality" />
        <Kpi
          label="Precision / recall"
          value={`${best?.precision?.toFixed(2) ?? "—"} / ${best?.recall?.toFixed(2) ?? "—"}`}
          note="at 0.50 cut-off"
          tone="warn"
        />
        <Kpi
          label="Anomaly PR-AUC"
          value={anomaly?.pr_auc?.toFixed(3) ?? "—"}
          note={`Isolation Forest · weight ${info.data?.anomaly_weight ?? 0.25}`}
        />
      </div>

      <Panel title="Candidate models" note="all trained on the same split" flush>
        <div style={{ padding: "12px 18px 16px" }}>
          <Table
            columns={["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier"]}
            rows={candidates.map(([name, row]) => [
              <span>
                {name}{" "}
                {name === selected && <Badge tone="info">selected</Badge>}
              </span>,
              <span className="num">{row.pr_auc.toFixed(3)}</span>,
              <span className="num">{row.roc_auc.toFixed(3)}</span>,
              <span className="num">{row.precision.toFixed(3)}</span>,
              <span className="num">{row.recall.toFixed(3)}</span>,
              <span className="num">{row.f1.toFixed(3)}</span>,
              <span className="num">{row.brier.toFixed(4)}</span>,
            ])}
          />
        </div>
      </Panel>

      <div className="split">
        <Panel title="Feature importance" note="gain-normalised, selected model">
          <BarList
            items={(metrics?.feature_importance ?? []).slice(0, 10).map((row) => ({
              label: row.feature,
              value: row.importance,
              note: `${(row.importance * 100).toFixed(1)}%`,
            }))}
          />
        </Panel>

        <div className="grid" style={{ gap: 16 }}>
          {best?.curves && (
            <Panel title="ROC curve" note={`${selected} · holdout`}>
              <CurveChart
                xLabel="false positive rate"
                yLabel="true positive rate"
                series={[{ name: selected, color: "#14543e", points: best.curves.roc }]}
              />
            </Panel>
          )}
          {best?.curves && (
            <Panel title="Precision-recall" note="area = PR-AUC">
              <CurveChart
                xLabel="recall"
                yLabel="precision"
                series={[{ name: selected, color: "#9a6400", points: best.curves.pr }]}
              />
            </Panel>
          )}
        </div>
      </div>

      <div className="grid cols-3">
        <Panel title="Dataset" note="training window">
          <div className="stack" style={{ gap: 4, fontSize: 12.5 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">Rows</span>
              <span className="mono">{String(metrics?.dataset?.rows ?? "—")}</span>
            </div>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">Train / test</span>
              <span className="mono">
                {String(metrics?.dataset?.train_rows ?? "—")} / {String(metrics?.dataset?.test_rows ?? "—")}
              </span>
            </div>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">Fraud rate</span>
              <span className="mono">
                {metrics?.dataset?.fraud_rate
                  ? `${(Number(metrics.dataset.fraud_rate) * 100).toFixed(2)}%`
                  : "—"}
              </span>
            </div>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">Features</span>
              <span className="mono">{String(metrics?.dataset?.features ?? "—")}</span>
            </div>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">Explainer</span>
              <span className="mono">{info.data?.explainer ?? "—"}</span>
            </div>
          </div>
        </Panel>

        <Panel title="Confusion matrix" note="at 0.50 cut-off">
          <Table
            columns={["", "Predicted fraud", "Predicted legit"]}
            rows={[
              [
                <span className="muted">Actual fraud</span>,
                <span className="num">{best?.confusion_matrix?.true_positive ?? 0}</span>,
                <span className="num">{best?.confusion_matrix?.false_negative ?? 0}</span>,
              ],
              [
                <span className="muted">Actual legit</span>,
                <span className="num">{best?.confusion_matrix?.false_positive ?? 0}</span>,
                <span className="num">{best?.confusion_matrix?.true_negative ?? 0}</span>,
              ],
            ]}
          />
        </Panel>

        <Panel title="Operating policy" note="applied by the scoring service">
          <div className="stack" style={{ gap: 6, fontSize: 12.5 }}>
            {(Object.entries(metrics?.thresholds ?? info.data?.model?.thresholds ?? {}) as [string, number][]).map(
              ([name, value]) => (
                <div key={name} className="row" style={{ justifyContent: "space-between" }}>
                  <span className="muted">{name}</span>
                  <span className="mono">{value}</span>
                </div>
              ),
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
