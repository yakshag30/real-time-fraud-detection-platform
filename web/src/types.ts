export interface Health {
  status: string;
  database: string | null;
  transactions: number;
  alerts: number;
  model: ModelInfo | null;
  thresholds: { review: number; block: number; alert: number; anomaly_weight: number };
}

export interface ModelInfo {
  model_name: string;
  trained_at: string;
  features: number;
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  isolation_forest: boolean;
}

export interface Overview {
  total_transactions: number;
  total_volume: number;
  transactions_24h: number;
  volume_24h: number;
  flagged_transactions: number;
  flagged_rate: number;
  flagged_amount: number;
  avg_score: number;
  alerts_total: number;
  alerts_open: number;
  alerts_critical: number;
  customers: number;
  last_transaction_at: string | null;
}

export interface Bucket {
  bucket: string;
  count: number;
  flagged: number;
  volume: number;
  avg_score: number;
}

export interface BreakdownRow {
  label: string;
  count: number;
  flagged: number;
  volume: number;
  avg_score: number;
}

export interface Transaction {
  transaction_id: string;
  timestamp: string;
  customer_id: string;
  customer_name?: string | null;
  amount: number;
  merchant_id?: string | null;
  merchant_category?: string | null;
  channel?: string | null;
  city?: string | null;
  country?: string | null;
  fraud_probability?: number;
  anomaly_score?: number;
  combined_score?: number;
  decision?: string;
  risk_level?: string;
  reasons?: string[];
  summary?: string | null;
  alert_id?: string | null;
  device_id?: string | null;
}

export interface Alert {
  alert_id: string;
  transaction_id: string | null;
  customer_id: string | null;
  severity: string | null;
  score: number | null;
  status: string | null;
  title: string | null;
  summary: string[] | string | null;
  created_at: string | null;
  updated_at: string | null;
  analyst_note: string | null;
}

export interface Contribution {
  feature: string;
  label: string;
  value: number;
  contribution: number;
  direction: string;
}

export interface ScoreResult {
  transaction_id: string;
  timestamp: string;
  customer_id: string;
  customer_name?: string | null;
  amount: number;
  channel?: string | null;
  city?: string | null;
  country?: string | null;
  merchant_name?: string | null;
  merchant_category?: string | null;
  fraud_probability: number;
  anomaly_score: number;
  combined_score: number;
  risk_level: string;
  decision: string;
  model_name: string;
  reasons: string[];
  summary: string;
  anomaly_note: string;
  explanation: Contribution[];
  explanation_method: string;
  base_value: number;
  graph_flags: string[];
  policy_floor: number;
  policy_flags: string[];
  alert_id: string | null;
  latency_ms: number;
}

export interface MetricsResponse {
  model: ModelInfo & { thresholds: Record<string, number> };
  metrics: {
    trained_at: string;
    selected_model: string;
    dataset: Record<string, unknown>;
    models: Record<string, ModelMetrics>;
    anomaly: ModelMetrics;
    feature_importance: { feature: string; importance: number }[];
    thresholds: Record<string, number>;
  };
  explainer: string;
  anomaly_weight: number;
}

export interface ModelMetrics {
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  brier: number;
  curves?: { roc: number[][]; pr: number[][] };
  confusion_matrix?: Record<string, number>;
}

export interface Ring {
  ring_id: string;
  members: string[];
  size: number;
  risk_score: number;
  flagged_members: string[];
  shared_devices: string[];
  shared_ips: string[];
  shared_merchants: string[];
  transactions: number;
  amount: number;
  evidence: string[];
}

export interface GraphNode {
  id: string;
  label: string;
  kind: string;
  risk: number;
  transactions: number;
  amount: number;
}

export interface GraphNetwork {
  nodes: GraphNode[];
  edges: { source: string; target: string; transactions: number }[];
  found: boolean;
  ring?: Ring | null;
}

export interface Options {
  customers: { customer_id: string; name: string; home_city: string; avg_amount: number }[];
  merchants: { merchant_id: string; name: string; category: string; risk_score: number }[];
  cities: { name: string; country: string; lat: number; lon: number }[];
  channels: string[];
  demo_scenario: {
    customer_id: string;
    customer_name: string;
    amount: number;
    timestamp: string;
    merchant_id: string;
    merchant_name: string;
    merchant_category: string;
    channel: string;
    city: string;
    country: string;
    latitude: number;
    longitude: number;
    device_id: string;
  };
}

export interface Investigation {
  customer: { customer_id: string; name: string; home_city: string; segment: string; avg_amount: number };
  risk_grade: string;
  summary: string;
  markdown: string;
}
