import React from 'react';

export interface PredictRequest {
  card_id: string;
  amount: number;
  hour: number;
}

export interface PredictResponse {
  prediction_id: string;
  is_fraud: boolean;
  fraud_probability: number;
  latency_ms: number;
}

export interface StreamTokenResponse {
  token: string;
}

export interface StreamMessage {
  type: 'replay' | 'live';
  prediction_id: string;
  card_id: string;
  amount: number;
  fraud_probability: number;
  is_flagged: boolean;
  latency_ms: number;
  created_at: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  model_loaded: boolean;
}

export interface SHAPResponse {
  prediction_id: string;
  shap_values: number[];
  feature_names: string[];
  base_value: number;
  prediction_value: number;
}

export interface ConfigResponse {
  key: string;
  value: string;
  description?: string;
}

export interface MetricsResponse {
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  throughput_rps: number;
  error_rate: number;
  queue_depth: number;
  active_connections: number;
}

export interface ClusterConfig {
  feature_names: string[];
  n_clusters: number;
}

export interface TransactionCardProps {
  data: StreamMessage;
  index: number;
}

export interface LiveFeedProps {
  messages: StreamMessage[];
  isConnected: boolean;
}

export interface SHAPWaterfallProps {
  data: SHAPResponse | null;
  loading: boolean;
}

export interface LatencyChartProps {
  data: number[];
  maxPoints?: number;
}

export interface ThresholdTunerProps {
  currentThreshold: number;
  onChange: (value: number) => Promise<void>;
}

export interface KPICardProps {
  title: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  unit?: string;
  icon?: React.ReactNode;
}

export interface ClusterBadgeProps {
  clusterId: number;
  config: ClusterConfig | null;
}

export type TabKey = 'overview' | 'analyst' | 'ops' | 'model' | 'replay';

export interface TabConfig {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
  description: string;
}

export interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastMessage: StreamMessage | null;
  messages: StreamMessage[];
  replayComplete: boolean;
}

export interface AuthState {
  apiKey: string | null;
  streamToken: string | null;
  isAuthenticated: boolean;
}