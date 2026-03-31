// ── A/B Testing ───────────────────────────────────────────────────────────────

export interface FrequentistRequest {
  control_clicks: number;
  control_total: number;
  variant_clicks: number;
  variant_total: number;
  alpha?: number;
}

export interface FrequentistResponse {
  z_statistic: number;
  p_value: number;
  confidence_interval: [number, number];
  is_significant: boolean;
  conclusion: string;
  control_rate: number;
  variant_rate: number;
  relative_uplift: number;
}

export interface BayesianRequest {
  control_successes: number;
  control_failures: number;
  variant_successes: number;
  variant_failures: number;
  prior_alpha?: number;
  prior_beta?: number;
  n_simulations?: number;
}

export interface BayesianResponse {
  probability_b_beats_a: number;
  expected_loss_a: number;
  expected_loss_b: number;
  posterior_a: number[];
  posterior_b: number[];
  recommendation: string;
}

export interface SampleSizeRequest {
  baseline_rate: number;
  min_detectable_effect: number;
  alpha?: number;
  power?: number;
}

export interface SampleSizeResponse {
  sample_size_per_group: number;
}

// ── CLT ───────────────────────────────────────────────────────────────────────

export interface CLTRequest {
  distribution: "normal" | "uniform" | "binomial" | "exponential";
  sample_sizes: number[];
  n_simulations: number;
  params?: Record<string, number>;
}

export interface CLTSizeResult {
  sample_size: number;
  sample_means: number[];
  empirical_mean: number;
  empirical_std: number;
  theoretical_mean: number;
  theoretical_std: number;
}

export interface CLTResponse {
  distribution: string;
  population_mean: number;
  population_variance: number;
  results: CLTSizeResult[];
}

// ── Clustering ────────────────────────────────────────────────────────────────

export interface PersonaRequest {
  n_users: number;
  k: number;
}

export interface PersonaInfo {
  label: string;
  size: number;
  means: Record<string, number>;
}

export interface UserRecord extends Record<string, number | string> {
  cluster: number;
}

export interface PersonaResponse {
  users: UserRecord[];
  personas: Record<string, PersonaInfo>;
  n_users: number;
  k: number;
}

export interface KMeansRequest {
  data: number[][];
  k: number;
  max_iterations?: number;
}

export interface KMeansResponse {
  labels: number[];
  centroids: number[][];
  inertia: number;
  silhouette_score: number | null;
  k: number;
  n_samples: number;
}

// ── Experiments ───────────────────────────────────────────────────────────────

export interface Variant {
  id: string;
  name: string;
  is_control: boolean;
  clicks: number;
  impressions: number;
  conversion_rate: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string | null;
  status: "draft" | "running" | "paused" | "completed";
  hypothesis: string | null;
  baseline_rate: number | null;
  min_detectable_effect: number | null;
  confidence_level: number;
  created_at: string | null;
  updated_at: string | null;
  variants: Variant[];
}

export interface UserEvent {
  id: string;
  variant_id: string;
  user_identifier: string | null;
  event_type: "click" | "page_view" | "conversion" | "bounce";
  value: number | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

export interface WebSocketMessage {
  experiment_id: string;
  variant_name: "control" | "variant";
  event_type: string;
  cumulative_clicks: number;
  cumulative_impressions: number;
  conversion_rate: number;
  timestamp: string;
}
