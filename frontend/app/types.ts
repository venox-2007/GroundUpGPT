export type Role = "user" | "assistant";

export interface MessageMetrics {
  promptTokens?: number;
  generatedTokens?: number;
  totalTokens?: number;
  activeContextTokens?: number;
  contextLimit?: number;
  contextCropped?: boolean;
  elapsedTimeSec?: number;
  tokensPerSecond?: number;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  isStreaming?: boolean;
  metrics?: MessageMetrics;
  error?: string;
  modelProfile?: string;
  mode?: "native" | "assisted";
  origin?: "neural" | "safe_calculator" | "local_fact" | "tool_error" | string;
  toolName?: string;
  factId?: string;
  provenance?: string;
  evaluatedExpression?: string;
  errorCode?: string;
  modelGenerated?: boolean;
  details?: Record<string, any>;
}

export interface ToolResultEventPayload {
  request_id: string;
  mode: "native" | "assisted";
  origin: "neural" | "safe_calculator" | "local_fact" | "tool_error" | string;
  answer_text: string;
  model_generated: boolean;
  model_profile?: string | null;
  tool_name?: string | null;
  fact_id?: string | null;
  provenance?: string | null;
  evaluated_expression?: string | null;
  stop_reason?: string | null;
  prompt_tokens?: number | null;
  generated_tokens?: number | null;
  total_tokens?: number | null;
  elapsed_time_sec: number;
  tokens_per_second?: number | null;
  error_code?: string | null;
  details?: Record<string, any> | null;
}

export interface ModelInfo {
  model_name: string;
  profile_name?: string | null;
  active_profile?: string | null;
  vocab_size: number;
  block_size: number;
  embed_dim: number;
  num_heads: number;
  num_layers: number;
  weight_tying: boolean;
  total_parameters: number;
  tokenizer_type: string;
  tokenizer_vocab_size: number;
  tokenizer_fingerprint: string;
  device: string;
  checkpoint_step?: number | null;
  checkpoint_val_loss?: number | null;
  original_val_loss?: number | null;
  expanded_val_loss?: number | null;
  checkpoint_perplexity?: number | null;
  global_step?: number | null;
  sft_step?: number | null;
  source_step?: number | null;
  source_val_loss?: number | null;
  is_sft?: boolean;
  model_mode?: string;
  sft_val_loss?: number | null;
  sft_val_perplexity?: number | null;
  lm_val_loss?: number | null;
  lm_val_perplexity?: number | null;
  pretrain_steps?: number | null;
  original_sft_steps?: number | null;
  anchor_sft_steps?: number | null;
  anchor_val_loss?: number | null;
  anchor_val_perplexity?: number | null;
  dolly_val_loss?: number | null;
  dolly_val_perplexity?: number | null;
  transfer_test_loss?: number | null;
  transfer_test_perplexity?: number | null;
  is_anchor_sft?: boolean;
}

export interface ModelProfileEntry {
  name: string;
  checkpoint_path: string;
  tokenizer_path: string;
  checkpoint_exists: boolean;
  tokenizer_exists: boolean;
  is_sft?: boolean;
}

export interface ModelProfilesResponse {
  active_model: string | null;
  active_profile?: string | null;
  profiles: Record<string, ModelProfileEntry>;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
}

export interface GenerationSettings {
  temperature: number;
  top_k: number | null;
  top_p: number | null;
  max_new_tokens: number;
  seed: number | null;
  repetition_penalty?: number | null;
}
