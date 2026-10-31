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
}

export interface ModelInfo {
  model_name: string;
  profile_name?: string | null;
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
}
