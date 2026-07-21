// types/index.ts — Shared TypeScript types for the entire frontend
// These mirror the Pydantic response models in the FastAPI backend.

export interface User {
  user_id: string
  username: string
  email: string
  created_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  username: string
  user_id: string
  message: string
}

export interface Source {
  file: string
  page: number
  total_pages: number | string
  chunk_index: number | string
  upload_time: string
  preview: string
  relevance_score?: number    // CrossEncoder reranking score (when reranking is enabled)
}

export type LLMMode = 'auto' | 'online' | 'offline'

export interface Message {
  id: string                  // client-generated UUID for React key prop
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
  isStreaming?: boolean       // true while the assistant is still typing
  error?: boolean             // true if the response was an error
  timestamp: Date
  provider?: 'ollama' | 'groq'  // which LLM provider answered (assistant only)
}

export interface DocumentInfo {
  filename: string
  uploaded_at: string
  pages: number
  chunks: number
  size_kb: number
}

export interface DocumentLibrary {
  total_documents: number
  total_pages: number
  total_chunks: number
  documents: DocumentInfo[]
}

export interface StatusResponse {
  ready: boolean
  username: string
  total_documents: number
  total_pages: number
  total_chunks: number
  documents: DocumentInfo[]
  active_sessions: number
}

export interface UploadResponse {
  status: string
  file: string
  pages: number
  chunks: number
  was_duplicate: boolean
  message: string
  total_documents: number
}

// ── Conversation types ────────────────────────────────────────────────────────

/** Matches backend list_conversations() response shape */
export interface ConversationSummary {
  id: string
  title: string
  updated_at: string
  message_count: number
  last_message_preview: string
}

/** Full conversation with messages, from GET /conversations/{id} */
export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages: ConversationMessage[]
}

/** Backend message shape (differs from frontend Message — mapped on load) */
export interface ConversationMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources: Source[] | null
  llm_provider_used: string | null
  created_at: string
}

// ── SSE event types streamed from POST /stream ────────────────────────────────

export type StreamEvent =
  | { type: 'token';    content: string }
  | { type: 'metadata'; sources: Source[]; conversation_id: string; provider: string }
  | { type: 'error';    content: string }

// ── Benchmark types ──────────────────────────────────────────────────────────

export interface BenchmarkItem {
  id: string
  run_group_id: string
  user_id: string
  model: string
  question: string
  answer: string
  latency_seconds: number
  token_count: number
  created_at: string
}

export interface ModelSummary {
  avg_latency: number
  min_latency?: number
  max_latency?: number
  total_questions: number
}

export interface BenchmarkGroup {
  run_group_id: string
  created_at: string
  models: string[]
  summary: Record<string, ModelSummary>
  results: BenchmarkItem[]
}

// ── Analytics types ──────────────────────────────────────────────────────────

export interface TopQuestion {
  question: string
  count: number
}

export interface MostCitedDoc {
  document: string
  citations: number
  unique_pages_cited: number
}

export interface TimelineDataPoint {
  date: string
  questions: number
}

export interface ProviderSplit {
  offline: number
  online: number
  other: number
  total: number
}

export interface AnalyticsSummary {
  total_conversations: number
  total_answers: number
  avg_latency_seconds: number
  top_questions: TopQuestion[]
  most_cited_documents: MostCitedDoc[]
  usage_over_time: TimelineDataPoint[]
  provider_split: ProviderSplit
}


