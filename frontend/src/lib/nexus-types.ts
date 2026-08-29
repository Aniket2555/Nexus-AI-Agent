/**
 * TypeScript mirrors of the NEXUS backend's state/response shapes that the
 * custom nexus/ and generative-ui/ components render. These intentionally
 * track the Python source of truth field-for-field rather than being
 * independently designed:
 *
 * - Citation            <- backend/app/rag/processing/citation_builder.py
 * - MemoryContext        <- backend/app/memory/manager.py
 * - RAGChatState (part)  <- backend/app/graphs/states/rag_chat_state.py
 * - SubTask / SpecialistResult / SupervisorState (part)
 *                        <- backend/app/graphs/states/{shared_state,supervisor_state}.py
 * - DocumentRecord / DocumentListResponse
 *                        <- backend/app/models/schemas.py
 * - CodeSpecialistDetails / DataSpecialistDetails
 *                        <- backend/app/graphs/specialists/{code,data}.py's SpecialistResult.details
 */

export type SpecialistType =
  "research" | "code" | "data" | "browser" | "vision" | "report";

export interface Citation {
  source: string;
  page: number | string | null;
  chunk_id: string;
  retrieval_score: number | null;
  rerank_score: number | null;
}

export interface MemoryFact {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  score: number;
}

export interface MemoryContext {
  long_term_facts: MemoryFact[];
  graph_context: MemoryFact[];
}

/** The subset of RAGChatState relevant to the nexus/ panels. */
export interface RAGChatStateSlice {
  citations?: Citation[];
  recalled_memory?: MemoryContext;
  compression_stats?: {
    method: string;
    original_tokens: number;
    compressed_tokens: number;
  };
  quality_score?: number;
  reflection_feedback?: string;
  iteration_count?: number;
}

export interface SubTask {
  id: string;
  description: string;
  specialist: SpecialistType;
  depends_on: string[];
  target: string | null;
}

export interface CodeSpecialistDetails {
  language: string;
  code: string;
  syntax_valid: boolean;
  syntax_error: string | null;
  executed: boolean;
  execution_stdout: string;
  execution_stderr: string;
  execution_timed_out: boolean;
  execution_exit_code: number | null;
  execution_truncated: boolean;
  note: string;
}

export interface DataSpecialistDetails {
  sql_query: string;
  row_count: number;
  rows: Record<string, unknown>[];
  error: string | null;
}

export interface SpecialistResult {
  task_id: string;
  specialist: SpecialistType;
  summary: string;
  success: boolean;
  citations: Citation[];
  details: Record<string, unknown>;
}

/** The subset of SupervisorState relevant to SupervisorPanel/AgentStatusPanel. */
export interface SupervisorStateSlice {
  current_plan?: string | null;
  sub_tasks?: SubTask[];
  agent_results?: SpecialistResult[];
  quality_score?: number;
  iteration_count?: number;
  review_passed?: boolean | null;
  review_feedback?: string;
  final_response?: string | null;
}

export type DocumentStatus = "pending" | "running" | "succeeded" | "failed";

export interface DocumentRecord {
  doc_id: string;
  tenant_id: string;
  filename: string;
  suffix: string;
  content_type: string | null;
  uploaded_at: string;
  status: DocumentStatus;
  chunk_count: number;
  vector_count: number;
  keyword_count: number;
  error: string | null;
  content_size_bytes: number | null;
}

export interface DocumentListResponse {
  tenant_id: string;
  documents: DocumentRecord[];
}

export interface UploadJobResponse {
  job_id: string;
  status: DocumentStatus;
  filename: string;
}
