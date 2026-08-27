export type RunStatus =
  | 'queued'
  | 'running'
  | 'waiting_input'
  | 'waiting_external'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface RungentEvent<T extends string = string, D = Record<string, unknown>> {
  readonly v: 1;
  readonly id: string;
  readonly seq: number;
  readonly type: T;
  readonly session_id: string;
  readonly run_id: string;
  readonly created_at: string;
  readonly data: D;
}

export interface InteractionOption {
  readonly id: string;
  readonly label: string;
  readonly description?: string | null;
  readonly recommended: boolean;
}

export interface InteractionQuestion {
  readonly id: string;
  readonly prompt: string;
  readonly kind: 'text' | 'choice';
  readonly options: readonly InteractionOption[];
  readonly multiple: boolean;
  readonly allow_custom: boolean;
  readonly required: boolean;
}

export interface Interaction {
  readonly id: string;
  readonly kind: 'text' | 'choice' | 'form' | 'approval';
  readonly prompt: string;
  readonly options: readonly InteractionOption[];
  readonly questions: readonly InteractionQuestion[];
  readonly multiple: boolean;
  readonly allow_custom: boolean;
  readonly allow_skip: boolean;
  readonly skip_label?: string | null;
  readonly tool_call_id: string;
  readonly tool?: {
    readonly name: string;
    readonly title: string;
    readonly arguments: Record<string, unknown>;
  };
}

export interface ChoiceResponse {
  readonly selected: readonly string[];
  readonly custom?: string;
  readonly skipped?: boolean;
}

export interface FormResponse {
  readonly answers: Readonly<Record<string, string | ChoiceResponse>>;
}

export interface SessionMessage {
  readonly id: string;
  readonly role: 'system' | 'user' | 'assistant' | 'tool';
  readonly content: string;
  readonly created_at: string;
}

export interface ContextUsageCategory {
  readonly id: string;
  readonly label: string;
  readonly tokens: number;
}

export interface ContextUsage {
  readonly budget_tokens: number;
  readonly used_tokens: number;
  readonly used_percent: number;
  readonly categories: readonly ContextUsageCategory[];
  readonly source: 'estimated' | 'provider';
  readonly estimated_tokens?: number;
  readonly prompt_tokens?: number;
}

export interface Session {
  readonly id: string;
  readonly agent_name: string;
  readonly subject_id: string;
  readonly tenant_id?: string | null;
  readonly title?: string | null;
  readonly resource: Record<string, unknown>;
  readonly created_at: string;
  readonly updated_at: string;
  readonly messages?: readonly SessionMessage[];
}

export interface Run {
  readonly id: string;
  readonly session_id: string;
  readonly status: RunStatus;
  readonly event_seq: number;
  readonly model_steps: number;
  readonly created_at: string;
  readonly updated_at: string;
}

export interface ModelActivity {
  readonly id: string;
  readonly kind: 'model';
  readonly step: number;
  readonly status: 'running' | 'completed' | 'failed';
  readonly outcome?: 'tools' | 'final' | 'empty';
  readonly retry?: number;
  readonly maxRetries?: number;
  readonly retryReason?: string;
  readonly createdAt: string;
}

export interface ProgressActivity {
  readonly id: string;
  readonly kind: 'progress';
  readonly step: number;
  readonly status: 'running' | 'completed' | 'failed';
  readonly message: string;
  readonly public?: unknown;
  readonly createdAt: string;
}

export interface ToolActivity {
  readonly id: string;
  readonly kind: 'tool';
  readonly callId: string;
  readonly name: string;
  readonly title?: string;
  readonly status: 'running' | 'completed' | 'failed';
  readonly code?: string;
  readonly message?: string;
  readonly public?: unknown;
  readonly deduplicated?: boolean;
  readonly createdAt: string;
}

export interface InteractionActivity {
  readonly id: string;
  readonly kind: 'interaction';
  readonly status: 'waiting' | 'completed' | 'failed';
  readonly interaction: Interaction;
  readonly createdAt: string;
}

export interface ExternalTaskActivity {
  readonly id: string;
  readonly kind: 'external_task';
  readonly taskId: string;
  readonly callId: string;
  readonly status: 'running' | 'completed' | 'failed';
  readonly message?: string;
  readonly public?: unknown;
  readonly createdAt: string;
}

export type Activity =
  | ModelActivity
  | ProgressActivity
  | ToolActivity
  | InteractionActivity
  | ExternalTaskActivity;

export interface RungentState {
  readonly status: RunStatus | 'idle';
  readonly runId?: string;
  readonly output: string;
  readonly interaction?: Interaction;
  readonly activities: readonly Activity[];
  readonly error?: string;
  readonly errorCode?: string;
  readonly retryable?: boolean;
  readonly lastSeq?: number;
}
