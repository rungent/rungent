import { consumeRungentStream } from './stream';
import type { RungentEvent, Run, Session } from './types';

export interface RungentClientOptions {
  readonly baseUrl?: string;
  readonly fetch?: typeof globalThis.fetch;
  readonly headers?: HeadersInit | (() => HeadersInit | Promise<HeadersInit>);
}

export interface CreateSessionInput {
  readonly agent?: string;
  readonly resource?: Record<string, unknown>;
}

export interface StreamOptions {
  readonly signal?: AbortSignal;
  readonly onEvent: (event: RungentEvent) => void;
}

export interface RunHandle {
  readonly run_id: string;
  readonly status: Run['status'];
}

export class RungentRequestError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(`Rungent request failed (${status})`);
  }
}

export class RungentClient {
  private readonly baseUrl: string;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly headerSource?: RungentClientOptions['headers'];

  constructor(options: RungentClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? '').replace(/\/$/, '');
    this.fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.headerSource = options.headers;
  }

  private async headers(): Promise<Headers> {
    const source =
      typeof this.headerSource === 'function' ? await this.headerSource() : this.headerSource;
    const headers = new Headers(source);
    headers.set('Content-Type', 'application/json');
    return headers;
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetcher(`${this.baseUrl}${path}`, {
      ...init,
      headers: await this.headers(),
    });
    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = undefined;
      }
      throw new RungentRequestError(response.status, detail);
    }
    return response.json() as Promise<T>;
  }

  async createSession(input: CreateSessionInput = {}): Promise<Session> {
    return this.json('/sessions', { method: 'POST', body: JSON.stringify(input) });
  }

  async getSession(sessionId: string): Promise<Session> {
    return this.json(`/sessions/${encodeURIComponent(sessionId)}`);
  }

  async listRuns(sessionId: string): Promise<readonly Run[]> {
    return this.json(`/sessions/${encodeURIComponent(sessionId)}/runs`);
  }

  async listEvents(runId: string, afterSeq = 0): Promise<readonly RungentEvent[]> {
    return this.json(
      `/runs/${encodeURIComponent(runId)}/events?after_seq=${encodeURIComponent(afterSeq)}`,
    );
  }

  async cancel(runId: string): Promise<{ readonly id: string; readonly status: string }> {
    return this.json(`/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' });
  }

  async createRun(
    sessionId: string,
    input: string,
    options: { readonly idempotencyKey?: string; readonly signal?: AbortSignal } = {},
  ): Promise<RunHandle> {
    const headers = await this.headers();
    if (options.idempotencyKey) headers.set('Idempotency-Key', options.idempotencyKey);
    const response = await this.fetcher(
      `${this.baseUrl}/sessions/${encodeURIComponent(sessionId)}/runs`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify({ input }),
        signal: options.signal,
      },
    );
    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = undefined;
      }
      throw new RungentRequestError(response.status, detail);
    }
    return response.json() as Promise<RunHandle>;
  }

  async streamRunEvents(
    runId: string,
    afterSeq: number,
    options: StreamOptions,
  ): Promise<void> {
    const response = await this.fetcher(
      `${this.baseUrl}/runs/${encodeURIComponent(runId)}/events/stream?after_seq=${encodeURIComponent(afterSeq)}`,
      { headers: await this.headers(), signal: options.signal },
    );
    if (!response.ok) throw new RungentRequestError(response.status, undefined);
    if (!response.body) throw new Error('Rungent stream has no response body');
    await consumeRungentStream(response.body, options.onEvent);
  }

  async respond(
    runId: string,
    interactionId: string,
    value: unknown,
    options: { readonly signal?: AbortSignal } = {},
  ): Promise<RunHandle> {
    const response = await this.fetcher(
      `${this.baseUrl}/runs/${encodeURIComponent(runId)}/responses`,
      {
      method: 'POST',
      headers: await this.headers(),
      body: JSON.stringify({ interaction_id: interactionId, value }),
      signal: options.signal,
      },
    );
    if (!response.ok) throw new RungentRequestError(response.status, undefined);
    return response.json() as Promise<RunHandle>;
  }
}
