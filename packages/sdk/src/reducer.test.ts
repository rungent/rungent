import { describe, expect, it } from 'vitest';

import { failRungentState, initialRungentState, reduceRungentEvent } from './reducer';
import type { RungentEvent } from './types';

function event(type: string, seq: number, data: Record<string, unknown> = {}): RungentEvent {
  return {
    v: 1,
    id: `evt-${type}`,
    seq,
    type,
    session_id: 's',
    run_id: 'r',
    created_at: `2026-08-03T00:00:0${seq}Z`,
    data,
  };
}

describe('reduceRungentEvent', () => {
  it('tracks text and pending interactions', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('message.delta', 2, { delta: 'Hello' }));
    state = reduceRungentEvent(
      state,
      event('interaction.requested', 3, {
        id: 'i',
        kind: 'choice',
        prompt: 'Pick',
        options: [],
        multiple: false,
        allow_custom: false,
        tool_call_id: 'c',
      }),
    );
    expect(state.output).toBe('Hello');
    expect(state.interaction?.id).toBe('i');
  });

  it('settles every running activity when a run fails', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('model.started', 2, { step: 1 }));
    state = reduceRungentEvent(
      state,
      event('tool.started', 3, { call_id: 'tool-1', name: 'search' }),
    );
    state = reduceRungentEvent(state, event('run.failed', 4, { error: 'Agent execution failed' }));

    expect(state.status).toBe('failed');
    expect(state.activities.map((activity) => activity.status)).toEqual(['failed', 'failed']);
    expect(state.activities.some((activity) => activity.status === 'running')).toBe(false);
  });

  it('projects stable tool failure codes for product filtering', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(
      state,
      event('tool.failed', 2, {
        call_id: 'bad-args',
        name: 'change_trip',
        code: 'invalid_arguments',
        message: 'Tool arguments failed validation',
      }),
    );

    expect(state.activities[0]).toMatchObject({
      kind: 'tool',
      status: 'failed',
      code: 'invalid_arguments',
    });
  });

  it('projects safe failure metadata for product recovery UI', () => {
    const state = reduceRungentEvent(
      initialRungentState,
      event('run.failed', 1, {
        error: 'The assistant could not finish this request.',
        code: 'model_step_limit_exceeded',
        retryable: true,
      }),
    );

    expect(state.error).toBe('The assistant could not finish this request.');
    expect(state.errorCode).toBe('model_step_limit_exceeded');
    expect(state.retryable).toBe(true);
  });

  it('preserves cancelled as a distinct terminal state', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('model.started', 2, { step: 1 }));
    state = reduceRungentEvent(state, event('run.cancelled', 3));

    expect(state.status).toBe('cancelled');
    expect(state.activities[0]?.status).toBe('failed');
  });

  it('replaces partial output and exposes provider retry progress', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('model.started', 2, { step: 1 }));
    state = reduceRungentEvent(state, event('message.delta', 3, { delta: 'partial' }));
    state = reduceRungentEvent(state, event('message.reset', 4));
    state = reduceRungentEvent(
      state,
      event('model.retrying', 5, {
        step: 1,
        retry: 1,
        max_retries: 3,
        reason: 'network',
      }),
    );

    expect(state.output).toBe('');
    expect(state.activities[0]).toMatchObject({
      kind: 'model',
      status: 'running',
      retry: 1,
      maxRetries: 3,
      retryReason: 'network',
    });
  });

  it('settles activities when the transport fails before a terminal event', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('model.started', 2, { step: 1 }));

    state = failRungentState(state, 'Network interrupted');

    expect(state.status).toBe('failed');
    expect(state.error).toBe('Network interrupted');
    expect(state.activities[0]?.status).toBe('failed');
  });

  it('ignores duplicate or stale events within a run', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('message.delta', 2, { delta: 'Hello' }));
    state = reduceRungentEvent(state, event('message.delta', 2, { delta: 'Hello' }));
    expect(state.output).toBe('Hello');
    expect(state.lastSeq).toBe(2);
  });

  it('projects model, progress, tool, and interaction events in order', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(state, event('model.started', 2, { step: 1 }));
    state = reduceRungentEvent(
      state,
      event('activity.updated', 3, {
        id: 'progress-1',
        step: 1,
        message: 'Checking the trip',
        public: { trip_changed: true },
      }),
    );
    state = reduceRungentEvent(
      state,
      event('tool.started', 4, { call_id: 'tool-1', name: 'move', title: 'Move place' }),
    );
    state = reduceRungentEvent(
      state,
      event('tool.completed', 5, {
        call_id: 'tool-1',
        name: 'move',
        title: 'Move place',
        message: 'Moved',
        public: { trip_changed: true },
        deduplicated: true,
      }),
    );

    expect(state.activities.map((activity) => activity.kind)).toEqual([
      'model',
      'progress',
      'tool',
    ]);
    const tool = state.activities[2];
    const progress = state.activities[1];
    expect(progress?.kind === 'progress' && progress.public).toEqual({ trip_changed: true });
    expect(tool?.kind === 'tool' && tool.public).toEqual({ trip_changed: true });
    expect(tool?.kind === 'tool' && tool.deduplicated).toBe(true);
  });

  it('updates one running progress activity and settles it at the terminal event', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(
      state,
      event('activity.updated', 2, {
        id: 'primary',
        message: 'Preparing',
        status: 'running',
        public: { stage: 'planning', elapsed_seconds: 3 },
      }),
    );
    state = reduceRungentEvent(
      state,
      event('activity.updated', 3, {
        id: 'primary',
        message: 'Still preparing',
        status: 'running',
        public: { stage: 'planning', elapsed_seconds: 8 },
      }),
    );

    expect(state.activities).toHaveLength(1);
    expect(state.activities[0]).toMatchObject({
      id: 'primary',
      kind: 'progress',
      status: 'running',
      message: 'Still preparing',
    });

    state = reduceRungentEvent(
      state,
      event('activity.updated', 4, {
        id: 'primary',
        message: 'Prepared',
        status: 'completed',
        public: { stage: 'prepared' },
      }),
    );
    expect(state.activities).toHaveLength(1);
    expect(state.activities[0]).toMatchObject({
      id: 'primary',
      status: 'completed',
      message: 'Prepared',
    });

    state = reduceRungentEvent(state, event('run.completed', 5));
    expect(state.activities[0]).toMatchObject({ status: 'completed' });
  });

  it('projects a durable external task through waiting, progress, and completion', () => {
    let state = reduceRungentEvent(initialRungentState, event('run.started', 1));
    state = reduceRungentEvent(
      state,
      event('external_task.started', 2, {
        task_id: 'job-1',
        call_id: 'tool-1',
        message: 'Calculating routes',
        public: { completed: 0, total: 2 },
      }),
    );
    state = reduceRungentEvent(state, event('run.waiting_external', 3));
    state = reduceRungentEvent(
      state,
      event('external_task.progress', 4, {
        task_id: 'job-1',
        call_id: 'tool-1',
        message: 'Calculated 1 of 2 routes',
        public: { completed: 1, total: 2 },
      }),
    );

    expect(state.status).toBe('waiting_external');
    expect(state.activities[0]).toMatchObject({
      kind: 'external_task',
      status: 'running',
      message: 'Calculated 1 of 2 routes',
      public: { completed: 1, total: 2 },
    });

    state = reduceRungentEvent(
      state,
      event('external_task.completed', 5, {
        task_id: 'job-1',
        call_id: 'tool-1',
        message: 'Routes updated',
      }),
    );
    expect(state.activities[0]).toMatchObject({ status: 'completed' });
  });
});
