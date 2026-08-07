import type { Activity, RungentEvent, RungentState, Interaction } from './types';

export const initialRungentState: RungentState = {
  status: 'idle',
  output: '',
  activities: [],
};

function updateActivity(
  activities: readonly Activity[],
  id: string,
  update: (current: Activity | undefined) => Activity,
): readonly Activity[] {
  const current = activities.find((item) => item.id === id);
  const next = update(current);
  return current
    ? activities.map((item) => (item.id === id ? next : item))
    : [...activities, next];
}

function settleActivities(
  activities: readonly Activity[],
  terminalStatus: 'completed' | 'failed',
): readonly Activity[] {
  return activities.map((activity) => {
    if (activity.kind === 'model' && activity.status === 'running') {
      return { ...activity, status: terminalStatus };
    }
    if (activity.kind === 'tool' && activity.status === 'running') {
      return { ...activity, status: terminalStatus };
    }
    if (activity.kind === 'progress' && activity.status === 'running') {
      return { ...activity, status: terminalStatus };
    }
    if (activity.kind === 'interaction' && activity.status === 'waiting') {
      return { ...activity, status: terminalStatus };
    }
    if (activity.kind === 'external_task' && activity.status === 'running') {
      return { ...activity, status: terminalStatus };
    }
    return activity;
  });
}

export function failRungentState(
  state: RungentState,
  error = 'Run failed',
  errorCode?: string,
  retryable?: boolean,
): RungentState {
  return {
    ...state,
    status: 'failed',
    interaction: undefined,
    activities: settleActivities(state.activities, 'failed'),
    error,
    errorCode,
    retryable,
  };
}

function applyRungentEvent(state: RungentState, event: RungentEvent): RungentState {
  switch (event.type) {
    case 'run.started':
      return { ...initialRungentState, status: 'running', runId: event.run_id };
    case 'model.started': {
      const step = Number(event.data.step);
      const id = `model:${step}`;
      return {
        ...state,
        output: '',
        activities: updateActivity(state.activities, id, () => ({
          id,
          kind: 'model',
          step,
          status: 'running',
          createdAt: event.created_at,
        })),
      };
    }
    case 'model.retrying': {
      const step = Number(event.data.step);
      const id = `model:${step}`;
      return {
        ...state,
        activities: updateActivity(state.activities, id, (current) => ({
          id,
          kind: 'model',
          step,
          status: 'running',
          retry: Number(event.data.retry),
          maxRetries: Number(event.data.max_retries),
          retryReason: String(event.data.reason ?? 'network'),
          createdAt: current?.createdAt ?? event.created_at,
        })),
      };
    }
    case 'model.completed': {
      const step = Number(event.data.step);
      const id = `model:${step}`;
      return {
        ...state,
        activities: updateActivity(state.activities, id, (current) => ({
          id,
          kind: 'model',
          step,
          status: 'completed',
          outcome:
            event.data.outcome === 'final'
              ? 'final'
              : event.data.outcome === 'empty'
                ? 'empty'
                : 'tools',
          createdAt: current?.createdAt ?? event.created_at,
        })),
      };
    }
    case 'activity.updated': {
      const id = String(event.data.id ?? `progress:${event.seq}`);
      return {
        ...state,
        activities: updateActivity(state.activities, id, (current) => ({
          id,
          kind: 'progress',
          step: Number(event.data.step ?? 0),
          status:
            event.data.status === 'running' || event.data.status === 'failed'
              ? event.data.status
              : 'completed',
          message: String(event.data.message ?? ''),
          public: event.data.public,
          createdAt: current?.createdAt ?? event.created_at,
        })),
      };
    }
    case 'message.delta':
      return { ...state, output: state.output + String(event.data.delta ?? '') };
    case 'message.reset':
      return { ...state, output: '' };
    case 'message.completed':
      return { ...state, output: String(event.data.content ?? state.output) };
    case 'run.waiting_input':
      return { ...state, status: 'waiting_input' };
    case 'run.waiting_external':
      return { ...state, status: 'waiting_external' };
    case 'interaction.requested': {
      const interaction = event.data as unknown as Interaction;
      return {
        ...state,
        interaction,
        activities: updateActivity(state.activities, interaction.id, () => ({
          id: interaction.id,
          kind: 'interaction',
          status: 'waiting',
          interaction,
          createdAt: event.created_at,
        })),
      };
    }
    case 'interaction.resolved': {
      const id = String(event.data.interaction_id);
      return {
        ...state,
        status: 'running',
        interaction: undefined,
        activities: state.activities.map((activity) =>
          activity.id === id && activity.kind === 'interaction'
            ? { ...activity, status: 'completed' }
            : activity,
        ),
      };
    }
    case 'tool.started': {
      const callId = String(event.data.call_id);
      return {
        ...state,
        activities: updateActivity(state.activities, callId, () => ({
          id: callId,
          kind: 'tool',
          callId,
          name: String(event.data.name),
          title: event.data.title ? String(event.data.title) : undefined,
          status: 'running',
          createdAt: event.created_at,
        })),
      };
    }
    case 'tool.completed':
    case 'tool.failed': {
      const callId = String(event.data.call_id);
      const status = event.type === 'tool.completed' ? 'completed' : 'failed';
      return {
        ...state,
        activities: updateActivity(state.activities, callId, (current) => ({
          id: callId,
          kind: 'tool',
          callId,
          name: String(event.data.name ?? (current?.kind === 'tool' ? current.name : '')),
          title: event.data.title
            ? String(event.data.title)
            : current?.kind === 'tool'
              ? current.title
              : undefined,
          status,
          code: event.data.code ? String(event.data.code) : undefined,
          message: event.data.message ? String(event.data.message) : undefined,
          public: event.data.public,
          deduplicated: event.data.deduplicated === true,
          createdAt: current?.createdAt ?? event.created_at,
        })),
      };
    }
    case 'external_task.started':
    case 'external_task.progress':
    case 'external_task.completed':
    case 'external_task.failed':
    case 'external_task.cancelled': {
      const taskId = String(event.data.task_id);
      const id = `external:${taskId}`;
      const status =
        event.type === 'external_task.completed'
          ? 'completed'
          : event.type === 'external_task.failed' || event.type === 'external_task.cancelled'
            ? 'failed'
            : 'running';
      return {
        ...state,
        activities: updateActivity(state.activities, id, (current) => ({
          id,
          kind: 'external_task',
          taskId,
          callId: String(event.data.call_id ?? ''),
          status,
          message: event.data.message
            ? String(event.data.message)
            : current?.kind === 'external_task'
              ? current.message
              : undefined,
          public:
            event.data.public !== undefined
              ? event.data.public
              : current?.kind === 'external_task'
                ? current.public
                : undefined,
          createdAt: current?.createdAt ?? event.created_at,
        })),
      };
    }
    case 'run.completed':
      return {
        ...state,
        status: 'completed',
        interaction: undefined,
        activities: settleActivities(state.activities, 'completed'),
      };
    case 'run.failed':
      return failRungentState(
        state,
        String(event.data.error ?? 'Run failed'),
        event.data.code ? String(event.data.code) : undefined,
        event.data.retryable === true,
      );
    case 'run.cancelled':
      return {
        ...state,
        status: 'cancelled',
        interaction: undefined,
        activities: settleActivities(state.activities, 'failed'),
        error: String(event.data.error ?? 'Run cancelled'),
      };
    default:
      return state;
  }
}

export function reduceRungentEvent(state: RungentState, event: RungentEvent): RungentState {
  if (state.runId === event.run_id && state.lastSeq !== undefined && event.seq <= state.lastSeq) {
    return state;
  }
  return { ...applyRungentEvent(state, event), lastSeq: event.seq };
}
