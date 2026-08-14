import { describe, expect, it, vi } from 'vitest';

import { RungentClient, RungentRequestError } from './client';

describe('RungentClient detached Run contract', () => {
  it('creates an idempotent Run and follows its persisted event stream', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        Response.json({ run_id: 'run-1', status: 'queued' }, { status: 202 }),
      )
      .mockResolvedValueOnce(
        new Response(
          'data: {"v":1,"id":"e1","seq":1,"type":"run.completed","session_id":"s1","run_id":"run-1","created_at":"2026-08-06T00:00:00Z","data":{}}\n\n',
          { status: 200 },
        ),
      );
    const client = new RungentClient({ baseUrl: '/assistant', fetch: fetcher });

    const run = await client.createRun('s1', 'Plan', { idempotencyKey: 'request-1' });
    const events: string[] = [];
    await client.streamRunEvents(run.run_id, 0, {
      onEvent: (event) => events.push(event.type),
    });

    expect(run).toEqual({ run_id: 'run-1', status: 'queued' });
    expect(new Headers(fetcher.mock.calls[0]?.[1]?.headers).get('Idempotency-Key')).toBe(
      'request-1',
    );
    expect(fetcher.mock.calls[1]?.[0]).toBe('/assistant/runs/run-1/events/stream?after_seq=0');
    expect(events).toEqual(['run.completed']);
  });

  it('lists and titles sessions', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json([{ id: 's1', title: 'Tokyo', agent_name: 'trip' }]))
      .mockResolvedValueOnce(Response.json({ id: 's1', title: 'Kyoto' }));
    const client = new RungentClient({ baseUrl: '/assistant', fetch: fetcher });

    await expect(client.listSessions()).resolves.toEqual([
      { id: 's1', title: 'Tokyo', agent_name: 'trip' },
    ]);
    await expect(client.updateSession('s1', { title: 'Kyoto' })).resolves.toEqual({
      id: 's1',
      title: 'Kyoto',
    });
    expect(fetcher.mock.calls[0]?.[0]).toBe('/assistant/sessions');
    expect(fetcher.mock.calls[1]?.[0]).toBe('/assistant/sessions/s1');
    expect(fetcher.mock.calls[1]?.[1]).toMatchObject({ method: 'PATCH' });
  });

  it('preserves structured active Run conflicts', async () => {
    const detail = {
      detail: { code: 'active_run_conflict', run_id: 'run-active', status: 'running' },
    };
    const client = new RungentClient({
      fetch: vi.fn<typeof fetch>().mockResolvedValue(Response.json(detail, { status: 409 })),
    });

    await expect(client.createRun('s1', 'Plan')).rejects.toMatchObject({
      status: 409,
      detail,
    } satisfies Partial<RungentRequestError>);
  });
});
