import { describe, expect, it } from 'vitest';

import { parseRungentEvent } from './stream';

describe('parseRungentEvent', () => {
  it('parses a versioned ACS event', () => {
    const event = parseRungentEvent(
      'data: {"v":1,"id":"evt_1","seq":1,"type":"message.delta","session_id":"ses_1","run_id":"run_1","created_at":"2026-08-03T00:00:00Z","data":{"delta":"hi"}}',
    );
    expect(event?.type).toBe('message.delta');
    expect(event?.data.delta).toBe('hi');
  });

  it('rejects an unsupported envelope', () => {
    expect(() => parseRungentEvent('data: {"v":2,"type":"x","seq":1}')).toThrow();
  });
});
