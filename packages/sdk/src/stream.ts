import type { RungentEvent } from './types';

function parseEnvelope(raw: string): RungentEvent | null {
  const data = raw
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');
  if (!data) return null;
  const parsed: unknown = JSON.parse(data);
  if (!parsed || typeof parsed !== 'object') throw new Error('Invalid Rungent event');
  const event = parsed as Partial<RungentEvent>;
  if (event.v !== 1 || typeof event.type !== 'string' || typeof event.seq !== 'number') {
    throw new Error('Unsupported Rungent event envelope');
  }
  return event as RungentEvent;
}

function flush(buffer: string, onEvent: (event: RungentEvent) => void): string {
  const chunks = buffer.split(/\r?\n\r?\n/);
  const remaining = chunks.pop() ?? '';
  for (const chunk of chunks) {
    const event = parseEnvelope(chunk);
    if (event) onEvent(event);
  }
  return remaining;
}

export async function consumeRungentStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: RungentEvent) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = flush(buffer + decoder.decode(value, { stream: true }), onEvent);
  }
  buffer = flush(buffer + decoder.decode(), onEvent);
  if (buffer.trim()) {
    const event = parseEnvelope(buffer);
    if (event) onEvent(event);
  }
}

export { parseEnvelope as parseRungentEvent };

