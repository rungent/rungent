import { pageText, source } from '@/lib/source';

export async function GET() {
  const pages = await Promise.all(source.getPages().map(pageText));
  return new Response(pages.join('\n\n---\n\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}

