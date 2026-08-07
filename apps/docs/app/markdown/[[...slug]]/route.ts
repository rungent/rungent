import { pageText, source } from '@/lib/source';

export async function GET(
  _request: Request,
  context: { params: Promise<{ slug?: string[] }> },
) {
  const { slug = [] } = await context.params;
  const page = source.getPage(slug);
  if (!page) return new Response('Not found', { status: 404 });

  return new Response(await pageText(page), {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
}
