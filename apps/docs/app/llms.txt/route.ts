import { source } from '@/lib/source';

export function GET(request: Request) {
  const origin = new URL(request.url).origin;
  const pages = source
    .getPages()
    .map((page) => {
      const rawUrl = page.url.replace(/^\/docs/, '/markdown');
      return `- [${page.data.title}](${origin}${rawUrl}): ${page.data.description ?? ''}`;
    })
    .join('\n');
  const body = `# Rungent\n\nTyped embedded agent runtime for Python applications.\n\n## AI loading order\n\n1. Read /llms-full.txt for complete context when possible.\n2. Read /docs/ai-reference for the compact public API contract.\n3. Read /docs/baseline before implementing application tools.\n\n## Documentation\n\n${pages}\n`;
  return new Response(body, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
