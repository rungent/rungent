import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://rungent.github.io').replace(
  /\/$/,
  '',
);
const docsDir = path.resolve(import.meta.dirname, '../../../docs');
const publicDir = path.resolve(import.meta.dirname, '../public');
const markdownDir = path.join(publicDir, 'markdown');

function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { title: '', description: '', body: raw.trim() };
  const meta = Object.fromEntries(
    match[1]
      .split('\n')
      .map((line) => line.match(/^(\w+):\s*(.*)$/))
      .filter(Boolean)
      .map(([, key, value]) => [key, value.replace(/^["']|["']$/g, '')]),
  );
  return {
    title: meta.title ?? '',
    description: meta.description ?? '',
    body: match[2].trim(),
  };
}

function markdownHref(pageId) {
  return `${siteUrl}/markdown/${pageId}.md`;
}

function docsHref(pageId) {
  return pageId === 'index' ? `${siteUrl}/docs/` : `${siteUrl}/docs/${pageId}/`;
}

const meta = JSON.parse(await readFile(path.join(docsDir, 'meta.json'), 'utf8'));
const pages = [];
for (const pageId of meta.pages) {
  const raw = await readFile(path.join(docsDir, `${pageId}.mdx`), 'utf8');
  const parsed = parseFrontmatter(raw);
  pages.push({ pageId, ...parsed });
}

await rm(markdownDir, { recursive: true, force: true });
await mkdir(markdownDir, { recursive: true });

for (const page of pages) {
  const text = `# ${page.title}\n\n${page.body}\n`;
  await writeFile(path.join(markdownDir, `${page.pageId}.md`), text);
}

const index = `# Rungent

Typed embedded agent runtime for Python applications.

## AI loading order

1. Read ${siteUrl}/llms-full.txt for complete context when possible.
2. Read ${siteUrl}/docs/ai-reference/ for the compact public API contract.
3. Read ${siteUrl}/docs/baseline/ before implementing application tools.

## Documentation

${pages.map((page) => `- [${page.title}](${markdownHref(page.pageId)}): ${page.description}`).join('\n')}
`;

await writeFile(path.join(publicDir, 'llms.txt'), index);
await writeFile(
  path.join(publicDir, 'llms-full.txt'),
  `${pages.map((page) => `# ${page.title}\n\n${page.body}`).join('\n\n---\n\n')}\n`,
);

const sitemapUrls = [`${siteUrl}/`, ...pages.map((page) => docsHref(page.pageId))];
await writeFile(
  path.join(publicDir, 'sitemap.xml'),
  `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapUrls.map((loc) => `  <url><loc>${loc}</loc></url>`).join('\n')}
</urlset>
`,
);
await writeFile(
  path.join(publicDir, 'robots.txt'),
  `User-agent: *
Allow: /

Sitemap: ${siteUrl}/sitemap.xml
`,
);
