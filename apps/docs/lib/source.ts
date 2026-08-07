import { docs } from 'collections/server';
import { loader } from 'fumadocs-core/source';

export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
});

export async function pageText(page: ReturnType<typeof source.getPages>[number]) {
  return `# ${page.data.title}\n\n${await page.data.getText('processed')}`;
}

