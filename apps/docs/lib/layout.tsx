import { CopyForLlm } from '@/components/copy-for-llm';
import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <span className="brand">
        <span className="brand-mark">C</span>
        <span>Rungent</span>
      </span>
    ),
  },
  links: [
    { type: 'button', text: 'Docs', url: '/docs', secondary: true },
    { type: 'custom', secondary: true, children: <CopyForLlm /> },
  ],
};
