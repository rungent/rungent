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
    { text: 'LLM context', url: '/llms.txt', external: true },
    { text: 'Full Markdown', url: '/llms-full.txt', external: true },
  ],
};

