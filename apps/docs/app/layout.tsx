import type { Metadata } from 'next';
import { RootProvider } from 'fumadocs-ui/provider/next';

import { siteUrl } from '@/lib/site';

import './global.css';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: 'Rungent', template: '%s · Rungent' },
  description: 'Typed agent runtime for Python applications',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
