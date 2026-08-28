import { HomeLayout } from 'fumadocs-ui/layouts/home';
import Link from 'next/link';

import { baseOptions } from '@/lib/layout';

const groups = [
  {
    title: 'Start here',
    links: [
      { href: '/docs', title: 'Overview', description: 'Scope, design rules, and what stays out.' },
      {
        href: '/docs/architecture',
        title: 'Architecture',
        description: 'Package map and host boundaries.',
      },
      {
        href: '/docs/getting-started',
        title: 'Getting started',
        description: 'Install and run the smallest useful agent.',
      },
    ],
  },
  {
    title: 'Integrate',
    links: [
      { href: '/docs/python-api', title: 'Python API', description: 'Agent, tools, runtime, models.' },
      { href: '/docs/fastapi', title: 'FastAPI', description: 'HTTP and SSE mounting.' },
      { href: '/docs/mcp', title: 'MCP', description: 'In-process adapter for the same tools.' },
      { href: '/docs/frontend', title: 'Frontend SDK', description: 'Consume the activity stream.' },
    ],
  },
  {
    title: 'Runtime',
    links: [
      {
        href: '/docs/external-agents',
        title: 'External agents',
        description: 'Remote MCP plus Skill, host-owned OAuth.',
      },
      {
        href: '/docs/interactions',
        title: 'Interactions',
        description: 'Clarify, approve, pause, and resume.',
      },
      { href: '/docs/persistence', title: 'Persistence', description: 'SQLAlchemy store port.' },
    ],
  },
  {
    title: 'Quality',
    links: [
      { href: '/docs/baseline', title: 'Baseline', description: 'Harness tests and application baselines.' },
      { href: '/docs/roasea', title: 'Roasea', description: 'Reference integration walkthrough.' },
      {
        href: '/docs/ai-reference',
        title: 'AI reference',
        description: 'Compact contract for coding agents.',
      },
    ],
  },
];

export default function HomePage() {
  return (
    <HomeLayout {...baseOptions}>
      <div className="home">
        <p className="eyebrow">Embedded agent runtime</p>
        <h1>Rungent</h1>
        <p className="lede">
          A small, typed agent runtime that embeds into an existing Python application. It
          standardizes tools, the harness loop, durable sessions, approvals, and SSE activity —
          without a sidecar, skill router, or prescribed chat UI.
        </p>
        <div className="home-actions">
          <Link className="home-button" href="/docs/getting-started">
            Get started
          </Link>
          <a className="home-button home-button-ghost" href="https://github.com/rungent/rungent">
            GitHub
          </a>
        </div>
        <pre className="home-install">
          <code>pip install rungent{'\n'}npm install @rungent/sdk</code>
        </pre>

        {groups.map((group) => (
          <section className="home-group" key={group.title}>
            <h2>{group.title}</h2>
            <div className="home-cards">
              {group.links.map((link) => (
                <Link className="home-card" href={link.href} key={link.href}>
                  <strong>{link.title}</strong>
                  <span>{link.description}</span>
                </Link>
              ))}
            </div>
          </section>
        ))}

        <section className="home-group">
          <h2>For LLMs</h2>
          <p className="home-note">
            Same docs, machine-readable. Prefer the full dump when the context window allows.
          </p>
          <div className="home-cards">
            <a className="home-card" href="/llms.txt">
              <strong>llms.txt</strong>
              <span>Index of every page with raw Markdown links.</span>
            </a>
            <a className="home-card" href="/llms-full.txt">
              <strong>llms-full.txt</strong>
              <span>All documentation concatenated for one-shot loading.</span>
            </a>
            <Link className="home-card" href="/docs/ai-reference">
              <strong>AI reference</strong>
              <span>Normative public API contract for coding agents.</span>
            </Link>
            <a className="home-card" href="https://github.com/rungent/rungent/blob/main/RUNGENT.md">
              <strong>RUNGENT.md</strong>
              <span>Short integration guide shipped in the repo and wheel.</span>
            </a>
          </div>
        </section>
      </div>
    </HomeLayout>
  );
}
