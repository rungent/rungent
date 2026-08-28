'use client';

import { useState } from 'react';

export function CopyForLlm() {
  const [copied, setCopied] = useState(false);

  async function copy() {
    const response = await fetch('/llms-full.txt');
    if (!response.ok) throw new Error('Failed to load documentation');
    await navigator.clipboard.writeText(await response.text());
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <button className="copy-for-llm" type="button" onClick={() => void copy()}>
      {copied ? 'Copied' : 'For LLM'}
    </button>
  );
}
