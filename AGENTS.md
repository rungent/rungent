# Rungent project instructions

Read `RUNGENT.md` before integrating or modifying Rungent. Published docs:
https://rungent.github.io — source is `docs/`; `docs/ai-reference.mdx` is the complete API
reference intended for coding agents. Load https://rungent.github.io/llms-full.txt when possible.

Project rules:

- Keep the embedded Python API smaller than the HTTP API.
- Every tool must declare types, effect, and approval policy.
- Safety decisions belong in runtime code, not only in prompts.
- Add or update deterministic baseline coverage for every harness behavior.
- Any public API change must update `docs/`, `RUNGENT.md`, Python tests, and SDK contract tests.
- Do not add sidecar, skill routing, a styled chat UI, or another model provider abstraction unless a
  real application requires it. `rungent.mcp` is an in-process adapter, not a sidecar or skill router.

