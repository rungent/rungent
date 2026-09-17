# Releasing Rungent

`rungent` (Acahti PyPI) and `@rungent/sdk` (Acahti npm) share one semver. A `v*`
tag on Acahti triggers [`.acahti/pipelines/pkg.yaml`](.acahti/pipelines/pkg.yaml).
Do not publish to pypi.org or registry.npmjs.org.

## Everyday release

```bash
./scripts/release.sh 0.3.7
git push acahti main
git push acahti v0.3.7
```

The script bumps the shared semver, refreshes `uv.lock`, commits, and tags.
Acahti `pkg` publishes. Do not `uv publish` / `npm publish` from a laptop.
