# Releasing Rungent

`rungent` (PyPI) and `@rungent/sdk` (npm) share one semver. A `v*` tag triggers
[`.github/workflows/release.yml`](.github/workflows/release.yml). Local machines never publish.

## Everyday release

```bash
./scripts/release.sh 0.2.0
git push origin main --tags
```

The script only bumps versions, commits, and tags. GitHub Actions publishes.

## One-time registry setup (Trusted Publishing)

No long-lived PyPI/npm tokens are stored in GitHub. Bind OIDC publishers once.

### 1. GitHub Environments

This repo already expects environments named **`pypi`** and **`npm`**
(Settings → Environments). Create them if missing.

### 2. PyPI

1. Sign in at https://pypi.org/
2. Open [Publishing](https://pypi.org/manage/account/publishing/) → **Add a new pending publisher**
3. Fill in:
   - PyPI project name: `rungent`
   - Owner: `rungent`
   - Repository: `rungent`
   - Workflow: `release.yml`
   - Environment name: `pypi`
4. Save. The first successful `uv publish` from that workflow creates the project.

### 3. npm

1. Create the `@rungent` org/scope on https://www.npmjs.com/ if needed
2. For `@rungent/sdk`, configure
   [Trusted Publishing](https://docs.npmjs.com/trusted-publishers) with:
   - Repository: `rungent/rungent`
   - Workflow: `release.yml`
   - Environment: `npm`
3. Ensure your npm account can publish under `@rungent`

After both publishers are saved, push a `v*` tag to publish.
