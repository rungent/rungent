#!/usr/bin/env bash
# Bump shared semver and create an annotated tag. Does NOT publish.
# Usage: ./scripts/release.sh 0.2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <version>" >&2
  echo "Example: $0 0.2.0" >&2
  exit 1
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid version: $VERSION" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash first." >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
  echo "Must be on main (current: $BRANCH)" >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
  echo "Tag v${VERSION} already exists." >&2
  exit 1
fi

if ! command -v uv >/dev/null; then
  echo "uv is required to refresh uv.lock before tagging." >&2
  exit 1
fi

python3 - "$VERSION" <<'PY'
import pathlib, re, sys
version = sys.argv[1]
root = pathlib.Path(".")
pyproject = root / "pyproject.toml"
text = pyproject.read_text()
new, n = re.subn(
    r'(?m)^version = "[^"]+"$',
    f'version = "{version}"',
    text,
    count=1,
)
if n != 1:
    raise SystemExit("Failed to update version in pyproject.toml")
pyproject.write_text(new)

pkg = root / "packages" / "sdk" / "package.json"
text = pkg.read_text()
new, n = re.subn(
    r'"version": "[^"]+"',
    f'"version": "{version}"',
    text,
    count=1,
)
if n != 1:
    raise SystemExit("Failed to update version in packages/sdk/package.json")
pkg.write_text(new)
print(f"Bumped versions to {version}")
PY

uv lock
git add pyproject.toml packages/sdk/package.json uv.lock
git commit -m "release: v${VERSION}"
git tag -a "v${VERSION}" -m "v${VERSION}"

echo
echo "Created commit and tag v${VERSION}."
echo "Push to publish via GitHub Actions:"
echo "  git push origin main --tags"
echo
echo "Do not run uv publish / npm publish locally."
