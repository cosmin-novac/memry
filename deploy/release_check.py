#!/usr/bin/env python3
"""Release-drift gate: shipped code must never change without a version bump.

Used by CI (.github/workflows/publish.yml), the pre-push hook
(deploy/git-hooks/pre-push) and the VPS deploy script, so all three agree on
one rule:

  * the version lives in exactly one place, src/memry/__init__.py
    (pyproject.toml reads it via hatch's dynamic version);
  * if tag v<version> already exists, none of the SHIPPED paths may differ
    from that tag. Changing them means the next push must carry a new
    version, otherwise PyPI, GitHub main and any deployment silently
    diverge (this happened between v0.2.25 on 2026-07-26 and 2026-08-13:
    mcp 2.0 broke `pip install memry` while main already had the fix).

Docs, tests, website and deploy scripts may change freely: they are not part
of the published package or the container image.

Exit codes: 0 = fine (prints "release=true|false" and "version=X" lines,
consumed by CI), 1 = drift or version mismatch.

Usage:
    python deploy/release_check.py            # check HEAD
    python deploy/release_check.py --ref REF  # check another commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everything that ends up in the wheel / sdist / Docker image.
SHIPPED_PATHS = ["src/memry", "pyproject.toml", "Dockerfile", "requirements-docker.txt"]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def version_at(ref: str) -> str:
    text = git("show", f"{ref}:src/memry/__init__.py")
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if not match:
        sys.exit("::error::__version__ not found in src/memry/__init__.py")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD", help="commit to check (default HEAD)")
    args = parser.parse_args()

    version = version_at(args.ref)
    tag = f"v{version}"

    # Guard against a stale pin: pyproject must not carry its own version.
    pyproject = git("show", f"{args.ref}:pyproject.toml")
    if re.search(r'^version\s*=', pyproject, re.M):
        print(
            "::error::pyproject.toml pins its own version; it must use "
            'dynamic = ["version"] and read src/memry/__init__.py'
        )
        return 1

    print(f"version={version}")

    tagged = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tagged.returncode != 0:
        print(f"{tag} not tagged yet: this ref is a release candidate")
        print("release=true")
        return 0

    changed = git("diff", "--name-only", tag, args.ref, "--", *SHIPPED_PATHS)
    if changed:
        files = "\n  ".join(changed.splitlines())
        print(
            f"::error::shipped code changed since {tag} but __version__ is still "
            f"{version}. Bump src/memry/__init__.py before pushing, otherwise "
            f"PyPI/GitHub/deployments diverge. Changed:\n  {files}"
        )
        return 1

    print(f"{tag} already released and shipped code is identical: nothing to release")
    print("release=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
