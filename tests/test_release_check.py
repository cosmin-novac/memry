"""deploy/release_check.py is the gate that keeps PyPI, GitHub and deployments
in step: shipped code may not change under an already-released version."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "release_check.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    # The script resolves the repo from its own location, so copy it in.
    target = repo / "deploy" / "release_check.py"
    target.parent.mkdir(exist_ok=True)
    shutil.copy(SCRIPT, target)
    return subprocess.run([sys.executable, str(target)], cwd=repo, capture_output=True, text=True)


def _write(repo: Path, version: str, code: str = "x = 1\n") -> None:
    pkg = repo / "src" / "memry"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n{code}', encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "memry"\ndynamic = ["version"]\n', encoding="utf-8"
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    _write(tmp_path, "0.1.0")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "v0.1.0")
    _git(tmp_path, "tag", "v0.1.0")
    return tmp_path


def test_released_and_unchanged_is_not_a_release(repo: Path) -> None:
    (repo / "README.md").write_text("docs only\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "docs")
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "release=false" in result.stdout


def test_shipped_change_without_bump_fails(repo: Path) -> None:
    _write(repo, "0.1.0", code="x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "code change, forgot bump")
    result = _run(repo)
    assert result.returncode == 1
    assert "Bump src/memry/__init__.py" in result.stdout
    assert "src/memry/__init__.py" in result.stdout


def test_bumped_version_is_a_release(repo: Path) -> None:
    _write(repo, "0.1.1", code="x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v0.1.1")
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "release=true" in result.stdout
    assert "version=0.1.1" in result.stdout


def test_pyproject_must_not_pin_version(repo: Path) -> None:
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "memry"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "pin")
    result = _run(repo)
    assert result.returncode == 1
    assert "dynamic" in result.stdout


def test_real_repo_uses_dynamic_version() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in project["project"]
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["hatch"]["version"] == {"path": "src/memry/__init__.py"}
