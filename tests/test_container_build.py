from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_requirements_match_pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    expected = [
        *project["project"]["dependencies"],
        *project["project"]["optional-dependencies"]["anthropic"],
    ]
    actual = [
        line.strip()
        for line in (ROOT / "requirements-docker.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert actual == expected

    assert project["build-system"] == {
        "requires": ["hatchling"],
        "build-backend": "hatchling.build",
    }


def test_container_installs_dependencies_before_source():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    builder_install = dockerfile.index("pip install --no-cache-dir hatchling")
    source_copy = dockerfile.index("COPY src ./src")
    wheel_build = dockerfile.index("pip wheel --no-cache-dir --no-deps")
    runtime_stage = dockerfile.index("FROM python:3.12-slim AS runtime")
    dependency_copy = dockerfile.index("COPY requirements-docker.txt")
    dependency_install = dockerfile.index("pip install --no-cache-dir -r")
    wheel_copy = dockerfile.index("COPY --from=package-builder /wheels /wheels")
    package_install = dockerfile.index(
        "pip install --no-cache-dir --no-deps /wheels/*.whl"
    )

    assert builder_install < source_copy < wheel_build < runtime_stage
    assert runtime_stage < dependency_copy < dependency_install < wheel_copy
    assert wheel_copy < package_install


def test_docker_context_excludes_local_data_and_credentials():
    patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert patterns[0] == "**"
    assert "!src/**" in patterns
    assert "!requirements-docker.txt" in patterns
    assert not any("local" in pattern for pattern in patterns)
