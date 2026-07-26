"""The dashboard's JavaScript must actually parse.

The dashboard is one big inline script built by string concatenation in
rest.py, so an editing slip can leave a stray brace and ship a page that loads,
returns 200, passes every API test, and then dies on the first line of script -
memories stuck on "loading" with a single console error. That happened. Python
tests cannot see it because the JS is opaque text to them.

`node --check` parses without executing. Where node is unavailable the test
skips rather than pretending to have checked.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to parse the dashboard JS"
)


def _dashboard_html() -> str:
    store = MemoryStore(
        Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64)
    )
    try:
        with TestClient(create_app(store)) as client:
            return client.get("/").text
    finally:
        store.close()


def _scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


def test_dashboard_javascript_parses(tmp_path):
    blocks = _scripts(_dashboard_html())
    assert blocks, "the dashboard should serve at least one inline script"
    for index, source in enumerate(blocks):
        path = tmp_path / f"dashboard_{index}.js"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"dashboard script block {index} is not valid JavaScript:\n"
            f"{result.stderr}"
        )


def test_every_onclick_handler_is_defined(tmp_path):
    """An `onclick` naming a function that does not exist is a dead button.

    Removing a feature is the usual way this happens: the handler goes, the
    markup that calls it stays, and nothing fails until someone clicks.
    """
    html = _dashboard_html()
    source = "\n".join(_scripts(html))
    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", source))
    defined |= set(re.findall(r"(?:const|let|var)\s+(\w+)\s*=", source))

    called = set(re.findall(r'on(?:click|change|input)="(\w+)\(', html))
    called |= set(re.findall(r"on(?:click|change|input)='(\w+)\(", html))
    missing = sorted(name for name in called if name not in defined)
    assert not missing, f"markup calls handlers that no longer exist: {missing}"
