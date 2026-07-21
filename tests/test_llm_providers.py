from __future__ import annotations

import json

from memry.config import LLMConfig
from memry.providers.llm import OpenAILLM


def _capture_post(captured: dict):
    def post(url, *, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "{}"}}]}

        return Response()

    return post


def test_openai_gpt5_sends_reasoning_effort(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("memry.providers.llm.httpx.post", _capture_post(captured))
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-5-mini", api_key="k"))
    llm.complete("sys", "user", json_schema={"type": "object"})
    assert captured["body"]["reasoning_effort"] == "low"
    assert captured["body"]["response_format"]["type"] == "json_schema"


def test_openai_non_reasoning_model_omits_effort(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("memry.providers.llm.httpx.post", _capture_post(captured))
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="k"))
    llm.complete("sys", "user")
    assert "reasoning_effort" not in captured["body"]
    assert "response_format" not in captured["body"]
