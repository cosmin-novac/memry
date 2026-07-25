from __future__ import annotations

from memry.config import LLMConfig
from memry.providers.llm import OpenAILLM


def _fake_client(captured: dict):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    class Client:
        def __init__(self, **kwargs):
            captured["clients"] = captured.get("clients", 0) + 1
            captured["client_kwargs"] = kwargs

        def post(self, url, *, headers=None, json=None):
            captured["url"] = url
            captured["body"] = json
            captured["posts"] = captured.get("posts", 0) + 1
            return Response()

        def close(self):
            captured["closed"] = True

    return Client


def test_openai_gpt5_sends_reasoning_effort(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("memry.providers.llm.httpx.Client", _fake_client(captured))
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-5-mini", api_key="k"))
    llm.complete("sys", "user", json_schema={"type": "object"})
    assert captured["body"]["reasoning_effort"] == "low"
    assert captured["body"]["response_format"]["type"] == "json_schema"


def test_openai_non_reasoning_model_omits_effort(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("memry.providers.llm.httpx.Client", _fake_client(captured))
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="k"))
    llm.complete("sys", "user")
    assert "reasoning_effort" not in captured["body"]
    assert "response_format" not in captured["body"]


def test_openai_reuses_and_closes_http_client(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("memry.providers.llm.httpx.Client", _fake_client(captured))
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="k"))
    llm.complete("sys", "one")
    llm.complete("sys", "two")
    llm.close()
    assert captured["clients"] == 1
    assert captured["posts"] == 2
    assert captured["closed"] is True
