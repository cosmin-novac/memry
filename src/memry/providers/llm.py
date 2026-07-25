"""LLM providers used by the intelligence layer (extraction + reconciliation).

The interface is deliberately tiny: ``complete(system, user, json_schema=None)``
returns text. Providers that support structured outputs use the schema to
guarantee valid JSON; others ignore it and the caller parses leniently.

Anthropic calls go through the official ``anthropic`` SDK (optional extra:
``pip install memry[anthropic]``). OpenAI and Ollama use plain HTTP via
httpx to keep the dependency footprint small.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import LLMConfig

# Models that support adaptive thinking + effort (Claude 4.6+ families).
_ADAPTIVE_PREFIXES = (
    "claude-fable-5",
    "claude-mythos",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
)

# Models that support structured outputs (output_config.format).
_STRUCTURED_PREFIXES = (
    "claude-fable-5",
    "claude-mythos",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
)


class LLM(ABC):
    name: str = "llm"
    available: bool = True

    @abstractmethod
    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        """Run one completion and return the text output."""

    def close(self) -> None:
        """Release reusable provider connections."""
        return None


class NoneLLM(LLM):
    """Placeholder when no LLM is configured; callers must check ``available``."""

    name = "none"
    available = False

    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        raise RuntimeError(
            "No LLM configured. Set ANTHROPIC_API_KEY / OPENAI_API_KEY, or "
            "MEMRY_LLM_PROVIDER=ollama for a local model."
        )


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, cfg: LLMConfig) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The Anthropic provider requires the anthropic SDK: "
                "pip install 'memry[anthropic]'"
            ) from exc
        kwargs: dict[str, Any] = {"timeout": cfg.timeout}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.model = cfg.resolved_model()
        self.max_tokens = cfg.max_tokens
        self.effort = cfg.effort

    def close(self) -> None:
        self._client.close()

    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {}
        output_config: dict[str, Any] = {}
        if self.model.startswith(_ADAPTIVE_PREFIXES):
            kwargs["thinking"] = {"type": "adaptive"}
            output_config["effort"] = self.effort
        if json_schema is not None and self.model.startswith(_STRUCTURED_PREFIXES):
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            kwargs["output_config"] = output_config

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, cfg: LLMConfig) -> None:
        import os

        self.model = cfg.resolved_model()
        self.base_url = (cfg.base_url or "https://api.openai.com").rstrip("/")
        self.api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = cfg.timeout
        self.effort = cfg.effort
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # gpt-5* are reasoning models that default to medium effort; extraction
        # is a structured task where low keeps latency and cost down. Other
        # models reject the parameter.
        if self.model.startswith("gpt-5"):
            body["reasoning_effort"] = self.effort
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": json_schema},
            }
        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        self.model = cfg.resolved_model()
        self.base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")
        self.timeout = cfg.timeout
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_schema is not None:
            body["format"] = json_schema
        resp = self._client.post(f"{self.base_url}/api/chat", json=body)
        resp.raise_for_status()
        return resp.json()["message"]["content"] or ""


def build_llm(cfg: LLMConfig) -> LLM:
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg)
    if cfg.provider == "openai":
        return OpenAILLM(cfg)
    if cfg.provider == "ollama":
        return OllamaLLM(cfg)
    return NoneLLM()
