from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from openbrain.domain.ports import EmbeddingProvider, StructuredGenerationProvider
from openbrain.settings import OpenBrainSettings


class OpenAICompatibleStructuredProvider(StructuredGenerationProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        api_key: str,
        base_url: str | None,
        openrouter_site_url: str | None = None,
        openrouter_app_name: str | None = None,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self._extra_headers: dict[str, str] | None = None
        if provider_name == "openrouter" and openrouter_site_url and openrouter_app_name:
            self._extra_headers = {
                "HTTP-Referer": openrouter_site_url,
                "X-Title": openrouter_app_name,
            }
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self._extra_headers:
            params["extra_headers"] = self._extra_headers
        response = self._client.chat.completions.create(**params)
        raw = (response.choices[0].message.content or "").strip()
        return json.loads(raw) if raw else {}


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        api_key: str,
        base_url: str | None,
        embedding_dims: int,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text: str) -> list[float]:
        params: dict[str, Any] = {"input": [text.replace("\n", " ")], "model": self.model_name}
        if self.provider_name == "openai":
            params["dimensions"] = self.embedding_dims
        response = self._client.embeddings.create(**params)
        return list(response.data[0].embedding)


class OllamaStructuredProvider(StructuredGenerationProvider):
    def __init__(self, *, model_name: str, base_url: str):
        try:
            from ollama import Client
        except ImportError as exc:
            raise ImportError("The 'ollama' package is required for OPENBRAIN_LLM_PROVIDER=ollama") from exc

        self.provider_name = "ollama"
        self.model_name = model_name
        self._client = Client(host=base_url)

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self._client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nReturn valid JSON only."},
            ],
            format="json",
            options={"temperature": 0},
        )
        if isinstance(response, dict):
            raw = response.get("message", {}).get("content", "")
        else:
            raw = getattr(getattr(response, "message", None), "content", "")
        return json.loads(raw) if raw else {}


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, model_name: str, base_url: str, embedding_dims: int):
        try:
            from ollama import Client
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is required for OPENBRAIN_EMBEDDING_PROVIDER=ollama"
            ) from exc

        self.provider_name = "ollama"
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self._client = Client(host=base_url)

    def embed(self, text: str) -> list[float]:
        if hasattr(self._client, "embed"):
            response = self._client.embed(model=self.model_name, input=text)
            if isinstance(response, dict):
                values = response.get("embeddings") or response.get("embedding") or []
            else:
                values = getattr(response, "embeddings", None) or getattr(response, "embedding", [])
            if values and isinstance(values[0], list):
                return list(values[0])
            return list(values)
        response = self._client.embeddings(model=self.model_name, prompt=text)
        return list(response["embedding"])


def build_structured_generation_provider(settings: OpenBrainSettings) -> StructuredGenerationProvider:
    if settings.llm.provider == "ollama":
        return OllamaStructuredProvider(
            model_name=settings.llm.model,
            base_url=settings.llm.base_url or "http://localhost:11434",
        )
    return OpenAICompatibleStructuredProvider(
        provider_name=settings.llm.provider,
        model_name=settings.llm.model,
        api_key=settings.llm.api_key or "",
        base_url=settings.llm.base_url,
        openrouter_site_url=settings.llm.openrouter_site_url,
        openrouter_app_name=settings.llm.openrouter_app_name,
    )


def build_embedding_provider(settings: OpenBrainSettings) -> EmbeddingProvider:
    if settings.embedding.provider == "ollama":
        return OllamaEmbeddingProvider(
            model_name=settings.embedding.model,
            base_url=settings.embedding.base_url or "http://localhost:11434",
            embedding_dims=settings.embedding.dims,
        )
    return OpenAICompatibleEmbeddingProvider(
        provider_name=settings.embedding.provider,
        model_name=settings.embedding.model,
        api_key=settings.embedding.api_key or "",
        base_url=settings.embedding.base_url,
        embedding_dims=settings.embedding.dims,
    )
