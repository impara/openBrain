from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from openbrain.infrastructure.providers import (
    OllamaEmbeddingProvider,
    OllamaStructuredProvider,
    build_embedding_provider,
    build_structured_generation_provider,
)
from openbrain.settings import DatabaseSettings, EmbeddingSettings, LLMSettings, OpenBrainSettings


def make_settings(llm_provider="openai", embedding_provider="openai"):
    return OpenBrainSettings(
        database=DatabaseSettings(
            dbname="db",
            user="user",
            password="pass",
            host="localhost",
            port=5432,
        ),
        llm=LLMSettings(
            provider=llm_provider,
            model="model-name",
            api_key="api-key" if llm_provider != "ollama" else None,
            base_url="http://localhost:11434" if llm_provider == "ollama" else "https://example.com/v1",
            openrouter_site_url="https://openbrain.local",
            openrouter_app_name="OpenBrain",
        ),
        embedding=EmbeddingSettings(
            provider=embedding_provider,
            model="embed-model",
            api_key="api-key" if embedding_provider != "ollama" else None,
            base_url="http://localhost:11434" if embedding_provider == "ollama" else "https://example.com/v1",
            dims=768 if embedding_provider == "ollama" else 1536,
        ),
        capture_mode="async",
        ingest_workers=1,
        ingest_poll_ms=250,
        graph_name="brain_graph_v2",
        log_level="INFO",
    )


def test_openai_compatible_structured_provider_uses_openrouter_headers():
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content='{"bullets":["x"]}'))]
    with patch("openbrain.infrastructure.providers.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = response
        provider = build_structured_generation_provider(make_settings(llm_provider="openrouter"))
        payload = provider.generate_json(system_prompt="sys", user_prompt="user")
    assert payload == {"bullets": ["x"]}
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["extra_headers"]["HTTP-Referer"] == "https://openbrain.local"
    assert kwargs["extra_headers"]["X-Title"] == "OpenBrain"


def test_openai_compatible_embedding_provider_uses_dimensions_for_openai():
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1, 0.2])]
    with patch("openbrain.infrastructure.providers.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = response
        provider = build_embedding_provider(make_settings(embedding_provider="openai"))
        embedding = provider.embed("hello")
    assert embedding == [0.1, 0.2]
    kwargs = mock_openai.return_value.embeddings.create.call_args.kwargs
    assert kwargs["dimensions"] == 1536


def test_ollama_structured_provider_supports_json_generation(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": '{"nodes": []}'}}
    ollama_module = types.ModuleType("ollama")
    ollama_module.Client = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "ollama", ollama_module)
    provider = OllamaStructuredProvider(model_name="llama3.1", base_url="http://localhost:11434")
    assert provider.generate_json(system_prompt="sys", user_prompt="user") == {"nodes": []}


def test_ollama_embedding_provider_supports_embed_endpoint(monkeypatch):
    fake_client = MagicMock()
    fake_client.embed.return_value = {"embeddings": [[0.5, 0.6]]}
    ollama_module = types.ModuleType("ollama")
    ollama_module.Client = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "ollama", ollama_module)
    provider = OllamaEmbeddingProvider(model_name="nomic-embed-text", base_url="http://localhost:11434", embedding_dims=768)
    assert provider.embed("hello") == [0.5, 0.6]


def test_build_embedding_provider_returns_ollama_provider(monkeypatch):
    fake_client = MagicMock()
    fake_client.embed.return_value = {"embeddings": [[0.5, 0.6]]}
    ollama_module = types.ModuleType("ollama")
    ollama_module.Client = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "ollama", ollama_module)
    provider = build_embedding_provider(make_settings(embedding_provider="ollama"))
    assert provider.provider_name == "ollama"
