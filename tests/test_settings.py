from __future__ import annotations

import pytest

from openbrain.settings import OpenBrainSettings


def _base_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("OPENBRAIN_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENBRAIN_LLM_API_KEY", "openai-key")
    monkeypatch.setenv("OPENBRAIN_EMBEDDING_PROVIDER", "openai")


def test_settings_use_neutral_openai_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("OPENBRAIN_LLM_MODEL", "gpt-4o-mini")
    settings = OpenBrainSettings.from_env()
    assert settings.llm.api_key == "openai-key"
    assert settings.llm.model == "gpt-4o-mini"
    assert settings.embedding.api_key == "openai-key"
    assert settings.embedding.dims == 1536
    assert settings.graph_name == "brain_graph_v2"


def test_openrouter_requires_explicit_embedding_dims(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("OPENBRAIN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENBRAIN_LLM_API_KEY", "router-key")
    monkeypatch.setenv("OPENBRAIN_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENBRAIN_EMBEDDING_API_KEY", "router-key")
    with pytest.raises(ValueError, match="OPENBRAIN_EMBEDDING_DIMS"):
        OpenBrainSettings.from_env()


def test_ollama_requires_embedding_dims(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("OPENBRAIN_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENBRAIN_EMBEDDING_PROVIDER", "ollama")
    with pytest.raises(ValueError, match="OPENBRAIN_EMBEDDING_DIMS"):
        OpenBrainSettings.from_env()


def test_invalid_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("OPENBRAIN_LLM_PROVIDER", "bad-provider")
    with pytest.raises(ValueError, match="OPENBRAIN_LLM_PROVIDER"):
        OpenBrainSettings.from_env()


def test_embedding_dims_are_bounded(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("OPENBRAIN_EMBEDDING_DIMS", "2501")
    with pytest.raises(ValueError, match="between 1 and 2000"):
        OpenBrainSettings.from_env()
