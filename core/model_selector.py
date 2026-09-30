"""Ollama availability checks and Qwen-first, Llama-fallback selection."""

from dataclasses import dataclass

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
SETUP_MESSAGE = (
    "Install Ollama, then run `ollama pull qwen2.5:3b` (recommended) or "
    "`ollama pull llama3.2:3b`. Start Ollama and refresh this page."
)


@dataclass(frozen=True)
class ModelStatus:
    connected: bool
    models: list[str]
    selected_model: str | None
    family: str | None
    error: str | None = None


def detect_ollama_models(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.5) -> list[str]:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [
        str(item.get("name") or item.get("model"))
        for item in payload.get("models", [])
        if item.get("name") or item.get("model")
    ]


def select_model(models: list[str]) -> tuple[str | None, str | None]:
    normalized = sorted(models, key=str.casefold)
    for family in ("qwen", "llama"):
        match = next((name for name in normalized if name.split(":", 1)[0].casefold().startswith(family)), None)
        if match:
            return match, family.title()
    return None, None


def get_model_status(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.5) -> ModelStatus:
    try:
        models = detect_ollama_models(base_url, timeout)
        selected, family = select_model(models)
        return ModelStatus(True, models, selected, family)
    except requests.RequestException:
        return ModelStatus(False, [], None, None, "Ollama is not responding at the configured local address.")
    except (ValueError, TypeError):
        return ModelStatus(False, [], None, None, "Ollama returned an unreadable model list.")