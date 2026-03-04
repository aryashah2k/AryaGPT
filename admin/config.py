"""Server-wide configuration store for AryaGPT admin panel."""

from __future__ import annotations

from typing import Dict, List

PROVIDER_MODELS: Dict[str, List[str]] = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "together": [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
}

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024

PROVIDER_ENV_KEYS: Dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openai": "OPENAI_API_KEY",
}

PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "groq": "Groq",
    "together": "Together AI",
    "openai": "OpenAI",
}
