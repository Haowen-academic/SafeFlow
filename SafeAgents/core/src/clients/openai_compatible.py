"""
Provider-agnostic OpenAI-compatible client utilities for SafeAgents.

Supports:
- Azure OpenAI via existing AZURE_* variables
- Generic OpenAI-compatible endpoints via OPENAI_COMPAT_* variables
"""
import os
from typing import Any, Dict

from dotenv import load_dotenv


def _normalize_base_url(raw_url: str) -> str:
    """
    Normalize OpenAI-compatible base URLs.

    The OpenAI Python SDK and related wrappers typically expect a base URL such as
    `https://host/v1`, while users often provide a full chat completions endpoint like
    `https://host/v1/chat/completions`.
    """
    url = raw_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    return url


def get_llm_provider() -> str:
    """
    Resolve the active LLM provider.

    Priority:
    1. Explicit `LLM_PROVIDER`
    2. Presence of `OPENAI_COMPAT_API_KEY`
    3. Fallback to Azure
    """
    load_dotenv()
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider in {"azure", "openai_compatible"}:
        return provider
    if os.getenv("OPENAI_COMPAT_API_KEY"):
        return "openai_compatible"
    return "azure"


def get_openai_compatible_config() -> Dict[str, Any]:
    """
    Get generic OpenAI-compatible LLM configuration from environment variables.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_COMPAT_API_KEY")
    raw_base_url = os.getenv("OPENAI_COMPAT_BASE_URL")
    model = os.getenv("OPENAI_COMPAT_MODEL")
    temperature = float(os.getenv("OPENAI_COMPAT_TEMPERATURE", "0.0"))
    timeout = float(os.getenv("OPENAI_COMPAT_TIMEOUT", "120.0"))
    max_retries = int(os.getenv("OPENAI_COMPAT_MAX_RETRIES", "2"))

    required_vars = [api_key, raw_base_url, model]
    if any(v is None or v == "" for v in required_vars):
        raise ValueError(
            "OpenAI-compatible environment variables "
            "(OPENAI_COMPAT_API_KEY, OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_MODEL, "
            "OPENAI_COMPAT_TEMPERATURE) are not set properly in the `.env`."
        )

    base_url = _normalize_base_url(raw_base_url)

    return {
        "provider": "openai_compatible",
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
        "model_capabilities": {
            "function_calling": True,
            "json_output": False,
            "vision": False,
            "structured_output": False,
        },
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    }


def get_llm_config() -> Dict[str, Any]:
    """
    Get the active LLM configuration based on provider selection.
    """
    provider = get_llm_provider()
    if provider == "openai_compatible":
        return get_openai_compatible_config()

    from .azure_openai import get_azure_config

    config = get_azure_config()
    config["provider"] = "azure"
    return config


def get_openai_client_sync():
    """
    Create a synchronous client for the active provider.
    """
    provider = get_llm_provider()
    if provider == "openai_compatible":
        from openai import OpenAI

        config = get_openai_compatible_config()
        return OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
            max_retries=config["max_retries"],
        )

    from .azure_openai import get_azure_openai_client_sync

    return get_azure_openai_client_sync()


def get_openai_client_async():
    """
    Create an async client for the active provider.
    """
    provider = get_llm_provider()
    if provider == "openai_compatible":
        from openai import AsyncOpenAI

        config = get_openai_compatible_config()
        return AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
            max_retries=config["max_retries"],
        )

    from .azure_openai import get_azure_openai_client_async

    return get_azure_openai_client_async()
