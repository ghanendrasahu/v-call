"""LLM factory + retry/fallback helpers for agent nodes.

Models are created once per process (``lru_cache``) and injected into nodes so
tests can substitute fakes without touching the network.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@lru_cache
def get_chat_model(model: str | None = None, temperature: float = 0.0) -> Any:
    """Cached ChatOpenAI instance (reads OPENAI_API_KEY from settings/env)."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": model or settings.openai_model,
        "temperature": temperature,
        "timeout": 90,
        "max_retries": 2,
    }
    if settings.openai_api_key is not None:
        kwargs["api_key"] = settings.openai_api_key.get_secret_value()
    return ChatOpenAI(**kwargs)


def get_structured_model(schema: type[SchemaT], *, model: str | None = None) -> Any:
    """Chat model bound to a Pydantic schema via function-calling."""
    return get_chat_model(model).with_structured_output(schema)


async def invoke_structured(llm_with_schema: Any, prompt: str) -> Any:
    """Invoke a structured-output model (async, isolating the LLM surface).

    Nodes only ever call this, so a fake object exposing ``ainvoke`` is enough
    to unit-test the whole graph offline.
    """
    if hasattr(llm_with_schema, "ainvoke"):
        return await llm_with_schema.ainvoke(prompt)
    return llm_with_schema.invoke(prompt)


def clear_llm_cache() -> None:
    """Drop cached models (tests)."""
    get_chat_model.cache_clear()
