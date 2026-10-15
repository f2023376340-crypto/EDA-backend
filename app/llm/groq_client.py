"""Single entry point for Groq chat models (§5.5): shared instances, central
timeout and retry settings.
"""

from functools import lru_cache

from langchain_groq import ChatGroq
from pydantic import SecretStr

from app.core.config import settings
from app.core.errors import LLMUnavailableError

_TIMEOUT_S = 30.0
_MAX_RETRIES = 2


@lru_cache(maxsize=8)
def get_chat_model(model: str, temperature: float) -> ChatGroq:
    if not settings.groq_api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not configured.")
    return ChatGroq(
        model_name=model,
        temperature=temperature,
        api_key=SecretStr(settings.groq_api_key),
        timeout=_TIMEOUT_S,
        max_retries=_MAX_RETRIES,
    )
