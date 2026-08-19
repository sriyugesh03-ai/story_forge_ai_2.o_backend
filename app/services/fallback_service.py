import logging
from app.services.llm_client import ask_llm

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "groq/openai/gpt-oss-120b"

def fallback_call(prompt: str) -> str:
    """Executes a fallback call using the configured fallback model when the primary model fails."""
    logger.warning("Primary LLM model failed. Switching to fallback model: %s", FALLBACK_MODEL)
    return ask_llm(
        prompt=prompt,
        model=FALLBACK_MODEL
    )
