import logging
from litellm import completion
from app.core.config import settings
from app.prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def ask_llm(prompt: str, model: str = None) -> str:
    """Send prompt to LLM (LiteLLM) using default configured settings or override model."""
    model = model or settings.DEFAULT_MODEL

    logger.info("Calling LLM: model=%s, tokens=%d, temperature=%s", model, settings.MAX_TOKENS, settings.TEMPERATURE)

    if model.startswith("gemini"):
        api_key = settings.GEMINI_API_KEY
    elif model.startswith("groq"):
        api_key = settings.GROQ_API_KEY
    else:
        raise ValueError(f"Unsupported model: {model}")

    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=settings.MAX_TOKENS,
        temperature=settings.TEMPERATURE,
        api_key=api_key
    )

    return response.choices[0].message.content