import logging

from app.core.config import settings
from app.rag.retriever import get_retriever
from app.prompts.rag_prompt import build_rag_prompt
from app.services.llm_client import ask_llm
from app.services.retry_service import retry_call
from app.services.fallback_service import fallback_call
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "groq/llama-3.3-70b-versatile"


def welcome_message():
    return {"message": "Welcome To Story Forge AI 2.0"}


async def generate_story(topic: str, story_type: str, debug: bool = False) -> dict:
    retriever = get_retriever()   # shared singleton — no extra model load
    evaluator = EvaluationService()

    evaluator.start_request()

    # ── Retrieve Context ──────────────────────────────────────────────
    evaluator.start_retrieval()
    chunks = retriever.retrieve(topic)
    evaluator.end_retrieval()

    # ── Build Prompt ──────────────────────────────────────────────────
    context = "\n\n".join(chunks)
    prompt = build_rag_prompt(topic, story_type, context)

    # ── Generate Story ────────────────────────────────────────────────
    evaluator.start_llm()

    try:
        result = await retry_call(lambda: ask_llm(prompt))
    except Exception as err:
        logger.warning("Primary model failed after retries: %s — invoking fallback", err)
        fallback_story = fallback_call(prompt)
        result = {
            "story": fallback_story,
            "retry_count": 3,
            "fallback_used": True,
        }

    evaluator.end_llm()

    # ── Build Metrics ──────────────────────────────────────────────────
    story = result["story"]
    retry_count = result["retry_count"]
    fallback_used = result["fallback_used"]

    model_used = FALLBACK_MODEL if fallback_used else settings.DEFAULT_MODEL

    metrics = evaluator.build_metrics(
        model_used=model_used,
        retrieved_chunks=len(chunks),
        retry_count=retry_count,
        fallback_used=fallback_used,
        sources=[{"player": topic, "source": "Wikipedia"}],
    )

    # ── Response ───────────────────────────────────────────────────────
    response = {
        "story": story,
        "evaluation": metrics,
    }

    # Raw chunks are large — only include when the caller explicitly asks for them
    if debug:
        response["retrieved_context"] = chunks

    return response