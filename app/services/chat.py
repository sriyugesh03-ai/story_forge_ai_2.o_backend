import logging

from app.core.config import settings
from app.rag.retriever import get_retriever
from app.rag.node import RAGNode
from app.prompts.rag_prompt import get_rag_prompt_template
from app.routing.router import (
    ROUTE_GENERAL,
    ROUTE_RAG,
    ROUTE_WIKIPEDIA,
    route_query,
)
from app.services.llm_client import ask_llm
from app.services.retry_service import retry_call
from app.services.fallback_service import fallback_call
from app.services.evaluation_service import EvaluationService
from app.tools.wikipedia_tool import aget_wikipedia_player

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "groq/openai/gpt-oss-120b"

GENERAL_DECLINE = (
    "I'm a sports storytelling assistant focused on athletes and sporting moments. "
    "I can't help with that topic. Try asking about a player like Lionel Messi, "
    "Cristiano Ronaldo, or MS Dhoni."
)


def welcome_message():
    return {"message": "Welcome To Story Forge AI 2.0"}


def _graceful_response(topic: str, evaluator, model_used: str, message: str,
                       source: str, route: str) -> dict:
    """Build a clean, human-readable fallback response (no stack traces)."""
    metrics = evaluator.build_metrics(
        model_used=model_used,
        retrieved_chunks=0,
        retry_count=0,
        fallback_used=False,
        sources=[{"player": topic, "source": source}],
    )
    return {"story": message, "evaluation": metrics, "route": route, "source": source}


async def generate_story(topic: str, story_type: str, debug: bool = False) -> dict:
    retriever = get_retriever()   # shared singleton — BaseRetriever instance
    rag_node = RAGNode(retriever=retriever)
    evaluator = EvaluationService()

    evaluator.start_request()

    route = ROUTE_RAG
    source_label = "RAG"
    context = ""
    retrieved_chunks = []
    retry_count = 0
    fallback_used = False
    model_used = settings.DEFAULT_MODEL

    try:
        # ── 1. ROUTER NODE ─────────────────────────────────────────────
        # Decides rag / wikipedia / general. Never generates the final answer.
        decision = await route_query(topic, retriever)
        route = decision.route

        # ── 2. SELECTED PATH ────────────────────────────────────────────
        if route == ROUTE_RAG:
            # RAG PATH: reuse existing retriever → context
            evaluator.start_retrieval()
            rag_result = await rag_node.retrieve(topic, player_name=decision.player_name)
            evaluator.end_retrieval()
            retrieved_chunks = rag_result.chunks

            if rag_result.is_empty():
                # Empty RAG result → try Wikipedia for the known player, else graceful.
                if decision.player_name:
                    route = ROUTE_WIKIPEDIA
                    wiki = await aget_wikipedia_player(decision.player_name)
                    if not wiki.ok:
                        return _graceful_response(
                            topic, evaluator, model_used, wiki.error,
                            source="Wikipedia", route=route,
                        )
                    context = wiki.content
                    source_label = "Wikipedia"
                    retrieved_chunks = [wiki.content]
                else:
                    return _graceful_response(
                        topic, evaluator, model_used,
                        "I couldn't find relevant information in the knowledge base for that query.",
                        source="RAG", route=ROUTE_RAG,
                    )
            else:
                context = rag_node.build_context(retrieved_chunks)
                source_label = "RAG"

        elif route == ROUTE_WIKIPEDIA:
            # WIKIPEDIA TOOL PATH: tool retrieves data, LLM writes the answer
            player_name = decision.player_name or topic
            wiki = await aget_wikipedia_player(player_name)
            if not wiki.ok:
                return _graceful_response(
                    topic, evaluator, model_used, wiki.error,
                    source="Wikipedia", route=route,
                )
            context = wiki.content
            source_label = "Wikipedia"
            retrieved_chunks = [wiki.content]

        else:
            # GENERAL PATH: not about a player → graceful decline, no tool call.
            return _graceful_response(
                topic, evaluator, model_used, GENERAL_DECLINE,
                source="General", route=ROUTE_GENERAL,
            )

        # ── 3. FINAL LLM GENERATION ────────────────────────────────────
        # Both paths feed their retrieved data into the SAME prompt as context,
        # and the LLM is instructed to use ONLY that context.
        prompt_template = get_rag_prompt_template(topic, story_type)
        prompt = prompt_template.format(context=context)

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

        story = result["story"]
        retry_count = result["retry_count"]
        fallback_used = result["fallback_used"]
        model_used = FALLBACK_MODEL if fallback_used else model_used

    except Exception as err:
        logger.error("generate_story failed: %s", err, exc_info=True)
        return _graceful_response(
            topic, evaluator, model_used,
            "Something went wrong while generating your story. Please try again.",
            source=source_label, route=route,
        )

    # ── 4. METRICS + RESPONSE ──────────────────────────────────────────
    metrics = evaluator.build_metrics(
        model_used=model_used,
        retrieved_chunks=len(retrieved_chunks),
        retry_count=retry_count,
        fallback_used=fallback_used,
        sources=[{"player": topic, "source": source_label}],
    )

    response = {
        "story": story,
        "evaluation": metrics,
        "route": route,
        "source": source_label,
    }

    # Raw context is large — only include when the caller explicitly asks for it
    if debug:
        response["retrieved_context"] = retrieved_chunks

    return response