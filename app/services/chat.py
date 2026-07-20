from app.services.llm_client import ask_llm
from app.services.retry_service import retry_call
from app.services.fallback_service import fallback_call

from app.rag.retriever import Retriever

from app.prompts.rag_prompt import build_rag_prompt

from app.services.evaluation_service import EvaluationService


retriever = Retriever()


def welcome_message():

    return {

        "message": "Welcome To Story Forge AI 2.0"

    }


def generate_story(topic, story_type):

    evaluator = EvaluationService()

    evaluator.start_request()

    # -------------------------
    # Retrieve Context
    # -------------------------

    evaluator.start_retrieval()

    chunks = retriever.retrieve(topic)

    evaluator.end_retrieval()

    # -------------------------
    # Build Prompt
    # -------------------------

    context = "\n\n".join(chunks)

    prompt = build_rag_prompt(

        topic,

        story_type,

        context

    )

    # -------------------------
    # Generate Story
    # -------------------------

    evaluator.start_llm()

    try:
        result = retry_call(
            lambda: ask_llm(prompt)
        )
    except Exception as err:
        print(f"[Chat Service] Primary model failed after retries: {err}")
        print("[Chat Service] Invoking fallback_call...")
        fallback_story = fallback_call(prompt)
        result = {
            "story": fallback_story,
            "retry_count": 3,
            "fallback_used": True
        }

    evaluator.end_llm()

    # -------------------------
    # Retry Information
    # -------------------------

    story = result["story"]

    retry_count = result["retry_count"]

    fallback_used = result["fallback_used"]

    # -------------------------
    # Build Metrics
    # -------------------------

    from app.core.config import settings
    model_used = "groq/llama-3.3-70b-versatile" if fallback_used else settings.DEFAULT_MODEL

    metrics = evaluator.build_metrics(

        model_used=model_used,

        retrieved_chunks=len(chunks),

        retry_count=retry_count,

        fallback_used=fallback_used,

        sources=[

            {

                "player": topic,

                "source": "Wikipedia"

            }

        ]

    )

    # -------------------------
    # Response
    # -------------------------

    return {

        "retrieved_context": chunks,

        "story": story,

        "evaluation": metrics

    }