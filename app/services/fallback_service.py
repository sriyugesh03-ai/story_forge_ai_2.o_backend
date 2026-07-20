from app.services.llm_client import ask_llm

FALLBACK_MODEL = "groq/llama-3.3-70b-versatile"


def fallback_call(prompt: str):

    print("=" * 50)
    print("[Fallback] Primary model failed.")
    print(f"[Fallback] Switching to {FALLBACK_MODEL}")
    print("=" * 50)

    return ask_llm(
        prompt=prompt,
        model=FALLBACK_MODEL
    )






