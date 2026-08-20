"""GitHub data answer prompt — used when the agent answers from GitHub tool data."""


def get_github_prompt(topic: str, context: str) -> str:
    """Return a self-contained prompt instructing the LLM to answer from the GitHub context only."""
    return f"""You are a GitHub data assistant.

Use ONLY the GitHub data provided in the context below.

If something is missing or not available, say so plainly.

====================
CONTEXT

{context}

====================
USER REQUEST

{topic}

Answer the user's request directly and concisely using ONLY the context above.
Do not invent repositories, files, issues, or numbers.
"""