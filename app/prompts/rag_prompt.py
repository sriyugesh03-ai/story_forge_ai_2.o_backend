from app.prompts.story_prompt import build_story_prompt

def build_rag_prompt(

    topic: str,

    story_type: str,

    context: str

):

    story_task = build_story_prompt(topic, story_type)

    return f"""
You are Sports Story Forge AI.

Use ONLY the information given in the context below.

If something is missing,
say it is not available in the retrieved knowledge.

====================

CONTEXT

{context}

====================

TASK

{story_task}

Keep the story:

- Factually correct

- Cinematic

- Emotional

- Engaging

Do not invent facts.

"""