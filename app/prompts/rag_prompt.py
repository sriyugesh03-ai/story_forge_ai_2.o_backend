from langchain_core.prompts import ChatPromptTemplate
from app.prompts.story_prompt import build_story_prompt


def get_rag_prompt_template(topic: str, story_type: str) -> ChatPromptTemplate:
    """Return a LangChain ChatPromptTemplate configured with the topic and story type."""
    story_task = build_story_prompt(topic, story_type)
    template_str = f"""You are Sports Story Forge AI.

Use ONLY the information given in the context below.

If something is missing, say it is not available in the retrieved knowledge.

====================
CONTEXT

{{context}}

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
    return ChatPromptTemplate.from_template(template_str)


def build_rag_prompt(topic: str, story_type: str, context: str) -> str:
    """Build formatted prompt string using LangChain ChatPromptTemplate."""
    prompt_template = get_rag_prompt_template(topic, story_type)
    return prompt_template.format(context=context)