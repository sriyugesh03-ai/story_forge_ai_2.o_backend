from app.prompts.templates import (
    BIOGRAPHY_TEMPLATE,
    DOCUMENTARY_TEMPLATE,
    CAREER_TIMELINE_TEMPLATE,
    REEL_TEMPLATE
)

def build_story_prompt(topic: str, story_type : str):

    if story_type.lower() == "biography":

        template = BIOGRAPHY_TEMPLATE

    elif story_type.lower() == "timeline":

        template = CAREER_TIMELINE_TEMPLATE

    elif story_type.lower() == "reel":

        template = REEL_TEMPLATE

    else:

        template = DOCUMENTARY_TEMPLATE

    return template.format(topic=topic)