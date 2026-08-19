from fastapi import APIRouter, Depends
from app.services.chat import welcome_message, generate_story
from app.routes.auth import get_current_user

router = APIRouter()


@router.get("/health")
def health_check():
    return welcome_message()


@router.get("/story")
async def story(
    topic: str,
    story_type: str,
    debug: bool = False,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate a sports story using RAG + LLM.

    - **topic**: Player or event name (e.g. "MS Dhoni", "Lionel Messi")
    - **story_type**: biography | timeline | reel | documentary
    - **debug**: Set to `true` to include the raw retrieved chunks in the response
    """
    return await generate_story(topic, story_type, debug=debug)