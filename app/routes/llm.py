from fastapi import APIRouter, Depends
from app.services.chat import welcome_message
from app.services.chat import generate_story
from app.routes.auth import get_current_user

router = APIRouter()


@router.get("/")
def landing_page():
    return welcome_message()


@router.get("/story")
def story(topic: str, story_type: str, current_user: dict = Depends(get_current_user)):

    return generate_story(topic, story_type)