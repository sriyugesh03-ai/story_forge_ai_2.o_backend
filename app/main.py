import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.llm import router as llm_router
from app.routes.auth import router as auth_router
from app.routes.players import router as players_router
# from app.routes.voice import router as voice_router
from app.db.mongo_db import connect_to_mongo, close_mongo_connection
from app.rag.rag_pipeline import ingest

# Load allowed origins from environment or default to common development origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")

if ALLOWED_ORIGINS:
    origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
        "https://storyforge-ten-taupe.vercel.app"

    ]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    await connect_to_mongo()
    try:
        await ingest()
    except Exception as e:
        import logging
        logging.getLogger("uvicorn").error(f"Error during startup ingestion: {e}")
    yield
    # Shutdown code
    await close_mongo_connection()

app = FastAPI(
    title="Story Forge AI 2.O",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(llm_router)
app.include_router(players_router)
#app.include_router(voice_router)