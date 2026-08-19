import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routes.llm import router as llm_router
from app.routes.auth import router as auth_router
from app.routes.players import router as players_router
# from app.routes.voice import router as voice_router
from app.db.mongo_db import connect_to_mongo, close_mongo_connection
from app.rag.rag_pipeline import ingest

# Option A: serve the built frontend from this backend (same origin) so the
# refresh-token cookie can use SameSite=strict.
STATIC_DIR = Path(__file__).resolve().parent / "static"


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
    lifespan=lifespan,
)

# CORS is primarily for local development (Vite dev server on another port).
# In production (Option A) the frontend is served from the same origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        settings.FRONTEND_URL,
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(llm_router)
app.include_router(players_router)
# app.include_router(voice_router)

# ── Static frontend (built with `npm run build`, copied into app/static) ──
if (STATIC_DIR / "index.html").exists():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")