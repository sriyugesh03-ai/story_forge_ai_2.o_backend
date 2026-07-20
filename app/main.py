from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.llm import router as llm_router
from app.routes.auth import router as auth_router
from app.routes.players import router as players_router
from app.core.db import init_db

app = FastAPI(
    
    title = "Story Forge AI 2.O",
    version = "1.0.0"
)

@app.on_event("startup")
def on_startup():
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(llm_router)
app.include_router(players_router)