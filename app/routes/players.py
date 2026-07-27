from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path

from app.rag.retriever import get_retriever
from app.routes.auth import get_current_user

router = APIRouter(prefix="/players", tags=["players"])

PDF_FOLDER = Path("data/pdfs")


# ------------------------------------------------------------------
# GET /players
# List all available players (from PDFs on disk + indexed stats)
# ------------------------------------------------------------------
@router.get("/")
async def list_players(current_user: dict = Depends(get_current_user)):
    """
    Returns all players available in the system.
    - `pdf_available`: player PDFs present on disk
    - `indexed_stats`: per-player chunk counts in the vector DB
    """
    retriever = get_retriever()  # shared singleton

    # PDFs on disk
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    pdf_players = [
        {"player": f.stem, "filename": f.name, "size_mb": round(f.stat().st_size / 1_048_576, 2)}
        for f in pdf_files
    ]

    # Indexed stats from vector DB
    indexed_stats = await retriever.get_player_stats()
    indexed_map = {s["player"]: s["chunks_indexed"] for s in indexed_stats}

    # Merge: annotate each PDF entry with indexing status
    for entry in pdf_players:
        chunks = indexed_map.get(entry["player"], 0)
        entry["indexed"] = chunks > 0
        entry["chunks_indexed"] = chunks

    return {
        "total_players": len(pdf_players),
        "total_indexed": len(indexed_stats),
        "players": pdf_players,
    }


# ------------------------------------------------------------------
# GET /players/{player_name}/chunks
# Return indexed chunks for a given player (paginated)
# ------------------------------------------------------------------
@router.get("/{player_name}/chunks")
async def get_player_chunks(
    player_name: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve indexed chunks in the vector DB for the given player.
    The player_name is matched flexibly (case-insensitive, underscore-tolerant).
    Use `limit` to control how many chunks are returned (default: 50).
    """
    retriever = get_retriever()  # shared singleton

    try:
        result = await retriever.retrieve_all_for_player(player_name, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return result
