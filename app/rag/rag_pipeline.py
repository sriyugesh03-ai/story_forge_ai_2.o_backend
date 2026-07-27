import logging
from pathlib import Path
from app.rag.loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingService
from app.rag.vectordb import VectorDatabase
from app.rag.retriever import invalidate_player_cache

logger = logging.getLogger("uvicorn")

async def ingest():
    """
    Scans the data/pdfs directory for documents, extracts their text,
    splits them into chunks, gets vector embeddings, and saves them to MongoDB.
    Prevents re-ingestion if the document first chunk is already in the database.
    """
    logger.info("⏳ [Ingest] Starting startup data ingestion check...")
    
    vectordb = VectorDatabase()
    
    try:
        # Fetch only the distinct source names (e.g. 'Carlos_Alcaraz') already in
        # MongoDB — far lighter than loading all chunk IDs into memory.
        ingested_sources = await vectordb.get_ingested_sources()
        logger.info(f"[Ingest] Found {len(ingested_sources)} already-ingested sources in vector store.")
    except Exception as e:
        logger.error(f"[Ingest] Failed to fetch ingested sources: {e}")
        ingested_sources = set()

    pdf_folder = Path("data/pdfs")
    if not pdf_folder.exists():
        logger.warning(f"⚠️ [Ingest] Folder '{pdf_folder}' not found. Ingestion skipped.")
        return

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    total = len(pdf_files)
    
    if total == 0:
        logger.info("ℹ️ [Ingest] No PDF files found to ingest.")
        return

    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = EmbeddingService()
    
    ingested_any = False
    
    for idx, pdf in enumerate(pdf_files, start=1):
        # Skip if this PDF's source name is already present in MongoDB.
        # pdf.stem is e.g. 'Carlos_Alcaraz' which matches metadata.player.
        if pdf.stem in ingested_sources:
            logger.info(f"[Ingest] [{idx}/{total}] Already ingested (skipped): {pdf.name}")
            continue

        logger.info(f"🚀 [Ingest] [{idx}/{total}] Processing: {pdf.name}")
        
        try:
            # 1. Load document text
            text = loader.load_pdf(str(pdf))
            if not text.strip():
                logger.warning(f"⚠️ [Ingest] Extracted text for {pdf.name} is empty. Skipping.")
                continue
            
            # 2. Split text into chunks
            chunks = chunker.split_text(text)
            if not chunks:
                logger.warning(f"⚠️ [Ingest] No chunks generated for {pdf.name}. Skipping.")
                continue
                
            # 3. Create embeddings for chunks (async)
            embeddings = await embedder.embed_texts(chunks)
            
            # 4. Prepare IDs and Metadatas
            ids = [f"{pdf.stem}_{i}" for i in range(len(chunks))]
            metadatas = [
                {"player": pdf.stem, "source": "Wikipedia"}
                for _ in chunks
            ]
            
            # 5. Store in vector database
            await vectordb.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            ingested_any = True
            logger.info(f"✅ [Ingest] Successfully ingested {len(chunks)} chunks for {pdf.name}")
            
        except Exception as e:
            logger.error(f"❌ [Ingest] Failed to process {pdf.name}: {e}", exc_info=True)

    if ingested_any:
        # Invalidate player cache in the retriever to force reload
        invalidate_player_cache()
        logger.info("🔄 [Ingest] Ingestion completed. Player retriever cache invalidated.")
    else:
        logger.info("ℹ️ [Ingest] Startup ingestion check completed. No new files needed ingestion.")
