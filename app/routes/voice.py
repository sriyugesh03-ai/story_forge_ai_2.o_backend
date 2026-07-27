# import os
# import tempfile
# import logging
# from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse
# from app.ai.voice import speech_to_text, text_to_speech
# from app.routes.auth import get_current_user

# logger = logging.getLogger(__name__)

# router = APIRouter(prefix="/voice", tags=["voice"])


# # ------------------------------------------------------------------
# # POST /voice/ask
# # Accept browser audio recording, transcribe, generate LLM answer
# # ------------------------------------------------------------------
# @router.post("/ask")
# async def voice_ask(
#     audio: UploadFile = File(...),
#     current_user: dict = Depends(get_current_user),
# ):
#     """
#     Accept an audio file recording from the browser (webm / mp3 / wav),
#     transcribe it with Groq Whisper, generate an AI answer with Gemini,
#     and return both the transcript and the generated answer as JSON.
#     """
#     # Determine suffix from the uploaded filename (browser sends webm)
#     original_name = audio.filename or "recording.webm"
#     suffix = os.path.splitext(original_name)[1] or ".webm"

#     # Write upload to a temp file so the voice module can open it
#     tmp_path: str | None = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             content = await audio.read()
#             if not content:
#                 raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
#             tmp.write(content)
#             tmp_path = tmp.name

#         logger.info("Voice ask: saved %d bytes to %s", len(content), tmp_path)

#         transcript, answer = await speech_to_text(tmp_path)

#         return {
#             "transcript": transcript,
#             "answer": answer,
#         }

#     except HTTPException:
#         raise
#     except Exception as exc:
#         logger.error("Voice ask failed: %s", exc)
#         raise HTTPException(status_code=500, detail=f"Voice processing failed: {exc}")
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# # ------------------------------------------------------------------
# # POST /voice/speak
# # Convert text to speech and stream the MP3 back to the browser
# # ------------------------------------------------------------------
# @router.post("/speak")
# async def voice_speak(
#     payload: dict,
#     current_user: dict = Depends(get_current_user),
# ):
#     """
#     Convert the provided text to speech using edge-tts.
#     Returns the generated MP3 file as a streaming audio response.
#     The file is deleted from disk after the response is sent.
#     """
#     text = payload.get("text", "").strip()
#     voice = payload.get("voice", "en-IN-PrabhatNeural")

#     if not text:
#         raise HTTPException(status_code=400, detail="Text field is required.")

#     try:
#         audio_path = await text_to_speech(text, voice=voice)
#         return FileResponse(
#             path=audio_path,
#             media_type="audio/mpeg",
#             filename="response.mp3",
#             background=None,
#         )
#     except Exception as exc:
#         logger.error("TTS failed: %s", exc)
#         raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {exc}")
