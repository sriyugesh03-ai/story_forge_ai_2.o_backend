# import os
# import litellm
# import edge_tts
# from litellm import completion
# from app.core.config import settings


# # -----------------------------------------
# # Speech To Text
# # -----------------------------------------
# async def speech_to_text(audio_path: str) -> tuple[str, str]:
#     """
#     Converts speech to text using Groq Whisper
#     and generates a response using Gemini.
#     Returns (transcript, answer)
#     """

#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio file not found: {audio_path}")

#     with open(audio_path, "rb") as audio_file:
#         transcription = await litellm.atranscription(
#             model="groq/whisper-large-v3-turbo",
#             file=audio_file,
#             api_key=settings.GROQ_API_KEY
#         )

#     transcript = transcription.text

#     response = completion(
#         model="gemini/gemini-2.5-flash",
#         api_key=settings.GEMINI_API_KEY,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Answer the user's question briefly and accurately."
#             },
#             {
#                 "role": "user",
#                 "content": transcript
#             }
#         ]
#     )

#     answer = response.choices[0].message.content

#     return transcript, answer


# # -----------------------------------------
# # Text To Speech
# # -----------------------------------------
# async def text_to_speech(
#     text: str,
#     voice: str = "en-IN-PrabhatNeural"
# ) -> str:
#     """
#     Converts text to speech and saves it as:
#     response_1.mp3
#     response_2.mp3
#     response_3.mp3
#     ...
#     """

#     output_dir = "outputs"

#     os.makedirs(output_dir, exist_ok=True)

#     # Find the next available filename
#     for i in range(1, 100000):

#         filename = f"response_{i}.mp3"

#         output_path = os.path.join(output_dir, filename)

#         if not os.path.exists(output_path):
#             break

#     communicate = edge_tts.Communicate(
#         text=text,
#         voice=voice
#     )

#     await communicate.save(output_path)

#     print(f"Audio saved successfully: {output_path}")

#     return output_path