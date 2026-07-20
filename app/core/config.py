
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:

    APP_NAME = os.getenv("APP_NAME")

    APP_VERSION = os.getenv("APP_VERSION")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    TEMPERATURE = float(

        os.getenv("TEMPERATURE")

    )

    MAX_TOKENS = int(

        os.getenv("MAX_TOKENS")

    )

settings = Settings()
