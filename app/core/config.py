import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application-wide configuration settings loaded from environment variables."""
    APP_NAME: str = os.getenv("APP_NAME", "Story Forge AI 2.O")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    XAI_API_KEY:str = os.getenv("XAI_API_KEY","")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "5000"))
    MANGO_DB_URL :str = os.getenv("MANGO_DB_URL")
    MONGO_DB_LOCAL : str = os.getenv("MONGO_DB_LOCAL")

settings = Settings()

