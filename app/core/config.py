import os
from pydantic_settings import BaseSettings
from typing import Dict

class Settings(BaseSettings):
    # --- General App Settings ---
    APP_NAME: str = "Multimodal RAG SaaS"

    # --- Google Gemini API Key ---
    # The environment variable GOOGLE_API_KEY must be set.
    GOOGLE_API_KEY: str

    # --- ChromaDB Settings ---
    # Path to the directory where ChromaDB data will be stored.
    CHROMA_DB_PATH: str = "output/chroma_db"

    # --- User Management (Simple API Key Auth) ---
    # A dictionary mapping static API keys to user IDs.
    # In a real application, this would come from a database.
    # Example format in .env: API_KEYS='{"some-secret-key": "user-1", "another-key": "user-2"}'
    API_KEYS: Dict[str, str] = {"test-key": "test-user"}

    # --- Checkpoint (Conversation History) Settings ---
    # Path to the directory for storing SQLite checkpoint files.
    CHECKPOINT_DIR: str = "output/checkpoints"

    class Config:
        # This tells Pydantic to load variables from a .env file.
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a single, reusable instance of the settings
settings = Settings()
