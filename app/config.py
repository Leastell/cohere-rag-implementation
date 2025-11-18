# app/config.py - .env load and project constant configuration

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_DIR = BASE_DIR / "documents"
PROMPT_DIR = BASE_DIR / "prompts"

EMBED_MODEL = os.getenv("EMBED_MODEL", "embed-english-light-v3.0")
CHAT_MODEL = os.getenv("CHAT_MODEL", "command-a-03-2025")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")