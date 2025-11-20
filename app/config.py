from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    document_dir: Path
    prompt_dir: Path

    embed_model: str
    chat_model: str
    cohere_api_key: str


def load_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent.parent

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is not set in the environment.")

    return AppConfig(
        base_dir=base_dir,
        document_dir=base_dir / "documents",
        prompt_dir=base_dir / "prompts",
        embed_model=os.getenv("EMBED_MODEL", "embed-english-light-v3.0"),
        chat_model=os.getenv("CHAT_MODEL", "command-a-03-2025"),
        cohere_api_key=cohere_api_key,
    )
