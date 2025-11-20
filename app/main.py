# app/main.py

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI

from .config import load_config
from .llm_client import LlmClient
from .rag_service import RagService, EmbeddingIndex
from .models import Document
from .api import router


def load_documents(document_dir: Path) -> List[Document]:
    if not document_dir.exists():
        raise FileNotFoundError(
            f"Document directory not found: {document_dir.resolve()}"
        )

    docs: List[Document] = []
    for file_path in document_dir.iterdir():
        if not file_path.is_file():
            continue

        docs.append(
            Document(
                id=file_path.name,
                path=file_path,
                text=file_path.read_text(encoding="utf-8"),
            )
        )

    return docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    # Create LLM client
    llm = LlmClient(
        api_key=config.cohere_api_key,
        embed_model=config.embed_model,
        chat_model=config.chat_model,
    )

    # Load documents and build embedding index
    docs = load_documents(config.document_dir)
    index = EmbeddingIndex.from_documents(docs, llm)

    # Create RAG service
    app.state.rag_service = RagService(
        index=index, llm=llm, prompt_dir=config.prompt_dir
    )

    # Hand control back to FastAPI
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
