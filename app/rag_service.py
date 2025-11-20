# app/rag_service.py

from typing import TypedDict
from pathlib import Path
import numpy as np
from .llm_client import LlmClient
from .models import Document


class EmbeddingIndex:
    def __init__(self, docs: list[dict], embeddings: np.ndarray):
        self.docs = docs
        self.embeddings = embeddings

    @classmethod
    def from_documents(cls, docs: list["Document"], llm: LlmClient) -> "EmbeddingIndex":
        embeddings = llm.generate_embeddings([d["text"] for d in docs])
        return cls(docs, embeddings)


class RagService:
    def __init__(self, index: EmbeddingIndex, llm: LlmClient, prompt_dir: Path):
        self.index = index
        self.llm_client = llm

        if not prompt_dir.exists():
            raise FileNotFoundError(
                f"Prompt directory not found: {prompt_dir.resolve()}"
            )
        self.prompt_dir = prompt_dir

    def _load_prompt(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.txt"
        return path.read_text(encoding="utf-8")

    def rank_document_relevancy(self, query_vec: np.ndarray, top_k: int = 3):
        doc_embs = self.index.embeddings  # (N, D)

        # cosine similarity
        dot = doc_embs @ query_vec  # (N,)
        doc_norms = np.linalg.norm(doc_embs, axis=1)  # (N,)
        q_norm = np.linalg.norm(query_vec)  # scalar
        scores = dot / (doc_norms * q_norm + 1e-10)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, i in enumerate(top_indices, start=1):
            doc = self.index.docs[i]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[i]),
                    "id": doc["id"],
                    "text": doc["text"],
                }
            )
        return results

    def answer_with_citations(self, query: str):
        # Generate query embedding vectors
        query_embedding_vec = self.llm_client.generate_embedding(query)

        # Rank documents to find the most relevant to the query
        ranked_docs = self.rank_document_relevancy(query_embedding_vec)

        # Load and template prompt
        prompt_template = self._load_prompt("response_prompt")

        context_parts = []
        for d in ranked_docs:
            # e.g. [1] refund.txt
            header = f"[{d['rank']}] {d['id']}"
            context_parts.append(f"{header}\n{d['text']}\n")
        context_block = "\n---\n".join(context_parts)

        prompt = prompt_template.format(context_block=context_block, query=query)

        # Generate and return response
        chat_response = self.llm_client.generate_chat_response(prompt)
        return {"answer": chat_response, "documents": ranked_docs}

        pass
