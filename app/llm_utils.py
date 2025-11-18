# app/llm-utils.py - Utilities related to cohere, llm processing, and embedding data structures

import cohere
import os
import numpy as np
from .config import COHERE_API_KEY, DOCUMENT_DIR, PROMPT_DIR, EMBED_MODEL, CHAT_MODEL

co = cohere.ClientV2(api_key=os.getenv(COHERE_API_KEY))

class EmbeddingIndex:
    def __init__(self, docs: list[dict], embeddings: np.ndarray):
        self.docs = docs
        self.embeddings = embeddings

def _load_prompt(name: str) -> str:
    if not PROMPT_DIR.exists():
        raise FileNotFoundError(f"Prompt directory not found: {PROMPT_DIR.resolve()}")
    
    path = PROMPT_DIR / f"{name}.txt"
    return path.read_text()
    

def _get_documents() -> list[dict]:
    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(f"Document directory not found: {DOCUMENT_DIR.resolve()}")
    
    docs = []
    for file_path in DOCUMENT_DIR.iterdir():      
        if not file_path.is_file():
            continue

        text = file_path.read_text(encoding="utf-8")
        docs.append(
            {
                "id": file_path.name,
                "path": file_path,
                "text": text
            }
        )

    return docs

def _create_embeddings(docs: list[dict]) -> np.ndarray:
    texts = [doc["text"] for doc in docs]

    embed_response = co.embed(
        texts=texts,
        model=EMBED_MODEL,
        input_type="search_document",
        embedding_types=["float"]
    )

    if not embed_response.embeddings.float:
        raise ValueError("Error fetching embeddings from Cohere API.", embed_response)

    return np.array(embed_response.embeddings.float)

def build_embedding_index() -> EmbeddingIndex:
    docs = _get_documents()
    embeddings = _create_embeddings(docs)
    return EmbeddingIndex(docs, embeddings)

def _rank_documents(query_vec: np.ndarray, index, top_k: int = 3) -> list[dict]:
    doc_embs = index.embeddings

    # cosine similarity
    dot = doc_embs @ query_vec
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    q_norm = np.linalg.norm(query_vec)
    scores = dot / (doc_norms * q_norm + 1e-10)

    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, i in enumerate(top_indices, start=1):
        doc = index.docs[i]
        results.append(
            {
                "rank": rank,
                "score": float(scores[i]),
                "id": doc["id"],
                "text": doc["text"],
            }
        )
    return results


def answer_with_citations(query:  str, embedding_index: dict):
    # 1. Generate query embedding vector
    query_embed_response = co.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_document",
        embedding_types=["float"]
    )
    query_vec = np.array(query_embed_response.embeddings.float[0], dtype=float)

    # 2. Rank documents to find the most relevant ones
    ranked_docs = _rank_documents(query_vec, embedding_index)

    # 3. Build prompt
    prompt = _load_prompt("response_prompt")

    context_parts = []
    for d in ranked_docs:
        # e.g. [1] refund.txt
        header = f"[{d['rank']}] {d['id']}"
        context_parts.append(f"{header}\n{d['text']}\n")

    context_block = "\n---\n".join(context_parts)

    prompt = prompt.format(
        context_block=context_block,
        query=query
    )

    # 4. Generate response with Cohere
    response = co.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content" : prompt}]
    )
    answer_text = response.message.content[0].text

    # 5. Return the response
    return {
        "answer": answer_text,
        "documents": ranked_docs,
    }