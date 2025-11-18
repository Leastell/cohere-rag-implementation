# RAG implementation - Exercise

This project implements a small Retrival-Augmented Generation (RAG) system using **Cosine Similarity** and **Cohere LLMs**. I built it on top of a **FastAPI** backend.

I wrote this as a research task to prepare for a job interview The goal was primarily to demonstrate professional code structure and testing practices while brushing up on LLMs.

---

## 📌 Overview

This service allows users to submit natural-language questions to the `/query` endpoint via POST request. The service will respond with an LLM-generated natural-language response that cites real documents.

At startup, the application generates embeddings for the files in the `documents/` directory using a [Cohere Embedding model](https://docs.cohere.com/docs/embeddings). These models are stored for later reference as requests are made.

For each incoming query, the user's prompt is embedded using the same method that we used to generate the document embeddings. Then, using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), the query embeddings are compared to the cached document embeddings. The top `k` most-similar documents are used to inform a chat response. This response is generated using the [Cohere chat endpoint](https://docs.cohere.com/reference/chat).

### Example

User prompt:

```
{
    "query": "Can I return my purchase?"
}
```

Service response[^1]:

```
{
    "answer": "Yes, you can return your purchase within 30 days as long as the item is in its original condition [1]. To initiate a return, contact customer support with your order number [1]. Note that return shipping costs may apply depending on the reason for the return [1].",
    "documents": [
        {
            "rank": 1,
            "score": 0.5211235585017844,
            "id": "returns.txt",
            "text": "Items can be returned within 30 days as long as they are in their original condition. \nTo start a return, contact customer support with your order number. \nReturn shipping costs may apply depending on the reason for return. \nExchanges are available for items of equal value."
        }
    ]
}
```

[^1]: topK is limited to 1 for demonstration purposes.

---

## Architecture

High-level response generation flow:

```
Query
    ↓
Embed query using Cohere
    ↓
Determine cosine similarity against stored document embeddings
    ↓
Select Top-K documents
    ↓
Build prompt with numbered citations
    ↓
Generate answer using Cohere
    ↓
Return JSON response with answer + citations + context docs
```

## Project structure

```
├── app
│   ├── api.py                  # API Routes
│   ├── config.py               # Environment variables and application config
│   ├── llm_utils.py            # Embeddings, retrieval, prompt building, Cohere calls
│   └── main.py                 # FastAPI initialization and lifespan
├── documents                   # .txt documents for RAG reference
│   └── *.txt
├── prompts
│   └── response_prompt.txt     # Prompt used for LLM response generation
├── README.md
└── tests
    └── test_llm_utils.py
```

## Running Locally

### 1. Set up `.venv` and install dependencies.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create .env

Create a `.env` file with the following variable:

```
COHERE_API_KEY=[Your API key]
```

You can optionally override the default embedding and chat models by setting the following variables:

```
    EMBED_MODEL=[Chosen Cohere embedding model]
    CHAT_MODEL=[Chosen Cohere chat model]
```

### 3. Start the server

Run this command in your (venv-activated) terminal to start the FastAPI server

```
fastapi dev app/main.py
```

Document embeddings are generated automatically on startup.

## Testing

Tests use `pytest` and `monkeypatch` to mock Cohere calls.
Run tests by executing the `pytest` command in your terminal.
