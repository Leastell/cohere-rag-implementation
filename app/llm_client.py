# app/llm_client.py

import cohere
import numpy as np


class LlmClient:
    def __init__(self, api_key: str, embed_model: str, chat_model: str):
        self.client = cohere.ClientV2(api_key=api_key)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        embed_response = self.client.embed(
            texts=texts,
            model=self.embed_model,
            input_type="search_document",
            embedding_types=["float"],
        )

        if not embed_response.embeddings.float:
            raise ValueError(
                "Error fetching embeddings from Cohere API.", embed_response
            )

        return np.array(embed_response.embeddings.float)

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.generate_embeddings([text])[0]

    def generate_chat_response(self, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.chat_model, messages=[{"role": "user", "content": prompt}]
            )
            # Safely extract the answer text
            message = getattr(response, "message", None)
            content = getattr(message, "content", None)
            if not content or not isinstance(content, list) or not content[0].text:
                raise ValueError("Invalid response format from Cohere API.", response)
            answer_text = content[0].text
            return answer_text
        except Exception as e:
            raise RuntimeError(f"Error generating chat response: {e}")
