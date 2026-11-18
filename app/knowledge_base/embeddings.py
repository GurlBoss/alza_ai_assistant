"""Text embeddings with Gemini on Vertex AI."""

import math
from typing import Protocol

from google import genai
from google.genai import types

DIMENSIONS = 768
TIMEOUT_MS = 10_000
BATCH_SIZE = 50


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def normalize(vector: list[float]) -> list[float]:
    """L2-normalizes, so the dot product equals cosine similarity."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class GeminiEmbedder:
    def __init__(self, project: str, location: str, model: str) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
        )
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors += self._embed(
                texts[start : start + BATCH_SIZE], "RETRIEVAL_DOCUMENT"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=DIMENSIONS
            ),
        )
        return [normalize(list(item.values)) for item in response.embeddings]
