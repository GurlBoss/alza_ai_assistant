"""Knowledge base: Vertex AI RAG Engine, used as a retrieval tool in the Gemini call.

The corpus (documents, chunking, embeddings, index) is managed by RAG Engine;
this module only builds the tool and logs what Gemini retrieved.
"""

import logging
from typing import Any

from google.genai import types as genai_types

from app.config import Settings

log = logging.getLogger(__name__)


def build_rag_tool(settings: Settings) -> genai_types.Tool | None:
    """The retrieval tool for RAG_CORPUS, or None when no corpus is configured."""
    if not settings.RAG_CORPUS:
        return None
    return genai_types.Tool(
        retrieval=genai_types.Retrieval(
            vertex_rag_store=genai_types.VertexRagStore(
                rag_resources=[
                    genai_types.VertexRagStoreRagResource(
                        rag_corpus=settings.RAG_CORPUS
                    )
                ],
                rag_retrieval_config=genai_types.RagRetrievalConfig(
                    top_k=settings.RAG_TOP_K,
                    filter=genai_types.RagRetrievalConfigFilter(
                        vector_distance_threshold=settings.RAG_DISTANCE_THRESHOLD
                    ),
                ),
            )
        )
    )


def log_grounding(response: Any, debug: bool = False) -> None:
    """Logs the retrieved chunks from the response's grounding metadata: their
    count and source documents, plus their text when debug is on."""
    candidates = getattr(response, "candidates", None) or []
    metadata = candidates[0].grounding_metadata if candidates else None
    contexts = [
        chunk.retrieved_context
        for chunk in (getattr(metadata, "grounding_chunks", None) or [])
        if chunk.retrieved_context is not None
    ]
    sources = sorted({_source_name(context) for context in contexts})
    log.info(
        "RAG: %d chunk(s) retrieved, sources: %s", len(contexts), sources or "none"
    )
    if debug:
        # The knowledge base contains no personal data, so the text may be logged.
        for number, context in enumerate(contexts, start=1):
            log.info(
                "RAG chunk %d from %s: %s", number, _source_name(context), context.text
            )


def _source_name(context: Any) -> str:
    """Document name: the title, or the last part of the URI."""
    if context.title:
        return context.title
    return (context.uri or "unknown").rstrip("/").rsplit("/", 1)[-1]
