"""Shows which knowledge base chunks RAG Engine retrieves for a question.

Run from the repository root: python -m scripts.test_rag "Jak dlouho trvá doručení?"
Uses RAG_CORPUS, RAG_TOP_K and RAG_DISTANCE_THRESHOLD from .env, the same values
the reply uses.
"""

import argparse

import google.auth
from google.auth.transport.requests import AuthorizedSession

from app.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    question = parser.parse_args().question

    settings = Settings()
    if not settings.RAG_CORPUS:
        raise SystemExit("Set RAG_CORPUS in .env")
    # projects/<project>/locations/<region>/ragCorpora/<id>
    parent = settings.RAG_CORPUS.split("/ragCorpora/")[0]
    region = parent.split("/locations/")[1]

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    response = AuthorizedSession(credentials).post(
        f"https://{region}-aiplatform.googleapis.com/v1/{parent}:retrieveContexts",
        json={
            "vertex_rag_store": {
                "rag_resources": [{"rag_corpus": settings.RAG_CORPUS}]
            },
            "query": {
                "text": question,
                "rag_retrieval_config": {
                    "top_k": settings.RAG_TOP_K,
                    "filter": {
                        "vector_distance_threshold": settings.RAG_DISTANCE_THRESHOLD
                    },
                },
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    contexts = response.json().get("contexts", {}).get("contexts", [])

    print(
        f"{len(contexts)} chunk(s), top_k={settings.RAG_TOP_K}, "
        f"distance threshold={settings.RAG_DISTANCE_THRESHOLD}\n"
    )
    for number, context in enumerate(contexts, start=1):
        source = context.get("sourceDisplayName") or context.get("sourceUri", "?")
        distance = context.get("distance", context.get("score"))
        print(f"--- {number}. {source}  distance={distance}")
        print(context.get("text", "").strip(), "\n")


if __name__ == "__main__":
    main()
