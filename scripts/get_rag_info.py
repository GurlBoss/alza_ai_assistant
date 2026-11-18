"""Prints what is in the RAG Engine corpus: its settings, its files and how many
chunks each file has. The console does not show chunk counts.

Run from the repository root: python -m scripts.get_rag_info [--chunks]
Uses RAG_CORPUS from .env and your gcloud application-default credentials.
--chunks also prints the beginning of every chunk.
"""

import argparse
from collections import Counter

import google.auth
from google.auth.transport.requests import AuthorizedSession

from app.config import Settings

# retrieveContexts returns at most this many chunks, so it is the most we can count.
MAX_CHUNKS = 100


def main() -> None:
    """Loads the corpus, its files and its chunks, and prints a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", action="store_true", help="print every chunk")
    show_chunks = parser.parse_args().chunks

    corpus_name = Settings().RAG_CORPUS
    # projects/<project>/locations/<region>/ragCorpora/<id>
    parent = corpus_name.split("/ragCorpora/")[0]
    api = f"https://{parent.split('/locations/')[1]}-aiplatform.googleapis.com/v1"
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)

    corpus = _get(session, f"{api}/{corpus_name}")
    files = _list_files(session, f"{api}/{corpus_name}/ragFiles")
    chunks = _all_chunks(session, f"{api}/{parent}:retrieveContexts", corpus_name)

    _print_corpus(corpus, len(files), len(chunks))
    _print_files(files, Counter(chunk.get("sourceDisplayName") for chunk in chunks))
    if show_chunks:
        _print_chunks(chunks)


def _get(session: AuthorizedSession, url: str, **params: str) -> dict:
    """One GET request to the Vertex AI API; raises on an error status."""
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _list_files(session: AuthorizedSession, url: str) -> list[dict]:
    """All files in the corpus, following the pages of the listing."""
    files: list[dict] = []
    page_token = ""
    while True:
        page = _get(session, url, pageSize="100", pageToken=page_token)
        files += page.get("ragFiles", [])
        page_token = page.get("nextPageToken", "")
        if not page_token:
            break
    return files


def _all_chunks(session: AuthorizedSession, url: str, corpus_name: str) -> list[dict]:
    """Up to MAX_CHUNKS chunks of the corpus: a retrieval with no distance limit
    returns every chunk when there are fewer than that."""
    body = {
        "vertex_rag_store": {"rag_resources": [{"rag_corpus": corpus_name}]},
        "query": {"text": "*", "rag_retrieval_config": {"top_k": MAX_CHUNKS}},
    }
    response = session.post(url, json=body, timeout=60)
    response.raise_for_status()
    return response.json().get("contexts", {}).get("contexts", [])


def _print_corpus(corpus: dict, file_count: int, chunk_count: int) -> None:
    """The corpus name, state, embedding model, vector store and totals."""
    vector_db = corpus.get("vectorDbConfig", {})
    endpoint = (
        vector_db.get("ragEmbeddingModelConfig", {})
        .get("vertexPredictionEndpoint", {})
        .get("endpoint", "")
    )
    stores = [key for key in vector_db if key != "ragEmbeddingModelConfig"]
    chunks = f"{chunk_count}" if chunk_count < MAX_CHUNKS else f"at least {MAX_CHUNKS}"
    print(f"Corpus:      {corpus.get('displayName')}  ({corpus.get('name')})")
    print(f"Description: {corpus.get('description', '-')}")
    print(f"State:       {corpus.get('corpusStatus', {}).get('state', '-')}")
    print(f"Embeddings:  {endpoint.rsplit('/', 1)[-1] or '-'}")
    print(f"Vector DB:   {', '.join(stores) or '-'}")
    print(f"Created:     {corpus.get('createTime', '-')}")
    print(f"Files:       {file_count}")
    print(f"Chunks:      {chunks}")


def _print_files(files: list[dict], chunks_per_file: Counter) -> None:
    """One line per file: chunk count, state, source and last update."""
    print("\nChunks  State     Source        Updated               File")
    for item in sorted(files, key=lambda f: f.get("displayName", "")):
        name = item.get("displayName", "-")
        source = next((key for key in item if key.endswith("Source")), "-")
        print(
            f"{chunks_per_file.get(name, 0):>6}  "
            f"{item.get('fileStatus', {}).get('state', '-'):<8}  "
            f"{source.removesuffix('Source'):<12}  "
            f"{item.get('updateTime', '-')[:19]:<20}  {name}"
        )


def _print_chunks(chunks: list[dict]) -> None:
    """The source, length and beginning of every chunk."""
    for number, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.get("text", "").split())
        print(
            f"\n--- chunk {number}: {chunk.get('sourceDisplayName')}, {len(text)} chars"
        )
        print(text[:300] + ("…" if len(text) > 300 else ""))


if __name__ == "__main__":
    main()
