"""Knowledge base entry point. The rest of the app uses only this module:
retrieve() for an email, index_file() / delete_file() / index_all() for files."""

import hashlib
import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.knowledge_base.chunk_store import Chunk, ChunkStore, FirestoreChunkStore
from app.knowledge_base.chunking import chunk_document
from app.knowledge_base.embeddings import Embedder, GeminiEmbedder
from app.knowledge_base.loader import GcsLoader, Loader, is_kb_file
from app.knowledge_base.vector_store import VectorStore, VertexVectorSearch

log = logging.getLogger(__name__)

TOP_K = 4

# Outcomes of index_file / delete_file
INDEXED = "INDEXED"
UNCHANGED = "UNCHANGED"
DELETED = "DELETED"
IGNORED = "IGNORED"


@dataclass
class Snippet:
    source: str  # document name shown in the prompt, e.g. "shipping.md"
    text: str
    score: float
    chunk_id: str


def is_kb_object(name: str) -> bool:
    """Whether a bucket object belongs to the knowledge base (docs/, md or txt)."""
    return is_kb_file(name)


class KnowledgeBase:
    def __init__(
        self,
        loader: Loader,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
        min_score: float,
    ) -> None:
        self._loader = loader
        self._embedder = embedder
        self._vectors = vector_store
        self._chunks = chunk_store
        self._min_score = min_score

    @classmethod
    def from_settings(cls, settings: Settings) -> "KnowledgeBase | None":
        """None when the knowledge base is not configured (replies go out without it)."""
        required = (
            "KB_BUCKET",
            "VS_INDEX_ID",
            "VS_ENDPOINT_ID",
            "VS_DEPLOYED_INDEX_ID",
        )
        missing = [name for name in required if not getattr(settings, name)]
        if missing:
            log.warning("Knowledge base disabled, missing: %s", ", ".join(missing))
            return None
        project = settings.GCP_PROJECT_ID
        return cls(
            loader=GcsLoader(project, settings.KB_BUCKET),
            embedder=GeminiEmbedder(project, settings.REGION, settings.EMBEDDING_MODEL),
            vector_store=VertexVectorSearch(
                project,
                settings.REGION,
                settings.VS_INDEX_ID,
                settings.VS_ENDPOINT_ID,
                settings.VS_DEPLOYED_INDEX_ID,
            ),
            chunk_store=FirestoreChunkStore(project),
            min_score=settings.MIN_SCORE,
        )

    # --- retrieval ---------------------------------------------------------------

    def retrieve(self, subject: str, body: str) -> list[Snippet]:
        """Relevant snippets for an email. Never raises: on any failure it logs a
        warning and returns nothing, so the reply goes out without KNOWLEDGE."""
        started = time.monotonic()
        try:
            vector = self._embedder.embed_query(f"{subject}\n\n{body}".strip())
            matches = self._vectors.query(vector, TOP_K)
            relevant = [
                (cid, score) for cid, score in matches if score >= self._min_score
            ]
            chunks = self._chunks.get([cid for cid, _ in relevant])
        except Exception as exc:
            log.warning(
                "RAG failed after %.1f s (%s), replying without knowledge",
                time.monotonic() - started,
                type(exc).__name__,
            )
            return []

        snippets = [
            Snippet(_label(chunks[cid].source), chunks[cid].text, score, cid)
            for cid, score in relevant
            if cid in chunks
        ]
        log.info(
            "RAG in %.1f s: %s; used %s (min score %.2f)",
            time.monotonic() - started,
            [f"{cid}={score:.3f}" for cid, score in matches] or "no matches",
            [s.chunk_id for s in snippets] or "nothing",
            self._min_score,
        )
        if not matches:
            log.warning("Knowledge base returned no matches; is it empty?")
        return snippets

    # --- indexing ----------------------------------------------------------------

    def index_file(self, name: str, force: bool = False) -> str:
        """Indexes one file; unchanged files (same hash) are skipped, so repeated
        events are harmless. A changed file replaces all its old chunks."""
        if not is_kb_file(name):
            return IGNORED
        started = time.monotonic()
        data = self._loader.read(name)
        if data is None:
            log.info("KB file %s no longer exists, removing it", name)
            return self.delete_file(name)

        file_hash = hashlib.sha256(data).hexdigest()
        if not force and self._chunks.file_hash(name) == file_hash:
            log.info("KB file %s unchanged, skipping", name)
            return UNCHANGED

        self._remove_chunks(name)
        texts = chunk_document(data.decode("utf-8", errors="replace"))
        chunks = [
            Chunk(f"{name}#{index}", name, index, text, file_hash)
            for index, text in enumerate(texts)
        ]
        if chunks:
            vectors = self._embedder.embed_documents([c.text for c in chunks])
            # Firestore first: a vector without its text would be dropped at query time.
            self._chunks.save(chunks)
            self._vectors.upsert([(c.id, v) for c, v in zip(chunks, vectors)])
        log.info(
            "KB file %s indexed: %d chunk(s) in %.1f s",
            name,
            len(chunks),
            time.monotonic() - started,
        )
        return INDEXED

    def delete_file(self, name: str) -> str:
        """Removes a deleted file's chunks. Overwriting a file also sends a delete
        event for the old version; a file that still exists is left alone."""
        if not is_kb_file(name):
            return IGNORED
        if self._loader.read(name) is not None:
            log.info("KB file %s still exists (overwritten), keeping it", name)
            return IGNORED
        removed = self._remove_chunks(name)
        log.info("KB file %s deleted: %d chunk(s) removed", name, removed)
        return DELETED

    def index_all(self, force: bool = False) -> dict[str, str]:
        """Indexes every file in the bucket and removes files that are gone."""
        files = self._loader.list_files()
        results = {name: self.index_file(name, force) for name in files}
        for name in sorted(self._chunks.sources() - set(files)):
            self._remove_chunks(name)
            results[name] = DELETED
        return results

    def _remove_chunks(self, name: str) -> int:
        ids = self._chunks.chunk_ids(name)
        if ids:
            # Vector Search first: a dangling vector is harmless, a missing text isn't.
            self._vectors.remove(ids)
            self._chunks.delete(ids)
        return len(ids)


def _label(source: str) -> str:
    return source.removeprefix("docs/")
