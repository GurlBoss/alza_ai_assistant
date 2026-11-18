"""Chunk texts and metadata in Firestore (kb_chunks), keyed by chunk ID."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from google.cloud import firestore

COLLECTION = "kb_chunks"


@dataclass
class Chunk:
    id: str  # "<file>#<index>"
    source: str  # file name inside the bucket, e.g. docs/shipping.md
    index: int
    text: str
    file_hash: str


class ChunkStore(Protocol):
    def file_hash(self, source: str) -> str | None:
        """Hash of the indexed version of the file; None when not indexed."""

    def chunk_ids(self, source: str) -> list[str]: ...

    def sources(self) -> set[str]: ...

    def save(self, chunks: list[Chunk]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def get(self, ids: list[str]) -> dict[str, Chunk]: ...


class FirestoreChunkStore:
    def __init__(self, project_id: str) -> None:
        self._client = firestore.Client(project=project_id)
        self._collection = self._client.collection(COLLECTION)

    def file_hash(self, source: str) -> str | None:
        docs = self._collection.where("source", "==", source).limit(1).stream()
        return next((doc.get("file_hash") for doc in docs), None)

    def chunk_ids(self, source: str) -> list[str]:
        query = self._collection.where("source", "==", source).select(["source"])
        return [doc.id for doc in query.stream()]

    def sources(self) -> set[str]:
        return {
            doc.get("source") for doc in self._collection.select(["source"]).stream()
        }

    def save(self, chunks: list[Chunk]) -> None:
        now = datetime.now(timezone.utc)
        for start in range(0, len(chunks), 400):  # batch limit is 500 writes
            batch = self._client.batch()
            for chunk in chunks[start : start + 400]:
                batch.set(
                    self._collection.document(_doc_id(chunk.id)),
                    {
                        "chunk_id": chunk.id,
                        "source": chunk.source,
                        "index": chunk.index,
                        "text": chunk.text,
                        "file_hash": chunk.file_hash,
                        "updated_at": now,
                    },
                )
            batch.commit()

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), 400):
            batch = self._client.batch()
            for chunk_id in ids[start : start + 400]:
                batch.delete(self._collection.document(_doc_id(chunk_id)))
            batch.commit()

    def get(self, ids: list[str]) -> dict[str, Chunk]:
        refs = [self._collection.document(_doc_id(chunk_id)) for chunk_id in ids]
        found = {}
        for doc in self._client.get_all(refs):
            if doc.exists:
                data = doc.to_dict()
                found[data["chunk_id"]] = Chunk(
                    data["chunk_id"],
                    data["source"],
                    data["index"],
                    data["text"],
                    data["file_hash"],
                )
        return found


def _doc_id(chunk_id: str) -> str:
    # Firestore document IDs cannot contain "/"; file names in docs/ do.
    return chunk_id.replace("/", "|")


class InMemoryChunkStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    def file_hash(self, source: str) -> str | None:
        return next(
            (c.file_hash for c in self.chunks.values() if c.source == source), None
        )

    def chunk_ids(self, source: str) -> list[str]:
        return [c.id for c in self.chunks.values() if c.source == source]

    def sources(self) -> set[str]:
        return {c.source for c in self.chunks.values()}

    def save(self, chunks: list[Chunk]) -> None:
        self.chunks.update({c.id: c for c in chunks})

    def delete(self, ids: list[str]) -> None:
        for chunk_id in ids:
            self.chunks.pop(chunk_id, None)

    def get(self, ids: list[str]) -> dict[str, Chunk]:
        return {i: self.chunks[i] for i in ids if i in self.chunks}
