"""Reads knowledge base files from gs://KB_BUCKET/docs/."""

from typing import Protocol

from google.cloud import storage

PREFIX = "docs/"
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


def is_kb_file(name: str) -> bool:
    """Only markdown and plain text files under docs/ are indexed."""
    return name.startswith(PREFIX) and name.lower().endswith(SUPPORTED_SUFFIXES)


class Loader(Protocol):
    def list_files(self) -> list[str]: ...

    def read(self, name: str) -> bytes | None:
        """File content, or None when the file does not exist."""


class GcsLoader:
    def __init__(self, project_id: str, bucket: str) -> None:
        self._bucket = storage.Client(project=project_id).bucket(bucket)

    def list_files(self) -> list[str]:
        blobs = self._bucket.client.list_blobs(self._bucket, prefix=PREFIX)
        return sorted(blob.name for blob in blobs if is_kb_file(blob.name))

    def read(self, name: str) -> bytes | None:
        blob = self._bucket.blob(name)
        try:
            return blob.download_as_bytes()
        except Exception as exc:  # google.api_core.exceptions.NotFound
            if getattr(exc, "code", None) == 404:
                return None
            raise


class InMemoryLoader:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = dict(files or {})

    def list_files(self) -> list[str]:
        return sorted(name for name in self.files if is_kb_file(name))

    def read(self, name: str) -> bytes | None:
        return self.files.get(name)
