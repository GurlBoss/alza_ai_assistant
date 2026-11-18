"""Processing state of Gmail messages, so each email is answered at most once.

Collection processed_emails, document ID = Gmail message ID.
Old documents are deleted by a Firestore TTL policy on updated_at with a 30-day
expiration offset, so no extra expiry field is stored.
"""

from datetime import datetime, timezone
from typing import Any, Protocol

from google.cloud import firestore

COLLECTION = "processed_emails"

PROCESSING = "PROCESSING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"


def now() -> datetime:
    return datetime.now(timezone.utc)


class EmailStore(Protocol):
    def claim(self, message_id: str, data: dict[str, Any]) -> bool:
        """Creates the document with status PROCESSING. False if it already exists."""

    def complete(self, message_id: str, reply_message_id: str) -> None:
        """Marks the email as answered."""

    def fail(
        self, message_id: str, error_type: str, reply_message_id: str | None = None
    ) -> None:
        """Marks the email as failed. error_type must not contain email content.

        reply_message_id is set when a fallback reply was sent.
        """


def new_record(data: dict[str, Any]) -> dict[str, Any]:
    return {**data, "status": PROCESSING, "updated_at": now()}


def failed_record(error_type: str, reply_message_id: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "status": FAILED,
        "error_type": error_type,
        "updated_at": now(),
    }
    if reply_message_id:
        record["reply_message_id"] = reply_message_id
    return record


@firestore.transactional
def _create_if_missing(
    transaction: firestore.Transaction,
    ref: firestore.DocumentReference,
    data: dict[str, Any],
) -> bool:
    if ref.get(transaction=transaction).exists:
        return False
    transaction.create(ref, new_record(data))
    return True


@firestore.transactional
def _save_if_newer(
    transaction: firestore.Transaction,
    ref: firestore.DocumentReference,
    history_id: str,
) -> None:
    # Concurrent notifications can finish out of order; never move backwards.
    snapshot = ref.get(transaction=transaction)
    stored = snapshot.get("history_id") if snapshot.exists else None
    if stored is None or int(history_id) > int(stored):
        transaction.set(ref, {"history_id": history_id, "updated_at": now()})


class FirestoreStore:
    def __init__(self, project_id: str) -> None:
        self._client = firestore.Client(project=project_id)
        self._collection = self._client.collection(COLLECTION)

    def claim(self, message_id: str, data: dict[str, Any]) -> bool:
        ref = self._collection.document(message_id)
        return _create_if_missing(self._client.transaction(), ref, data)

    def complete(self, message_id: str, reply_message_id: str) -> None:
        self._collection.document(message_id).update(
            {
                "status": COMPLETED,
                "reply_message_id": reply_message_id,
                "updated_at": now(),
            }
        )

    def fail(
        self, message_id: str, error_type: str, reply_message_id: str | None = None
    ) -> None:
        self._collection.document(message_id).update(
            failed_record(error_type, reply_message_id)
        )


class InMemoryStore:
    """Same behaviour as FirestoreStore, for tests."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def claim(self, message_id: str, data: dict[str, Any]) -> bool:
        if message_id in self.docs:
            return False
        self.docs[message_id] = new_record(data)
        return True

    def complete(self, message_id: str, reply_message_id: str) -> None:
        self.docs[message_id].update(
            status=COMPLETED,
            reply_message_id=reply_message_id,
            updated_at=now(),
        )

    def fail(
        self, message_id: str, error_type: str, reply_message_id: str | None = None
    ) -> None:
        self.docs[message_id].update(failed_record(error_type, reply_message_id))


SYNC_COLLECTION = "gmail_sync"
SYNC_DOCUMENT = "state"


class SyncState(Protocol):
    def get_history_id(self) -> str | None:
        """The last Gmail historyId that was fully processed."""

    def save_history_id(self, history_id: str) -> None:
        """Stores history_id unless a newer one is already stored."""


class FirestoreSyncState:
    """gmail_sync/state document: {history_id, updated_at}."""

    def __init__(self, project_id: str) -> None:
        self._client = firestore.Client(project=project_id)
        self._ref = self._client.collection(SYNC_COLLECTION).document(SYNC_DOCUMENT)

    def get_history_id(self) -> str | None:
        snapshot = self._ref.get()
        return snapshot.get("history_id") if snapshot.exists else None

    def save_history_id(self, history_id: str) -> None:
        _save_if_newer(self._client.transaction(), self._ref, history_id)


class InMemorySyncState:
    def __init__(self, history_id: str | None = None) -> None:
        self.history_id = history_id

    def get_history_id(self) -> str | None:
        return self.history_id

    def save_history_id(self, history_id: str) -> None:
        if self.history_id is None or int(history_id) > int(self.history_id):
            self.history_id = history_id
