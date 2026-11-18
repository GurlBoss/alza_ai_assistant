"""All communication with Firestore: the app's whole persistent state.

- processed_emails/<Gmail message ID>: one record per email, so each email is
  answered at most once. Old records are deleted by a Firestore TTL policy on
  updated_at with a 30-day expiration offset.
- gmail_history/state: the history ID, the last Gmail historyId fully processed.
"""

from datetime import datetime, timezone
from typing import Any, Protocol

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore

EMAILS_COLLECTION = "processed_emails"
HISTORY_ID_COLLECTION = "gmail_history"
HISTORY_ID_DOCUMENT = "state"

PROCESSING = "PROCESSING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"  # did not get through the filters, deliberately not answered


def now() -> datetime:
    """The current time in UTC."""
    return datetime.now(timezone.utc)


class Storage(Protocol):
    """The app's state: email records and the history ID."""

    def create_and_claim_message(
        self, message_id: str, data: dict[str, Any], retry_failed: bool = False
    ) -> bool:
        """Creates the record with status PROCESSING. False if it already exists,
        unless retry_failed and the existing record can_retry."""

    def complete_message(self, message_id: str, reply_message_id: str) -> None:
        """Marks the email as answered."""

    def skip_message(self, message_id: str, data: dict[str, Any], reason: str) -> None:
        """Records an email the filters rejected, unless it already has a record."""

    def fail_message(
        self, message_id: str, error_type: str, reply_message_id: str | None = None
    ) -> None:
        """Marks the email as failed. error_type must not contain email content;
        reply_message_id is set when a fallback reply was sent."""

    def get_history_id(self) -> str | None:
        """The last Gmail historyId that was fully processed."""

    def save_history_id(self, history_id: str) -> None:
        """Stores history_id unless a newer one is already stored."""


def new_record(data: dict[str, Any]) -> dict[str, Any]:
    """A new processed_emails record in the PROCESSING state."""
    return {**data, "status": PROCESSING, "updated_at": now()}


def can_retry(record: dict[str, Any]) -> bool:
    """A FAILED email that got no reply at all (e.g. sending failed); a recovery may
    claim it again. A fallback reply sets reply_message_id, so it is never resent."""
    return record.get("status") == FAILED and not record.get("reply_message_id")


def skipped_record(data: dict[str, Any], reason: str) -> dict[str, Any]:
    """A processed_emails record for an email the filters rejected, with the reason."""
    return {**data, "status": SKIPPED, "skip_reason": reason, "updated_at": now()}


def completed_record(reply_message_id: str) -> dict[str, Any]:
    """Fields for an email that was answered."""
    return {
        "status": COMPLETED,
        "reply_message_id": reply_message_id,
        "updated_at": now(),
    }


def failed_record(error_type: str, reply_message_id: str | None) -> dict[str, Any]:
    """Fields for an email that failed; reply_message_id is set when a fallback reply
    was sent."""
    record: dict[str, Any] = {
        "status": FAILED,
        "error_type": error_type,
        "updated_at": now(),
    }
    if reply_message_id:
        record["reply_message_id"] = reply_message_id
    return record


@firestore.transactional
def _claim(
    transaction: firestore.Transaction,
    ref: firestore.DocumentReference,
    data: dict[str, Any],
    retry_failed: bool,
) -> bool:
    """Creates the record in a transaction, so only one request claims the email; an
    existing record is taken over only when retry_failed and it can_retry."""
    snapshot = ref.get(transaction=transaction)
    claimed = True
    if not snapshot.exists:
        transaction.create(ref, new_record(data))
    elif retry_failed and can_retry(snapshot.to_dict()):
        transaction.set(ref, new_record(data))
    else:
        claimed = False
    return claimed


@firestore.transactional
def _save_if_newer(
    transaction: firestore.Transaction,
    ref: firestore.DocumentReference,
    history_id: str,
) -> None:
    """Saves the historyId in a transaction, never moving it backwards."""
    # Concurrent notifications can finish out of order; never move backwards.
    snapshot = ref.get(transaction=transaction)
    stored = snapshot.get("history_id") if snapshot.exists else None
    if stored is None or int(history_id) > int(stored):
        transaction.set(ref, {"history_id": history_id, "updated_at": now()})


class FirestoreStorage:
    """Storage in Firestore, with one client for the email records and the history
    ID."""

    def __init__(self, project_id: str) -> None:
        self._client = firestore.Client(project=project_id)
        self._emails = self._client.collection(EMAILS_COLLECTION)
        self._history_id_ref = self._client.collection(HISTORY_ID_COLLECTION).document(
            HISTORY_ID_DOCUMENT
        )

    # --- email records (processed_emails) -------------------------------------------

    def create_and_claim_message(
        self, message_id: str, data: dict[str, Any], retry_failed: bool = False
    ) -> bool:
        """Claims the email for processing; False if another request already has it.
        With retry_failed, a FAILED email that got no reply is claimed again."""
        ref = self._emails.document(message_id)
        return _claim(self._client.transaction(), ref, data, retry_failed)

    def complete_message(self, message_id: str, reply_message_id: str) -> None:
        """Marks the email as answered."""
        self._emails.document(message_id).update(completed_record(reply_message_id))

    def fail_message(
        self, message_id: str, error_type: str, reply_message_id: str | None = None
    ) -> None:
        """Marks the email as failed, with only the error type."""
        self._emails.document(message_id).update(
            failed_record(error_type, reply_message_id)
        )

    def skip_message(self, message_id: str, data: dict[str, Any], reason: str) -> None:
        """Records an email the filters rejected; an existing record is kept, so a
        repeated check never overwrites another state."""
        try:
            self._emails.document(message_id).create(skipped_record(data, reason))
        except AlreadyExists:
            pass

    # --- history ID (gmail_history/state) -----------------------------------------

    def get_history_id(self) -> str | None:
        """The stored history ID, or None before the first inbox check."""
        snapshot = self._history_id_ref.get()
        return snapshot.get("history_id") if snapshot.exists else None

    def save_history_id(self, history_id: str) -> None:
        """Stores the historyId unless a newer one is already stored."""
        _save_if_newer(self._client.transaction(), self._history_id_ref, history_id)
