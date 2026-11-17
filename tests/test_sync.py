import pytest

from app.mail_processing.store import COMPLETED, FAILED
from app.mail_processing.sync import sync_mailbox
from tests.fakes import BOT, FakeGmail, make_email, make_mailbox


def test_processes_every_new_message_and_saves_history_id() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    gmail.add(make_email("m2"))
    mailbox = make_mailbox(gmail, history_id="100")

    result = sync_mailbox(mailbox, notified_history_id="102")

    assert result.results == {"m1": COMPLETED, "m2": COMPLETED}
    assert len(gmail.sent) == 2
    assert gmail.read == ["m1", "m2"]
    assert mailbox.sync_state.get_history_id() == "102"


def test_only_messages_after_stored_history_id() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("old"))  # historyId 101
    gmail.add(make_email("new"))  # historyId 102
    mailbox = make_mailbox(gmail, history_id="101")

    result = sync_mailbox(mailbox, "102")

    assert list(result.results) == ["new"]


def test_duplicate_notification_sends_no_second_reply() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail, history_id="100")
    sync_mailbox(mailbox, "101")

    # Pub/Sub redelivers the same notification, and history is re-read from 100.
    mailbox.sync_state.history_id = "100"
    result = sync_mailbox(mailbox, "101")

    assert len(gmail.sent) == 1
    assert result.results["m1"] in {"SKIPPED", "FILTERED:already read"}


def test_no_stored_history_id_starts_from_notification() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail, history_id=None)

    result = sync_mailbox(mailbox, "101")

    assert result.results == {}
    assert gmail.sent == []
    assert mailbox.sync_state.get_history_id() == "101"


def test_expired_history_falls_back_to_unread_inbox() -> None:
    gmail = FakeGmail([make_email("m1")], history_id="500")
    gmail.history_expired = True
    mailbox = make_mailbox(gmail, history_id="1")

    result = sync_mailbox(mailbox, None)

    assert result.results == {"m1": COMPLETED}
    assert mailbox.sync_state.get_history_id() == "500"


def test_deleted_message_is_skipped() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("gone"))
    del gmail.emails["gone"]
    mailbox = make_mailbox(gmail)

    assert sync_mailbox(mailbox, "101").results == {"gone": "DELETED"}


def test_unrecorded_error_keeps_history_id() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail, history_id="100")

    def firestore_down(*args, **kwargs):
        raise ConnectionError("Firestore unavailable")

    mailbox.store.claim = firestore_down

    with pytest.raises(ConnectionError):
        sync_mailbox(mailbox, "101")
    assert mailbox.sync_state.get_history_id() == "100"


def test_send_failure_is_recorded_and_history_moves_on() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    gmail.send = lambda body: (_ for _ in ()).throw(ConnectionError("Gmail down"))
    mailbox = make_mailbox(gmail, history_id="100")

    result = sync_mailbox(mailbox, "101")

    assert result.results == {"m1": FAILED}
    assert mailbox.store.docs["m1"]["status"] == FAILED
    assert mailbox.sync_state.get_history_id() == "101"


def test_filtered_messages_are_not_answered() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("own", sender=f"Bot <{BOT}>"))
    gmail.add(make_email("spam", label_ids=["INBOX", "UNREAD", "SPAM"]))
    gmail.add(make_email("ok"))
    mailbox = make_mailbox(gmail)

    result = sync_mailbox(mailbox, "103")

    assert result.results == {
        "own": "FILTERED:own message",
        "spam": "FILTERED:label SPAM",
        "ok": COMPLETED,
    }
    assert len(gmail.sent) == 1
