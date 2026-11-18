import base64
from email import message_from_bytes, policy

import pytest

from app.ai_core import llm
from app.mail_processing.mailbox import FALLBACK_REPLY
from app.utils.firestore_storage import COMPLETED, FAILED
from tests.fakes import FakeGmail, make_email, make_mailbox


def test_processes_every_new_message_and_saves_history_id() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    gmail.add(make_email("m2"))
    mailbox = make_mailbox(gmail, history_id="100")

    result = mailbox.check_inbox(notified_history_id="102")

    assert result.results == {"m1": COMPLETED, "m2": COMPLETED}
    assert len(gmail.sent) == 2
    assert gmail.read == ["m1", "m2"]
    assert mailbox.storage.get_history_id() == "102"


def test_duplicate_notification_sends_no_second_reply() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail, history_id="100")
    mailbox.check_inbox("101")

    # Pub/Sub redelivers the same notification, and history is re-read from 100.
    mailbox.storage.history_id = "100"
    result = mailbox.check_inbox("101")

    assert len(gmail.sent) == 1
    assert result.results["m1"] in {"DUPLICATE", "SKIPPED:already read"}


def test_no_stored_history_id_answers_unread_mail() -> None:
    """The email whose notification arrives first must not be lost."""
    gmail = FakeGmail([make_email("read", label_ids=["INBOX"])])
    gmail.add(make_email("m1"))  # historyId 101, the email of the notification
    mailbox = make_mailbox(gmail, history_id=None)

    result = mailbox.check_inbox("101")

    assert result.results == {"m1": COMPLETED}  # the read email is not checked
    assert len(gmail.sent) == 1
    assert mailbox.storage.get_history_id() == "101"


def test_unrecorded_error_keeps_history_id() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail, history_id="100")

    def firestore_down(*args, **kwargs):
        raise ConnectionError("Firestore unavailable")

    mailbox.storage.create_and_claim_message = firestore_down

    with pytest.raises(ConnectionError):
        mailbox.check_inbox("101")
    assert mailbox.storage.get_history_id() == "100"


def failing_send(body: dict) -> dict:
    raise ConnectionError("Gmail down")


def test_recovery_retries_an_email_whose_send_failed() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail, history_id="100")
    gmail.send = failing_send
    assert mailbox.check_inbox("101").results == {"m1": FAILED}
    assert gmail.read == []  # still unread: the customer got nothing

    del gmail.send  # Gmail works again
    mailbox.storage.history_id = "100"
    assert mailbox.check_inbox("101").results == {"m1": "DUPLICATE"}  # normal check

    result = mailbox.check_inbox(recovery=True)

    assert result.results == {"m1": COMPLETED}
    assert len(gmail.sent) == 1
    assert mailbox.storage.docs["m1"]["status"] == COMPLETED


def test_recovery_does_not_resend_after_a_fallback_reply() -> None:
    """Fallback reply sent, but marking as read failed: unread and FAILED."""
    gmail = FakeGmail([make_email("m1")])
    mailbox = make_mailbox(gmail)
    mailbox.storage.create_and_claim_message("m1", {})
    mailbox.storage.fail_message("m1", "Gemini:TimeoutError", "fallback-reply-id")

    assert mailbox.check_inbox(recovery=True).results == {"m1": "DUPLICATE"}
    assert gmail.sent == []


def test_filtered_email_is_recorded_as_skipped_and_marked_read() -> None:
    gmail = FakeGmail([])
    gmail.add(make_email("news", headers={"list-unsubscribe": "<mailto:u@x>"}))
    mailbox = make_mailbox(gmail)

    result = mailbox.check_inbox("101")

    assert result.results == {"news": "SKIPPED:mailing list"}
    doc = mailbox.storage.docs["news"]
    assert (doc["status"], doc["skip_reason"]) == ("SKIPPED", "mailing list")
    assert gmail.read == ["news"]
    assert gmail.sent == []


def test_gemini_failure_sends_the_fallback_reply_and_sets_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)  # no real backoff
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail)
    mailbox.gemini.models.generate_content.side_effect = TimeoutError("prompt text")

    assert mailbox.check_inbox("101").results == {"m1": FAILED}

    assert mailbox.gemini.models.generate_content.call_count == llm.ATTEMPTS
    sent = message_from_bytes(
        base64.urlsafe_b64decode(gmail.sent[0]["raw"]), policy=policy.default
    )
    assert FALLBACK_REPLY in sent.get_body(("plain",)).get_content()
    doc = mailbox.storage.docs["m1"]
    assert (doc["status"], doc["error_type"]) == (FAILED, "Gemini:TimeoutError")
    assert doc["reply_message_id"] == "sent-1"
    assert "prompt text" not in str(doc)
    assert gmail.read == ["m1"]  # the customer got a reply, so it is marked read
