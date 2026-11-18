from app.mail_processing.filters import (
    MAX_BOT_REPLIES_PER_THREAD,
    skip_reason,
    thread_limit_reached,
)
from tests.fakes import BOT, FakeGmail, make_email


def test_skip_reasons() -> None:
    """Each case changes one detail of a normal customer email."""
    cases = [
        ({}, None),  # a normal customer email is answered
        ({"headers": {"auto-submitted": "no"}}, None),  # "no" means written by a human
        ({"sender": f"Alza Bot <{BOT.upper()}>"}, "own message"),
        ({"sender": "Shop <no-reply@shop.example>"}, "no-reply sender"),
        ({"sender": "MAILER-DAEMON@example.com"}, "no-reply sender"),
        ({"label_ids": ["INBOX", "UNREAD", "SENT"]}, "label SENT"),
        ({"label_ids": ["INBOX", "UNREAD", "SPAM"]}, "label SPAM"),
        (
            {"label_ids": ["INBOX", "UNREAD", "CATEGORY_PROMOTIONS"]},
            "label CATEGORY_PROMOTIONS",
        ),
        ({"label_ids": ["INBOX"]}, "already read"),
        ({"headers": {"auto-submitted": "auto-replied"}}, "auto-submitted"),
        ({"headers": {"x-autoreply": "yes"}}, "auto-reply header"),
        ({"headers": {"precedence": "Bulk"}}, "bulk precedence"),
        ({"headers": {"list-unsubscribe": "<mailto:u@x>"}}, "mailing list"),
        ({"headers": {"return-path": "<>"}}, "bounce"),
    ]
    for overrides, reason in cases:
        assert skip_reason(make_email(**overrides), BOT) == reason, overrides


def test_thread_reply_limit() -> None:
    gmail = FakeGmail([])
    email = make_email()
    gmail.sent_per_thread[email.thread_id] = MAX_BOT_REPLIES_PER_THREAD - 1
    assert not thread_limit_reached(gmail, email)
    gmail.sent_per_thread[email.thread_id] = MAX_BOT_REPLIES_PER_THREAD
    assert thread_limit_reached(gmail, email)
