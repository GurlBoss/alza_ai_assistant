import base64

import pytest

from app.config import Settings
from app.mail_processing.attachments import (
    DOWNLOAD_FAILED,
    EXTRA_AUDIO,
    TOO_LARGE,
    TOTAL_TOO_LARGE,
    UNSUPPORTED,
    detect_type,
    is_signature_image,
    load_attachments,
)
from app.mail_processing.gmail import AttachmentRef, parse_message
from app.mail_processing.store import COMPLETED
from app.mail_processing.sync import sync_mailbox
from tests import samples
from tests.fakes import FakeGmail, make_email, make_mailbox

MB = 1024 * 1024
SETTINGS = Settings(MAX_ATTACHMENT_MB=10, MAX_TOTAL_ATTACHMENTS_MB=20)


def ref(
    attachment_id: str,
    filename: str,
    mime_type: str,
    size: int,
    inline: bool = False,
    content_id: str = "",
) -> AttachmentRef:
    return AttachmentRef(attachment_id, filename, mime_type, size, inline, content_id)


def gmail_with(*files: tuple[AttachmentRef, bytes]) -> tuple[FakeGmail, object]:
    """FakeGmail holding one email with the given attachments."""
    email = make_email("m1", attachments=[r for r, _ in files])
    gmail = FakeGmail([email])
    for attachment, data in files:
        gmail.attachment_data[attachment.attachment_id] = data
    return gmail, email


# --- type detection by magic bytes ---------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (samples.PDF, "application/pdf"),
        (samples.JPEG, "image/jpeg"),
        (samples.png(), "image/png"),
        (samples.wav(), "audio/wav"),
        (samples.MP3_ID3, "audio/mpeg"),
        (samples.MP3_FRAME, "audio/mpeg"),
        (samples.WEBP, "image/webp"),
        (samples.M4A, "audio/mp4"),
        # Everything else is unsupported:
        (samples.office("word"), None),  # DOCX
        (samples.OGG_OPUS, None),
        (samples.MP4_VIDEO, None),  # MP4 container, but a video brand
        (samples.EXE, None),
        (b"", None),
    ],
)
def test_detect_type(data: bytes, expected: str | None) -> None:
    assert detect_type(data) == expected


def test_text_needs_text_content_and_declared_plain_text() -> None:
    assert detect_type(samples.text(), "text/plain") == "text/plain"
    assert detect_type(samples.text(), "text/plain; charset=utf-8") == "text/plain"
    assert detect_type(samples.text(), "application/octet-stream") is None
    assert detect_type(samples.EXE, "text/plain") is None  # binary renamed to .txt


# --- loading -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "declared", "data", "verified"),
    [
        ("invoice.pdf", "application/pdf", samples.PDF, "application/pdf"),
        ("photo.jpg", "image/jpeg", samples.JPEG, "image/jpeg"),
        ("screen.png", "image/png", samples.png(), "image/png"),
        ("voice.mp3", "audio/mpeg", samples.MP3_ID3, "audio/mpeg"),
        ("voice.wav", "audio/wav", samples.wav(), "audio/wav"),
        ("photo.webp", "image/webp", samples.WEBP, "image/webp"),
        ("memo.m4a", "audio/x-m4a", samples.M4A, "audio/mp4"),
        ("note.txt", "text/plain", samples.text(), "text/plain"),
    ],
)
def test_supported_attachment_is_accepted(
    filename: str, declared: str, data: bytes, verified: str
) -> None:
    gmail, email = gmail_with((ref("a1", filename, declared, len(data)), data))

    loaded = load_attachments(gmail, email, SETTINGS)

    assert [(a.filename, a.mime_type, a.data) for a in loaded.accepted] == [
        (filename, verified, data)
    ]
    assert loaded.rejected == []


def test_multiple_attachments() -> None:
    gmail, email = gmail_with(
        (ref("a1", "a.pdf", "application/pdf", 100), samples.PDF),
        (ref("a2", "b.png", "image/png", 100), samples.png()),
        (ref("a3", "c.wav", "audio/wav", 100), samples.wav()),
        (ref("a4", "d.txt", "text/plain", 100), samples.text()),
    )

    loaded = load_attachments(gmail, email, SETTINGS)

    assert [a.mime_type for a in loaded.accepted] == [
        "application/pdf",
        "image/png",
        "audio/wav",
        "text/plain",
    ]


def test_only_the_first_audio_file_is_used() -> None:
    gmail, email = gmail_with(
        (ref("a1", "first.mp3", "audio/mpeg", 100), samples.MP3_ID3),
        (ref("a2", "photo.png", "image/png", 100), samples.png()),
        (ref("a3", "second.m4a", "audio/mp4", 100), samples.M4A),
    )

    loaded = load_attachments(gmail, email, SETTINGS)

    assert [a.filename for a in loaded.accepted] == ["first.mp3", "photo.png"]
    assert [(r.filename, r.reason) for r in loaded.rejected] == [
        ("second.m4a", EXTRA_AUDIO)
    ]


@pytest.mark.parametrize(
    ("filename", "declared", "data"),
    [
        (
            "offer.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            samples.office("word"),
        ),
        ("voice.ogg", "audio/ogg", samples.OGG_OPUS),
        ("clip.m4a", "audio/mp4", samples.MP4_VIDEO),  # video renamed to .m4a
    ],
)
def test_other_formats_are_unsupported(
    filename: str, declared: str, data: bytes
) -> None:
    gmail, email = gmail_with((ref("a1", filename, declared, len(data)), data))

    loaded = load_attachments(gmail, email, SETTINGS)

    assert loaded.accepted == []
    assert [(r.filename, r.reason) for r in loaded.rejected] == [
        (filename, UNSUPPORTED)
    ]


# --- signature images ----------------------------------------------------------


def test_small_inline_image_is_skipped() -> None:
    logo = ref("a1", "logo.png", "image/png", 5_000, inline=True)
    gmail, email = gmail_with((logo, samples.png()))

    loaded = load_attachments(gmail, email, SETTINGS)

    assert loaded.accepted == [] and loaded.rejected == []
    assert gmail.downloaded == []


def test_image_referenced_by_cid_is_skipped() -> None:
    logo = ref("a1", "logo.png", "image/png", 50_000, content_id="logo@sig")
    assert is_signature_image(logo, {"logo@sig"})
    assert not is_signature_image(logo, set())


def test_large_inline_image_is_not_a_signature() -> None:
    photo = ref("a1", "photo.jpg", "image/jpeg", 200_000, inline=True)
    assert not is_signature_image(photo, set())


def test_inline_pdf_is_not_a_signature() -> None:
    doc = ref("a1", "doc.pdf", "application/pdf", 5_000, inline=True)
    assert not is_signature_image(doc, set())


# --- parsing the Gmail message ---------------------------------------------------


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def test_parse_message_collects_attachments_from_nested_parts() -> None:
    message = {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "1794950000000",
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [{"name": "From", "value": "jan@example.com"}],
            "parts": [
                {
                    "mimeType": "multipart/related",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": b64("Hello")}},
                        {
                            "mimeType": "text/html",
                            "body": {"data": b64('<img src="cid:Logo@Sig">')},
                        },
                        {
                            "mimeType": "image/png",
                            "filename": "logo.png",
                            "headers": [
                                {"name": "Content-ID", "value": "<Logo@Sig>"},
                                {"name": "Content-Disposition", "value": "inline"},
                            ],
                            "body": {"attachmentId": "att-logo", "size": 4000},
                        },
                    ],
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "invoice.pdf",
                    "headers": [
                        {
                            "name": "Content-Disposition",
                            "value": 'attachment; filename="invoice.pdf"',
                        }
                    ],
                    "body": {"attachmentId": "att-pdf", "size": 120000},
                },
            ],
        },
    }

    email = parse_message(message)

    assert email.body == "Hello"
    assert email.referenced_cids == {"logo@sig"}
    assert email.attachments == [
        AttachmentRef("att-logo", "logo.png", "image/png", 4000, True, "logo@sig"),
        AttachmentRef("att-pdf", "invoice.pdf", "application/pdf", 120000, False, ""),
    ]


# --- end to end through the processor --------------------------------------------


def test_email_with_attachments_is_answered_with_files_and_note() -> None:
    gmail = FakeGmail([])
    gmail.add(
        make_email(
            "m1",
            attachments=[
                ref("a1", "invoice.pdf", "application/pdf", 100),
                ref("a2", "setup.exe", "application/octet-stream", 100),
                ref("a3", "logo.png", "image/png", 3000, inline=True),
            ],
        )
    )
    gmail.attachment_data = {"a1": samples.PDF, "a2": samples.EXE}
    mailbox = make_mailbox(gmail)

    result = sync_mailbox(mailbox, "101")

    assert result.results == {"m1": COMPLETED}
    contents = mailbox.gemini.models.generate_content.call_args.kwargs["contents"]
    prompt = contents[0].text
    assert "invoice.pdf (application/pdf)" in prompt
    assert "setup.exe: not processed (unsupported file type)" in prompt
    assert "Add one polite sentence" in prompt
    assert "logo.png" not in prompt
    assert len(contents) == 2
    assert contents[1].inline_data.mime_type == "application/pdf"
    assert contents[1].inline_data.data == samples.PDF
    assert gmail.downloaded == ["a1", "a2"]


def test_second_audio_is_mentioned_in_the_prompt() -> None:
    gmail = FakeGmail([])
    gmail.add(
        make_email(
            "m1",
            attachments=[
                ref("a1", "first.mp3", "audio/mpeg", 100),
                ref("a2", "second.wav", "audio/wav", 100),
            ],
        )
    )
    gmail.attachment_data = {"a1": samples.MP3_ID3, "a2": samples.wav()}
    mailbox = make_mailbox(gmail)

    sync_mailbox(mailbox, "101")

    contents = mailbox.gemini.models.generate_content.call_args.kwargs["contents"]
    assert [p.inline_data.mime_type for p in contents[1:]] == ["audio/mpeg"]
    assert (
        "second.wav: not processed (only one audio file per email can be listened to)"
        in contents[0].text
    )
