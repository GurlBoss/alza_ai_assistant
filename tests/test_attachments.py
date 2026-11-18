import base64

from app.config import Settings
from app.mail_processing.attachments import (
    ACCEPTED,
    REJECTED,
    Attachment,
    load_attachments,
)
from app.mail_processing.filters import (
    TOO_LARGE,
    UNSUPPORTED,
    filter_type,
    is_signature_image,
)
from app.mail_processing.gmail import AttachmentRef, parse_message
from app.utils.firestore_storage import COMPLETED
from tests import samples
from tests.fakes import FakeGmail, make_email, make_mailbox

MB = 1024 * 1024
SETTINGS = Settings(MAX_ATTACHMENT_MB=10, MAX_TOTAL_ATTACHMENTS_MB=20)


def accepted(attachments: list[Attachment]) -> list[Attachment]:
    return [item for item in attachments if item.state == ACCEPTED]


def rejected(attachments: list[Attachment]) -> list[Attachment]:
    return [item for item in attachments if item.state == REJECTED]


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


def test_filter_type() -> None:
    """The type comes from the content; everything not supported is None."""
    cases = [
        (samples.PDF, "application/pdf"),
        (samples.JPEG, "image/jpeg"),
        (samples.png(), "image/png"),
        (samples.WEBP, "image/webp"),
        (samples.wav(), "audio/wav"),
        (samples.MP3_ID3, "audio/mpeg"),
        (samples.MP3_FRAME, "audio/mpeg"),
        (samples.M4A, "audio/mp4"),
        (samples.office("word"), None),  # DOCX
        (samples.OGG_OPUS, None),
        (samples.MP4_VIDEO, None),  # MP4 container, but a video brand
        (samples.EXE, None),
        (b"", None),
    ]
    for data, expected in cases:
        assert filter_type(data) == expected, data[:12]
    assert filter_type(samples.text(), "text/plain") == "text/plain"
    assert filter_type(samples.text(), "application/octet-stream") is None
    assert filter_type(samples.EXE, "text/plain") is None  # binary renamed to .txt


# --- loading -------------------------------------------------------------------


def test_fake_extension_is_rejected() -> None:
    """An .exe renamed to .pdf and declared as PDF is still not a PDF."""
    gmail, email = gmail_with(
        (ref("a1", "invoice.pdf", "application/pdf", 100), samples.EXE)
    )

    loaded = load_attachments(gmail, email, SETTINGS)

    assert accepted(loaded) == []
    assert [(r.filename, r.reason) for r in rejected(loaded)] == [
        ("invoice.pdf", UNSUPPORTED)
    ]


def test_oversized_file_is_rejected_without_download() -> None:
    gmail, email = gmail_with(
        (ref("a1", "big.pdf", "application/pdf", 11 * MB), samples.PDF)
    )

    loaded = load_attachments(gmail, email, SETTINGS)

    assert rejected(loaded)[0].reason == TOO_LARGE
    assert gmail.downloaded == []


# --- signature images ----------------------------------------------------------


def test_signature_images() -> None:
    """Only small images shown inline or by the HTML are signature images."""
    cases = [
        (ref("a1", "logo.png", "image/png", 5_000, inline=True), set(), True),
        (ref("a1", "logo.png", "image/png", 25_000, content_id="sig"), {"sig"}, True),
        (ref("a1", "logo.png", "image/png", 25_000, content_id="sig"), set(), False),
        # A screenshot pasted into the body is inline and referenced, but large.
        (ref("a1", "shot.png", "image/png", 654_770, True, "ii_1"), {"ii_1"}, False),
        (ref("a1", "icon.png", "image/png", 5_000), set(), False),  # attached
        (ref("a1", "doc.pdf", "application/pdf", 5_000, inline=True), set(), False),
    ]
    for attachment, cids, expected in cases:
        assert is_signature_image(attachment, cids) is expected, attachment


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

    result = mailbox.check_inbox("101")

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
