"""Builds and sends a reply in the original Gmail thread."""

import base64
import html
from email.message import EmailMessage

from app.mail_processing.gmail import Email, GmailClient

# Added to the end of every reply, Gemini's and the fallback alike.
AI_GENERATED_TEXT = "AI Digital Mark: Tento e-mail byl vygenerován umělou inteligencí."
# How the mark looks in the HTML version: italic, smaller and gray.
AI_MARK_STYLE = "font-style:italic;font-size:12px;color:#888888;"


def reply_subject(subject: str) -> str:
    """The original subject with "Re: ", unless it already starts with it."""
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def build_reply(original: Email, text: str, own_address: str) -> dict[str, str]:
    """Returns the Gmail API send body for a reply to the original's sender only."""
    message = EmailMessage()
    message["From"] = own_address
    message["To"] = original.sender
    message["Subject"] = reply_subject(original.subject)
    if original.message_id:
        message["In-Reply-To"] = original.message_id
        message["References"] = " ".join(
            filter(None, [original.references, original.message_id])
        )
    # Plain text for clients without HTML, HTML (with the styled mark) for the rest.
    message.set_content(f"{text}\n\n--\n{AI_GENERATED_TEXT}")
    message.add_alternative(reply_html(text), subtype="html")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw, "threadId": original.thread_id}


def reply_html(text: str) -> str:
    """The reply as HTML: the text with its line breaks, then the AI mark styled
    as AI_MARK_STYLE. The text is escaped, so nothing in it becomes markup."""
    body = html.escape(text).replace("\n", "<br>\n")
    mark = html.escape(AI_GENERATED_TEXT)
    return (
        f"<html><body>\n<div>{body}</div>\n"
        f'<p style="{AI_MARK_STYLE}">{mark}</p>\n</body></html>\n'
    )


def send_reply(gmail: GmailClient, original: Email, text: str, own_address: str) -> str:
    """Sends the reply and returns the id of the sent message."""
    return gmail.send(build_reply(original, text, own_address))["id"]
