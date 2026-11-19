"""Generates the reply text with Gemini on Vertex AI."""

import logging
import re
import time
from typing import Any, Sequence

from google import genai
from google.genai import errors, types

from app.ai_core.instructions import (
    ATTACHMENT_DESCRIPTIONS_END,
    ATTACHMENT_DESCRIPTIONS_START,
    ATTACHMENT_END,
    ATTACHMENT_START,
    DESCRIBE_INSTRUCTION,
    DESCRIBE_TASK,
    EMAIL_END,
    EMAIL_START,
    NOT_PROCESSED_NOTE,
    SYSTEM_INSTRUCTION,
    TASK,
)
from app.ai_core.rag import log_grounding
from app.config import Settings
from app.mail_processing.attachments import ACCEPTED, REJECTED, Attachment

log = logging.getLogger(__name__)

TIMEOUT_MS = 60_000
# Attempts per Gemini call: the reply (with the knowledge base) and the attachment
# description before it.
REPLY_ATTEMPTS = 3
DESCRIBE_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0

# Anything that looks like one of our block markers, e.g. "=== ATTACHMENT 2 END ===".
MARKER_PATTERN = re.compile(r"===\s*(?:EMAIL|ATTACHMENT)\b[^=\n]*===", re.IGNORECASE)
TEXT_ATTACHMENT_MAX_CHARS = 50_000
# Reason shown for file attachments when describing them failed.
DESCRIPTION_FAILED = "could not be read"

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


class EmptyResponseError(Exception):
    """Gemini returned no text, e.g. because the answer was blocked."""


def make_client(settings: Settings) -> genai.Client:
    """Gemini client on Vertex AI for GCP_PROJECT_ID and GEMINI_REGION, with a 60 s
    timeout."""
    return genai.Client(
        vertexai=True,
        project=settings.GCP_PROJECT_ID,
        location=settings.GEMINI_REGION,
        http_options=types.HttpOptions(timeout=TIMEOUT_MS),
    )


def strip_markers(text: str) -> str:
    """Removes block markers, so sender text cannot close or fake a block."""
    return MARKER_PATTERN.sub("", text).strip()


def build_prompt(
    subject: str,
    body: str,
    attachments: Sequence[Attachment] = (),
    descriptions: str = "",
) -> str:
    """The whole request as text: the email, text attachments in ATTACHMENT blocks,
    and the descriptions of the other accepted files (from describe_attachments);
    rejected attachments are listed as not processed."""
    accepted = [item for item in attachments if item.state == ACCEPTED]
    not_processed = [item for item in attachments if item.state == REJECTED]
    lines = [f"Subject: {strip_markers(subject)}", "", strip_markers(body)]
    text_blocks = []
    # File names come from the sender, so they stay inside the EMAIL block.
    if attachments:
        lines += ["", "Attachments:"]
        file_number = 0
        for item in accepted:
            name = f"- {_file_name(item.filename)} ({item.mime_type})"
            if item.mime_type == "text/plain":
                number = len(text_blocks) + 1
                lines.append(f"{name}: in ATTACHMENT {number} below")
                text_blocks.append(_text_block(number, item.data))
            else:
                file_number += 1
                lines.append(f"{name}: file {file_number}, described below")
        for item in not_processed:
            lines.append(
                f"- {_file_name(item.filename)}: not processed ({item.reason})"
            )

    parts = [
        TASK,
        f"{EMAIL_START}\n" + "\n".join(lines) + f"\n{EMAIL_END}",
        *text_blocks,
    ]
    if descriptions:
        parts.append(
            f"{ATTACHMENT_DESCRIPTIONS_START}\n{strip_markers(descriptions)}\n"
            f"{ATTACHMENT_DESCRIPTIONS_END}"
        )
    if not_processed:
        parts.append(NOT_PROCESSED_NOTE)
    return "\n\n".join(parts)


def _text_block(number: int, data: bytes) -> str:
    """A text attachment as a delimited ATTACHMENT block, without fake markers and cut
    to TEXT_ATTACHMENT_MAX_CHARS."""
    text = strip_markers(data.decode("utf-8", errors="replace"))
    if len(text) > TEXT_ATTACHMENT_MAX_CHARS:
        text = text[:TEXT_ATTACHMENT_MAX_CHARS] + "\n[... truncated]"
    start = ATTACHMENT_START.format(number=number)
    end = ATTACHMENT_END.format(number=number)
    return f"{start}\n{text}\n{end}"


def _file_name(name: str) -> str:
    """A file name safe to put in the prompt: one line, short, no markers."""
    return " ".join(strip_markers(name).split())[:100] or "unnamed file"


def remove_email_urls(answer: str, email_text: str) -> str:
    """Safety net for the "no URLs from the email" rule."""
    for url in set(URL_PATTERN.findall(email_text)):
        answer = answer.replace(url, "[link removed]")
    return answer


def _is_retryable(exc: Exception) -> bool:
    """Whether a retry can help: not for 4xx errors (except 429) or an empty answer."""
    # 4xx means the request itself is wrong (except rate limiting); retrying won't help.
    bad_request = isinstance(exc, errors.ClientError) and exc.code != 429
    return not bad_request and not isinstance(exc, EmptyResponseError)


def generate_reply(
    client: Any,
    model: str,
    subject: str,
    body: str,
    attachments: Sequence[Attachment] = (),
    rag_tool: types.Tool | None = None,
    rag_debug: bool = False,
) -> str:
    """Returns Gemini's reply. Raises the last error when all REPLY_ATTEMPTS fail.

    The knowledge base tool only works with text-only requests, so file
    attachments (documents, images, audio) are first described as text by a
    separate call; the reply is then one text request with the tool. If the
    description fails, the files are listed as not processed.
    """
    files = [
        item
        for item in attachments
        if item.state == ACCEPTED and item.mime_type != "text/plain"
    ]
    descriptions = ""
    if files:
        try:
            descriptions = describe_attachments(client, model, subject, body, files)
        except Exception as exc:
            log.warning(
                "Describing %d attachment(s) failed: %s", len(files), type(exc).__name__
            )
            attachments = [
                (
                    Attachment(item.filename, state=REJECTED, reason=DESCRIPTION_FAILED)
                    if item in files
                    else item
                )
                for item in attachments
            ]

    prompt = build_prompt(subject, body, attachments, descriptions)
    response, tool = _generate(
        client,
        model,
        [types.Part.from_text(text=prompt)],
        SYSTEM_INSTRUCTION,
        rag_tool,
        REPLY_ATTEMPTS,
    )
    if tool:
        log_grounding(response, rag_debug)
    # The prompt holds all sender text (email, text attachments, descriptions).
    return remove_email_urls(response.text.strip(), prompt)


def describe_attachments(
    client: Any, model: str, subject: str, body: str, files: Sequence[Attachment]
) -> str:
    """Step 1 for emails with file attachments: Gemini describes them as text (with
    DESCRIBE_INSTRUCTION), without the knowledge base tool. It also gets the email,
    so it describes precisely what the sender asks about the files."""
    listing = "\n".join(
        f"File {number}: {_file_name(item.filename)} ({item.mime_type})"
        for number, item in enumerate(files, start=1)
    )
    email = f"Subject: {strip_markers(subject)}\n\n{strip_markers(body)}"
    prompt = f"{DESCRIBE_TASK}\n{listing}\n\n{EMAIL_START}\n{email}\n{EMAIL_END}"
    contents = [types.Part.from_text(text=prompt)] + [
        types.Part.from_bytes(data=item.data, mime_type=item.mime_type)
        for item in files
    ]
    response, _ = _generate(
        client, model, contents, DESCRIBE_INSTRUCTION, None, DESCRIBE_ATTEMPTS
    )
    log.info("Described %d attachment(s)", len(files))
    return response.text.strip()


def _generate(
    client: Any,
    model: str,
    contents: list,
    instruction: str,
    rag_tool: types.Tool | None,
    attempts: int,
) -> tuple[Any, types.Tool | None]:
    """Calls Gemini up to `attempts` times; returns the response and the tool it
    used. With rag_tool, the last attempt goes without it (with a warning), and so
    does every attempt after a bad request (4xx), so a broken knowledge base does
    not block the reply."""
    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        tool = None if last else rag_tool
        if last and rag_tool:
            log.warning(
                "Gemini attempt %d/%d without the knowledge base (RAG tool)",
                attempt,
                attempts,
            )
        started = time.monotonic()
        try:
            response = _call(client, model, contents, instruction, tool)
            break
        except Exception as exc:
            log.warning(
                "Gemini attempt %d/%d failed after %.1f s: %s",
                attempt,
                attempts,
                time.monotonic() - started,
                type(exc).__name__,
            )
            # A bad request without the tool would fail again; with the tool, it may
            # be the tool's fault, so the next attempts go without it.
            if last or (tool is None and not _is_retryable(exc)):
                raise
            if _is_retryable(exc):
                time.sleep(BACKOFF_SECONDS * attempt)
            elif tool:
                log.warning("Bad request with the knowledge base, retrying without it")
                rag_tool = None

    log.info(
        "Gemini answered in %.1f s (model=%s, attempt %d, knowledge base %s)",
        time.monotonic() - started,
        model,
        attempt,
        "on" if tool else "off",
    )
    return response, tool


def _call(
    client: Any,
    model: str,
    contents: list,
    instruction: str,
    tool: types.Tool | None,
) -> Any:
    """One Gemini request, with the knowledge base tool when given."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=instruction, tools=[tool] if tool else None
        ),
    )
    if not response.text:
        raise EmptyResponseError()
    return response
