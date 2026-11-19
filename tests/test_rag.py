import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.ai_core import llm
from app.ai_core.llm import generate_reply
from app.ai_core.rag import build_rag_tool
from app.config import Settings

CORPUS = "projects/p/locations/europe-west1/ragCorpora/123"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)


def response(text: str = "Answer", chunks: list | None = None) -> SimpleNamespace:
    """A Gemini response with grounding metadata from RAG Engine."""
    grounding_chunks = [
        SimpleNamespace(
            retrieved_context=SimpleNamespace(title=title, uri=uri, text=chunk_text)
        )
        for title, uri, chunk_text in (chunks or [])
    ]
    metadata = SimpleNamespace(grounding_chunks=grounding_chunks)
    return SimpleNamespace(
        text=text, candidates=[SimpleNamespace(grounding_metadata=metadata)]
    )


def gemini(*results) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = list(results)
    return client


def tools_per_call(client: MagicMock) -> list:
    return [
        call.kwargs["config"].tools
        for call in client.models.generate_content.call_args_list
    ]


# --- the Gemini call -------------------------------------------------------------


def test_last_attempt_goes_without_the_tool_with_a_warning(caplog) -> None:
    tool = build_rag_tool(Settings(RAG_CORPUS=CORPUS))
    failures = [TimeoutError()] * (llm.REPLY_ATTEMPTS - 1)
    client = gemini(*failures, response("Without knowledge"))

    with caplog.at_level(logging.WARNING):
        answer = generate_reply(client, "m", "s", "b", rag_tool=tool)

    assert answer == "Without knowledge"
    assert tools_per_call(client) == [[tool]] * (llm.REPLY_ATTEMPTS - 1) + [None]
    assert (
        f"Gemini attempt {llm.REPLY_ATTEMPTS}/{llm.REPLY_ATTEMPTS} without the knowledge base"
        in caplog.text
    )


# --- logging ---------------------------------------------------------------------

CHUNKS = [
    ("shipping.md", "gs://kb/docs/shipping.md", "Doprava zdarma nad 1500 Kč."),
    ("", "gs://kb/docs/returns.md", "Vrácení do 30 dnů."),
    ("shipping.md", "gs://kb/docs/shipping.md", "Doručení do 2 dnů."),
]


def test_grounding_is_logged_without_texts(caplog) -> None:
    tool = build_rag_tool(Settings(RAG_CORPUS=CORPUS))
    client = gemini(response("Reply text", CHUNKS))

    with caplog.at_level(logging.INFO):
        generate_reply(client, "m", "Secret subject", "Secret body", rag_tool=tool)

    assert "RAG: 3 chunk(s) retrieved, sources: ['returns.md', 'shipping.md']" in (
        caplog.text
    )
    for private in ("Doprava zdarma", "Secret subject", "Secret body", "Reply text"):
        assert private not in caplog.text
