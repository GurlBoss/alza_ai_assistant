import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors

from app.ai_core import llm
from app.ai_core.llm import generate_reply
from app.config import Settings
from app.knowledge_base.rag import build_rag_tool, log_grounding

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


# --- the tool --------------------------------------------------------------------


def test_tool_is_built_from_settings() -> None:
    tool = build_rag_tool(
        Settings(RAG_CORPUS=CORPUS, RAG_TOP_K=3, RAG_DISTANCE_THRESHOLD=0.4)
    )

    store = tool.retrieval.vertex_rag_store
    assert store.rag_resources[0].rag_corpus == CORPUS
    assert store.rag_retrieval_config.top_k == 3
    assert store.rag_retrieval_config.filter.vector_distance_threshold == 0.4


def test_defaults() -> None:
    settings = Settings(RAG_CORPUS=CORPUS)
    assert (settings.RAG_TOP_K, settings.RAG_DISTANCE_THRESHOLD) == (5, 0.5)
    assert settings.RAG_DEBUG is False


def test_no_tool_without_corpus() -> None:
    assert build_rag_tool(Settings(RAG_CORPUS="")) is None


# --- the Gemini call -------------------------------------------------------------


def test_tool_is_added_when_corpus_is_set() -> None:
    tool = build_rag_tool(Settings(RAG_CORPUS=CORPUS))
    client = gemini(response())

    assert generate_reply(client, "m", "s", "b", rag_tool=tool) == "Answer"

    assert tools_per_call(client) == [[tool]]
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.system_instruction == llm.SYSTEM_INSTRUCTION


def test_no_tool_when_corpus_is_empty() -> None:
    client = gemini(response())

    generate_reply(client, "m", "s", "b", rag_tool=build_rag_tool(Settings()))

    assert tools_per_call(client) == [None]
    config = client.models.generate_content.call_args.kwargs["config"]
    assert "Do not add a source line" in config.system_instruction


def test_instruction_has_no_concrete_example_source() -> None:
    assert "shipping.md" not in llm.SYSTEM_INSTRUCTION


@pytest.mark.parametrize(
    "error", [errors.ClientError(400, {}), TimeoutError(), ConnectionError()]
)
def test_failure_with_tool_retries_without_it(error, caplog) -> None:
    tool = build_rag_tool(Settings(RAG_CORPUS=CORPUS))
    client = gemini(error, response("Without knowledge"))

    with caplog.at_level(logging.WARNING):
        answer = generate_reply(client, "m", "s", "b", rag_tool=tool)

    assert answer == "Without knowledge"
    assert tools_per_call(client) == [[tool], None]
    assert "Retrying without the knowledge base" in caplog.text
    instructions = [
        call.kwargs["config"].system_instruction
        for call in client.models.generate_content.call_args_list
    ]
    assert llm.NO_KNOWLEDGE_INSTRUCTION not in instructions[0]
    assert instructions[1].endswith(llm.NO_KNOWLEDGE_INSTRUCTION)


def test_failure_without_tool_after_fallback_raises() -> None:
    tool = build_rag_tool(Settings(RAG_CORPUS=CORPUS))
    client = gemini(TimeoutError(), TimeoutError(), response())

    with pytest.raises(TimeoutError):
        generate_reply(client, "m", "s", "b", rag_tool=tool)
    assert client.models.generate_content.call_count == 2


def test_attachments_are_kept_with_the_tool() -> None:
    from app.mail_processing.attachments import Attachment

    tool = build_rag_tool(Settings(RAG_CORPUS=CORPUS))
    client = gemini(response())

    generate_reply(
        client,
        "m",
        "s",
        "b",
        [Attachment("a.pdf", "application/pdf", b"%PDF-1")],
        rag_tool=tool,
    )

    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents[1].inline_data.mime_type == "application/pdf"


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


def test_debug_logs_chunk_texts(caplog) -> None:
    with caplog.at_level(logging.INFO):
        log_grounding(response(chunks=CHUNKS), debug=True)

    assert "RAG chunk 1 from shipping.md: Doprava zdarma nad 1500 Kč." in caplog.text
    assert "RAG chunk 2 from returns.md: Vrácení do 30 dnů." in caplog.text


def test_no_grounding_metadata(caplog) -> None:
    empty = SimpleNamespace(
        text="x", candidates=[SimpleNamespace(grounding_metadata=None)]
    )

    with caplog.at_level(logging.INFO):
        log_grounding(empty)

    assert "RAG: 0 chunk(s) retrieved, sources: none" in caplog.text


def test_reply_flow_passes_the_tool_from_settings() -> None:
    from tests.fakes import FakeGmail, make_email, make_mailbox
    from app.mail_processing.sync import sync_mailbox

    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    mailbox = make_mailbox(gmail)
    mailbox.settings = Settings(GEMINI_MODEL="gemini-test", RAG_CORPUS=CORPUS)
    mailbox.gemini.models.generate_content.return_value = response()

    assert sync_mailbox(mailbox, "101").results == {"m1": "COMPLETED"}
    tools = mailbox.gemini.models.generate_content.call_args.kwargs["config"].tools
    assert tools[0].retrieval.vertex_rag_store.rag_resources[0].rag_corpus == CORPUS
