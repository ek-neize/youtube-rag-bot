"""Pipeline tests using fake LLM/embeddings, so they run offline and cost nothing."""

import dataclasses

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from core.config import Settings
from core.rag import VideoQA, format_context

SETTINGS = dataclasses.replace(Settings(), chunk_size=120, chunk_overlap_segments=1, top_k=3)


def make_qa(segments, responses=("fake answer [0:05]",), settings=SETTINGS):
    return VideoQA(
        "vid12345678",
        segments,
        llm=FakeListChatModel(responses=list(responses)),
        embeddings=DeterministicFakeEmbedding(size=32),
        settings=settings,
    )


def test_index_built(segments):
    qa = make_qa(segments)
    assert qa.vector_store.index.ntotal == len(qa.chunks)


def test_retrieve_respects_k(segments):
    qa = make_qa(segments)
    assert len(qa.retrieve("what is FAISS?", k=2)) == 2
    # k larger than the number of chunks is clamped
    assert len(qa.retrieve("anything", k=999)) == len(qa.chunks)


def test_answer_returns_text_and_sorted_sources(segments):
    qa = make_qa(segments)
    result = qa.answer("How are vectors stored?")
    assert result.text == "fake answer [0:05]"
    starts = [d.metadata["start"] for d in result.sources]
    assert starts == sorted(starts) and len(starts) == SETTINGS.top_k


def test_format_context_labels_time_ranges(segments):
    qa = make_qa(segments)
    ctx = format_context(qa.chunks[:2])
    assert ctx.startswith("[0:00-")
    assert ctx.count("[") == 2


def test_summarize_single_pass(segments):
    qa = make_qa(segments, responses=["a summary"])
    assert qa.summarize() == "a summary"


def test_summarize_map_reduce_for_long_transcripts(segments):
    settings = dataclasses.replace(SETTINGS, max_single_pass_chars=200)
    qa = make_qa(segments, settings=settings)

    calls = []

    def recording_llm(prompt_value):
        system = prompt_value.to_messages()[0].content
        step = "reduce" if system.startswith("You are given summaries") else "map"
        calls.append(step)
        return "combined" if step == "reduce" else "part"

    qa.llm = RunnableLambda(recording_llm)
    assert qa.summarize() == "combined"
    assert calls.count("map") > 1 and calls.count("reduce") == 1


def test_empty_transcript_rejected():
    with pytest.raises(ValueError):
        make_qa([])