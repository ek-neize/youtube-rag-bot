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

# --- conversational memory -------------------------------------------------
def recording_llm(calls, condensed="What is FAISS used for?", answer="an answer"):
    """A fake LLM that records which prompt it got and replies accordingly."""

    def _llm(prompt_value):
        messages = prompt_value.to_messages()
        if messages[0].content.startswith("You rewrite follow-up questions"):
            calls.append(("condense", messages[-1].content))
            return condensed
        calls.append(("answer", messages[-1].content))
        return answer

    return RunnableLambda(_llm)


HISTORY = [
    {"role": "user", "content": "How are vectors stored?"},
    {"role": "assistant", "content": "They are stored in a FAISS index [0:15]."},
]


def test_first_question_skips_condense(segments):
    qa = make_qa(segments)
    calls = []
    qa.llm = recording_llm(calls)
    result = qa.answer("How are vectors stored?")
    assert [c[0] for c in calls] == ["answer"]
    assert result.search_query == "How are vectors stored?"


def test_follow_up_is_condensed_and_used_for_retrieval(segments, monkeypatch):
    qa = make_qa(segments)
    calls = []
    qa.llm = recording_llm(calls)
    searched = []
    original_retrieve = qa.retrieve
    monkeypatch.setattr(
        qa, "retrieve", lambda q, k=None: searched.append(q) or original_retrieve(q, k)
    )

    result = qa.answer("What is it used for?", history=HISTORY)

    assert [c[0] for c in calls] == ["condense", "answer"]
    assert "FAISS index" in calls[0][1]  # the history reached the rewrite prompt
    assert searched == ["What is FAISS used for?"]  # retrieval used the rewrite
    assert result.search_query == "What is FAISS used for?"
    assert "Recent conversation" in calls[1][1]  # history also reaches the answer prompt
    assert "What is it used for?" in calls[1][1]  # but the user's own words are answered


def test_empty_rewrite_falls_back_to_original(segments):
    qa = make_qa(segments)
    qa.llm = recording_llm([], condensed="   ")
    assert qa.answer("And then?", history=HISTORY).search_query == "And then?"


def test_format_history_limits_and_trims():
    from core.rag import format_history

    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    assert format_history(history, max_messages=3).splitlines() == [
        "User: q7", "User: q8", "User: q9"
    ]
    long = [{"role": "assistant", "content": "x" * 1000}]
    assert format_history(long, max_messages=6, max_chars=50) == "Assistant: " + "x" * 50 + "…"
    assert format_history(history, max_messages=0) == ""
