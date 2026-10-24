"""End-to-end UI tests with Streamlit's AppTest, using fakes instead of YouTube/OpenAI."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

from core.rag import VideoQA

APP = str(Path(__file__).resolve().parents[1] / "ytbot.py")
URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return AppTest.from_file(APP, default_timeout=30)


def fake_builder(video_id, segments, settings):
    llm = FakeListChatModel(responses=["Vectors live in FAISS [0:15]."])
    return VideoQA(video_id, segments, llm, DeterministicFakeEmbedding(size=32), settings)


def test_prompts_for_url(app):
    app.run()
    assert not app.exception
    assert "Paste a YouTube link" in app.info[0].value


def test_rejects_bad_url(app):
    app.run()
    app.text_input(key="url").input("https://vimeo.com/123").run()
    assert "doesn't look like a YouTube link" in app.error[0].value


def test_full_flow_answer_with_sources(app, segments):
    with (
        patch("core.transcript.fetch_transcript", return_value=segments),
        patch("core.rag.build_openai_qa", side_effect=fake_builder),
    ):
        app.run()
        app.text_input(key="url").input(URL).run()
        assert not app.exception, app.exception
        assert not app.error

        app.chat_input[0].set_value("Where are vectors stored?").run()
        assert not app.exception, app.exception

    contents = [m.markdown[0].value for m in app.chat_message]
    assert contents == ["Where are vectors stored?", "Vectors live in FAISS [0:15]."]
    # source "jump" buttons were rendered for the answer
    assert any(b.label.startswith("▶") for b in app.button)