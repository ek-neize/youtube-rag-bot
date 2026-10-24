import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.transcript import Segment  # noqa: E402


@pytest.fixture
def segments():
    lines = [
        "Welcome to this video about retrieval augmented generation.",
        "First we load documents and split them into chunks.",
        "Then each chunk is turned into an embedding vector.",
        "The vectors are stored in a FAISS index for fast search.",
        "At question time we retrieve the most similar chunks.",
        "Finally the language model answers using that context.",
        "Thanks for watching and see you next time.",
    ]
    return [Segment(text=t, start=i * 5.0, duration=5.0) for i, t in enumerate(lines)]