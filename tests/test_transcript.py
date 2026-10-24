from types import SimpleNamespace

import pytest

from core.transcript import (
    Segment,
    TranscriptError,
    build_chunks,
    fetch_transcript,
    format_timestamp,
    get_video_id,
    transcript_to_text,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_get_video_id_valid(url):
    assert get_video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", ["", None, "https://vimeo.com/123", "not a url"])
def test_get_video_id_invalid(url):
    assert get_video_id(url) is None


@pytest.mark.parametrize(
    "seconds,expected", [(0, "0:00"), (65.9, "1:05"), (3600, "1:00:00"), (3725, "1:02:05")]
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_transcript_to_text(segments):
    text = transcript_to_text(segments[:2])
    assert text.splitlines()[1].startswith("[0:05] First we load")
    assert "[" not in transcript_to_text(segments[:2], with_timestamps=False)


def test_build_chunks_metadata_and_coverage(segments):
    docs = build_chunks(segments, "vid12345678", chunk_size=120, overlap_segments=1)
    assert len(docs) > 1
    assert docs[0].metadata["start"] == 0.0
    assert docs[-1].metadata["end"] == segments[-1].end
    for d in docs:
        assert d.metadata["video_id"] == "vid12345678"
        assert d.metadata["start"] < d.metadata["end"]
    # every segment's text appears in at least one chunk
    joined = " ".join(d.page_content for d in docs)
    assert all(s.text in joined for s in segments)


def test_build_chunks_overlap(segments):
    docs = build_chunks(segments, "v", chunk_size=120, overlap_segments=1)
    for prev, nxt in zip(docs, docs[1:], strict=False):
        assert nxt.metadata["start"] < prev.metadata["end"]  # overlapping time ranges


def test_build_chunks_oversized_segment_still_progresses():
    segs = [Segment("x" * 500, 0, 1), Segment("short", 1, 1)]
    docs = build_chunks(segs, "v", chunk_size=100, overlap_segments=2)
    assert [d.page_content for d in docs] == ["x" * 500, "short"]


def test_build_chunks_rejects_bad_size(segments):
    with pytest.raises(ValueError):
        build_chunks(segments, "v", chunk_size=0)


# --- fetch_transcript with a fake API --------------------------------------
def _track(code, generated, snippets, translatable=False):
    track = SimpleNamespace(
        language_code=code,
        is_generated=generated,
        is_translatable=translatable,
        fetch=lambda: [SimpleNamespace(text=t, start=s, duration=1.0) for t, s in snippets],
    )
    track.translate = lambda lang: _track(lang, True, [("translated", 0.0)])
    return track


class FakeApi:
    def __init__(self, tracks):
        self.tracks = tracks

    def list(self, video_id):
        return self.tracks


def test_fetch_prefers_manual_english():
    api = FakeApi([_track("en", True, [("auto", 0)]), _track("en", False, [("manual", 0)])])
    assert fetch_transcript("v", api=api)[0].text == "manual"


def test_fetch_falls_back_to_translation():
    api = FakeApi([_track("fr", False, [("bonjour", 0)], translatable=True)])
    assert fetch_transcript("v", api=api)[0].text == "translated"


def test_fetch_cleans_noise_tags():
    api = FakeApi([_track("en", False, [("[Music]  hello   world", 0), ("[Applause]", 1)])])
    segs = fetch_transcript("v", api=api)
    assert [s.text for s in segs] == ["hello world"]


def test_fetch_no_tracks_raises():
    with pytest.raises(TranscriptError):
        fetch_transcript("v", api=FakeApi([]))