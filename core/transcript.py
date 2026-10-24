"""Loading YouTube transcripts and turning them into time-stamped LangChain Documents."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})"
)
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


class TranscriptError(Exception):
    """Raised with a user-friendly message when a transcript can't be loaded."""


@dataclass(frozen=True)
class Segment:
    """One caption line from a transcript."""

    text: str
    start: float
    duration: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------
def get_video_id(url: str | None) -> str | None:
    """Extract the 11-character video ID from a YouTube URL (or accept a bare ID).

    Supports watch?v=, youtu.be/, /shorts/, /embed/ and /live/ links.
    """
    if not url:
        return None
    url = url.strip()
    if _BARE_ID_RE.match(url):
        return url
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def format_timestamp(seconds: float) -> str:
    """Format seconds as m:ss, or h:mm:ss for long videos."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def timestamp_url(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_transcript(video_id: str, api: YouTubeTranscriptApi | None = None) -> list[Segment]:
    """Fetch a transcript, preferring English.

    Order of preference: manual English > auto-generated English >
    any transcript machine-translated to English > any transcript as-is.
    """
    api = api or YouTubeTranscriptApi()
    try:
        transcripts = list(api.list(video_id))
        chosen = _choose_transcript(transcripts)
        if chosen is None:
            raise TranscriptError("This video has no transcripts available.")
        fetched = chosen.fetch()
    except TranscriptError:
        raise
    except TranscriptsDisabled as e:
        raise TranscriptError("Transcripts are disabled for this video.") from e
    except NoTranscriptFound as e:
        raise TranscriptError("No transcript was found for this video.") from e
    except VideoUnavailable as e:
        raise TranscriptError(
            "This video is unavailable (private, removed or region-locked)."
        ) from e
    except (RequestBlocked, IpBlocked) as e:
        raise TranscriptError(
            "YouTube is blocking transcript requests from this network. This often happens "
            "on cloud servers; try running the app locally or configure a proxy."
        ) from e
    except CouldNotRetrieveTranscript as e:
        raise TranscriptError(f"Couldn't retrieve the transcript: {e.__class__.__name__}") from e

    segments = [
        Segment(text=_clean(s.text), start=float(s.start), duration=float(s.duration))
        for s in fetched
    ]
    return [s for s in segments if s.text]


def _choose_transcript(transcripts: list):
    english = [t for t in transcripts if t.language_code.lower().startswith("en")]
    manual = [t for t in english if not t.is_generated]
    if manual:
        return manual[0]
    if english:
        return english[0]
    for t in transcripts:
        if getattr(t, "is_translatable", False):
            try:
                return t.translate("en")
            except Exception:  # translation not offered for this track
                continue
    return transcripts[0] if transcripts else None


def _clean(text: str) -> str:
    text = re.sub(r"\[(?:music|applause|laughter)\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def transcript_to_text(segments: Iterable[Segment], with_timestamps: bool = True) -> str:
    """Render segments as plain text, one line per segment."""
    if with_timestamps:
        return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments)
    return " ".join(s.text for s in segments)


def build_chunks(
    segments: list[Segment],
    video_id: str,
    chunk_size: int = 1000,
    overlap_segments: int = 2,
) -> list[Document]:
    """Group consecutive caption segments into chunks of roughly `chunk_size` characters.

    Unlike a plain character splitter, chunk boundaries always fall between captions,
    so every chunk knows exactly where it starts and ends in the video. That metadata
    is what lets the app cite clickable timestamps for each answer.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap_segments = max(0, overlap_segments)

    docs: list[Document] = []
    i = 0
    while i < len(segments):
        j, length = i, 0
        while j < len(segments) and (j == i or length + len(segments[j].text) + 1 <= chunk_size):
            length += len(segments[j].text) + 1
            j += 1
        window = segments[i:j]
        docs.append(
            Document(
                page_content=" ".join(s.text for s in window),
                metadata={
                    "video_id": video_id,
                    "start": window[0].start,
                    "end": window[-1].end,
                    "chunk": len(docs),
                },
            )
        )
        if j >= len(segments):
            break
        # step forward, keeping a few segments of overlap for context continuity
        i = max(j - overlap_segments, i + 1)
    return docs