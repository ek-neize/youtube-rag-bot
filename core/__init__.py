"""Core RAG logic for the YouTube QA bot, kept independent of the UI so it can be tested."""

from core.config import Settings
from core.rag import Answer, VideoQA
from core.transcript import (
    Segment,
    TranscriptError,
    build_chunks,
    fetch_transcript,
    format_timestamp,
    get_video_id,
    transcript_to_text,
)

__all__ = [
    "Answer",
    "Segment",
    "Settings",
    "TranscriptError",
    "VideoQA",
    "build_chunks",
    "fetch_transcript",
    "format_timestamp",
    "get_video_id",
    "transcript_to_text",
]