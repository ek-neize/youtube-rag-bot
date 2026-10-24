"""Runtime settings, read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All tunable knobs in one place. Override any of them via environment variables."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    # Chunking: target characters per chunk, and how many transcript segments to overlap
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap_segments: int = field(
        default_factory=lambda: _env_int("CHUNK_OVERLAP_SEGMENTS", 2)
    )
    # Retrieval: how many chunks to hand the LLM per question
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    # Transcripts longer than this (characters) are summarized map-reduce style
    max_single_pass_chars: int = field(
        default_factory=lambda: _env_int("MAX_SINGLE_PASS_CHARS", 100_000)
    )