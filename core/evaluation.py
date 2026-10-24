"""Retrieval evaluation: does the retriever find the part of the video that answers a question?

Each test case is a question plus the time range (in seconds) where the answer is spoken.
A retrieved chunk counts as correct if its own time range overlaps the answer's range,
which works no matter how the transcript was chunked.

Metrics
-------
hit rate@k  share of questions with at least one correct chunk in the top k results
MRR@k       mean reciprocal rank: 1 if the first correct chunk ranks 1st, 1/2 if 2nd, ...
            and 0 if none is in the top k. Rewards putting the right chunk first.
coverage    average share of the video's running time the top k chunks span. Big chunks
            and big k raise hit rate partly by retrieving more of the video; coverage
            makes that cost visible (more text for the LLM to read, more tokens, more noise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.transcript import Segment, build_chunks


@dataclass(frozen=True)
class EvalCase:
    question: str
    answer_start: float  # seconds
    answer_end: float  # seconds

    def matches(self, doc: Document) -> bool:
        """True if the chunk's time range overlaps the answer's time range."""
        return doc.metadata["start"] < self.answer_end and doc.metadata["end"] > self.answer_start


@dataclass
class EvalSet:
    video_id: str
    title: str
    cases: list[EvalCase]

    @classmethod
    def load(cls, path: str | Path) -> EvalSet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [
            EvalCase(c["question"], _seconds(c["answer_start"]), _seconds(c["answer_end"]))
            for c in data["cases"]
        ]
        for c in cases:
            if c.answer_end <= c.answer_start:
                raise ValueError(f"answer_end must be after answer_start: {c.question!r}")
        return cls(video_id=data["video_id"], title=data.get("title", ""), cases=cases)


@dataclass
class CaseResult:
    case: EvalCase
    rank: int | None  # 1-based rank of the first correct chunk, None if not retrieved
    retrieved: list[Document] = field(default_factory=list)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by possibly overlapping (start, end) intervals."""
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


@dataclass
class EvalReport:
    chunk_size: int
    k: int
    num_chunks: int
    results: list[CaseResult]
    video_duration: float = 0.0

    @property
    def coverage(self) -> float:
        if not self.results or self.video_duration <= 0:
            return 0.0
        spans = [
            _union_length([(d.metadata["start"], d.metadata["end"]) for d in r.retrieved])
            for r in self.results
        ]
        return sum(spans) / len(spans) / self.video_duration

    @property
    def hit_rate(self) -> float:
        hits = [r for r in self.results if r.rank is not None and r.rank <= self.k]
        return len(hits) / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        scores = [
            1 / r.rank if r.rank is not None and r.rank <= self.k else 0.0 for r in self.results
        ]
        return sum(scores) / len(scores)

    @property
    def misses(self) -> list[CaseResult]:
        return [r for r in self.results if r.rank is None or r.rank > self.k]


def _seconds(value: float | int | str) -> float:
    """Accept 125, 125.5, "2:05" or "1:02:05"."""
    if isinstance(value, int | float):
        return float(value)
    total = 0.0
    for part in str(value).split(":"):
        total = total * 60 + float(part)
    return total


def first_correct_rank(case: EvalCase, docs: list[Document]) -> int | None:
    for i, doc in enumerate(docs, start=1):
        if case.matches(doc):
            return i
    return None


def evaluate(
    segments: list[Segment],
    video_id: str,
    cases: list[EvalCase],
    embeddings: Embeddings,
    chunk_sizes: list[int],
    ks: list[int],
    overlap_segments: int = 2,
) -> list[EvalReport]:
    """Score retrieval for every chunk size x k combination.

    The index is built once per chunk size and searched once per question with the
    largest k; smaller k values are scored from the same ranked list, so the number of
    embedding calls stays low.
    """
    if not cases:
        raise ValueError("Need at least one evaluation case.")
    max_k = max(ks)
    duration = segments[-1].end - segments[0].start if segments else 0.0
    reports: list[EvalReport] = []
    for size in chunk_sizes:
        chunks = build_chunks(
            segments, video_id, chunk_size=size, overlap_segments=overlap_segments
        )
        store = FAISS.from_documents(chunks, embeddings)
        depth = min(max_k, len(chunks))
        ranked = {c: store.similarity_search(c.question, k=depth) for c in cases}
        for k in sorted(ks):
            results = [
                CaseResult(c, first_correct_rank(c, ranked[c]), ranked[c][:k]) for c in cases
            ]
            reports.append(
                EvalReport(
                    chunk_size=size,
                    k=k,
                    num_chunks=len(chunks),
                    results=results,
                    video_duration=duration,
                )
            )
    return reports


def format_table(reports: list[EvalReport]) -> str:
    """A Markdown table of results, ready to paste into the README."""
    lines = [
        "| Chunk size | Chunks | k | Hit rate | MRR | Video retrieved |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.chunk_size} | {r.num_chunks} | {r.k} | {r.hit_rate:.0%} | {r.mrr:.2f} "
            f"| {r.coverage:.0%} |"
        )
    return "\n".join(lines)


def best_report(reports: list[EvalReport], max_coverage: float = 0.25) -> EvalReport:
    """Pick the best setting: highest hit rate, then MRR, among those that retrieve at
    most `max_coverage` of the video (so 'retrieve everything' can't win).
    Falls back to all reports if none qualify."""
    eligible = [r for r in reports if r.coverage <= max_coverage] or reports
    return max(eligible, key=lambda r: (r.hit_rate, r.mrr, -r.coverage))
