"""Evaluation tests, run offline with a keyword-matching fake embedding."""

import json
import math
import re
import zlib
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.evaluation import (
    CaseResult,
    EvalCase,
    EvalReport,
    EvalSet,
    _union_length,
    best_report,
    evaluate,
    first_correct_rank,
    format_table,
)
from core.transcript import Segment

STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "what", "how", "does", "it"}


class KeywordEmbeddings(Embeddings):
    """Bag-of-words vectors: texts sharing words point in similar directions."""

    def __init__(self, size: int = 512):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in STOPWORDS:
                vec[zlib.crc32(word.encode()) % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# A tiny "video": one topic every 10 seconds
TOPICS = [
    "Neurons hold a number between zero and one called the activation.",
    "The input layer has 784 neurons, one per pixel of the image.",
    "Weights are attached to every connection between layers.",
    "The sigmoid function squishes any number into the range zero to one.",
    "The bias shifts the weighted sum before the sigmoid.",
    "ReLU replaced sigmoid because deep networks train more easily with it.",
]
SEGMENTS = [Segment(text=t, start=i * 10.0, duration=10.0) for i, t in enumerate(TOPICS)]
CASES = [
    EvalCase("What does the sigmoid function do?", 30, 40),
    EvalCase("How many neurons are in the input layer?", 10, 20),
    EvalCase("Why is ReLU used instead of sigmoid?", 50, 60),
]


def doc(start, end):
    return Document(page_content="x", metadata={"start": start, "end": end})


# --- matching and ranking ---------------------------------------------------
@pytest.mark.parametrize(
    "start,end,expected",
    [
        (30, 40, True),  # exact
        (25, 35, True),  # overlaps the beginning
        (35, 50, True),  # overlaps the end
        (20, 30, False),  # touches but doesn't overlap
        (40, 45, False),  # after
    ],
)
def test_case_matches_on_time_overlap(start, end, expected):
    assert EvalCase("q", 30, 40).matches(doc(start, end)) is expected


def test_first_correct_rank():
    case = EvalCase("q", 30, 40)
    assert first_correct_rank(case, [doc(0, 10), doc(32, 38), doc(30, 40)]) == 2
    assert first_correct_rank(case, [doc(0, 10)]) is None


def test_hit_rate_and_mrr_math():
    cases = [EvalCase(f"q{i}", 0, 1) for i in range(4)]
    report = EvalReport(
        chunk_size=100,
        k=3,
        num_chunks=10,
        results=[
            CaseResult(cases[0], rank=1),
            CaseResult(cases[1], rank=2),
            CaseResult(cases[2], rank=4),  # beyond k=3, so a miss
            CaseResult(cases[3], rank=None),
        ],
    )
    assert report.hit_rate == 0.5
    assert report.mrr == pytest.approx((1 + 0.5) / 4)
    assert [r.case.question for r in report.misses] == ["q2", "q3"]


# --- loading test sets -------------------------------------------------------
def test_eval_set_load_parses_timestamps(tmp_path):
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps(
            {
                "video_id": "abc12345678",
                "cases": [
                    {"question": "q1", "answer_start": "2:05", "answer_end": "2:30"},
                    {"question": "q2", "answer_start": 3725, "answer_end": "1:02:10"},
                ],
            }
        )
    )
    s = EvalSet.load(path)
    assert s.video_id == "abc12345678"
    assert (s.cases[0].answer_start, s.cases[0].answer_end) == (125, 150)
    assert (s.cases[1].answer_start, s.cases[1].answer_end) == (3725, 3730)


def test_eval_set_rejects_backwards_ranges(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {"video_id": "v", "cases": [{"question": "q", "answer_start": 50, "answer_end": 40}]}
        )
    )
    with pytest.raises(ValueError):
        EvalSet.load(path)


def test_bundled_test_set_is_valid():
    s = EvalSet.load(Path(__file__).resolve().parents[1] / "evals" / "neural_networks.json")
    assert s.video_id == "aircAruvnKk"
    assert len(s.cases) >= 15
    assert len({c.question for c in s.cases}) == len(s.cases)  # no duplicate questions


# --- the full grid ----------------------------------------------------------
def test_evaluate_finds_the_right_chunks():
    reports = evaluate(
        SEGMENTS, "v", CASES, KeywordEmbeddings(), chunk_sizes=[80], ks=[1, 3]
    )
    by_k = {r.k: r for r in reports}
    assert by_k[1].num_chunks == len(TOPICS)  # each topic fits in its own chunk
    assert by_k[3].hit_rate == 1.0
    assert by_k[1].hit_rate >= 2 / 3
    assert by_k[3].mrr >= by_k[1].mrr


def test_evaluate_grid_shape_and_bigger_k_never_hurts():
    reports = evaluate(
        SEGMENTS, "v", CASES, KeywordEmbeddings(), chunk_sizes=[80, 200], ks=[3, 1, 5]
    )
    assert [(r.chunk_size, r.k) for r in reports] == [
        (80, 1), (80, 3), (80, 5), (200, 1), (200, 3), (200, 5)
    ]
    for size in (80, 200):
        rates = [r.hit_rate for r in reports if r.chunk_size == size]
        assert rates == sorted(rates)  # hit rate can only go up as k grows


def test_evaluate_needs_cases():
    with pytest.raises(ValueError):
        evaluate(SEGMENTS, "v", [], KeywordEmbeddings(), chunk_sizes=[80], ks=[1])


def test_format_table():
    reports = evaluate(SEGMENTS, "v", CASES, KeywordEmbeddings(), chunk_sizes=[80], ks=[3])
    table = format_table(reports).splitlines()
    assert table[0] == "| Chunk size | Chunks | k | Hit rate | MRR | Video retrieved |"
    assert table[2].startswith("| 80 | 6 | 3 | 100% |")
    assert table[2].endswith("| 50% |")  # 3 of 6 equal-length chunks


# --- coverage ---------------------------------------------------------------
def test_union_length_merges_overlaps():
    assert _union_length([]) == 0
    assert _union_length([(0, 10), (5, 15), (20, 30)]) == 25
    assert _union_length([(20, 30), (0, 10)]) == 20


def test_coverage_grows_with_k():
    reports = evaluate(
        SEGMENTS, "v", CASES, KeywordEmbeddings(), chunk_sizes=[80], ks=[1, 3, 6]
    )
    cov = {r.k: r.coverage for r in reports}
    assert cov[1] == pytest.approx(1 / 6)
    assert cov[3] == pytest.approx(3 / 6)
    assert cov[6] == pytest.approx(1.0)  # retrieving everything covers the whole video


def test_best_report_ignores_settings_that_retrieve_too_much():
    reports = evaluate(
        SEGMENTS, "v", CASES, KeywordEmbeddings(), chunk_sizes=[80], ks=[1, 6]
    )
    by_k = {r.k: r for r in reports}
    assert by_k[6].coverage == pytest.approx(1.0)  # k=6 retrieves the whole video
    # k=6 is never chosen under the coverage limit
    assert best_report(reports, max_coverage=0.25) is by_k[1]
    # with no limit, a tie on hit rate and MRR goes to the setting that retrieves less
    assert by_k[1].hit_rate == by_k[6].hit_rate
    assert best_report(reports, max_coverage=1.0) is by_k[1]
