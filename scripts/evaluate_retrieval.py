"""Measure how well retrieval finds the right part of a video.

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --chunk-sizes 500 1000 2000 --k 1 3 5 10
    python scripts/evaluate_retrieval.py --show-misses --save evals/results.md

Uses your OPENAI_API_KEY for embeddings only (no chat model), so a full run costs
a fraction of a cent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Settings  # noqa: E402
from core.evaluation import EvalSet, best_report, evaluate, format_table  # noqa: E402
from core.transcript import fetch_transcript, format_timestamp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default="evals/neural_networks.json", help="test set JSON")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.25,
        help="ignore settings that retrieve more than this share of the video (default 0.25)",
    )
    parser.add_argument("--show-misses", action="store_true", help="list questions not found")
    parser.add_argument("--save", help="also write the results table to this Markdown file")
    args = parser.parse_args()

    settings = Settings()
    if not settings.openai_api_key:
        print("Set OPENAI_API_KEY in your .env file first.", file=sys.stderr)
        return 1

    from langchain_openai import OpenAIEmbeddings

    eval_set = EvalSet.load(args.cases)
    print(f"Video: {eval_set.title or eval_set.video_id}")
    print(f"Questions: {len(eval_set.cases)}   Embeddings: {settings.embedding_model}\n")

    segments = fetch_transcript(eval_set.video_id)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    reports = evaluate(
        segments,
        eval_set.video_id,
        eval_set.cases,
        embeddings,
        chunk_sizes=args.chunk_sizes,
        ks=args.k,
        overlap_segments=settings.chunk_overlap_segments,
    )

    table = format_table(reports)
    print(table)

    best = best_report(reports, max_coverage=args.max_coverage)
    print(
        f"\nBest (retrieving at most {args.max_coverage:.0%} of the video): "
        f"chunk size {best.chunk_size}, k={best.k} "
        f"-> hit rate {best.hit_rate:.0%}, MRR {best.mrr:.2f}, "
        f"video retrieved {best.coverage:.0%}"
    )

    if args.show_misses:
        print(f"\nMissed at chunk size {best.chunk_size}, k={best.k}:")
        for miss in best.misses or []:
            c = miss.case
            got = ", ".join(format_timestamp(d.metadata["start"]) for d in miss.retrieved)
            print(
                f"  - {c.question}\n"
                f"      answer at {format_timestamp(c.answer_start)}"
                f"-{format_timestamp(c.answer_end)}, retrieved: {got}"
            )
        if not best.misses:
            print("  (none)")

    if args.save:
        Path(args.save).write_text(
            f"# Retrieval evaluation\n\n{eval_set.title}  \n"
            f"{len(eval_set.cases)} questions, embeddings: `{settings.embedding_model}`\n\n"
            f"{table}\n",
            encoding="utf-8",
        )
        print(f"\nSaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
