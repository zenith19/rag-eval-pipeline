"""Compare two retrievers on the frozen eval set, with paired intervals.

The comparison this project is built to make, and the one it is easy to get wrong.
Both systems answer the SAME questions, so the question that matters is not "is
0.52 bigger than 0.39" but "is the per-query difference distinguishable from the
questions happening to suit one of them".

Absolute 95% intervals here are roughly ±0.13 wide, so any two retrievers'
intervals overlap heavily and overlap gets misread as "no difference". The paired
interval cancels the difficulty both systems share; a change that helps 9 questions
and harms 2 shows an interval clear of zero while the absolute intervals sit on top
of each other. So the verdict below is read off the paired interval, never off the
two means.

    python -m evaluation.compare_retrievers
    python -m evaluation.compare_retrievers --pool 20 --pool 50 --pool 100
"""

import argparse
import time

from evaluation.eval_harness import (
    DEFAULT_EVAL_SET,
    bootstrap_delta_ci,
    evaluate,
    load_eval_set,
)
from rag.config import RERANK_POOL
from rag.retriever import CrossEncoderRetriever, DenseRetriever

K_VALUES = (1, 3, 5, 10)
HEADLINE = ("recall@1", "recall@5", "recall@10", "ndcg@10", "mrr")


def _verdict(lo: float, hi: float) -> str:
    """What the interval licenses saying out loud."""
    if lo > 0:
        return "better"
    if hi < 0:
        return "WORSE"
    return "not distinguishable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pool",
        type=int,
        action="append",
        help="candidate pool size; repeatable to sweep. Default: the configured one.",
    )
    args = ap.parse_args()
    pools = args.pool or [RERANK_POOL]

    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    dense = DenseRetriever()

    t0 = time.perf_counter()
    base = evaluate(dense, eval_set, k_values=K_VALUES)
    dense_s = (time.perf_counter() - t0) / len(eval_set)

    print(f"{len(eval_set)} questions, dense baseline at {dense_s * 1000:.0f} ms/query\n")

    for pool in pools:
        # Reuse the dense retriever: a second instance would load the embedding
        # model again and change the latency being measured.
        rr = CrossEncoderRetriever(dense=dense, pool=pool)
        t0 = time.perf_counter()
        cand = evaluate(rr, eval_set, k_values=K_VALUES)
        rr_s = (time.perf_counter() - t0) / len(eval_set)

        print(f"=== pool {pool} — {rr_s * 1000:.0f} ms/query ({rr_s / dense_s:.0f}x dense) ===")
        print(f"  {'metric':<11} {'dense':>7} {'+rerank':>8} {'delta':>7}   95% CI of delta")
        for metric in HEADLINE:
            b = sum(base.per_query[metric]) / base.n_queries
            c = sum(cand.per_query[metric]) / cand.n_queries
            lo, hi = bootstrap_delta_ci(base.per_query[metric], cand.per_query[metric])
            print(
                f"  {metric:<11} {b:>7.3f} {c:>8.3f} {c - b:>+7.3f}   "
                f"[{lo:>+6.3f}, {hi:>+6.3f}]  {_verdict(lo, hi)}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
