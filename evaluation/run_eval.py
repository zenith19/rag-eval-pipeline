"""Score the dense retriever against the labelled eval set."""

from evaluation.eval_harness import DEFAULT_EVAL_SET, evaluate, load_eval_set
from rag.retriever import DenseRetriever


def main() -> None:
    eval_set = load_eval_set(DEFAULT_EVAL_SET)
    report = evaluate(DenseRetriever(), eval_set, k_values=(1, 3, 5, 10))
    print(report.pretty())


if __name__ == "__main__":
    main()
