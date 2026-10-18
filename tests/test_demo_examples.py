"""The demo page's example questions must be ones the corpus can answer.

Two of the original three could not: "embedding collapse" appears in none of the 48
papers, and back-translation never shares a chunk with coreference. The model said
"I don't know" to both, which was the faithful answer to a question the page should
not have suggested. Found in the first live Langfuse traces (docs/decisions/014).

Each example must now be a question from the frozen eval set, where retrieval
quality is measured rather than hoped for. That ties the page to evidence without
needing Qdrant or Bedrock to check it.
"""

import html
import re

from evaluation.eval_harness import DEFAULT_EVAL_SET, load_eval_set


def test_every_example_question_comes_from_the_measured_eval_set():
    from api.main import _PAGE

    examples = [
        html.unescape(q) for q in re.findall(r'onclick="fill\(this\)">([^<]+)</button>', _PAGE)
    ]
    measured = {ex.query for ex in load_eval_set(DEFAULT_EVAL_SET)}

    assert len(examples) == 3
    assert not [q for q in examples if q not in measured]
