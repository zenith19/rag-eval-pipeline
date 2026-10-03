"""Provenance: page is exact, section is a heuristic, neither may move the split.

The expensive property — that annotating provenance left all 249 original chunk
texts byte-identical — was verified against a fingerprint taken before the change
and is recorded in docs/decisions/004. These tests cover the parts that can be
checked quickly, plus a real single-PDF read.
"""

from pathlib import Path

import pytest

from rag.ingest import (
    _clean_heading,
    _page_for_offset,
    _plausible_section_number,
    _read_pdf,
    make_chunk_id,
    make_document_id,
)

CORPUS = Path(__file__).resolve().parent.parent / "data"
SAMPLE = CORPUS / "luo2005_ceaf.pdf"


def test_document_id_is_stable_and_distinct():
    assert make_document_id("a.pdf") == make_document_id("a.pdf")
    assert make_document_id("a.pdf") != make_document_id("b.pdf")


def test_document_id_is_independent_of_chunk_id():
    """Two different handles for two different jobs: one names the file, one names a piece of it."""
    assert make_document_id("a.pdf") != make_chunk_id("a.pdf", 0)


@pytest.mark.parametrize(
    "number,ok",
    [
        ("1", True), ("2.6.2", True), ("12", True),
        ("0.63532", False),   # a results-table row seen in the real corpus
        ("08", False),        # a reference line
        ("31", False),        # beyond any plausible section count here
        ("1.2.3.4", False),   # too deeply nested to be a heading
    ],
)
def test_section_numbers_reject_table_rows_and_references(number, ok):
    assert _plausible_section_number(number) is ok


def test_small_caps_extraction_artifacts_are_repaired():
    # PDFs render small-capped headings as a large initial plus a capital run,
    # so extraction yields "I NTRODUCTION".
    assert _clean_heading("I NTRODUCTION") == "Introduction"
    assert _clean_heading("L OW-R ANK ADAPTATION") == "Low-Rank Adaptation"
    assert _clean_heading("Related Work") == "Related Work"


def test_page_lookup_maps_offsets_to_pages():
    spans = [(1, 0, 100), (2, 101, 250), (3, 251, 400)]
    assert _page_for_offset(spans, 0) == 1
    assert _page_for_offset(spans, 150) == 2
    assert _page_for_offset(spans, 399) == 3
    assert _page_for_offset(spans, 10_000) == 3  # past the end: last page, not a crash


@pytest.mark.skipif(not SAMPLE.exists(), reason="corpus not fetched (scripts/fetch_corpus.py)")
def test_real_pdf_text_is_unchanged_by_page_tracking():
    """The spans are metadata: the text must still be exactly the joined pages."""
    from pypdf import PdfReader

    text, spans = _read_pdf(str(SAMPLE))
    expected = "\n".join(page.extract_text() or "" for page in PdfReader(str(SAMPLE)).pages)
    assert text == expected
    assert [p for p, _, _ in spans] == list(range(1, len(spans) + 1))
    for _page, start, end in spans:
        assert 0 <= start <= end <= len(text)
