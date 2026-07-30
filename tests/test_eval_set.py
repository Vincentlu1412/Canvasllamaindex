"""Guardrails for the evaluation set and knowledge base.

A RAG benchmark leaks silently: if ground_truth answers are verbatim copies
of the source docs, or the docs contain the eval questions' own wording,
both Hit Rate and judge scores are inflated and the pipeline looks far
better than it is. These tests pin down the invariants that prevent it.
"""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = ROOT / "eval_set.json"
SOURCE_DOCS_DIR = ROOT / "data" / "source_docs"


@pytest.fixture(scope="module")
def eval_items() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_doc_text() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in SOURCE_DOCS_DIR.glob("*.md")}


def test_eval_set_is_valid_json() -> None:
    json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def test_no_duplicate_ids(eval_items: list[dict]) -> None:
    ids = [item["id"] for item in eval_items]
    assert len(ids) == len(set(ids)), "duplicate question ids in eval_set.json"


def test_answerable_items_have_a_source_file(eval_items: list[dict]) -> None:
    for item in eval_items:
        if item.get("is_answerable", True):
            source = item.get("expected_source_metadata", {}).get("source_file")
            assert source, f"{item['id']} is answerable but has no expected_source_metadata.source_file"


def test_unanswerable_items_have_no_source_file(eval_items: list[dict]) -> None:
    for item in eval_items:
        if not item.get("is_answerable", True):
            assert "expected_source_metadata" not in item, (
                f"{item['id']} is marked unanswerable but tags a source file — "
                "that would make hit-rate scoring meaningless for it"
            )


def test_expected_source_files_exist(eval_items: list[dict], source_doc_text: dict[str, str]) -> None:
    for item in eval_items:
        source = item.get("expected_source_metadata", {}).get("source_file")
        if source:
            assert source in source_doc_text, f"{item['id']} points at missing file {source}"


def test_question_text_does_not_leak_into_source_docs(eval_items: list[dict], source_doc_text: dict[str, str]) -> None:
    """Regression test for the original leakage bug: a doc must not contain
    the eval question verbatim, which used to let retrieval "cheat" via
    literal question/question string matching instead of real semantic
    matching against the answer content."""
    for item in eval_items:
        question = item["question"].rstrip("?").lower()
        for name, text in source_doc_text.items():
            assert question not in text.lower(), (
                f"{item['id']}'s question text leaks verbatim into {name}"
            )


def test_ground_truth_is_not_a_verbatim_doc_copy(eval_items: list[dict], source_doc_text: dict[str, str]) -> None:
    """Regression test: ground_truth must be a paraphrase, not a copy-paste
    of the source doc, otherwise the LLM judge is graded on parroting, not
    understanding."""
    for item in eval_items:
        if not item.get("is_answerable", True):
            continue
        ground_truth = item["ground_truth"].lower()
        source = item["expected_source_metadata"]["source_file"]
        doc_text = source_doc_text[source].lower()
        assert ground_truth not in doc_text, (
            f"{item['id']}'s ground_truth is copied verbatim from {source}"
        )


def test_app_compiles() -> None:
    py_compile.compile(str(ROOT / "hello_streamlit.py"), doraise=True)
