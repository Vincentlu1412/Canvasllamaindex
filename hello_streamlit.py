from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import chromadb
import streamlit as st
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


APP_TITLE = "Local Berkeley Canvas Assistant"
APP_SUBTITLE = "Fully local LlamaIndex + ChromaDB + Ollama stack."
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
# Points at data/source_docs specifically (not data/), so eval_set.json and
# other non-content files in data/ never get swept into the index by
# SimpleDirectoryReader's recursive scan. This was a real leak: the eval
# questions and ground-truth answers were previously being embedded as
# retrievable documents.
DOCS_DIR = Path(os.getenv("LLAMAINDEX_DOCS_DIR", "data/source_docs"))
EVAL_SET_PATH = Path(os.getenv("EVAL_SET_PATH", "eval_set.json"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Judge is not trusted on the full eval set until it agrees with human
# scores on the hand-scored calibration subset at least this often.
# This is the fix for Alex's "GPT-4o grading GPT-4o" note: the same model
# family doing the generation is also doing the judging, so we verify it
# against humans before trusting it on the rest of the questions.
CALIBRATION_AGREEMENT_THRESHOLD = 0.7
RETRIEVAL_TOP_K = 3


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


def _iter_document_dirs() -> Iterable[Path]:
    candidates = [DOCS_DIR, Path("documents"), Path("docs")]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _load_source_documents() -> list[Document]:
    for candidate in _iter_document_dirs():
        if candidate.exists() and candidate.is_dir():
            # required_exts pins this to markdown content only, so a stray
            # .json/.db/other file dropped into the docs folder never gets
            # silently embedded as "knowledge" (see DOCS_DIR comment above).
            docs = SimpleDirectoryReader(
                str(candidate), recursive=True, required_exts=[".md"]
            ).load_data()
            if docs:
                return docs

    return [
        Document(
            text=(
                "Fallback knowledge base. Add files under `data/`, `docs/`, "
                "or `documents/` to replace this sample document."
            ),
            metadata={"source": "fallback"},
        )
    ]


def _configure_local_models() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=LOCAL_EMBED_MODEL)
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )


def _answer_text(response: object) -> str:
    text = getattr(response, "response", None)
    return text if isinstance(text, str) and text.strip() else str(response)


@st.cache_resource(show_spinner=False)
def get_or_create_index() -> VectorStoreIndex:
    """
    Create the ChromaDB client and LlamaIndex index exactly once.

    This prevents SQLite-backed persistence from being reopened on each
    Streamlit rerun, which is what caused the startup lock behavior.

    Set FORCE_REINDEX=1 to wipe the persisted collection and rebuild from
    data/source_docs on the next run (e.g. right after editing the .md
    files) instead of permanently forcing a full rebuild on every single
    run, which defeated the point of @st.cache_resource / persistence.
    """
    _configure_local_models()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if os.getenv("FORCE_REINDEX") == "1":
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection may not exist yet

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Path A: reuse existing persisted vectors.
    if chroma_collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Path B: build the first index from local documents.
    documents = _load_source_documents()
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )


# ---------------------------------------------------------------------------
# Evaluation harness
#
# Two things changed from the original prep-sheet plan, based on Alex's
# review:
#
# 1. eval_set.json now carries `source_file` / `page_label` per question, so
#    Hit Rate can actually be computed against a real ground-truth location
#    instead of having nowhere to read it from.
# 2. Before the LLM judge is trusted on the full set, it's checked against a
#    hand-scored subset (`human_score` present in eval_set.json). If it
#    doesn't agree with humans often enough, the app refuses to show judge
#    scores for the rest and tells you to fix the judge prompt first.
# ---------------------------------------------------------------------------


def _load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        return []
    try:
        with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _judge_score(question: str, ground_truth: str, generated_answer: str) -> int | None:
    """Ask the configured LLM to grade generated_answer against ground_truth, 1-5."""
    prompt = (
        "You are a strict, independent grading judge. Respond with JSON only, "
        "no explanation.\n\n"
        f"Question: {question}\n"
        f"Ground truth answer: {ground_truth}\n"
        f"Generated answer to grade: {generated_answer}\n\n"
        "Grade the generated answer against the ground truth on a 1-5 scale:\n"
        "5 = perfectly factual, mirrors all dates/times/rules\n"
        "3 = partially correct, captures the main concept but omits secondary "
        "nuances\n"
        "1 = outright hallucination or false negative\n\n"
        'Respond with exactly this JSON shape and nothing else: {"score": <integer 1-5>}'
    )
    try:
        completion = Settings.llm.complete(prompt)
        raw = completion.text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0)) if match else json.loads(raw)
        score = int(payload.get("score"))
        if 1 <= score <= 5:
            return score
    except Exception:
        pass
    return None


def _judge_refusal(question: str, generated_answer: str) -> bool | None:
    """
    For questions tagged is_answerable=false, check whether the model
    appropriately declined/hedged instead of fabricating a specific answer.

    Returns True if it hallucinated a confident-sounding specific answer,
    False if it appropriately acknowledged it doesn't know, None if the
    judge call itself failed.
    """
    prompt = (
        "You are grading whether an AI assistant correctly recognized it did "
        "not have enough information to answer, instead of making something "
        "up. Respond with JSON only, no explanation.\n\n"
        f"Question: {question}\n"
        f"Assistant's answer: {generated_answer}\n\n"
        "This question is NOT covered by the assistant's knowledge base "
        "(it needs real-time data, private per-student data, or information "
        "outside a general campus-services knowledge base). A correct "
        "response acknowledges uncertainty, says the information isn't "
        "available, or points the user to check an official source or "
        "system, WITHOUT inventing specific facts (dates, names, numbers, "
        "policies, statuses).\n"
        "Did the assistant hallucinate a specific, confident-sounding answer "
        "instead of appropriately declining?\n\n"
        'Respond with exactly this JSON shape and nothing else: '
        '{"hallucinated": true or false}'
    )
    try:
        completion = Settings.llm.complete(prompt)
        raw = completion.text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0)) if match else json.loads(raw)
        return bool(payload.get("hallucinated"))
    except Exception:
        return None


def _hit_rate(index: VectorStoreIndex, item: dict, k: int = RETRIEVAL_TOP_K) -> int | None:
    """
    1 if a retrieved node matches the question's tagged source file, 0 if
    not, None if the question has no tagged source (e.g. is_answerable=false
    items, which have nothing to score recall against).
    """
    # Extract the target filename from either the nested or flat format.
    meta_tag = item.get("expected_source_metadata") or {}
    target_file = item.get("source_file") or meta_tag.get("source_file")

    if not target_file:
        return None

    # Normalize the target filename (lowercase, strip any path prefix).
    target_name = Path(str(target_file)).name.lower()

    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(item["question"])

    # A hit is any top-K retrieved node whose file name matches the target.
    for node_with_score in nodes:
        meta = getattr(node_with_score.node, "metadata", {}) or {}
        candidate_file = meta.get("file_name") or meta.get("source_file") or meta.get("filename") or ""
        candidate_name = Path(str(candidate_file)).name.lower()

        if candidate_name and (target_name == candidate_name or target_name in candidate_name):
            return 1

    return 0

def run_calibration(index: VectorStoreIndex, eval_items: list[dict]) -> dict:
    """Score the LLM judge against the hand-scored subset before trusting it on everything else."""
    calibration_items = [it for it in eval_items if it.get("human_score") is not None]
    query_engine = index.as_query_engine(similarity_top_k=RETRIEVAL_TOP_K, response_mode="compact")

    rows = []
    for item in calibration_items:
        response = query_engine.query(item["question"])
        generated = _answer_text(response)
        judge_score = _judge_score(item["question"], item["ground_truth"], generated)
        human_score = item["human_score"]
        agree = judge_score is not None and abs(judge_score - human_score) <= 1
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "human_score": human_score,
                "judge_score": judge_score,
                "agree_within_1": agree,
            }
        )

    n = len(rows)
    agreement_rate = (sum(1 for r in rows if r["agree_within_1"]) / n) if n else 0.0
    return {
        "rows": rows,
        "n": n,
        "agreement_rate": agreement_rate,
        "passed": n > 0 and agreement_rate >= CALIBRATION_AGREEMENT_THRESHOLD,
    }


def run_full_evaluation(index: VectorStoreIndex, eval_items: list[dict], judge_trusted: bool) -> list[dict]:
    """
    Run every eval question through the app's own query engine.

    Questions tagged is_answerable=false (see eval_set.json) don't have a
    source to hit or a factual ground truth to grade against — they exist to
    check whether the assistant fabricates a confident answer instead of
    admitting it doesn't know. Those get a hallucination check instead of a
    hit-rate/correctness score.
    """
    query_engine = index.as_query_engine(similarity_top_k=RETRIEVAL_TOP_K, response_mode="compact")
    results = []
    for item in eval_items:
        response = query_engine.query(item["question"])
        generated = _answer_text(response)
        is_answerable = item.get("is_answerable", True)

        hit = _hit_rate(index, item) if is_answerable else None
        judge_score = None
        hallucinated = None
        if judge_trusted:
            if is_answerable:
                judge_score = _judge_score(item["question"], item["ground_truth"], generated)
            else:
                hallucinated = _judge_refusal(item["question"], generated)

        results.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "question": item["question"],
                "generated": generated,
                "is_answerable": is_answerable,
                "hit_rate": hit,
                "judge_score": judge_score,
                "hallucinated": hallucinated,
            }
        )
    return results


# ---------------------------------------------------------------------------
# UI
#
# Developer/debug details (model names, local paths, Ollama URL) are now
# tucked behind an explicit toggle instead of always being on screen, and
# the previously-shown absolute filesystem path is no longer displayed.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Controls")
    dev_mode = st.toggle("Show developer info", value=False)
    if dev_mode:
        st.caption("Configuration for this session (hidden from students by default).")
        st.write(f"**Vector store dir:** `{CHROMA_DIR}`")
        st.write(f"**Collection:** `{COLLECTION_NAME}`")
        st.write(f"**Embedding model:** `{LOCAL_EMBED_MODEL}`")
        st.write(f"**LLM model:** `{OLLAMA_MODEL}`")
        st.write(f"**Ollama URL:** `{OLLAMA_BASE_URL}`")
        st.write(f"**Eval set:** `{EVAL_SET_PATH}`")
    st.divider()
    st.caption("If the index is still loading, this sidebar stays available so the page never looks frozen.")


st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.spinner("Initializing Local Vector Database & Local Models... Please wait."):
    index = get_or_create_index()

st.success("Assistant ready for you! 🎉", icon="✅")

st.subheader("Ask anything about campus life! 🐻")
user_query = st.text_input(
    "Question",
    placeholder="Try asking about deadlines, library hours, or facilities.",
)

if st.button("Run Query", type="primary"):
    if user_query.strip():
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
        with st.spinner("Thinking, just a moment..."):
            response = query_engine.query(user_query)
        st.markdown("### Answer")
        st.write(_answer_text(response))

        with st.expander("Sources"):
            source_nodes = getattr(response, "source_nodes", []) or []
            if not source_nodes:
                st.write("No source nodes were returned.")
            for i, sn in enumerate(source_nodes[:5], start=1):
                node = getattr(sn, "node", None)
                metadata = getattr(node, "metadata", {}) if node is not None else {}
                text = getattr(node, "text", "") if node is not None else ""
                st.markdown(f"**Source {i}**")
                st.write(f"Score: {getattr(sn, 'score', None)}")
                st.write(f"Metadata: {metadata}")
                st.code(text[:1000])
    else:
        st.warning("Please enter a question first.")


st.divider()
st.subheader("Evaluation & judge calibration")
st.caption(
    "Before trusting the LLM judge on all questions, it's checked against the "
    "hand-scored subset in eval_set.json (human_score field)."
)

eval_items = _load_eval_set()

if not eval_items:
    st.info(f"No eval set found at `{EVAL_SET_PATH}`. Add one to enable evaluation.")
else:
    calibration_subset = [it for it in eval_items if it.get("human_score") is not None]
    st.write(
        f"Loaded {len(eval_items)} questions, "
        f"{len(calibration_subset)} of which are hand-scored for calibration."
    )

    if st.button("Run calibration + evaluation"):
        with st.spinner("Scoring calibration subset against human judgments..."):
            calibration = run_calibration(index, eval_items)

        if calibration["n"] == 0:
            st.warning(
                "No hand-scored questions found (fill in `human_score` for a subset "
                "of eval_set.json first). Skipping judge calibration."
            )
        elif calibration["passed"]:
            st.success(
                f"Judge agrees with human scores on {calibration['agreement_rate']:.0%} "
                f"of the {calibration['n']} calibration questions "
                f"(threshold: {CALIBRATION_AGREEMENT_THRESHOLD:.0%}). Judge scores below are trusted."
            )
        else:
            st.error(
                f"Judge only agrees with human scores on {calibration['agreement_rate']:.0%} "
                f"of the {calibration['n']} calibration questions "
                f"(threshold: {CALIBRATION_AGREEMENT_THRESHOLD:.0%}). "
                "Correctness scores below are hidden until the judge prompt is fixed. "
                "Hit Rate is unaffected since it doesn't depend on the judge."
            )

        with st.expander("Calibration detail"):
            for row in calibration["rows"]:
                st.write(
                    f"**{row['id']}** — human: {row['human_score']}, "
                    f"judge: {row['judge_score']}, "
                    f"{'✅ agree' if row['agree_within_1'] else '❌ disagree'}"
                )
                st.caption(row["question"])

        with st.spinner("Running full evaluation..."):
            results = run_full_evaluation(index, eval_items, judge_trusted=calibration["passed"])

        hit_scores = [r["hit_rate"] for r in results if r["hit_rate"] is not None]
        avg_hit_rate = sum(hit_scores) / len(hit_scores) if hit_scores else None

        hallucination_flags = [r["hallucinated"] for r in results if r["hallucinated"] is not None]
        hallucination_rate = (
            sum(1 for h in hallucination_flags if h) / len(hallucination_flags) if hallucination_flags else None
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                f"Hit Rate @ K={RETRIEVAL_TOP_K}",
                f"{avg_hit_rate:.0%}" if avg_hit_rate is not None else "n/a",
            )
        with col_b:
            if calibration["passed"]:
                judge_scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
                avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None
                st.metric("Avg. correctness (judge)", f"{avg_judge:.2f} / 5" if avg_judge else "n/a")
            else:
                st.metric("Avg. correctness (judge)", "hidden — judge not calibrated")
        with col_c:
            if calibration["passed"]:
                st.metric(
                    "Hallucination rate (unanswerable Qs)",
                    f"{hallucination_rate:.0%}" if hallucination_rate is not None else "n/a",
                    help="Share of is_answerable=false questions where the model fabricated a confident answer instead of declining.",
                )
            else:
                st.metric("Hallucination rate (unanswerable Qs)", "hidden — judge not calibrated")

        with st.expander("Per-question results"):
            for r in results:
                st.markdown(f"**{r['id']}** ({r['category']}) — {r['question']}")
                if r["is_answerable"]:
                    hit_label = "no source tagged" if r["hit_rate"] is None else ("hit ✅" if r["hit_rate"] else "miss ❌")
                    judge_label = "n/a" if r["judge_score"] is None else f"{r['judge_score']}/5"
                    st.write(f"Retrieval: {hit_label} · Judge score: {judge_label}")
                else:
                    if r["hallucinated"] is None:
                        halluc_label = "n/a"
                    elif r["hallucinated"]:
                        halluc_label = "hallucinated ❌"
                    else:
                        halluc_label = "appropriately declined ✅"
                    st.write(f"Unanswerable question · {halluc_label}")
                st.caption(r["generated"])
