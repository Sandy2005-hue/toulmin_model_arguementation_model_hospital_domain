"""Shared policy retrievers for the health-insurance claim assistant.

The application retains its normal BM25 fallback. End-to-end benchmarks can set
REQUIRE_DENSE_RETRIEVAL=1 to stop if Chroma/dense retrieval is unavailable.
"""
from __future__ import annotations

import json
import os
from typing import Callable

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CANDIDATE_POOL = 10
DEFAULT_RRF_CONST = 60


def _positive_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1; got {parsed}.")
    return parsed


def _build(
    k: int,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    rrf_const: int = DEFAULT_RRF_CONST,
    require_dense: bool = False,
):
    k = _positive_int(k, "k")
    candidate_pool = max(_positive_int(candidate_pool, "candidate_pool"), k)
    rrf_const = _positive_int(rrf_const, "rrf_const")

    root = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root, "chroma_db")
    chunks_path = os.path.join(root, "data", "policy_chunks.json")
    with open(chunks_path, encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", [])
    bm25_docs = [
        Document(
            page_content=f"{c.get('section_title', '')} {c.get('title', '')} {c.get('text', '')}",
            metadata={"id": str(c["id"]), "citation": str(c.get("citation", ""))},
        )
        for c in chunks
        if c.get("id") and str(c.get("text", "")).strip()
    ]
    if not bm25_docs:
        raise ValueError(f"No policy chunks with text were found in {chunks_path}")

    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = candidate_pool

    chroma = None
    dense_problem = None
    marker = os.path.join(db_path, "chroma.sqlite3")
    if os.path.isfile(marker):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            store = Chroma(persist_directory=db_path, embedding_function=embeddings)
            count = store._collection.count()
            if count > 0:
                if require_dense and count != len(bm25_docs):
                    raise RuntimeError(
                        "Chroma index/policy chunk count mismatch: "
                        f"index has {count}, policy_chunks.json has {len(bm25_docs)}. "
                        "Rebuild with `python ingest.py` before benchmarking."
                    )
                chroma = store.as_retriever(search_kwargs={"k": candidate_pool})
                print(f"[retrieval] Chroma index loaded: {count} docs; candidate_pool={candidate_pool}.")
            else:
                dense_problem = "Chroma index exists but is empty."
        except Exception as exc:
            dense_problem = f"Could not load Chroma index ({type(exc).__name__}: {exc})."
    else:
        dense_problem = f"Chroma index not found at {db_path}."

    if chroma is None:
        if require_dense:
            raise RuntimeError(
                "Dense retrieval is required for this benchmark, but unavailable. "
                f"{dense_problem} Check/rebuild the index with `python ingest.py`."
            )
        print(f"[retrieval] {dense_problem} Using BM25-only fallback for application use.")

    def unique(docs):
        result, seen = [], set()
        for doc in docs:
            doc_id = str(doc.metadata.get("id", "")).strip()
            key = doc_id or doc.page_content
            if key not in seen:
                seen.add(key)
                result.append(doc)
        return result

    def dense_hits(query: str):
        nonlocal chroma
        if chroma is None:
            if require_dense:
                raise RuntimeError("Dense retrieval unavailable; refusing benchmark fallback.")
            return []
        try:
            hits = unique(chroma.invoke(query))
            if require_dense and not hits:
                raise RuntimeError("Dense retrieval returned zero documents for a non-empty query.")
            if require_dense and any(not d.metadata.get("id") for d in hits):
                raise RuntimeError("Dense results contain documents without policy chunk IDs.")
            return hits
        except Exception as exc:
            if require_dense:
                raise RuntimeError(
                    "Dense query failed while dense retrieval was required; "
                    "benchmark stopped instead of using BM25 fallback."
                ) from exc
            print(f"[retrieval] Dense query failed ({type(exc).__name__}); disabling dense fallback.")
            chroma = None
            return []

    def dense_search(query: str):
        return dense_hits(query)[:k]

    def bm25_search(query: str):
        return unique(bm25.invoke(query))[:k]

    def interleave_search(query: str):
        sparse = unique(bm25.invoke(query))
        semantic = unique(dense_hits(query))
        if not semantic:
            if require_dense:
                raise RuntimeError("Dense results empty; refusing to score hybrid as BM25-only.")
            return sparse[:k]
        merged, seen = [], set()
        for index in range(max(len(sparse), len(semantic))):
            for ranked in (sparse, semantic):
                if index < len(ranked):
                    doc = ranked[index]
                    key = str(doc.metadata.get("id", "")).strip() or doc.page_content
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)
        return merged[:k]

    def rrf_search(query: str, *, rrf_const_override: int | None = None):
        constant = rrf_const if rrf_const_override is None else _positive_int(rrf_const_override, "rrf_const_override")
        sparse = unique(bm25.invoke(query))
        semantic = unique(dense_hits(query))
        if not semantic:
            if require_dense:
                raise RuntimeError("Dense results empty; refusing to score RRF as BM25-only.")
            return sparse[:k]
        scores, docs_by_id = {}, {}
        for ranked_hits in (sparse, semantic):
            for rank, doc in enumerate(ranked_hits, start=1):
                key = str(doc.metadata.get("id", "")).strip() or doc.page_content
                scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
                docs_by_id.setdefault(key, doc)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        return [docs_by_id[key] for key in ordered[:k]]

    return dense_search, bm25_search, interleave_search, rrf_search


def get_hybrid_retriever(
    k: int = 4,
    strategy: str | None = None,
    candidate_pool: int | None = None,
    rrf_const: int | None = None,
    require_dense: bool = False,
):
    selected = (strategy or os.getenv("RETRIEVAL_STRATEGY", "interleave")).strip().lower()
    if candidate_pool is None:
        candidate_pool = os.getenv("RETRIEVAL_CANDIDATE_POOL", str(DEFAULT_CANDIDATE_POOL))
    if rrf_const is None:
        rrf_const = os.getenv("RRF_CONST", str(DEFAULT_RRF_CONST))

    # Benchmark script sets this before toulmin_agent imports the retriever.
    require_dense = require_dense or os.getenv("REQUIRE_DENSE_RETRIEVAL", "0").strip() == "1"

    dense, bm25, interleave, rrf = _build(k, candidate_pool, rrf_const, require_dense)
    options: dict[str, Callable] = {
        "interleave": interleave,
        "hybrid": interleave,
        "rrf": rrf,
        "hybrid_rrf": rrf,
        "dense": dense,
        "bm25": bm25,
    }
    if selected not in options:
        raise ValueError(f"Unknown RETRIEVAL_STRATEGY={selected!r}. Choose dense, rrf, interleave, or bm25.")
    print(
        f"[retrieval] strategy={selected} | candidate_pool={candidate_pool} "
        f"| rrf_const={rrf_const} | require_dense={require_dense}"
    )
    return options[selected]


def get_retrievers(
    k: int = 4,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    rrf_const: int = DEFAULT_RRF_CONST,
    require_dense: bool = False,
):
    dense, bm25, interleave, rrf = _build(k, candidate_pool, rrf_const, require_dense)
    return {"dense": dense, "bm25": bm25, "hybrid": interleave, "hybrid_rrf": rrf}
