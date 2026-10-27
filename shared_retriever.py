"""Shared policy retrievers for the health-insurance claim assistant.

The local Chroma index is intentionally git-ignored because it can be rebuilt
from data/policy_chunks.json. On hosts where that index has not been built,
retrieval safely falls back to BM25 instead of returning no clauses.
"""
import json
import os

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def _build(k, pool=10, rrf_const=60):
    """Build BM25 and, when available, dense/hybrid retrievers.

    Missing or empty Chroma storage is not fatal: hybrid/RRF searches then use
    BM25 results. Chunk IDs remain intact, so sibling IDs such as G.2#1/G.2#2
    are not collapsed.
    """
    pool = max(pool, k)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    chunks_file = os.path.join(current_dir, "data", "policy_chunks.json")

    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    bm25_docs = [
        Document(
            page_content=(
                f"{c.get('section_title', '')} {c.get('title', '')} {c.get('text', '')}"
            ),
            metadata={"id": str(c["id"]), "citation": str(c.get("citation", ""))},
        )
        for c in chunks
        if c.get("id") and str(c.get("text", "")).strip()
    ]
    if not bm25_docs:
        raise ValueError(f"No policy chunks with text were found in {chunks_file}")

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = pool

    chroma_retriever = None
    db_marker = os.path.join(db_path, "chroma.sqlite3")
    if os.path.isfile(db_marker):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
            )
            count = vector_store._collection.count()
            if count > 0:
                chroma_retriever = vector_store.as_retriever(search_kwargs={"k": pool})
                print(f"[retrieval] Chroma index loaded: {count} documents.")
            else:
                print("[retrieval] Chroma index exists but is empty; using BM25-only fallback.")
        except Exception as exc:
            print(
                f"[retrieval] Could not load Chroma index ({type(exc).__name__}: {str(exc)[:180]}). "
                "Using BM25-only fallback."
            )
    else:
        print("[retrieval] chroma_db index not found; using BM25-only fallback from policy_chunks.json.")

    def _unique(docs):
        out, seen = [], set()
        for doc in docs:
            doc_id = str(doc.metadata.get("id", "")).strip()
            key = doc_id if doc_id else doc.page_content
            if key not in seen:
                seen.add(key)
                out.append(doc)
        return out

    def _dense_hits(query):
        nonlocal chroma_retriever
        if chroma_retriever is None:
            return []
        try:
            return chroma_retriever.invoke(query)
        except Exception as exc:
            print(
                f"[retrieval] Chroma query failed ({type(exc).__name__}: {str(exc)[:180]}). "
                "Disabling dense search for this process and falling back to BM25."
            )
            chroma_retriever = None
            return []

    def dense_search(query):
        # A dense-only strategy cannot fabricate semantic results if the vector DB is absent.
        return _unique(_dense_hits(query))[:k]

    def bm25_search(query):
        return _unique(bm25_retriever.invoke(query))[:k]

    def hybrid_search(query):
        """Interleave sparse and dense hits; gracefully support BM25-only hosting."""
        sparse_hits = _unique(bm25_retriever.invoke(query))
        dense_hits = _unique(_dense_hits(query))
        if not dense_hits:
            return sparse_hits[:k]
        merged_docs, seen_ids = [], set()
        for idx in range(max(len(sparse_hits), len(dense_hits))):
            for hits in (sparse_hits, dense_hits):
                if idx >= len(hits):
                    continue
                doc = hits[idx]
                doc_id = str(doc.metadata.get("id", "")).strip()
                key = doc_id if doc_id else doc.page_content
                if key not in seen_ids:
                    seen_ids.add(key)
                    merged_docs.append(doc)
        return merged_docs[:k]

    def hybrid_rrf_search(query):
        """Fuse sparse/dense ranks; when dense storage is absent, return BM25 rank."""
        hits_lists = [bm25_retriever.invoke(query), _dense_hits(query)]
        scores, docs = {}, {}
        for hits in hits_lists:
            seen_in_list = set()
            rank = 0
            for doc in hits:
                doc_id = str(doc.metadata.get("id", "")).strip()
                key = doc_id if doc_id else doc.page_content
                if key in seen_in_list:
                    continue
                seen_in_list.add(key)
                rank += 1
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_const + rank)
                docs.setdefault(key, doc)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [docs[key] for key in best]

    return dense_search, bm25_search, hybrid_search, hybrid_rrf_search


def get_hybrid_retriever(k=4, strategy=None):
    """Return the selected live retriever."""
    selected = (strategy or os.getenv("RETRIEVAL_STRATEGY", "interleave")).strip().lower()
    dense, bm25, hybrid, hybrid_rrf = _build(k)
    retrievers = {
        "interleave": hybrid,
        "hybrid": hybrid,
        "rrf": hybrid_rrf,
        "hybrid_rrf": hybrid_rrf,
        "dense": dense,
        "bm25": bm25,
    }
    if selected not in retrievers:
        choices = ", ".join(("interleave", "rrf", "dense", "bm25"))
        raise ValueError(f"Unknown RETRIEVAL_STRATEGY={selected!r}. Choose one of: {choices}.")
    print(f"[retrieval] strategy={selected}")
    return retrievers[selected]


def get_retrievers(k=4):
    """Return each available retrieval strategy for offline evaluation."""
    dense, bm25, hybrid, hybrid_rrf = _build(k)
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid, "hybrid_rrf": hybrid_rrf}
