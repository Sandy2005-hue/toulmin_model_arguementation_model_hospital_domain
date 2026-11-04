"""Controlled RRF parameter tuning on claim-level gold clauses (no LLM calls).

This experiment keeps the query construction, documents, top-k, and dense
baseline fixed. It varies ONLY:
  1. RRF rank constant (rrf_const)
  2. Candidate pool size (number of candidates from BM25 and dense before RRF)

Default dataset is the corrected DEVELOPMENT set. Do not use the holdout set to
choose parameters. Once a configuration is selected, evaluate it on a genuinely
untouched test set.

Usage:
    python compare_retrieval_strategies.py
    python compare_retrieval_strategies.py --claims data/claims_gold_corrected.json
    python compare_retrieval_strategies.py --pools 10,20,40 --rrf-constants 10,30,60,100

No Gemini API calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from shared_retriever import get_retrievers


def base_id(value: str) -> str:
    """Treat child chunks such as G.2#1 as belonging to parent clause G.2."""
    return str(value).split("#", 1)[0].strip()


def document_ids(docs) -> list[str]:
    """Return unique parent clause IDs while preserving retrieval order."""
    output = []
    seen = set()
    for doc in docs:
        raw_id = doc.metadata.get("id")
        if raw_id:
            clause_id = base_id(raw_id)
            if clause_id and clause_id not in seen:
                seen.add(clause_id)
                output.append(clause_id)
    return output


def ordered_union(*items: list[str]) -> list[str]:
    output = []
    seen = set()
    for values in items:
        for value in values:
            if value not in seen:
                seen.add(value)
                output.append(value)
    return output


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int_list(raw: str, arg_name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{arg_name} must be comma-separated integers.") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{arg_name} values must all be positive integers.")
    # Stable de-duplication.
    return list(dict.fromkeys(values))


def evaluate_claims(claims, search_fn, top_k: int):
    """Evaluate identical two-query claim retrieval for one configuration."""
    rows = []
    complete = 0
    any_gold = 0
    recall_sum = 0.0
    returned_count_sum = 0

    for claim in claims:
        text = str(claim["text"]).strip()
        gold = {base_id(value) for value in claim.get("gold_clause_ids", []) if value}
        if not gold:
            raise ValueError(f"Claim {claim.get('id', '?')} has no gold_clause_ids.")

        # This query plan is deliberately fixed for every configuration.
        primary = document_ids(search_fn(text))
        exclusion = document_ids(search_fn(f"{text} exclusion waiting period"))
        retrieved = ordered_union(primary, exclusion)
        retrieved_set = set(retrieved)
        found = gold & retrieved_set
        missing = sorted(gold - retrieved_set)

        is_complete = gold.issubset(retrieved_set)
        has_any = bool(found)
        complete += int(is_complete)
        any_gold += int(has_any)
        recall_sum += len(found) / len(gold)
        returned_count_sum += len(retrieved)

        rows.append({
            "id": claim["id"],
            "gold_clause_ids": sorted(gold),
            "retrieved_clause_ids": retrieved,
            "missing_clause_ids": missing,
            "complete_gold_retrieved": is_complete,
            "any_gold_retrieved": has_any,
            "gold_recall": len(found) / len(gold),
            "retrieved_clause_count": len(retrieved),
        })

    count = len(claims)
    return {
        "claims": count,
        "all_gold_count": complete,
        "all_gold_pct": 100.0 * complete / count if count else 0.0,
        "any_gold_count": any_gold,
        "any_gold_pct": 100.0 * any_gold / count if count else 0.0,
        "mean_gold_recall_pct": 100.0 * recall_sum / count if count else 0.0,
        "avg_retrieved_clauses": returned_count_sum / count if count else 0.0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune only RRF candidate pool and rank constant.")
    parser.add_argument(
        "--claims",
        default="data/claims_gold_corrected.json",
        help="Development claims JSON path, relative to this project directory.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Final docs returned per query (default: 4).")
    parser.add_argument("--pools", default="10,20,40", help="Candidate-pool grid (default: 10,20,40).")
    parser.add_argument(
        "--rrf-constants",
        default="10,30,60,100",
        help="RRF constant grid (default: 10,30,60,100).",
    )
    parser.add_argument(
        "--out",
        default="rrf_tuning_results.json",
        help="Output JSON filename (default: rrf_tuning_results.json).",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    try:
        pools = parse_int_list(args.pools, "--pools")
        rrf_constants = parse_int_list(args.rrf_constants, "--rrf-constants")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    # _build() never allows the candidate pool to be smaller than top-k.
    # Normalize labels here so the saved metadata reports the effective pool.
    pools = list(dict.fromkeys(max(value, args.top_k) for value in pools))

    here = os.path.dirname(os.path.abspath(__file__))
    claims_path = args.claims if os.path.isabs(args.claims) else os.path.join(here, args.claims)
    output_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    if not os.path.isfile(claims_path):
        parser.error(f"Claims file not found: {claims_path}")

    with open(claims_path, "r", encoding="utf-8") as file:
        claims = json.load(file)
    if not isinstance(claims, list) or not claims:
        parser.error("Claims file must contain a non-empty JSON list.")

    print(f"\nDevelopment claims: {len(claims)} | file: {os.path.basename(claims_path)}")
    print(f"Top-k per query: {args.top_k} | query plan: claim text + exclusion/waiting-period query")
    print("Loading dense retrieval in STRICT mode. If Chroma is unavailable, this run will stop.\n")

    # Dense results are the fixed baseline. Top-k dense output is unchanged by
    # the larger candidate pools, so build it from the smallest requested pool.
    baseline_pool = min(pools)
    baseline_retrievers = get_retrievers(
        k=args.top_k,
        candidate_pool=baseline_pool,
        rrf_const=60,
        require_dense=True,
    )
    dense_result = evaluate_claims(claims, baseline_retrievers["dense"], args.top_k)
    print(
        f"Dense baseline: all gold {dense_result['all_gold_count']}/{dense_result['claims']} "
        f"({dense_result['all_gold_pct']:.1f}%), "
        f"mean recall {dense_result['mean_gold_recall_pct']:.1f}%, "
        f"avg clauses {dense_result['avg_retrieved_clauses']:.2f}"
    )

    rrf_results = []
    for pool in pools:
        # One retriever build per pool; override only the RRF constant per run.
        retrievers = (
            baseline_retrievers
            if pool == baseline_pool
            else get_retrievers(
                k=args.top_k,
                candidate_pool=pool,
                rrf_const=60,
                require_dense=True,
            )
        )
        rrf_search = retrievers["hybrid_rrf"]

        for rrf_constant in rrf_constants:
            print(f"Evaluating RRF: candidate_pool={pool}, rrf_const={rrf_constant} ...")

            def configured_rrf(query, _search=rrf_search, _constant=rrf_constant):
                return _search(query, rrf_const_override=_constant)

            metrics = evaluate_claims(claims, configured_rrf, args.top_k)
            rrf_results.append({
                "strategy": "RRF",
                "candidate_pool": pool,
                "rrf_const": rrf_constant,
                **metrics,
            })

    # Primary selection criterion: all-gold-clause retrieval. Tie-break with
    # macro gold recall, then fewer returned clauses (less context to the LLM).
    best = max(
        rrf_results,
        key=lambda result: (
            result["all_gold_pct"],
            result["mean_gold_recall_pct"],
            -result["avg_retrieved_clauses"],
        ),
    )

    print("\nRRF parameter results")
    print(
        f"{'Pool':>6} {'RRF const':>10} {'All gold':>12} {'Any gold':>12} "
        f"{'Mean recall':>13} {'Avg clauses':>13}"
    )
    for result in sorted(
        rrf_results,
        key=lambda item: (item["candidate_pool"], item["rrf_const"]),
    ):
        print(
            f"{result['candidate_pool']:>6} {result['rrf_const']:>10} "
            f"{result['all_gold_count']:>4}/{result['claims']:<7} "
            f"{result['any_gold_count']:>4}/{result['claims']:<7} "
            f"{result['mean_gold_recall_pct']:>11.1f}% "
            f"{result['avg_retrieved_clauses']:>13.2f}"
        )

    print(
        "\nBest development configuration: "
        f"RRF candidate_pool={best['candidate_pool']}, rrf_const={best['rrf_const']}"
    )
    print(
        "Compare it against the dense baseline above. This is a development-set choice, "
        "not a final generalisation result. Do not repeatedly tune on the holdout set."
    )

    output = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claims_file": os.path.relpath(claims_path, here),
            "claims_sha256": sha256_file(claims_path),
            "policy_chunks_sha256": sha256_file(os.path.join(here, "data", "policy_chunks.json")),
            "top_k_per_query": args.top_k,
            "query_plan": ["claim_text", "claim_text + exclusion waiting period"],
            "candidate_pool_grid": pools,
            "rrf_constant_grid": rrf_constants,
            "embedding_model": "all-MiniLM-L6-v2",
            "dense_required": True,
            "selection_rule": "max all_gold_pct, then mean_gold_recall_pct, then fewer avg_retrieved_clauses",
            "note": "Offline retrieval-only tuning; no LLM calls. Dense is a fixed baseline.",
        },
        "dense_baseline": dense_result,
        "rrf_results": rrf_results,
        "best_development_configuration": {
            "candidate_pool": best["candidate_pool"],
            "rrf_const": best["rrf_const"],
            "all_gold_pct": best["all_gold_pct"],
            "mean_gold_recall_pct": best["mean_gold_recall_pct"],
            "avg_retrieved_clauses": best["avg_retrieved_clauses"],
        },
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(f"\nDetailed per-claim results saved to: {output_path}")


if __name__ == "__main__":
    main()
