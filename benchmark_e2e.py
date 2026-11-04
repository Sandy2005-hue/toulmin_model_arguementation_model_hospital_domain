"""Controlled end-to-end benchmark for Dense vs RRF retrieval.

API/model request failures returned by toulmin_agent as a fail-closed
'Needs Human Review' object are treated as NOT SCORED, not as model decisions.
Such failures are retried and can be recovered with --resume.

Examples:
  python benchmark_e2e.py --strategy dense --no-rewriter --no-fallback --claims data/claims_gold_corrected.json --out e2e_dense_controlled.json
  python benchmark_e2e.py --strategy rrf --candidate-pool 10 --rrf-const 10 --no-rewriter --no-fallback --claims data/claims_gold_corrected.json --out e2e_rrf_controlled.json

After a stopped/partial run, repeat the identical command with --resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LABELS = ["Claim Approved", "Claim Denied", "Needs Human Review"]
MAX_RETRIES = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled Dense vs RRF end-to-end benchmark.")
    parser.add_argument("--strategy", choices=("dense", "rrf"), required=True)
    parser.add_argument("--candidate-pool", type=int, default=10)
    parser.add_argument("--rrf-const", type=int, default=10)
    parser.add_argument("--claims", default="data/claims_gold_corrected.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-rewriter", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gap", type=int, default=int(os.getenv("E2E_GAP", "8")))
    parser.add_argument("--wait", type=int, default=int(os.getenv("E2E_WAIT", "60")))
    args = parser.parse_args()
    if args.candidate_pool < 1 or args.rrf_const < 1 or args.gap < 0 or args.wait < 0:
        parser.error("candidate-pool and rrf-const must be positive; gap and wait cannot be negative.")
    if args.strategy == "dense":
        args.rrf_const = 10  # normalized; irrelevant to Dense
    return args


def norm(value):
    value = str(value).strip()
    return value if value in LABELS else "OTHER"


def base_ids(ids):
    return {str(value).split("#")[0] for value in (ids or [])}


def is_retryable_error(error) -> bool:
    message = str(error).lower()
    return any(token in message for token in (
        "429", "503", "504", "unavailable", "deadline_exceeded",
        "deadline exceeded", "exhausted", "quota", "temporarily",
    ))


def is_returned_model_failure(result) -> bool:
    """Detect toulmin_agent's fail-closed response to an API/model failure."""
    if not isinstance(result, dict):
        return False
    status = str(result.get("Guardrail_Status", "")).lower()
    reason = str(result.get("Decision_Override_Reason", "")).lower()
    return (
        result.get("Model_Used") is None
        and ("model request failed" in status or "model request failed" in reason)
    )


def is_old_technical_failure_row(row) -> bool:
    """Drop API-failure rows created by the earlier benchmark patch on resume."""
    status = str(row.get("citation_check", "")).lower()
    return "model request failed" in status


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    claims_path = Path(args.claims)
    if not claims_path.is_absolute():
        claims_path = root / claims_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    if not claims_path.is_file():
        raise SystemExit(f"Claims file not found: {claims_path}")

    # Set environment before importing toulmin_agent; it builds its retriever at import time.
    os.environ["RETRIEVAL_STRATEGY"] = args.strategy
    os.environ["RETRIEVAL_CANDIDATE_POOL"] = str(args.candidate_pool)
    os.environ["RRF_CONST"] = str(args.rrf_const)
    os.environ["REQUIRE_DENSE_RETRIEVAL"] = "1"
    if args.no_rewriter:
        os.environ["USE_REWRITER"] = "0"
    if args.no_fallback:
        os.environ["MODEL_FALLBACK"] = "0"

    import toulmin_agent  # noqa: E402

    warnings.filterwarnings("ignore")
    with claims_path.open(encoding="utf-8") as stream:
        claims = json.load(stream)
    if not isinstance(claims, list) or not claims:
        raise SystemExit("Claims file must contain a non-empty JSON list.")

    policy_path = root / "data" / "policy_chunks.json"
    agent_path = root / "toulmin_agent.py"
    retriever_path = root / "shared_retriever.py"
    manifest = {
        "strategy": args.strategy,
        "candidate_pool": args.candidate_pool,
        "rrf_const": args.rrf_const,
        "top_k": 4,
        "query_rewriting": bool(toulmin_agent.USE_REWRITER),
        "model_fallback": bool(toulmin_agent.USE_FALLBACK),
        "model_chain": list(toulmin_agent.MODEL_CHAIN),
        "claims_file": str(claims_path.relative_to(root)) if claims_path.is_relative_to(root) else str(claims_path),
        "claims_sha256": sha256_file(claims_path),
        "policy_chunks_sha256": sha256_file(policy_path),
        "toulmin_agent_sha256": sha256_file(agent_path),
        "shared_retriever_sha256": sha256_file(retriever_path),
        "require_dense": True,
    }

    rows = []
    failures = []
    if args.resume and out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("metadata") != manifest:
            raise SystemExit("Cannot resume: saved run metadata differs from this configuration/dataset/code.")
        old_rows = previous.get("rows", [])
        rows = [row for row in old_rows if not is_old_technical_failure_row(row)]
        removed = len(old_rows) - len(rows)
        failures = previous.get("failures", [])
        print(f"Resuming {len(rows)} valid scored claims from {out_path.name}.")
        if removed:
            print(f"Removed {removed} old API-failure row(s); those claims will be retried.")
    elif out_path.exists():
        raise SystemExit(f"Output already exists: {out_path}. Choose another --out or use --resume.")

    def save():
        out_path.write_text(json.dumps({
            "metadata": manifest,
            "rows": rows,
            "failures": failures,
        }, indent=2), encoding="utf-8")

    done = {row["id"] for row in rows}
    save()
    print("\nCONTROLLED END-TO-END RUN")
    print(json.dumps(manifest, indent=2))
    print(f"Claims: {len(claims)} | output: {out_path.name}\n")

    consecutive_failures = 0
    for item in claims:
        claim_id = item["id"]
        if claim_id in done:
            continue

        result = None
        error_text = None
        attempts_used = 0
        for attempt in range(1, MAX_RETRIES + 1):
            attempts_used = attempt
            try:
                candidate = toulmin_agent.run_toulmin_pipeline(item["text"])
                if is_returned_model_failure(candidate):
                    # The agent catches the provider exception and returns a human-review
                    # object. This is fail-closed for the app, but it is not a completed
                    # benchmark prediction and must not enter the accuracy denominator.
                    error_text = candidate.get("Decision_Override_Reason") or candidate.get("Guardrail_Status")
                    if attempt < MAX_RETRIES:
                        print(f"{claim_id}: model/API request failed inside agent; waiting {args.wait}s "
                              f"(retry {attempt}/{MAX_RETRIES - 1}).")
                        time.sleep(args.wait)
                        continue
                    break
                result = candidate
                error_text = None
                break
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if is_retryable_error(exc) and attempt < MAX_RETRIES:
                    print(f"{claim_id}: transient API error; waiting {args.wait}s "
                          f"(retry {attempt}/{MAX_RETRIES - 1}).")
                    time.sleep(args.wait)
                    continue
                break

        if result is None:
            consecutive_failures += 1
            failures = [entry for entry in failures if entry.get("id") != claim_id]
            failures.append({
                "id": claim_id,
                "attempts": attempts_used,
                "reason": str(error_text or "Model/API request did not complete"),
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
            save()
            print(f"{claim_id:<8} | NOT SCORED — model/API failure after {attempts_used} attempt(s): "
                  f"{str(error_text)[:140]}")
            if consecutive_failures >= 3:
                print("Stopping after three consecutive model/API failures. Retry the identical command "
                      "with --resume after the provider recovers.")
                break
        else:
            consecutive_failures = 0
            expected = item["expected_verdict"]
            predicted = norm(result.get("Claim", ""))
            retrieved_ids = result.get("Retrieved_IDs") or []
            gold_found = bool(base_ids(item.get("gold_clause_ids")) & base_ids(retrieved_ids))
            rows.append({
                "id": claim_id,
                "expected": expected,
                "predicted": result.get("Claim"),
                "citation_check": result.get("Guardrail_Status"),
                "gold_clause_retrieved": gold_found,
                "model": result.get("Model_Used") or "?",
                "rewriter_model": result.get("Rewriter_Model"),
                "retrieved_ids": retrieved_ids,
                "gold_clause_ids": item.get("gold_clause_ids"),
            })
            failures = [entry for entry in failures if entry.get("id") != claim_id]
            done.add(claim_id)
            print(
                f"{claim_id:<8} | {expected:<20} | "
                f"{'OK' if expected == predicted else 'XX'} {predicted:<20} | "
                f"cite={'Y' if 'Verified' in str(result.get('Guardrail_Status', '')) else 'N'} | "
                f"gold={'Y' if gold_found else 'N'} | {result.get('Model_Used') or '?'}"
            )
            save()
        time.sleep(args.gap)

    scored = {row["id"] for row in rows}
    unscored = [claim["id"] for claim in claims if claim["id"] not in scored]
    n = len(rows)
    confusion = Counter((row["expected"], norm(row["predicted"])) for row in rows)
    correct = sum(row["expected"] == norm(row["predicted"]) for row in rows)
    cited = sum("Verified" in str(row["citation_check"]) for row in rows)
    gold_context = sum(bool(row["gold_clause_retrieved"]) for row in rows)
    models = Counter(row["model"] for row in rows)
    denied = sum(confusion[("Claim Denied", label)] for label in LABELS + ["OTHER"])
    reviews = sum(confusion[(expected, "Needs Human Review")] for expected in LABELS)

    print("\nConfusion matrix (rows=expected; columns=predicted)")
    columns = LABELS + ["OTHER"]
    print(f"{'':<22}" + "".join(f"{label[:13]:>16}" for label in columns))
    for expected in LABELS:
        print(f"{expected:<22}" + "".join(f"{confusion[(expected, label)]:>16}" for label in columns))
    print(f"\nClaims: {len(claims)} | valid scored: {n} | unscored: {len(unscored)} {unscored or ''}")
    print(f"Accuracy on valid scored claims: {correct}/{n} = {100 * correct / max(n, 1):.1f}%")
    false_approvals = confusion[("Claim Denied", "Claim Approved")]
    print(f"False approvals: {false_approvals}/{denied}")
    print(f"Human review (valid pipeline outputs): {reviews}/{n}")
    print(f"Citation check passed: {cited}/{n}")
    print(f"Gold clause present in final retrieved IDs: {gold_context}/{n}")
    print(f"Models used: {dict(models)}")
    print(f"Outstanding technical failures: {len(failures)}")
    print(f"Saved: {out_path}")
    if unscored:
        print("Unscored claims remain. Retry with the exact same command plus --resume.")
    if len(models) > 1:
        print("WARNING: multiple models appear in results; do not compare this run as single-model evidence.")


if __name__ == "__main__":
    main()
