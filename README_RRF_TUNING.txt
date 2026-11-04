RRF TUNING PATCH — TWO FILES ONLY
================================

Replace these two files in your project root:
  1. shared_retriever.py
  2. compare_retrieval_strategies.py

Do NOT replace or edit toulmin_agent.py, app.py, policy_chunks.json,
ChromaDB, claim labels, Gemini prompts, or benchmark_e2e.py for this first test.

WHAT CHANGED
------------
- RRF constant is configurable per search call.
- Candidate pool size is configurable when constructing retrievers.
- Application use keeps its existing graceful BM25 fallback by default.
- Offline tuning runs with require_dense=True, so a missing/broken Chroma
  index stops the benchmark instead of falsely reporting BM25-only results
  as RRF.
- RRF ties are sorted deterministically.
- The tuning script compares a fixed Dense baseline against a small RRF grid:
    candidate_pool = 10, 20, 40
    rrf_const      = 10, 30, 60, 100
- The tuning script makes no Gemini/API calls.
- Results are saved to rrf_tuning_results.json with dataset and policy hashes.

BEFORE RUNNING
--------------
1. Make a backup copy of the two original Python files.
2. Replace only those two files.
3. Make sure chroma_db exists and matches data/policy_chunks.json.
   If the strict check reports a count mismatch or missing index, rebuild it:
       python ingest.py
   Rebuilding Chroma may take time and requires the embedding dependencies.
4. Ensure the corrected DEVELOPMENT claims file exists:
       data/claims_gold_corrected.json

RUN
---
From the project root:
    python compare_retrieval_strategies.py

Optional explicit command:
    python compare_retrieval_strategies.py --claims data/claims_gold_corrected.json --pools 10,20,40 --rrf-constants 10,30,60,100 --top-k 4

This is an offline retrieval experiment. It does not call Gemini.

HOW TO READ THE RESULT
----------------------
Primary metric: all_gold_pct (percentage of claims for which every gold
policy clause is retrieved across the two fixed queries).
Tie-breakers: mean_gold_recall_pct, then avg_retrieved_clauses.

Do not assume RRF won just because it is RRF. If Dense baseline is equal or
better, keep Dense. If RRF wins only marginally on 25 claims, treat the result
as a development-set signal, not proof of generalisation.

IMPORTANT SECURITY NOTE
-----------------------
The previous project ZIP included a populated .env file. Revoke the exposed
Google API key and do not include .env in future ZIPs or commits. This patch
contains no credentials.
