"""
Latency benchmark for Person 2's slice of the pipeline: query embedding +
FAISS retrieval + guardrail evaluation, per chunking strategy.

Explicitly NOT a single best-case timing run: queries are drawn from a mix
of (a) real corpus content (should retrieve confidently) and (b) off-topic
probes (should trigger the guardrail's should_answer=False path), across
all four languages, run in randomized order, with a warm-up pass excluded
from the reported numbers (JIT/cache warm-up on the first few calls is not
representative of steady-state latency).

Usage:
    python3 -m benchmark.run_benchmark
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_msmarco_xi import load_synthetic, SUPPORTED_LANGUAGES
from chunking.semantic import split_sentences
from embeddings.embedder import TfidfEmbedder
from pipeline import RetrievalPipeline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OFF_TOPIC_PROBES = [
    "What is the best recipe for chocolate chip cookies?",
    "How do I fix a flat tire on my bicycle?",
    "What's the weather forecast for tomorrow in Paris?",
    "Can you recommend a good sci-fi movie from the 90s?",
    "How does compound interest work on a savings account?",
    "What is the offside rule in football?",
    "quantum entanglement explained simply",
    "best budget laptops under $500 in 2026",
]


def build_query_set(documents, n_per_language: int = 60, seed: int = 7):
    """Real queries = sentences pulled straight out of corpus documents
    (so we know *something* grounded should exist for them). Off-topic
    queries = fixed probe set, replicated across the run so the guardrail
    gets meaningfully exercised too, not just the happy path."""
    rng = random.Random(seed)
    by_lang = {}
    for doc in documents:
        by_lang.setdefault(doc.language, []).append(doc)

    queries = []  # (text, language_hint, expected_answerable)
    for lang in SUPPORTED_LANGUAGES:
        docs = by_lang.get(lang, [])
        rng.shuffle(docs)
        picked = 0
        for doc in docs:
            sents = split_sentences(doc.text)
            if not sents:
                continue
            q = rng.choice(sents)
            queries.append((q, lang, True))
            picked += 1
            if picked >= n_per_language:
                break

    for probe in OFF_TOPIC_PROBES:
        queries.append((probe, None, False))

    rng.shuffle(queries)
    return queries


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def run():
    print("=== Loading dataset (offline synthetic, MSMARCO-XI-shaped) ===")
    documents = load_synthetic(docs_per_language=150)
    print(f"Loaded {len(documents)} documents across languages: {SUPPORTED_LANGUAGES}")

    print("\n=== Building chunking strategies + FAISS indexes ===")
    pipeline = RetrievalPipeline(embedder=TfidfEmbedder(max_features=20000))
    build_reports = pipeline.build_all(documents)

    print("\n=== Preparing query set ===")
    queries = build_query_set(documents, n_per_language=60)
    print(f"Total queries: {len(queries)} "
          f"({sum(1 for *_, ok in queries if ok)} grounded / "
          f"{sum(1 for *_, ok in queries if not ok)} off-topic)")

    strategies = list(pipeline.indexes.keys())
    all_rows = []
    per_strategy_latencies = {s: [] for s in strategies}

    WARMUP = 5
    print("\n=== Running warm-up queries (excluded from reported stats) ===")
    for q, _, _ in queries[:WARMUP]:
        for strategy in strategies:
            pipeline.query(q, strategy=strategy, k=5)

    print("\n=== Running timed benchmark ===")
    t_start = time.perf_counter()
    for q, lang_hint, expected_answerable in queries:
        for strategy in strategies:
            outcome = pipeline.query(q, strategy=strategy, k=5)
            per_strategy_latencies[strategy].append(outcome.total_ms)
            all_rows.append({
                "query": q,
                "strategy": strategy,
                "expected_answerable": expected_answerable,
                "guardrail_should_answer": outcome.verdict.should_answer,
                "guardrail_reason": outcome.verdict.reason,
                "top_score": outcome.verdict.top_score,
                "embed_ms": round(outcome.embed_ms, 4),
                "search_ms": round(outcome.search_ms, 4),
                "total_ms": round(outcome.total_ms, 4),
            })
    t_end = time.perf_counter()
    wall_s = t_end - t_start

    # -- Latency percentiles per strategy -----------------------------------
    print("\n=== Latency report (per chunking strategy) ===")
    summary = {}
    for strategy in strategies:
        vals = sorted(per_strategy_latencies[strategy])
        p50, p70, p100 = percentile(vals, 50), percentile(vals, 70), percentile(vals, 100)
        mean = sum(vals) / len(vals)
        under_200 = sum(1 for v in vals if v < 200.0) / len(vals) * 100.0
        summary[strategy] = {
            "n_queries": len(vals),
            "p50_ms": round(p50, 4), "p70_ms": round(p70, 4),
            "p100_ms_max": round(p100, 4), "mean_ms": round(mean, 4),
            "pct_under_200ms": round(under_200, 2),
        }
        print(f"\n[{strategy}]  n={len(vals)}")
        print(f"  P50  : {p50:.3f} ms")
        print(f"  P70  : {p70:.3f} ms")
        print(f"  P100 : {p100:.3f} ms  (max observed)")
        print(f"  mean : {mean:.3f} ms")
        print(f"  % under 200ms budget : {under_200:.1f}%")

    # -- Guardrail sanity: did off-topic probes actually get rejected? ------
    off_topic_rows = [r for r in all_rows if not r["expected_answerable"]]
    correctly_rejected = sum(1 for r in off_topic_rows if not r["guardrail_should_answer"])
    grounded_rows = [r for r in all_rows if r["expected_answerable"]]
    correctly_answered = sum(1 for r in grounded_rows if r["guardrail_should_answer"])

    print("\n=== Guardrail sanity check ===")
    print(f"Off-topic probes correctly rejected: {correctly_rejected}/{len(off_topic_rows)} "
          f"({100.0 * correctly_rejected / max(1, len(off_topic_rows)):.1f}%)")
    print(f"Grounded queries answered:            {correctly_answered}/{len(grounded_rows)} "
          f"({100.0 * correctly_answered / max(1, len(grounded_rows)):.1f}%)")

    # -- Persist raw + summary results ---------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "latency_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = os.path.join(RESULTS_DIR, "latency_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "build_reports": [asdict(r) for r in build_reports],
            "latency_summary": summary,
            "guardrail": {
                "off_topic_total": len(off_topic_rows),
                "off_topic_correctly_rejected": correctly_rejected,
                "grounded_total": len(grounded_rows),
                "grounded_correctly_answered": correctly_answered,
            },
            "total_queries_run": len(all_rows),
            "wall_clock_seconds": round(wall_s, 3),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved raw results -> {csv_path}")
    print(f"Saved summary      -> {summary_path}")
    print(f"\nTotal wall-clock time for {len(all_rows)} timed retrievals: {wall_s:.2f}s")

    return summary, build_reports


if __name__ == "__main__":
    run()
