import asyncio
import httpx
import time
import numpy as np
from tabulate import tabulate

BASE_URL = "http://localhost:8000"

# Synthetic test queries simulating MSMARCO-XI Indic subsets + Person 2 outputs
BENCHMARK_DATASET = [
    {
        "query": "भारत की राजधानी क्या है?",
        "query_language": "hi",
        "stt_latency_ms": 65.4,
        "retrieval_latency_ms": 28.2,
        "retrieved_chunks": [
            {"chunk_id": "c1", "text": "नई दिल्ली भारत की आधिकारिक राजधानी है।", "score": 0.89, "language": "hi"}
        ]
    },
    {
        "query": "தமிழ்நாட்டின் தலைநகரம் எது?",
        "query_language": "ta",
        "stt_latency_ms": 72.1,
        "retrieval_latency_ms": 31.0,
        "retrieved_chunks": [
            {"chunk_id": "c2", "text": "சென்னை தமிழ்நாட்டின் தலைநகரமாகும்.", "score": 0.92, "language": "ta"}
        ]
    },
    {
        "query": "Who wrote the play Hamlet?",
        "query_language": "en",
        "stt_latency_ms": 55.0,
        "retrieval_latency_ms": 24.5,
        "retrieved_chunks": [
            {"chunk_id": "c3", "text": "The tragedy of Hamlet was written by William Shakespeare.", "score": 0.85, "language": "en"}
        ]
    },
    # Edge Case: Irrelevant context (Testing hallucination guardrail)
    {
        "query": "What is the speed of light in vacuum?",
        "query_language": "en",
        "stt_latency_ms": 60.0,
        "retrieval_latency_ms": 35.0,
        "retrieved_chunks": [
            {"chunk_id": "c4", "text": "Photosynthesis is the process by which plants convert sunlight to energy.", "score": 0.22, "language": "en"}
        ]
    }
]


async def run_benchmark(num_runs: int = 15):
    print(f"[*] Starting benchmark run across {len(BENCHMARK_DATASET) * num_runs} iterations...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check health
        health = await client.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("[!] API service is unreachable. Start FastAPI first.")
            return

        for i in range(num_runs):
            for item in BENCHMARK_DATASET:
                res = await client.post(f"{BASE_URL}/v1/process_turn", json=item)
                if res.status_code != 200:
                    print(f"[!] Error on run {i}: {res.text}")

        # Fetch telemetry
        analytics_res = await client.get(f"{BASE_URL}/v1/analytics")
        data = analytics_res.json()

        table_rows = []
        for stage, stats in data.items():
            if isinstance(stats, dict) and "p50" in stats:
                table_rows.append([
                    stage,
                    f"{stats['p50']} ms",
                    f"{stats['p70']} ms",
                    f"{stats['p90']} ms",
                    f"{stats['p100']} ms",
                    stats['count']
                ])

        print("\n" + "=" * 60)
        print("         FINAL BENCHMARK LATENCY REPORT")
        print("=" * 60)
        print(tabulate(
            table_rows,
            headers=["Pipeline Stage", "P50", "P70", "P90", "P100 (Max)", "Samples"],
            tablefmt="fancy_grid"
        ))


if __name__ == "__main__":
    asyncio.run(run_benchmark(num_runs=10))