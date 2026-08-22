"""
Demo-safety preflight: hits the LIVE /v1/chat endpoint with candidate demo
questions (one set of grounded candidates + one deliberately off-topic
question per language) and reports which are actually safe to say on stage.

Why this exists: against the real ai4bharat/MSMARCO-XI corpus, roughly half
of real gold-labeled queries hit the retrieval guardrail's fallback_no_context
state (MS MARCO's queries are intentionally paraphrased away from their
answer passage's wording -- a real lexical gap, not a bug). Picking a demo
query "blind" is a coin flip. This script removes the guesswork.

Pacing: the Groq key backing this deployment has a CONFIRMED 8000 tokens/
minute (TPM) cap -- seen verbatim in a live rate-limit error during testing:

    "Rate limit reached for model `openai/gpt-oss-120b` in organization
    ... service tier `on_demand` on tokens per minute (TPM): Limit 8000,
    Used 7563, Requested 3681."

/v1/chat's response schema doesn't expose per-call token usage, so this
can't run a true adaptive token-bucket from the client side. Instead it uses
PACING_SECONDS, an interval empirically proven safe: a 32-query batch run at
9s spacing completed with 0/32 rate-limit hits. 10s is used here for a small
extra margin, not a fresh guess.

Usage:
    python3 scripts/preflight_demo_queries.py
"""
from __future__ import annotations

import json
import time
import urllib.request

API_URL = "http://localhost:8000/v1/chat"
PACING_SECONDS = 10  # see module docstring: proven-safe interval + margin, not a guess
CONFIDENCE_BAR = 0.7

# Grounded candidates pulled from real, gold-labeled (is_selected=True) rows
# of the live ai4bharat/MSMARCO-XI corpus (validation split) via
# data.load_msmarco_xi.load_real(). Off-topic candidates are hand-written,
# clearly out-of-corpus questions per language.
CANDIDATES = {
    "hi": {
        "grounded": [
            "स्टबहब टोल फ्री नंबर",
            "बाज़ कितनी तेजी से यात्रा करता है",
            "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा",
        ],
        "off_topic": ["पिज़्ज़ा के लिए सबसे अच्छा टॉपिंग कौन सा है?"],
    },
    "ta": {
        "grounded": [
            "ஸ்டப்ஹப் டோல் இலவச எண்",
            "கழுகு எவ்வளவு வேகமாக பயணிக்கிறது?",
            "நேர்மை அல்லது நேர்மையின் வரையறை",
        ],
        "off_topic": ["பீட்சாவிற்கு சிறந்த டாப்பிங் எது?"],
    },
    "te": {
        "grounded": [
            "స్టబ్‌హబ్ టోల్ ఉచిత నంబర్",
            "గద్దలు ఎంత వేగంగా ప్రయాణిస్తాయో",
            "బార్టర్ సిస్టమ్ మరియు దాని సమస్యలు ఏమిటి",
        ],
        "off_topic": ["పిజ్జాకు ఉత్తమమైన టాపింగ్ ఏది?"],
    },
    "bn": {
        "grounded": [
            "স্টাবহাব টোল ফ্রি নম্বর",
            "একটি ঈগল কত দ্রুত ভ্রমণ করে।",
            "সততা বা সততার সংজ্ঞা",
        ],
        "off_topic": ["পিৎজার জন্য সেরা টপিং কোনটি?"],
    },
}


def call(query: str, lang: str) -> dict:
    body = json.dumps({"query": query, "query_language": lang, "stt_latency_ms": 250.0}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            data["http_status"] = resp.status
            return data
    except Exception as exc:
        return {"http_status": "EXC", "status": f"client_exception:{exc}",
                "confidence_score": 0.0, "answer": None, "latencies": {}}


def main():
    all_calls = [(lang, kind, q) for lang, groups in CANDIDATES.items()
                 for kind, queries in groups.items() for q in queries]

    print(f"Preflighting {len(all_calls)} candidate queries against {API_URL}, "
          f"{PACING_SECONDS}s apart (confirmed Groq TPM cap: 8000)\n")

    results = []
    for i, (lang, kind, q) in enumerate(all_calls, 1):
        resp = call(q, lang)
        row = {
            "lang": lang, "kind": kind, "query": q,
            "http_status": resp.get("http_status"),
            "pipeline_status": resp.get("status"),
            "confidence": resp.get("confidence_score"),
            "answer": resp.get("answer"),
            "e2e_ms": (resp.get("latencies") or {}).get("total_e2e_ms"),
        }
        results.append(row)
        print(f"[{i:2d}/{len(all_calls)}] {lang}/{kind:9s} http={row['http_status']} "
              f"status={str(row['pipeline_status']):24s} conf={row['confidence']}  {q[:40]}")
        if i < len(all_calls):
            time.sleep(PACING_SECONDS)

    safe_grounded, safe_guardrail_demo, avoid = [], [], []
    for r in results:
        if r["kind"] == "grounded":
            if r["pipeline_status"] == "success" and (r["confidence"] or 0) >= CONFIDENCE_BAR:
                safe_grounded.append(r)
            else:
                avoid.append(r)
        else:  # off_topic: safe means the guardrail correctly declined it
            if r["pipeline_status"] in ("fallback_no_context", "fallback_ungrounded"):
                safe_guardrail_demo.append(r)
            else:
                avoid.append(r)

    print("\n" + "=" * 72)
    print(f"SAFE TO USE LIVE -- grounded, status=success & confidence >= {CONFIDENCE_BAR} "
          f"({len(safe_grounded)}):")
    for r in safe_grounded:
        print(f"  [{r['lang']}] conf={r['confidence']}  Q: {r['query']}")
        print(f"      A: {r['answer']}")

    print(f"\nSAFE TO USE LIVE -- off-topic, guardrail correctly declines ({len(safe_guardrail_demo)}):")
    for r in safe_guardrail_demo:
        print(f"  [{r['lang']}] status={r['pipeline_status']}  Q: {r['query']}")

    print(f"\nDO NOT USE LIVE ({len(avoid)}):")
    for r in avoid:
        print(f"  [{r['lang']}/{r['kind']}] http={r['http_status']} status={r['pipeline_status']} "
              f"conf={r['confidence']}  Q: {r['query']}")

    out_path = "scripts/preflight_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved -> {out_path}")


if __name__ == "__main__":
    main()
