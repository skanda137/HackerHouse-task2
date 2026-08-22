# Person 2 — Chunking + Retrieval + Vector DB

**HH Goa 2026 — Task 2: Voice-Enabled RAG**
Owner: Person 2 — dataset chunking, indexing, and retrieval latency.

This is the retrieval backbone the rest of the pipeline (speech-to-text →
**[this component]** → answer generation) sits on. It takes raw
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
passages, chunks them three different ways, indexes each chunk set into a
local FAISS vector store, and serves retrieval with millisecond-level
latency and an explicit "don't answer" guardrail.

```
documents (hi/ta/te/bn) ─▶ chunking (×3 strategies) ─▶ embed ─▶ FAISS index
                                                                     │
question ──────────────▶ embed query ──────────────────▶ search ──▶│
                                                                     ▼
                                                       RetrievalGuardrail
                                                        (should_answer?)
                                                                     │
                                                                     ▼
                                                     QueryOutcome (typed,
                                                     handed to generation)
```

## Why the dataset shapes the design

MSMARCO-XI isn't one blob of text — it ships as **separate per-language
configs** (`hi`, `ta`, `te`, `bn`, ...). That's treated as a first-class
axis throughout this component, not an afterthought:

- Chunk size budgets are tuned per language (Tamil/Telugu are agglutinative
  and run longer per sentence than Hindi/Bengali for the same amount of
  meaning — see `chunking/metadata_aware.py::LANGUAGE_PROFILES`).
- Sentence splitting handles both Latin punctuation (`. ! ?`) *and* the
  Devanagari/Indic danda (`। ॥`) used across Hindi/Bengali/Marathi-family
  scripts — a naive English-only regex silently breaks on this dataset.
- Every chunk carries `language`, `script`, and `source` metadata, so
  retrieval can filter/boost by language at query time.

## The three chunking strategies

The brief explicitly warns against a single naive fixed-size approach.
This repo ships three, plus an optional fourth for embedding-based
semantic drift detection:

| Strategy | File | Idea |
|---|---|---|
| **Fixed-size + overlap** | `chunking/fixed_size.py` | Baseline: hard character budget (220 chars) with 40-char overlap so facts straddling a boundary aren't lost. Falls back to a single whole-passage chunk for MSMARCO's many already-short passages instead of pointlessly splitting them. |
| **Semantic (sentence-boundary)** | `chunking/semantic.py` → `SentenceBoundaryChunker` | Never cuts mid-sentence. Splits on real sentence terminators (Latin + Indic danda), then greedily packs sentences into a size budget. |
| **Metadata-aware (language-routed)** | `chunking/metadata_aware.py` | Wraps the semantic chunker but **routes each document to a per-language-tuned budget** and stamps every chunk with `language`, `script`, and `source` metadata for filtering. This is the default/primary strategy used at query time. |
| *(bonus)* **Embedding semantic drift** | `chunking/semantic.py` → `EmbeddingSemanticChunker` | Embeds sentences and merges them while cosine similarity to the running chunk centroid stays above a threshold; cuts a new chunk on topic drift. Pluggable with any embedder. Not used in the default benchmark (adds embedding cost during chunking itself) but included and tested as the "go further" option. |

All four share one `Chunk` schema (`chunking/base.py`) so nothing
downstream needs to know which strategy produced a given chunk.

## Vector DB: FAISS, chosen for the latency budget

The 200ms end-to-end cap is the reason a **local, in-process FAISS index**
was picked over a hosted vector DB. A hosted DB (Pinecone, Weaviate Cloud,
etc.) adds a network round-trip that's outside our control and routinely
*alone* eats the whole budget. FAISS's `IndexFlatIP` (exact, brute-force
cosine/inner-product search) runs in-memory — the only latency left is CPU
math, which is exactly what got profiled and optimized. `index/faiss_index.py`
also documents a one-line swap to `IndexIVFFlat` for if the corpus ever
grows past the point where exact search is the right tradeoff (well beyond
this task's realistic scale).

## Embeddings: pluggable, two implementations

`embeddings/embedder.py` defines one interface, two implementations:

- **`SentenceTransformerEmbedder`** (production default) — multilingual
  `intfloat/multilingual-e5-small`, so hi/ta/te/bn all land in one vector
  space. Needs `pip install sentence-transformers` and network access to
  the HF Hub *once*, on first run, to download weights; fully offline
  after that.
- **`TfidfEmbedder`** (offline fallback, used for the numbers in this
  README) — pure scikit-learn, character n-gram TF-IDF (char n-grams, not
  word n-grams, because word tokenization differs a lot across
  Devanagari/Tamil/Telugu/Bengali scripts). Zero downloads, runs anywhere.

**Why the benchmark numbers below use TF-IDF, not the transformer:** the
architecture, retry/harness logic, index, guardrail, and latency-measurement
code are all identical either way — swapping embedders is a one-line change
(`RetrievalPipeline(embedder=...)`). TF-IDF is what's actually deployed
right now; `SentenceTransformerEmbedder` is implemented and importable but
hasn't been swapped in yet (network access for the model download and time
were the constraints, not the code — see
[Switching to the production embedder](#switching-to-the-production-embedder)
below). The dataset itself, however, **is** the real one — see below.

## Harness (requirement #5)

`pipeline.py` wraps every fallible step — embedding a batch, adding to the
index, running a search — in a generic `retry()` with backoff
(`RetrievalPipeline` / `retrieve()` in `retrieval/retriever.py`). Every
stage returns a **typed dataclass**, never a bare string or dict:
`IndexBuildReport`, `RetrievalResult`, `RetrievalVerdict`, `QueryOutcome`.
A failed retrieval doesn't crash the run — it degrades to a structured
`should_answer=False` outcome with the error captured in `reason`, so the
generation layer downstream always gets a well-formed contract to branch
on.

## Guardrails (requirement #6) — the retrieval half

`guardrails/retrieval_guardrail.py` turns raw FAISS scores into an
explicit verdict *before* generation is ever invoked:

1. **No hits** → refuse (empty/misconfigured index).
2. **Top score below threshold** (`0.28` cosine similarity, default) →
   refuse — this is the off-topic / out-of-domain signal.
3. **Score-gap check** (optional, disabled by default — corpus-size
   dependent) → refuse on diffuse, low-confidence matches.

Generation only ever receives `verdict.grounded_chunks` — chunks that
already cleared the confidence bar — so an ungrounded answer is
structurally harder to produce, not just discouraged by a prompt.

## Latency results (requirement #4)

Measured with `benchmark/run_benchmark.py` against the **real
`ai4bharat/MSMARCO-XI` dataset** (validation split, loaded via
`load_real()` — not synthetic data): **248 queries** (240 real MS MARCO
queries, drawn from each document's own gold-labeled `query` field across
all 4 languages — expected answerable — + 8 off-topic probes, replicated
across all 3 strategies = **744 total timed retrievals**), after a 5-query
warm-up excluded from the reported numbers. Corpus: 1,600 real documents
(400/language), producing 2,173–3,289 chunks depending on strategy.

| Strategy | n | P50 | P70 | P100 (max) | mean | % under 200ms |
|---|---|---|---|---|---|---|
| `fixed_size` | 248 | **34.70 ms** | **36.57 ms** | 49.02 ms | 30.71 ms | 100% |
| `semantic_sentence_boundary` | 248 | **25.37 ms** | **26.96 ms** | 42.84 ms | 22.78 ms | 100% |
| `metadata_aware` (default) | 248 | **23.48 ms** | **25.34 ms** | 32.94 ms | 20.96 ms | 100% |

Every strategy clears the 200ms retrieval budget with room to spare — real
retrieval against the real (larger, real-text) index costs low tens of
milliseconds, not the sub-2ms seen on the earlier synthetic corpus, but
still nowhere near the 200ms ceiling. These retrieval-only numbers are
consistent with what the live `/v1/analytics` endpoint reports for
`retrieval_ms` under real HTTP traffic. **This is the part of the pipeline
fully in our control; it is not where the 200ms target is actually spent —
see [Known limitations](#known-limitations) below for the honest number on
the full pipeline.**

Raw per-query data: `results/latency_results.csv`
Machine-readable summary: `results/latency_summary.json` (`data_source` field
records whether a given run used real or synthetic data)
Chart: `results/latency_chart.png`

Guardrail sanity check from the same run: **24/24 off-topic probes
correctly rejected**, **696/720 grounded queries correctly answered**
(96.7%, across all 3 strategies) — this measures the retrieval guardrail's
own `0.28` similarity threshold, which is more lenient than the live
`/v1/chat` pipeline's full funnel (retrieval guardrail → harness's stricter
`0.45` re-check → LLM generation → post-generation grounding check). See
[Known limitations](#known-limitations) for why the live success rate is
noticeably lower than 96.7%.

## Known limitations

Stated plainly, not buried in commit messages:

- **Full end-to-end latency does not meet the 200ms target, and can't with
  the current architecture.** Live-endpoint batch testing across 32 real
  queries (`/v1/analytics`, `total_e2e_ms`) measured **P50 1328ms / P70
  1697ms / P100 2094ms** — dominated almost entirely by the external LLM API
  call (`total_generation_ms` alone: P50 1422ms). This is architecturally
  external to this codebase: no amount of retrieval optimization closes a
  gap that's coming from a third-party API round-trip. **Retrieval — the
  component actually in our control — meets the 200ms target with a wide
  margin**, at P50 23-35ms / P100 33-49ms even against the real MS MARCO
  index (see the table above).
- **Roughly half of real corpus queries hit the grounding guardrail's
  `fallback_no_context` / `fallback_ungrounded` state**, not because
  retrieval or the guardrail are broken, but because MS MARCO's queries are
  *intentionally* paraphrased away from their answer passage's wording —
  that lexical gap is the dataset's whole point, and a purely lexical
  TF-IDF embedder feels it more than a semantic one would. The guardrail is
  correctly detecting "not confidently grounded" here, not misfiring. A
  demo-safety preflight script (`scripts/preflight_demo_queries.py`) exists
  specifically to pick queries that are confirmed to clear the full
  pipeline live, rather than assuming any gold-labeled MS MARCO query will.
- **No real Sarvam STT session has been run in this environment** — no
  `SARVAM_API_KEY` was available here. `voice/node/server.js` (the real
  bridge) and `App.jsx` are both implemented and were proven correct against
  a mocked STT bridge (`voice/node/mock-stt-bridge.test.js`) that replays
  Sarvam's exact documented realtime message contract (`event`,
  `text`, `language` fields). That is a verified integration against the
  real contract, not a verified integration against the real service —
  don't let the submission imply otherwise. Run it against a real key before
  relying on it live.

## Switching to the production embedder

Everything below needs outbound network access for the model download, but
is fully wired up and ready to run (the dataset side of this no longer
needs a swap — `app/main.py` already loads real data by default):

```bash
pip install -r requirements.txt   # adds datasets + sentence-transformers

python3 -c "
from data.load_msmarco_xi import load_real
from embeddings.embedder import SentenceTransformerEmbedder
from pipeline import RetrievalPipeline

docs = load_real(languages=['hi','ta','te','bn'], max_docs_per_language=2000)
pipeline = RetrievalPipeline(embedder=SentenceTransformerEmbedder())
pipeline.build_all(docs)

out = pipeline.query('भारत की राजधानी क्या है?', strategy='metadata_aware', language='hi')
print(out.verdict.should_answer, out.verdict.top_score, out.total_ms)
"
```

Re-run `python3 -m benchmark.run_benchmark` after that swap (it already uses
`load_real()`; construct `RetrievalPipeline(embedder=SentenceTransformerEmbedder())`
in place of the default `TfidfEmbedder()`) to get updated P50/P70/P100
numbers with the production embedder instead of TF-IDF.

## Repo layout

```
chunking/            Chunk/Document schema + all 4 chunking strategies
embeddings/          Pluggable embedder interface (TF-IDF + sentence-transformer)
index/               FAISS wrapper (IndexFlatIP, with IVF upgrade path)
retrieval/           Retriever with per-stage latency instrumentation
guardrails/          Retrieval-side "should I even try to answer?" logic
data/                MSMARCO-XI loader (real HF path + offline synthetic fallback)
pipeline.py          Harnessed orchestration tying it all together
benchmark/           Latency benchmark + chart generator
demo.py              Minimal runnable example
tests/               Unit tests for chunking correctness
results/             Generated: latency_results.csv, latency_summary.json, latency_chart.png
```

## Running it yourself

```bash
pip install -r requirements.txt        # numpy/sklearn/faiss are enough for the offline path
python3 -m pytest tests/ -v            # 8 tests, chunking correctness
python3 demo.py                        # 4 example queries, grounded + off-topic
python3 -m benchmark.run_benchmark     # full latency benchmark -> results/
python3 -m benchmark.plot_latency      # -> results/latency_chart.png
```
