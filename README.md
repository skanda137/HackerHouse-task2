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

**Why the benchmark numbers below use TF-IDF, not the transformer:** this
was built and tested in a network-restricted sandbox that can't reach
`huggingface.co`. The architecture, retry/harness logic, index, guardrail,
and latency-measurement code are all identical either way — swapping
embedders is a one-line change (`RetrievalPipeline(embedder=...)`). See
[Switching to the real dataset + production embedder](#switching-to-the-real-dataset--production-embedder) below.

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

Measured with `benchmark/run_benchmark.py`: **248 queries** (240 pulled
verbatim from corpus sentences across all 4 languages — expected
answerable — + 8 off-topic probes, replicated across all 3 strategies =
**744 total timed retrievals**), after a 5-query warm-up excluded from the
reported numbers. Corpus: 600 synthetic MSMARCO-XI-shaped documents
(150/language), producing 907–1,210 chunks depending on strategy.

| Strategy | n | P50 | P70 | P100 (max) | mean | % under 200ms |
|---|---|---|---|---|---|---|
| `fixed_size` | 248 | **1.310 ms** | **1.334 ms** | 1.585 ms | 1.312 ms | 100% |
| `semantic_sentence_boundary` | 248 | **1.119 ms** | **1.142 ms** | 2.503 ms | 1.129 ms | 100% |
| `metadata_aware` (default) | 248 | **1.098 ms** | **1.116 ms** | 1.439 ms | 1.100 ms | 100% |

Every strategy clears the 200ms budget with roughly **two orders of
magnitude of headroom** — even the worst single observed query (P100) is
under 2.5ms. That headroom is deliberate: it's there to absorb the extra
cost of the production embedder (a transformer forward pass is slower than
TF-IDF) plus whatever latency speech-to-text and generation add elsewhere
in the pipeline, while the *whole system* still targets sub-200ms.

Raw per-query data: `results/latency_results.csv`
Machine-readable summary: `results/latency_summary.json`
Chart: `results/latency_chart.png`

Guardrail sanity check from the same run: **24/24 off-topic probes
correctly rejected**, **720/720 grounded queries correctly answered**
(100% each, across all 3 strategies).

### An honest limitation, on display in `demo.py`

Run `python3 demo.py` and one of the four example queries — a Tamil
question about the Himalayas, phrased differently from the passage's
wording — gets **correctly declined** by the guardrail (score 0.247, just
under the 0.28 threshold), even though a relevant passage exists in the
corpus. That's TF-IDF's char n-gram matching being weaker than real
semantic embeddings at handling morphological variation in agglutinative
languages like Tamil — exactly the kind of gap `SentenceTransformerEmbedder`
closes. Left in deliberately rather than cherry-picking only the queries
that work: it shows the guardrail doing its job (declining on a genuinely
marginal match) and gives an honest, reproducible reason the production
embedder matters, not just an assertion that it would.

## Switching to the real dataset + production embedder

Everything below needs outbound network access (blocked in the sandbox
this was built in) but is fully wired up and ready to run:

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

Re-run `python3 -m benchmark.run_benchmark` after that swap (edit the two
lines noted with `# offline synthetic` in `benchmark/run_benchmark.py` to
call `load_real()` instead of `load_synthetic()`, and construct
`RetrievalPipeline(embedder=SentenceTransformerEmbedder())`) to get final
submission-ready P50/P70/P100 numbers against the real dataset.

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
