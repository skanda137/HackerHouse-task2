import json
import logging
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.schemas import PipelineInput, PipelineOutput
from app.generator import GenerationHarness
from app.analytics import telemetry
from pipeline import RetrievalPipeline
from app.schemas import RetrievedChunk, PipelineInput
from data.load_msmarco_xi import load_real, load_synthetic
from embeddings.embedder import TfidfEmbedder
from dotenv import load_dotenv; load_dotenv()

logger = logging.getLogger("main")

retrieval_pipeline = RetrievalPipeline(embedder=TfidfEmbedder())
try:
    _startup_docs = load_real(max_docs_per_language=400)
    if not _startup_docs:
        raise RuntimeError("load_real() returned zero documents")
    logger.warning(
        "STARTUP: loaded REAL ai4bharat/MSMARCO-XI data (validation split) -- %d documents across %s",
        len(_startup_docs), sorted({d.language for d in _startup_docs}),
    )
except Exception as exc:
    logger.error(
        "STARTUP: FAILED to load real ai4bharat/MSMARCO-XI data (%s) -- "
        "FALLING BACK TO SYNTHETIC DATA. This is NOT the real dataset.",
        exc, exc_info=True,
    )
    _startup_docs = load_synthetic(docs_per_language=150)
retrieval_pipeline.build_all(_startup_docs)

app = FastAPI(
    title="Hacker House Goa 2026 - RAG Generation & Harness Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

harness = GenerationHarness()

# Demo-reliability cache: keyed on normalized (query, language), only ever
# populated with a response that was actually produced by a real successful
# call (status == "success") -- never precomputed or faked. Exists so a
# query rehearsed once doesn't re-risk a live timeout/rate-limit hit on the
# exact same question during the actual demo. Deliberately NOT recorded into
# telemetry on a hit, so it can't distort /v1/analytics latency reporting.
_demo_response_cache: dict[tuple[str, str], PipelineOutput] = {}


def _demo_cache_key(query: str, language: str) -> tuple[str, str]:
    return (query.strip().lower(), language)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "generation_guardrails_analytics"}

class UserQueryRequest(BaseModel):
    query: str
    query_language: str = "hi"
    stt_latency_ms: float = 0.0

@app.post("/v1/chat", response_model=PipelineOutput)
async def chat_end_to_end(payload: UserQueryRequest):
    cache_key = _demo_cache_key(payload.query, payload.query_language)
    cached = _demo_response_cache.get(cache_key)
    if cached is not None:
        logger.info("demo cache hit: query=%r lang=%s", payload.query, payload.query_language)
        return cached

    outcome = retrieval_pipeline.query(
        text=payload.query,
        language=payload.query_language
    )

    chunks = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            score=outcome.verdict.top_score or 0.0,
            language=c.language,
            source=getattr(c, "source", None)
        )
        for c in outcome.verdict.grounded_chunks
    ]

    pipeline_input = PipelineInput(
        query=payload.query,
        query_language=payload.query_language,
        retrieved_chunks=chunks,
        stt_latency_ms=payload.stt_latency_ms,
        retrieval_latency_ms=outcome.total_ms
    )

    result = await harness.execute_unary(pipeline_input)
    telemetry.record(result.latencies, status=result.status)
    if result.status == "success":
        _demo_response_cache[cache_key] = result
    return result
    
@app.post("/v1/process_turn", response_model=PipelineOutput)
async def process_turn(payload: PipelineInput):
    """
    Synchronous end-to-end harness execution with complete latency accounting.
    """
    try:
        result = await harness.execute_unary(payload)
        telemetry.record(result.latencies, status=result.status)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/stream_turn")
async def stream_turn(payload: PipelineInput):
    """
    SSE stream enabling Person 1's TTS engine to start speaking on token 1.
    """
    async def event_generator():
        # Validate chunks first
        valid_chunks = [c for c in payload.retrieved_chunks if c.score >= harness.min_retrieval_score_threshold]
        if not valid_chunks:
            yield f"data: {json.dumps({'token': harness._get_fallback(payload.query_language), 'is_final': True})}\n\n"
            return

        messages = harness._build_prompt(payload.query, valid_chunks, payload.query_language)
        
        try:
            stream = await harness.client.chat.completions.create(
                model=harness.model,
                messages=messages,
                temperature=0.0,
                max_tokens=60,
                reasoning_effort="low",
                stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'token': delta, 'is_final': False})}\n\n"
            
            yield f"data: {json.dumps({'token': '', 'is_final': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'is_final': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/v1/analytics")
async def get_analytics():
    """
    Returns live P50/P70/P90/P100 latency percentiles across all recorded
    turns, plus a count of outcomes by status (success / guardrail declines /
    generation failures) so failure modes are visible as their own category,
    not folded into a single crash count.
    """
    return {
        "latency_percentiles": telemetry.get_percentiles(),
        "status_counts": telemetry.get_status_counts(),
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2)