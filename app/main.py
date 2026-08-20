import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.schemas import PipelineInput, PipelineOutput
from app.generator import GenerationHarness
from app.analytics import telemetry
from pipeline import RetrievalPipeline  
from app.schemas import RetrievedChunk, PipelineInput

# Initialize once at startup (warm up index/embedder)
retrieval_pipeline = RetrievalPipeline()
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


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "generation_guardrails_analytics"}

class UserQueryRequest(BaseModel):
    query: str
    query_language: str = "hi"
    stt_latency_ms: float = 0.0

@app.post("/v1/chat", response_model=PipelineOutput)
async def chat_end_to_end(payload: UserQueryRequest):
    # 1. Execute Retrieval via Person 2's engine
    retrieval_result = retrieval_pipeline.query(
        query=payload.query, 
        language=payload.query_language
    )

    # 2. Map Person 2's Chunk dataclass to Person 3's Pydantic schema
    chunks = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
            language=c.language,
            source=getattr(c, "source", None)
        )
        for c in retrieval_result.chunks
    ]

    pipeline_input = PipelineInput(
        query=payload.query,
        query_language=payload.query_language,
        retrieved_chunks=chunks,
        stt_latency_ms=payload.stt_latency_ms,
        retrieval_latency_ms=retrieval_result.latency_ms
    )

    # 3. Generate + Ground + Record Telemetry
    result = await harness.execute_unary(pipeline_input)
    telemetry.record(result.latencies)
    return result
    
@app.post("/v1/process_turn", response_model=PipelineOutput)
async def process_turn(payload: PipelineInput):
    """
    Synchronous end-to-end harness execution with complete latency accounting.
    """
    try:
        result = await harness.execute_unary(payload)
        telemetry.record(result.latencies)
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
    Returns live P50/P70/P90/P100 latency percentiles across all recorded turns.
    """
    return telemetry.get_percentiles()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2)