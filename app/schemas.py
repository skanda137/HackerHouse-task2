from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(..., description="Cosine similarity or retrieval score")
    language: str = Field(..., description="ISO 639-1 code: hi, ta, te, bn, en")
    source: Optional[str] = None


class PipelineInput(BaseModel):
    query: str
    query_language: str = "hi"
    retrieved_chunks: List[RetrievedChunk]
    stt_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    conversation_id: Optional[str] = "default-session"


class StageLatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    ttft_ms: float = 0.0
    total_generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_e2e_ms: float = 0.0


class PipelineOutput(BaseModel):
    answer: str
    is_grounded: bool
    confidence_score: float
    status: Literal[
        "success", "fallback_no_context", "fallback_ungrounded", "error", "fallback_generation_error"
    ]
    latencies: StageLatencyBreakdown
    used_chunks: List[str] = Field(default_factory=list)