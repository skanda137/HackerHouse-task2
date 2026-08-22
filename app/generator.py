import os
import time
import asyncio
import logging
from typing import AsyncGenerator, Tuple, List
from openai import AsyncOpenAI
from app.schemas import PipelineInput, PipelineOutput, StageLatencyBreakdown
from app.grounding import LightweightGroundingEngine

logger = logging.getLogger("generator")

# Fallback templates in supported Indic languages
FALLBACK_RESPONSES = {
    "hi": "मुझे प्रदान किए गए संदर्भ में इस प्रश्न का सटीक उत्तर नहीं मिला।",
    "ta": "வழங்கப்பட்ட சூழலில் இந்த கேள்விக்கான சரியான பதில் கிடைக்கவில்லை.",
    "te": "అందించిన సందర్భంలో ఈ ప్రశ్నకు ఖచ్చితమైన సమాధానం దొరకలేదు.",
    "bn": "প্রদত্ত প্রসঙ্গে এই প্রশ্নের সঠিক উত্তর পাওয়া যায়নি।",
    "en": "I do not have sufficient context from the verified records to answer this accurately."
}


class GenerationHarness:
    def __init__(self):
        # Using Groq / Together / DeepInfra for ultra-low Time-To-First-Token
        self.api_key = os.getenv("LLM_API_KEY", "your-api-key")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.grounding_engine = LightweightGroundingEngine(token_overlap_threshold=0.60)
        self.min_retrieval_score_threshold = 0.45

        # Minimum spacing between outbound LLM calls. Doesn't remove the
        # provider's TPM cap, just reduces the odds of tripping it during a
        # live demo/judge session where requests can otherwise land back-to-back.
        self._min_call_interval_s = float(os.getenv("LLM_MIN_CALL_INTERVAL_S", "4.0"))
        self._throttle_lock = asyncio.Lock()
        self._last_call_monotonic = 0.0

    async def _throttle(self):
        async with self._throttle_lock:
            now = time.monotonic()
            wait_s = self._min_call_interval_s - (now - self._last_call_monotonic)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_call_monotonic = time.monotonic()

    def _build_prompt(self, query: str, chunks: list, language: str) -> List[dict]:
        context_text = "\n---\n".join([f"Passage [{i+1}]: {c.text}" for i, c in enumerate(chunks)])
        
        system_instruction = (
            f"You are a strict, low-latency factual voice assistant answering in language code '{language}'. "
            "Follow these rules strictly:\n"
            "1. Answer ONLY using the facts stated in the provided passages.\n"
            "2. Keep the answer under 35 words so it can be spoken quickly.\n"
            "3. If the context does not clearly contain the answer, say: 'NO_CONTEXT'.\n"
            "4. Do not speculate or add background knowledge."
        )

        user_content = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    def _get_fallback(self, lang: str) -> str:
        return FALLBACK_RESPONSES.get(lang, FALLBACK_RESPONSES["en"])

    async def execute_unary(self, payload: PipelineInput) -> PipelineOutput:
        start_gen = time.perf_counter()
        lang = payload.query_language.lower()
        
        # 1. Retrieval confidence gate: check if Person 2 found anything useful
        valid_chunks = [c for c in payload.retrieved_chunks if c.score >= self.min_retrieval_score_threshold]
        if not valid_chunks:
            return PipelineOutput(
                answer=self._get_fallback(lang),
                is_grounded=True,
                confidence_score=0.0,
                status="fallback_no_context",
                latencies=StageLatencyBreakdown(
                    stt_ms=payload.stt_latency_ms,
                    retrieval_ms=payload.retrieval_latency_ms,
                    total_e2e_ms=payload.stt_latency_ms + payload.retrieval_latency_ms
                )
            )

        messages = self._build_prompt(payload.query, valid_chunks, lang)
        generated_text = ""
        ttft_recorded = False
        ttft_ms = 0.0

        try:
            await self._throttle()
            # Enforce strict 1.8s timeout to prevent hanging the voice interface
            async with asyncio.timeout(1.8):
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=60,
                    stream=True
                )
                
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if not ttft_recorded:
                            ttft_ms = (time.perf_counter() - start_gen) * 1000.0
                            ttft_recorded = True
                        generated_text += delta

        except asyncio.TimeoutError:
            return PipelineOutput(
                answer=self._get_fallback(lang),
                is_grounded=False,
                confidence_score=0.0,
                status="error",
                latencies=StageLatencyBreakdown(
                    stt_ms=payload.stt_latency_ms,
                    retrieval_ms=payload.retrieval_latency_ms,
                    total_generation_ms=1800.0,
                    total_e2e_ms=payload.stt_latency_ms + payload.retrieval_latency_ms + 1800.0
                )
            )
        except Exception as exc:
            # Any other LLM-call failure (bad model name, auth expiry, rate
            # limit, malformed response, provider outage, ...). The whole
            # point of the guardrail requirement is that a failure here
            # degrades to a structured decline, not a raw 500 -- the client
            # must never see an unhandled crash for something as ordinary as
            # a provider hiccup. Full exception logged server-side; the
            # client gets the same shape fallback_no_context already uses,
            # tagged with its own status so it's countable separately.
            logger.error("LLM generation call failed: %s", exc, exc_info=True)
            elapsed_ms = (time.perf_counter() - start_gen) * 1000.0
            return PipelineOutput(
                answer=self._get_fallback(lang),
                is_grounded=False,
                confidence_score=0.0,
                status="fallback_generation_error",
                latencies=StageLatencyBreakdown(
                    stt_ms=payload.stt_latency_ms,
                    retrieval_ms=payload.retrieval_latency_ms,
                    total_generation_ms=round(elapsed_ms, 2),
                    total_e2e_ms=round(payload.stt_latency_ms + payload.retrieval_latency_ms + elapsed_ms, 2)
                )
            )

        total_gen_ms = (time.perf_counter() - start_gen) * 1000.0
        generated_text = generated_text.strip()

        # 2. Check if the model triggered our explicit fallback refusal
        if "NO_CONTEXT" in generated_text:
            return PipelineOutput(
                answer=self._get_fallback(lang),
                is_grounded=True,
                confidence_score=1.0,
                status="fallback_no_context",
                latencies=StageLatencyBreakdown(
                    stt_ms=payload.stt_latency_ms,
                    retrieval_ms=payload.retrieval_latency_ms,
                    ttft_ms=round(ttft_ms, 2),
                    total_generation_ms=round(total_gen_ms, 2),
                    total_e2e_ms=round(payload.stt_latency_ms + payload.retrieval_latency_ms + total_gen_ms, 2)
                )
            )

        # 3. Output Guardrail / Grounding Check
        is_grounded, conf_score, ground_ms = self.grounding_engine.verify_grounding(generated_text, valid_chunks)

        final_answer = generated_text if is_grounded else self._get_fallback(lang)
        final_status = "success" if is_grounded else "fallback_ungrounded"

        total_e2e = payload.stt_latency_ms + payload.retrieval_latency_ms + total_gen_ms + ground_ms

        return PipelineOutput(
            answer=final_answer,
            is_grounded=is_grounded,
            confidence_score=conf_score,
            status=final_status,
            used_chunks=[c.chunk_id for c in valid_chunks],
            latencies=StageLatencyBreakdown(
                stt_ms=payload.stt_latency_ms,
                retrieval_ms=payload.retrieval_latency_ms,
                ttft_ms=round(ttft_ms, 2),
                total_generation_ms=round(total_gen_ms, 2),
                grounding_ms=round(ground_ms, 2),
                total_e2e_ms=round(total_e2e, 2)
            )
        )