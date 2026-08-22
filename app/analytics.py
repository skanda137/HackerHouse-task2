import numpy as np
from typing import Dict, List, Optional
from collections import deque, Counter
from app.schemas import StageLatencyBreakdown


class TelemetryTracker:
    def __init__(self, window_size: int = 1000):
        self.history: deque = deque(maxlen=window_size)
        self.status_counts: Counter = Counter()

    def record(self, latencies: StageLatencyBreakdown, status: Optional[str] = None):
        self.history.append(latencies.model_dump())
        if status:
            self.status_counts[status] += 1

    def get_status_counts(self) -> Dict[str, int]:
        return dict(self.status_counts)

    def get_percentiles(self) -> Dict[str, Dict[str, float]]:
        if not self.history:
            return {"status": "No data recorded yet"}

        metrics = ["stt_ms", "retrieval_ms", "ttft_ms", "total_generation_ms", "grounding_ms", "total_e2e_ms"]
        results = {}

        for m in metrics:
            values = [entry[m] for entry in self.history if entry[m] > 0]
            if not values:
                results[m] = {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p100": 0.0}
                continue
            
            results[m] = {
                "p50": round(float(np.percentile(values, 50)), 2),
                "p70": round(float(np.percentile(values, 70)), 2),
                "p90": round(float(np.percentile(values, 90)), 2),
                "p100": round(float(np.max(values)), 2),
                "count": len(values)
            }

        return results


telemetry = TelemetryTracker()