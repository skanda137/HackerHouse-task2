"""Generates results/latency_chart.png from results/latency_summary.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "latency_summary.json"), encoding="utf-8") as f:
    data = json.load(f)

summary = data["latency_summary"]
strategies = list(summary.keys())
p50 = [summary[s]["p50_ms"] for s in strategies]
p70 = [summary[s]["p70_ms"] for s in strategies]
p100 = [summary[s]["p100_ms_max"] for s in strategies]

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(strategies))
width = 0.25

bars50 = ax.bar([i - width for i in x], p50, width, label="P50", color="#065A82")
bars70 = ax.bar([i for i in x], p70, width, label="P70", color="#1C7293")
bars100 = ax.bar([i + width for i in x], p100, width, label="P100 (max)", color="#B23A2F")

ax.axhline(200, color="#C9962C", linestyle="--", linewidth=1.5, label="200ms budget")

ax.set_xticks(list(x))
ax.set_xticklabels(strategies, rotation=10)
ax.set_ylabel("Latency (ms)")
ax.set_title("Retrieval Latency by Chunking Strategy (embed + FAISS search)")
ax.legend()
ax.set_ylim(0, max(max(p100), 20) * 1.3)

for bars in (bars50, bars70, bars100):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

fig.tight_layout()
out_path = os.path.join(RESULTS_DIR, "latency_chart.png")
fig.savefig(out_path, dpi=150)
print(f"Saved chart -> {out_path}")
