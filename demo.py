"""
Standalone demo: build the pipeline on synthetic MSMARCO-XI-shaped data and
run a few example queries (grounded + off-topic) so you can see the
structured QueryOutcome contract this hands off to generation.

    python3 demo.py
"""
from data.load_msmarco_xi import load_synthetic
from embeddings.embedder import TfidfEmbedder
from pipeline import RetrievalPipeline

if __name__ == "__main__":
    docs = load_synthetic(docs_per_language=150)
    pipeline = RetrievalPipeline(embedder=TfidfEmbedder())
    pipeline.build_all(docs)

    demo_queries = [
        ("भारत की राजधानी क्या है?", "hi"),                # grounded (Hindi: what is India's capital)
        ("গঙ্গা নদী কোথায় প্রবাহিত হয়?", "bn"),              # grounded (Bengali: where does the Ganges flow)
        ("இமயமலை பற்றி சொல்லுங்கள்", "ta"),                  # grounded (Tamil: tell me about the Himalayas)
        ("What is the best pizza topping combination?", None),  # off-topic
    ]

    for query, lang in demo_queries:
        outcome = pipeline.query(query, strategy="metadata_aware", k=3, language=lang)
        print("=" * 80)
        print(f"Query: {query}")
        print(f"should_answer: {outcome.verdict.should_answer}  (reason: {outcome.verdict.reason})")
        print(f"top_score: {outcome.verdict.top_score}")
        print(f"latency: embed={outcome.embed_ms:.3f}ms search={outcome.search_ms:.3f}ms "
              f"total={outcome.total_ms:.3f}ms")
        if outcome.verdict.should_answer:
            print("Grounded chunks handed to generation:")
            for c in outcome.verdict.grounded_chunks[:3]:
                print(f"  - [{c.language}] {c.text[:90]}...")
        print()
