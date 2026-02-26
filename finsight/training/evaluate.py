"""Evaluation harness for comparing base model vs fine-tuned model.

Runs a set of benchmark financial questions and scores the responses.
"""

import json
import time
from datetime import datetime
from pathlib import Path

EVAL_QUESTIONS = [
    {
        "question": "Explain the relationship between DXY strength and emerging market currencies",
        "category": "forex",
        "expected_keywords": ["dollar", "emerging", "debt", "capital", "outflow"],
    },
    {
        "question": "What typically happens to JPY during risk-off environments and why?",
        "category": "forex",
        "expected_keywords": ["carry", "safe", "haven", "repatriate", "unwind"],
    },
    {
        "question": "How do rising US real yields affect gold prices? Explain the mechanism.",
        "category": "commodities",
        "expected_keywords": ["inverse", "opportunity cost", "treasury", "inflation"],
    },
    {
        "question": "What is the carry trade and which currency pairs are typically involved?",
        "category": "forex",
        "expected_keywords": ["borrow", "low", "interest", "high", "AUD", "JPY"],
    },
    {
        "question": "Explain how a surprise CPI print above expectations affects FX markets",
        "category": "macro",
        "expected_keywords": ["rate", "hawkish", "yield", "strengthen", "reprice"],
    },
    {
        "question": "What does an inverted yield curve signal about economic expectations?",
        "category": "macro",
        "expected_keywords": ["recession", "short", "long", "cut", "slowdown"],
    },
    {
        "question": "Why did EUR/USD fall when the ECB cut rates faster than the Fed?",
        "category": "forex",
        "expected_keywords": ["differential", "carry", "divergence", "capital"],
    },
    {
        "question": "What happens to oil prices when OPEC announces production cuts?",
        "category": "commodities",
        "expected_keywords": ["supply", "price", "increase", "barrel", "production"],
    },
    {
        "question": "How does quantitative tightening differ from rate hikes in market impact?",
        "category": "macro",
        "expected_keywords": ["balance sheet", "liquidity", "bond", "yield", "gradual"],
    },
    {
        "question": "Explain the concept of the Fed put and its implications for equity markets",
        "category": "equities",
        "expected_keywords": ["floor", "intervention", "moral hazard", "volatility", "cut"],
    },
]


def evaluate_model(model_name: str = "finsight-qwen14b") -> dict:
    """Run all evaluation questions against a model and score responses."""
    import ollama as ollama_client

    print(f"=== Evaluating model: {model_name} ===\n")
    results = []

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"Q{i}/{len(EVAL_QUESTIONS)}: {q['question'][:60]}...")

        start = time.time()
        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial markets expert. Answer precisely and cite mechanisms.",
                    },
                    {"role": "user", "content": q["question"]},
                ],
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            answer = response["message"]["content"]
            latency = time.time() - start
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append({
                "question": q["question"],
                "category": q["category"],
                "error": str(e),
                "score": 0,
            })
            continue

        keyword_hits = sum(
            1 for kw in q["expected_keywords"] if kw.lower() in answer.lower()
        )
        keyword_score = keyword_hits / len(q["expected_keywords"])

        length_score = min(len(answer) / 500, 1.0)
        combined_score = keyword_score * 0.7 + length_score * 0.3

        results.append({
            "question": q["question"],
            "category": q["category"],
            "answer_length": len(answer),
            "keyword_hits": keyword_hits,
            "keyword_total": len(q["expected_keywords"]),
            "keyword_score": round(keyword_score, 2),
            "length_score": round(length_score, 2),
            "combined_score": round(combined_score, 2),
            "latency_s": round(latency, 1),
            "answer_preview": answer[:200],
        })

        print(f"   Score: {combined_score:.0%} | Keywords: {keyword_hits}/{len(q['expected_keywords'])} | {latency:.1f}s")

    avg_score = sum(r.get("combined_score", 0) for r in results) / len(results) if results else 0
    avg_latency = sum(r.get("latency_s", 0) for r in results) / len(results) if results else 0

    summary = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "total_questions": len(EVAL_QUESTIONS),
        "average_score": round(avg_score, 2),
        "average_latency_s": round(avg_latency, 1),
        "results": results,
    }

    output_path = Path(__file__).parent / f"eval_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Average Score: {avg_score:.0%}")
    print(f"Average Latency: {avg_latency:.1f}s")
    print(f"Results saved to: {output_path}")

    return summary


def compare_models(base_model: str = "qwen2.5:14b", finetuned_model: str = "finsight-qwen14b"):
    """Compare base model vs fine-tuned model side by side."""
    print("Running comparison evaluation...\n")

    base_results = evaluate_model(base_model)
    print("\n" + "=" * 60 + "\n")
    ft_results = evaluate_model(finetuned_model)

    print("\n" + "=" * 60)
    print("=== COMPARISON ===")
    print(f"Base ({base_model}):      {base_results['average_score']:.0%} avg score, {base_results['average_latency_s']:.1f}s avg latency")
    print(f"Fine-tuned ({finetuned_model}): {ft_results['average_score']:.0%} avg score, {ft_results['average_latency_s']:.1f}s avg latency")
    improvement = ft_results["average_score"] - base_results["average_score"]
    print(f"Improvement: {improvement:+.0%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_models()
    else:
        model = sys.argv[1] if len(sys.argv) > 1 else "finsight-qwen14b"
        evaluate_model(model)
