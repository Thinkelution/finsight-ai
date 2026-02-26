#!/usr/bin/env python3
"""Orchestrate the full historical data collection and training data generation pipeline.

Usage:
    python -m finsight.historical.run_collection --start 2016-01-01 --end 2026-02-26
    python -m finsight.historical.run_collection --step collect   # Only collect raw data
    python -m finsight.historical.run_collection --step build     # Only build training pairs
    python -m finsight.historical.run_collection --step index     # Only index patterns
    python -m finsight.historical.run_collection --step all       # Full pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def step_collect(start: str, end: str):
    """Step 1: Collect raw historical data from all sources."""
    print("\n" + "=" * 60)
    print("  STEP 1: Collecting Historical Data")
    print("=" * 60)

    print("\n--- Yahoo Finance: Market Prices ---")
    from finsight.historical.collectors.yahoo_historical import download_all
    market_df = download_all(start, end)
    print(f"  Market data: {len(market_df)} rows")

    print("\n--- FRED: Economic Indicators ---")
    from finsight.historical.collectors.fred_data import download_all as download_fred
    econ_df = download_fred(start, end)
    print(f"  Economic data: {len(econ_df)} rows")

    print("\n--- Wikipedia: Current Events ---")
    from finsight.historical.collectors.wikipedia_events import collect_range as wiki_collect
    wiki_count = wiki_collect(start, end)
    print(f"  Wikipedia events: {wiki_count}")

    print("\n--- GDELT: News Articles ---")
    from finsight.historical.collectors.gdelt_collector import collect_range as gdelt_collect
    gdelt_count = gdelt_collect(start, end)
    print(f"  GDELT articles: {gdelt_count}")

    print(f"\n  Collection complete!")
    return {
        "market_rows": len(market_df),
        "econ_rows": len(econ_df),
        "wiki_events": wiki_count,
        "gdelt_articles": gdelt_count,
    }


def step_build(start: str, end: str, use_gpt: bool = True):
    """Step 2: Build training dataset from collected data."""
    print("\n" + "=" * 60)
    print("  STEP 2: Building Training Dataset")
    print("=" * 60)

    from finsight.historical.dataset_builder import build_dataset, combine_datasets

    count = build_dataset(start, end, use_gpt=use_gpt)
    print(f"  Generated {count} training pairs")

    combined = combine_datasets()
    print(f"  Combined dataset: {combined}")

    pair_count = sum(1 for _ in open(combined))
    print(f"  Total training examples: {pair_count}")

    return {"pairs": count, "combined_path": str(combined), "total": pair_count}


def step_index():
    """Step 3: Index historical patterns into Qdrant for similarity search."""
    print("\n" + "=" * 60)
    print("  STEP 3: Indexing Historical Patterns")
    print("=" * 60)

    from finsight.historical.pattern_matcher import index_historical_patterns

    count = index_historical_patterns()
    print(f"  Indexed {count} patterns")

    return {"indexed": count}


def step_test():
    """Step 4: Test the prediction engine."""
    print("\n" + "=" * 60)
    print("  STEP 4: Testing Predictions")
    print("=" * 60)

    from finsight.historical.trend_predictor import predict_trends

    test_scenarios = [
        "Federal Reserve raises interest rates by 25 basis points. "
        "Inflation data comes in above expectations at 3.5%. "
        "Oil prices surge due to Middle East tensions.",

        "Major tech companies report earnings above expectations. "
        "AI chip demand surges. NVIDIA stock hits new all-time high. "
        "Consumer spending remains strong despite rate environment.",

        "Trade tensions escalate between US and China. "
        "New tariffs announced on semiconductor imports. "
        "Global supply chain disruptions reported.",
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test Scenario {i} ---")
        print(f"  Context: {scenario[:80]}...")

        try:
            result = predict_trends(scenario)
            print(f"  Confidence: {result['confidence']}%")
            print(f"  Parallels found: {len(result['parallels'])}")
            for pred in result.get("predictions", [])[:3]:
                print(f"    {pred['asset']}: {pred['direction']} "
                      f"({pred['confidence']}%)")
        except Exception as e:
            print(f"  Error: {e}")

    return {"tested": len(test_scenarios)}


def main():
    parser = argparse.ArgumentParser(
        description="FinSight Historical Data Pipeline"
    )
    parser.add_argument(
        "--start", default="2016-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default="2026-02-26",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--step", default="all",
        choices=["collect", "build", "index", "test", "all"],
        help="Which pipeline step to run",
    )
    parser.add_argument(
        "--no-gpt", action="store_true",
        help="Use template-based analysis instead of GPT-4o-mini",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  FinSight AI — Historical Data Pipeline")
    print("=" * 60)
    print(f"  Date range: {args.start} → {args.end}")
    print(f"  Step: {args.step}")
    print(f"  GPT analysis: {'OFF' if args.no_gpt else 'ON'}")

    start_time = time.time()
    results = {}

    try:
        if args.step in ("collect", "all"):
            results["collect"] = step_collect(args.start, args.end)

        if args.step in ("build", "all"):
            results["build"] = step_build(
                args.start, args.end, use_gpt=not args.no_gpt
            )

        if args.step in ("index", "all"):
            results["index"] = step_index()

        if args.step in ("test", "all"):
            results["test"] = step_test()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved. Re-run to resume.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    for step_name, step_result in results.items():
        print(f"  {step_name}: {step_result}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
