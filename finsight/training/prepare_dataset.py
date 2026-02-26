"""Convert raw financial text and datasets into instruction-input-output JSONL format.

Supports:
- HuggingFace datasets (FinGPT, FiQA, FinQA)
- Custom Q&A from scraped news + market reactions
- Manual instruction pairs
"""

import json
import os
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
FORMATTED_DIR = DATASETS_DIR / "formatted"
OUTPUT_FILE = FORMATTED_DIR / "financial_qa.jsonl"


def download_huggingface_datasets():
    """Download and format public financial QA datasets from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    all_examples = []

    print("Downloading FinGPT FiQA QA dataset...")
    try:
        ds = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")
        for row in ds:
            instruction = row.get("input", row.get("question", ""))
            output = row.get("output", row.get("answer", ""))
            if instruction and output and len(output) > 20:
                all_examples.append({
                    "instruction": instruction.strip(),
                    "input": "",
                    "output": output.strip(),
                    "source": "fingpt_fiqa",
                })
        print(f"  FinGPT: {len([e for e in all_examples if e['source'] == 'fingpt_fiqa'])} examples")
    except Exception as e:
        print(f"  FinGPT download failed: {e}")

    print("Downloading FiQA dataset...")
    try:
        ds = load_dataset("financial_phrasebank", "sentences_50agree", split="train")
        for row in ds:
            sentence = row.get("sentence", "")
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            label = label_map.get(row.get("label", 1), "neutral")
            if sentence and len(sentence) > 20:
                all_examples.append({
                    "instruction": "What is the financial sentiment of this statement?",
                    "input": sentence.strip(),
                    "output": f"The sentiment is {label}. {sentence}",
                    "source": "financial_phrasebank",
                })
        print(f"  Phrasebank: {len([e for e in all_examples if e['source'] == 'financial_phrasebank'])} examples")
    except Exception as e:
        print(f"  Financial phrasebank download failed: {e}")

    print("Downloading FinQA dataset...")
    try:
        ds = load_dataset("dreamerdeo/finqa", split="train")
        for row in ds:
            question = row.get("question", "")
            answer = row.get("answer", "")
            if question and answer:
                all_examples.append({
                    "instruction": question.strip(),
                    "input": row.get("context", "").strip()[:1000],
                    "output": str(answer).strip(),
                    "source": "finqa",
                })
        print(f"  FinQA: {len([e for e in all_examples if e['source'] == 'finqa'])} examples")
    except Exception as e:
        print(f"  FinQA download failed: {e}")

    return all_examples


def generate_market_event_pairs() -> list[dict]:
    """Generate instruction pairs from known major market events."""
    return [
        {
            "instruction": "Explain why the US dollar strengthened today",
            "input": "Fed minutes released hawkish tone. US CPI came in at 3.4%, above 3.1% forecast. BoJ Governor speech was dovish. EUR/USD down 0.8%.",
            "output": "The USD strengthened across the board today driven by two key factors: (1) Hawkish Fed minutes signalled fewer rate cuts than markets priced in, boosting USD demand. (2) US CPI at 3.4% surprised to the upside, reinforcing the Fed's cautious stance. EUR/USD fell 0.8% as the ECB is on a faster cutting path than the Fed — widening the rate differential in USD's favour.",
        },
        {
            "instruction": "What is the relationship between gold and real yields?",
            "input": "",
            "output": "Gold has a strong inverse relationship with US real yields (the 10yr Treasury yield minus inflation expectations). When real yields fall — either because nominal yields drop or inflation expectations rise — the opportunity cost of holding non-yielding gold falls, making it more attractive. The 2020-2022 gold surge corresponded with deeply negative real yields. The 2022-2023 gold selloff aligned with the Fed hiking rates faster than inflation, driving real yields sharply positive.",
        },
        {
            "instruction": "Explain the relationship between DXY strength and emerging market currencies",
            "input": "",
            "output": "A stronger US Dollar Index (DXY) typically weakens emerging market (EM) currencies. Many EM nations have dollar-denominated debt, so a stronger USD increases their debt servicing costs, triggering capital outflows. Additionally, a rising DXY often reflects tighter US monetary policy, which reduces global liquidity and risk appetite — both negative for EM assets. Key transmission channels: (1) trade deficit widening for commodity importers, (2) foreign debt burden increase, (3) portfolio rebalancing away from EM into safer US assets.",
        },
        {
            "instruction": "What typically happens to JPY during risk-off environments and why?",
            "input": "",
            "output": "The Japanese yen (JPY) typically strengthens during risk-off episodes due to three mechanisms: (1) Carry trade unwinding — investors who borrowed in low-yielding JPY to invest in higher-yielding assets reverse positions, buying back JPY. (2) Japan's massive net foreign asset position (~$3.3T) means Japanese investors repatriate foreign earnings to JPY during uncertainty. (3) JPY is perceived as a safe-haven currency due to Japan's current account surplus, deep government bond market, and low inflation. Classic examples: COVID crash (March 2020) saw USD/JPY drop from 112 to 101.",
        },
        {
            "instruction": "How does a surprise CPI print above expectations affect FX markets?",
            "input": "US CPI released at 4.1% vs 3.8% expected. Core CPI 3.9% vs 3.6% expected.",
            "output": "A surprise upside CPI print strengthens the currency of the reporting country through the interest rate channel: (1) Markets immediately reprice rate cut expectations lower / rate hike expectations higher. (2) Short-term interest rate futures adjust, widening yield differentials. (3) The USD strengthens as Treasury yields rise on hawkish repricing. In this case, CPI at 4.1% vs 3.8% expected is a significant beat (+0.3pp). Expected impact: USD up 0.5-1%, US 2-year yield up 10-15bps, equity indices sell off 0.5-1% on tighter financial conditions. Gold likely weakens on higher real yield expectations.",
        },
        {
            "instruction": "What does an inverted yield curve signal about economic expectations?",
            "input": "",
            "output": "An inverted yield curve (short-term rates above long-term rates) has historically been one of the most reliable recession predictors. The inversion signals that: (1) The bond market expects the central bank will need to cut rates in the future due to economic weakness. (2) Short rates are high because the central bank is fighting inflation, while long rates are lower because the market sees a slowdown ahead. (3) The 2yr-10yr spread inversion has preceded every US recession since 1969, typically by 6-18 months. However, the lag is variable — the curve inverted in July 2022 but the US avoided recession into 2024, suggesting other factors (strong labour market, fiscal spending) can delay the signal.",
        },
        {
            "instruction": "Why did EUR/USD fall when the ECB cut rates faster than the Fed?",
            "input": "ECB cut rates by 25bps to 3.75%. Fed held rates at 5.25-5.50%. EUR/USD dropped from 1.0950 to 1.0870.",
            "output": "EUR/USD fell because the rate differential widened in the USD's favour. With the ECB at 3.75% and the Fed at 5.25-5.50%, the 150bps+ gap makes USD-denominated assets more attractive for carry trades. Capital flows toward higher yields, increasing USD demand. The 0.75% EUR/USD decline reflects: (1) immediate repricing of carry returns, (2) signal that ECB sees European growth weakness requiring stimulus while US economy remains resilient, (3) options market hedging as EUR put demand increases. This divergence theme tends to persist until the Fed also begins cutting.",
        },
        {
            "instruction": "What is the carry trade and which currency pairs are typically involved?",
            "input": "",
            "output": "The carry trade involves borrowing in a low-interest-rate currency to invest in a high-interest-rate currency, profiting from the rate differential. Classic carry trade pairs: (1) AUD/JPY — borrowing in JPY (~0-0.25%) to invest in AUD (~4.35%), (2) NZD/JPY — similar dynamic with New Zealand's higher rates, (3) USD/JPY — post-2022 with Fed funds at 5%+ vs BoJ at 0%, (4) MXN/JPY — Mexico's high rates (~11%) vs Japan. Risks: carry trades blow up during risk-off events when JPY strengthens rapidly (carry trade unwind), causing cascading losses. The August 2024 JPY carry unwind caused a 7% USD/JPY decline in three weeks.",
        },
        {
            "instruction": "Analyze the impact of a Fed rate hold when markets expected a cut",
            "input": "Fed held rates at 5.25-5.50%. CME FedWatch had priced in 80% probability of a 25bps cut. Dot plot shifted hawkish with median showing only 1 cut in 2024 vs 3 previously.",
            "output": "This is a hawkish surprise with significant market implications: (1) USD strength — the dollar rallies as the rate differential stays wide for longer than expected. DXY likely +0.5-1%. (2) Equity selloff — S&P 500 and NASDAQ likely drop 1-2% as the discount rate remains elevated and the 'Fed put' is deferred. (3) Bond market — short-end yields spike (2yr up 15-25bps) while long-end less affected (curve flattens). (4) Gold weakens on higher real yields. (5) EM currencies under pressure as the carry trade calculus shifts. The dot plot revision from 3 cuts to 1 is the key signal — it resets market expectations for the entire rate path, not just the next meeting.",
        },
    ]


def convert_raw_files_to_jsonl():
    """Convert any raw text files in datasets/raw/ to instruction format."""
    examples = []
    raw_dir = RAW_DIR

    if not raw_dir.exists():
        return examples

    for f in raw_dir.glob("*.txt"):
        text = f.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            continue

        examples.append({
            "instruction": "Summarize the key financial insights from this article",
            "input": text[:2000],
            "output": "",  # needs manual annotation
            "source": f"raw_{f.stem}",
            "needs_annotation": True,
        })

    return examples


def write_jsonl(examples: list[dict], output_path: Path | None = None):
    """Write examples to JSONL format."""
    path = output_path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    valid = [e for e in examples if e.get("output") and not e.get("needs_annotation")]
    with open(path, "w") as f:
        for ex in valid:
            record = {
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(valid)} examples to {path}")
    return len(valid)


def prepare_full_dataset():
    """Download all sources and merge into a single training JSONL."""
    print("=== Preparing FinSight Training Dataset ===\n")

    all_examples = []

    print("Step 1: Generating market event pairs...")
    manual = generate_market_event_pairs()
    all_examples.extend(manual)
    print(f"  Manual pairs: {len(manual)}\n")

    print("Step 2: Downloading HuggingFace datasets...")
    hf_examples = download_huggingface_datasets()
    if hf_examples:
        all_examples.extend(hf_examples)
    print()

    print("Step 3: Converting raw files...")
    raw = convert_raw_files_to_jsonl()
    all_examples.extend(raw)
    print(f"  Raw files: {len(raw)}\n")

    print(f"Total examples collected: {len(all_examples)}")
    count = write_jsonl(all_examples)
    print(f"\n=== Dataset preparation complete: {count} training examples ===")
    return count


if __name__ == "__main__":
    prepare_full_dataset()
