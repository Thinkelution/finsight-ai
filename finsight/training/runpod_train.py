#!/usr/bin/env python3
"""Self-contained training script designed to run end-to-end on a RunPod instance.

This script:
1. Installs required packages
2. Prepares the training dataset (downloads from HuggingFace + manual pairs)
3. Runs LoRA fine-tuning on Qwen 2.5 14B
4. Merges the adapter and quantizes to GGUF
5. Uploads results to persistent storage

Usage on RunPod:
    python runpod_train.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path("/workspace")
REPO_DIR = WORKSPACE / "finsight-ai"
DATASET_DIR = WORKSPACE / "datasets"
DATASET_FILE = DATASET_DIR / "financial_qa.jsonl"
CHECKPOINT_DIR = WORKSPACE / "checkpoints"
ADAPTER_DIR = WORKSPACE / "lora_adapter"
MERGED_DIR = WORKSPACE / "merged_model"
GGUF_OUTPUT = WORKSPACE / "finsight_qwen14b_q4.gguf"

MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/Qwen2.5-14B-bnb-4bit"


def run_cmd(cmd, check=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def step_install():
    """Install all required packages."""
    print("\n" + "=" * 60)
    print("STEP 1: Installing packages")
    print("=" * 60)

    run_cmd("pip install --upgrade pip")
    run_cmd(
        "pip install 'unsloth[colab-new]' trl peft datasets bitsandbytes "
        "transformers accelerate sentencepiece huggingface_hub"
    )


def step_prepare_dataset():
    """Download HuggingFace datasets and create training JSONL."""
    print("\n" + "=" * 60)
    print("STEP 2: Preparing training dataset")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    all_examples = []

    # Manual financial reasoning pairs
    manual_pairs = [
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
            "output": "The Japanese yen (JPY) typically strengthens during risk-off episodes due to three mechanisms: (1) Carry trade unwinding — investors who borrowed in low-yielding JPY to invest in higher-yielding assets reverse positions, buying back JPY. (2) Japan's massive net foreign asset position (~$3.3T) means Japanese investors repatriate foreign earnings to JPY during uncertainty. (3) JPY is perceived as a safe-haven currency due to Japan's current account surplus, deep government bond market, and low inflation.",
        },
        {
            "instruction": "How does a surprise CPI print above expectations affect FX markets?",
            "input": "US CPI released at 4.1% vs 3.8% expected. Core CPI 3.9% vs 3.6% expected.",
            "output": "A surprise upside CPI print strengthens the currency of the reporting country through the interest rate channel: (1) Markets immediately reprice rate cut expectations lower / rate hike expectations higher. (2) Short-term interest rate futures adjust, widening yield differentials. (3) The USD strengthens as Treasury yields rise on hawkish repricing. In this case, CPI at 4.1% vs 3.8% expected is a significant beat (+0.3pp). Expected impact: USD up 0.5-1%, US 2-year yield up 10-15bps, equity indices sell off 0.5-1% on tighter financial conditions.",
        },
        {
            "instruction": "What does an inverted yield curve signal about economic expectations?",
            "input": "",
            "output": "An inverted yield curve (short-term rates above long-term rates) has historically been one of the most reliable recession predictors. The inversion signals that: (1) The bond market expects the central bank will need to cut rates in the future due to economic weakness. (2) Short rates are high because the central bank is fighting inflation, while long rates are lower because the market sees a slowdown ahead. (3) The 2yr-10yr spread inversion has preceded every US recession since 1969, typically by 6-18 months.",
        },
        {
            "instruction": "Why did EUR/USD fall when the ECB cut rates faster than the Fed?",
            "input": "ECB cut rates by 25bps to 3.75%. Fed held rates at 5.25-5.50%. EUR/USD dropped from 1.0950 to 1.0870.",
            "output": "EUR/USD fell because the rate differential widened in the USD's favour. With the ECB at 3.75% and the Fed at 5.25-5.50%, the 150bps+ gap makes USD-denominated assets more attractive for carry trades. Capital flows toward higher yields, increasing USD demand. The 0.75% EUR/USD decline reflects: (1) immediate repricing of carry returns, (2) signal that ECB sees European growth weakness requiring stimulus while US economy remains resilient, (3) options market hedging as EUR put demand increases.",
        },
        {
            "instruction": "What is the carry trade and which currency pairs are typically involved?",
            "input": "",
            "output": "The carry trade involves borrowing in a low-interest-rate currency to invest in a high-interest-rate currency, profiting from the rate differential. Classic carry trade pairs: (1) AUD/JPY — borrowing in JPY (~0-0.25%) to invest in AUD (~4.35%), (2) NZD/JPY — similar dynamic with New Zealand's higher rates, (3) USD/JPY — post-2022 with Fed funds at 5%+ vs BoJ at 0%, (4) MXN/JPY — Mexico's high rates (~11%) vs Japan. Risks: carry trades blow up during risk-off events when JPY strengthens rapidly (carry trade unwind), causing cascading losses.",
        },
        {
            "instruction": "Analyze the impact of a Fed rate hold when markets expected a cut",
            "input": "Fed held rates at 5.25-5.50%. CME FedWatch had priced in 80% probability of a 25bps cut. Dot plot shifted hawkish with median showing only 1 cut in 2024 vs 3 previously.",
            "output": "This is a hawkish surprise with significant market implications: (1) USD strength — the dollar rallies as the rate differential stays wide for longer than expected. DXY likely +0.5-1%. (2) Equity selloff — S&P 500 and NASDAQ likely drop 1-2% as the discount rate remains elevated. (3) Bond market — short-end yields spike (2yr up 15-25bps) while long-end less affected. (4) Gold weakens on higher real yields. (5) EM currencies under pressure as the carry trade calculus shifts.",
        },
    ]

    all_examples.extend(manual_pairs)
    print(f"  Manual pairs: {len(manual_pairs)}")

    # FinGPT FiQA dataset
    print("  Downloading FinGPT FiQA...")
    try:
        ds = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")
        count = 0
        for row in ds:
            instruction = row.get("input", row.get("question", ""))
            output = row.get("output", row.get("answer", ""))
            if instruction and output and len(output) > 20:
                all_examples.append({
                    "instruction": instruction.strip(),
                    "input": "",
                    "output": output.strip(),
                })
                count += 1
        print(f"    FinGPT FiQA: {count} examples")
    except Exception as e:
        print(f"    FinGPT download failed: {e}")

    # Financial PhraseBank
    print("  Downloading Financial PhraseBank...")
    try:
        ds = load_dataset("financial_phrasebank", "sentences_50agree", split="train")
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        count = 0
        for row in ds:
            sentence = row.get("sentence", "")
            label = label_map.get(row.get("label", 1), "neutral")
            if sentence and len(sentence) > 20:
                all_examples.append({
                    "instruction": "What is the financial sentiment of this statement?",
                    "input": sentence.strip(),
                    "output": f"The sentiment is {label}. {sentence}",
                })
                count += 1
        print(f"    PhraseBank: {count} examples")
    except Exception as e:
        print(f"    PhraseBank failed: {e}")

    # FinQA
    print("  Downloading FinQA...")
    try:
        ds = load_dataset("dreamerdeo/finqa", split="train")
        count = 0
        for row in ds:
            question = row.get("question", "")
            answer = row.get("answer", "")
            if question and answer:
                all_examples.append({
                    "instruction": question.strip(),
                    "input": row.get("context", "").strip()[:1000],
                    "output": str(answer).strip(),
                })
                count += 1
        print(f"    FinQA: {count} examples")
    except Exception as e:
        print(f"    FinQA failed: {e}")

    # Write JSONL
    valid = [e for e in all_examples if e.get("output")]
    with open(DATASET_FILE, "w") as f:
        for ex in valid:
            record = {
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"\n  Total training examples: {len(valid)}")
    print(f"  Saved to: {DATASET_FILE}")
    return len(valid)


def step_train():
    """Run LoRA fine-tuning."""
    print("\n" + "=" * 60)
    print("STEP 3: LoRA Fine-Tuning")
    print("=" * 60)

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    if not DATASET_FILE.exists():
        print(f"ERROR: Dataset not found at {DATASET_FILE}")
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Model: {MODEL_NAME}")
    print(f"  Dataset: {DATASET_FILE}")

    print("\n  Loading base model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("  Attaching LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    print("  Loading dataset...")
    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    print(f"  Examples: {len(dataset)}")

    def format_prompt(example):
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        if inp:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}<|endoftext|>"
            )
        else:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}<|endoftext|>"
            )
        return {"text": text}

    dataset = dataset.map(format_prompt)

    print("  Starting training...")
    start_time = time.time()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            output_dir=str(CHECKPOINT_DIR),
            save_steps=100,
            logging_steps=10,
            save_total_limit=3,
            report_to="none",
        ),
    )

    stats = trainer.train()
    elapsed = time.time() - start_time

    print(f"\n  Training complete!")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Steps: {stats.global_step}")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    print(f"\n  Saving LoRA adapter to {ADAPTER_DIR}...")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))


def step_merge_and_quantize():
    """Merge LoRA adapter and convert to GGUF."""
    print("\n" + "=" * 60)
    print("STEP 4: Merge & Quantize to GGUF")
    print("=" * 60)

    from unsloth import FastLanguageModel

    if not ADAPTER_DIR.exists():
        print(f"ERROR: Adapter not found at {ADAPTER_DIR}")
        return

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"  Saving merged model to {MERGED_DIR}...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )

    # Try GGUF conversion via llama.cpp
    llama_cpp = WORKSPACE / "llama.cpp"
    if not llama_cpp.exists():
        print("  Cloning llama.cpp for GGUF conversion...")
        run_cmd(f"cd {WORKSPACE} && git clone https://github.com/ggerganov/llama.cpp", check=False)
        if llama_cpp.exists():
            run_cmd(f"cd {llama_cpp} && pip install -r requirements.txt", check=False)

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if convert_script.exists():
        print(f"  Converting to GGUF Q4_K_M...")
        ret = run_cmd(
            f"python {convert_script} {MERGED_DIR} --outtype q4_k_m --outfile {GGUF_OUTPUT}",
            check=False,
        )
        if ret == 0 and GGUF_OUTPUT.exists():
            size_gb = GGUF_OUTPUT.stat().st_size / (1024 ** 3)
            print(f"  GGUF created: {GGUF_OUTPUT} ({size_gb:.1f} GB)")
        else:
            print("  GGUF conversion failed — adapter and merged model still available")
    else:
        print("  llama.cpp not available, skipping GGUF. Adapter saved for manual conversion.")


def main():
    print("=" * 60)
    print("  FinSight AI - RunPod Training Pipeline")
    print("=" * 60)
    print(f"  Workspace: {WORKSPACE}")
    print(f"  GPU: ", end="")
    run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)

    step_install()
    count = step_prepare_dataset()
    if count < 10:
        print("WARNING: Very few training examples. Consider adding more data.")
    step_train()
    step_merge_and_quantize()

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Adapter:     {ADAPTER_DIR}")
    print(f"  Merged:      {MERGED_DIR}")
    print(f"  GGUF:        {GGUF_OUTPUT}")
    print(f"  Dataset:     {DATASET_FILE}")
    print("\nNext: scp the GGUF file to your Mac and register in Ollama:")
    print(f"  scp root@<pod-ip>:{GGUF_OUTPUT} ~/models/")
    print("  ./finsight/scripts/pull_model.sh ~/models/finsight_qwen14b_q4.gguf")


if __name__ == "__main__":
    main()
