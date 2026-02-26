"""LoRA fine-tuning script for Qwen 2.5 14B using Unsloth + QLoRA.

Designed to run on RunPod GPU instances (RTX 4090 24GB VRAM recommended).
"""

import os
import sys
from pathlib import Path

MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/Qwen2.5-14B-bnb-4bit"
DATASET_PATH = str(Path(__file__).parent / "datasets" / "formatted" / "financial_qa.jsonl")
OUTPUT_DIR = str(Path(__file__).parent / "checkpoints")
ADAPTER_DIR = str(Path(__file__).parent / "lora_adapter")


def train():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print(f"=== FinSight LoRA Training ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Run: python -m finsight.training.prepare_dataset")
        sys.exit(1)

    print("\n1. Loading base model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("2. Attaching LoRA adapter...")
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
    print(f"   Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    print("3. Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files=DATASET_PATH,
        split="train",
    )
    print(f"   Examples: {len(dataset)}")

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

    print("4. Starting training...")
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
            fp16=True,
            output_dir=OUTPUT_DIR,
            save_steps=100,
            logging_steps=10,
            save_total_limit=3,
            report_to="none",
        ),
    )

    stats = trainer.train()
    print(f"\n5. Training complete!")
    print(f"   Loss: {stats.training_loss:.4f}")
    print(f"   Steps: {stats.global_step}")
    print(f"   Runtime: {stats.metrics['train_runtime']:.0f}s")

    print(f"\n6. Saving LoRA adapter to {ADAPTER_DIR}...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("=== Training complete ===")


if __name__ == "__main__":
    train()
