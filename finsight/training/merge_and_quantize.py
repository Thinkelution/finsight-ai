"""Merge LoRA adapter into base model and convert to GGUF for local serving.

After running this script, transfer the GGUF file to your Mac mini and
register it in Ollama using scripts/pull_model.sh.
"""

import os
import subprocess
import sys
from pathlib import Path

ADAPTER_DIR = str(Path(__file__).parent / "lora_adapter")
MERGED_DIR = str(Path(__file__).parent / "merged_model")
GGUF_OUTPUT = str(Path(__file__).parent / "finsight_qwen14b_q4.gguf")
LLAMA_CPP_DIR = "llama.cpp"


def merge_adapter():
    """Merge LoRA adapter weights back into the base model."""
    from unsloth import FastLanguageModel

    print("=== Step 1: Merging LoRA adapter ===")

    if not os.path.exists(ADAPTER_DIR):
        print(f"ERROR: Adapter not found at {ADAPTER_DIR}")
        print("Run training first: python -m finsight.training.train_lora")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merge complete.\n")


def quantize_to_gguf():
    """Convert merged HuggingFace model to GGUF Q4_K_M format."""
    print("=== Step 2: Quantizing to GGUF ===")

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"ERROR: llama.cpp not found at {LLAMA_CPP_DIR}")
        print("Clone it: git clone https://github.com/ggerganov/llama.cpp")
        sys.exit(1)

    if not os.path.exists(MERGED_DIR):
        print(f"ERROR: Merged model not found at {MERGED_DIR}")
        print("Run merge first.")
        sys.exit(1)

    cmd = [
        sys.executable,
        convert_script,
        MERGED_DIR,
        "--outtype", "q4_k_m",
        "--outfile", GGUF_OUTPUT,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"GGUF conversion failed:\n{result.stderr}")
        sys.exit(1)

    file_size = os.path.getsize(GGUF_OUTPUT) / (1024 ** 3)
    print(f"GGUF file created: {GGUF_OUTPUT} ({file_size:.1f} GB)")
    print("\nNext steps:")
    print(f"  1. Transfer to Mac: scp {GGUF_OUTPUT} user@mac-mini:~/models/")
    print(f"  2. Register in Ollama: ./scripts/pull_model.sh ~/models/{os.path.basename(GGUF_OUTPUT)}")


def main():
    merge_adapter()
    quantize_to_gguf()
    print("\n=== Merge & quantize complete ===")


if __name__ == "__main__":
    main()
