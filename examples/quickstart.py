"""
SubNeutralize Quickstart Example
================================
Demonstrates how to attach the runtime governor to an autoregressive model in 2 lines.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import SubNeutralize


def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # 1. Initialize SubNeutralize on the model
    governor = SubNeutralize(model, tokenizer)

    prompt = "Write a thread-safe token bucket rate limiter with microsecond refills in Python."
    print("Generating governed response with SubNeutralize...")

    # 2. Run governed inference (stops overthinking automatically)
    output = governor.generate(prompt, max_new_tokens=1024)

    print(f"\nCompleted in {output.total_tokens} tokens ({output.thinking_tokens} thinking + {output.code_tokens} code).")
    print(f"Consensus reached: {output.consensus_reached} at token {output.consensus_step}")
    print(f"Wall-clock time: {output.wall_clock_seconds:.2f}s")
    print(f"\n--- Clean Extracted Code ---\n{output.clean_code}")


if __name__ == "__main__":
    main()
