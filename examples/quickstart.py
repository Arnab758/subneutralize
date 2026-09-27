"""
SubNeutralize Quickstart Example
================================
Demonstrates how to attach the runtime governor to an autoregressive model in under 15 lines.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import ConsensusEntropyGovernor

def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

    # 1. Initialize and attach the governor
    governor = ConsensusEntropyGovernor(model, tokenizer)
    governor.attach()

    prompt = "Solve for x: 3x + 15 = 42. Show step-by-step reasoning."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating response with SubNeutralize...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        stopping_criteria=[governor.as_stopping_criteria()],
        return_dict_in_generate=True,
        output_scores=True
    )

    # 2. Detach when finished
    governor.detach()

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    tokens_used = outputs.sequences[0].shape[-1] - inputs.input_ids.shape[-1]
    print(f"\nCompleted in {tokens_used} tokens.")
    print(f"Response:\n{generated_text}")

if __name__ == "__main__":
    main()
