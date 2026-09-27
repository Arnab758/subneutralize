"""
Side-by-side benchmark evaluation script: Vanilla vs. SubNeutralize.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import ConsensusEntropyGovernor

SAMPLE_PROBLEMS = [
    "A bakery sells cakes for $18 each and cookies for $3 each. If Sarah buys 4 cakes and 12 cookies, how much does she spend in total?",
    "Find the roots of the quadratic equation 2x^2 - 8x + 6 = 0.",
    "If a train travels at 90 km/h for 2.5 hours and then at 60 km/h for 1.5 hours, what is its average speed for the entire journey?",
]

def run_benchmark(model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing benchmark on {device} using {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    governor = ConsensusEntropyGovernor(model, tokenizer)

    total_vanilla_tokens = 0
    total_governed_tokens = 0

    print("\n" + "=" * 80)
    print(f"{'PROMPT ID':<10} | {'VANILLA TOKENS':<16} | {'SUBNEUTRALIZE':<16} | {'REDUCTION':<10}")
    print("=" * 80)

    for i, prompt in enumerate(SAMPLE_PROBLEMS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[-1]

        # 1. Vanilla Run
        with torch.no_grad():
            out_vanilla = model.generate(**inputs, max_new_tokens=1024)
        tokens_vanilla = out_vanilla.shape[-1] - prompt_len
        total_vanilla_tokens += tokens_vanilla

        # 2. SubNeutralize Run
        governor.attach()
        with torch.no_grad():
            out_gov = model.generate(
                **inputs,
                max_new_tokens=1024,
                stopping_criteria=[governor.as_stopping_criteria()],
                return_dict_in_generate=True,
                output_scores=True
            )
        governor.detach()
        tokens_gov = out_gov.sequences[0].shape[-1] - prompt_len
        total_governed_tokens += tokens_gov

        reduction = (1.0 - tokens_gov / max(1, tokens_vanilla)) * 100.0
        print(f"Problem {i+1:<3} | {tokens_vanilla:<16} | {tokens_gov:<16} | {reduction:>6.1f}%")

    overall_reduction = (1.0 - total_governed_tokens / max(1, total_vanilla_tokens)) * 100.0
    print("=" * 80)
    print(f"TOTAL:      | {total_vanilla_tokens:<16} | {total_governed_tokens:<16} | {overall_reduction:>6.1f}% SAVINGS\n")

if __name__ == "__main__":
    run_benchmark()
