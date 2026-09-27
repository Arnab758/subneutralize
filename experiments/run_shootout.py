"""
Phase 2 Side-by-Side Shootout:
Compares Vanilla Reasoning Generation vs. Subspace Neutralized Generation
across token count, latency, and answer correctness.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hooks import ResidualStreamProbe
from src.bias_subspace import BiasSubspaceAnalyzer
from src.neutralizer import SubspaceNeutralizationController
from src.benchmark_loader import load_diagnostic_probes, format_r1_prompt


def run_shootout(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 60)
    print("PHASE 2: SIDE-BY-SIDE SHOOTOUT (Vanilla vs. Subspace Neutralized)")
    print(f"Model: {model_name} on {device}")
    print("=" * 60)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    probes = load_diagnostic_probes()
    test_cases = probes[:3]  # Test top 3 cognitive bias problems

    results = []

    for idx, test in enumerate(test_cases, 1):
        print(f"\n[{idx}/3] Testing: {test['id']}")
        print(f"Question: {test['question']}")
        print(f"Ground Truth: {test['ground_truth']} | Documented Bias: {test['intuitive_bias']}")

        prompt = format_r1_prompt(test['question'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[-1]

        # ---------------------------------------------------------
        # RUN A: VANILLA (Uncontrolled / Standard Generation)
        # ---------------------------------------------------------
        print("\n  -> Running Vanilla Generation...")
        with torch.no_grad():
            vanilla_out = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        vanilla_tokens = vanilla_out[0].shape[-1] - prompt_len
        vanilla_text = tokenizer.decode(vanilla_out[0][prompt_len:], skip_special_tokens=True)

        # ---------------------------------------------------------
        # RUN B: SUBSPACE NEUTRALIZED (Intervention)
        # ---------------------------------------------------------
        print("  -> Running Subspace Neutralized Generation...")
        controller = SubspaceNeutralizationController(
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            velocity_threshold=0.04,
            min_reasoning_tokens=35,
            max_reflection_tokens=80
        )

        with torch.no_grad():
            neut_out = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                stopping_criteria=[controller],
                pad_token_id=tokenizer.eos_token_id
            )
        neut_tokens = neut_out[0].shape[-1] - prompt_len
        neut_text = tokenizer.decode(neut_out[0][prompt_len:], skip_special_tokens=True)

        # Calculate Token Savings
        saved_tokens = vanilla_tokens - neut_tokens
        savings_pct = (saved_tokens / vanilla_tokens) * 100 if vanilla_tokens > 0 else 0.0

        print(f"\n  [RESULTS FOR {test['id']}]:")
        print(f"  * Vanilla Tokens:     {vanilla_tokens}")
        print(f"  * Neutralized Tokens: {neut_tokens}")
        print(f"  * Tokens Saved:       {saved_tokens} ({savings_pct:.1f}%)")

        results.append({
            "probe_id": test["id"],
            "vanilla_tokens": vanilla_tokens,
            "neut_tokens": neut_tokens,
            "savings_pct": savings_pct,
            "vanilla_preview": vanilla_text[-150:].replace("\n", " "),
            "neut_preview": neut_text[-150:].replace("\n", " ")
        })

    # Summary Benchmark Table
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SHOOTOUT SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Probe ID':<25} | {'Vanilla':<10} | {'Neutralized':<12} | {'Savings (%)':<12}")
    print("-" * 70)
    total_v = 0
    total_n = 0
    for r in results:
        print(f"{r['probe_id']:<25} | {r['vanilla_tokens']:<10} | {r['neut_tokens']:<12} | {r['savings_pct']:>10.1f}%")
        total_v += r["vanilla_tokens"]
        total_n += r["neut_tokens"]
    
    total_savings = ((total_v - total_n) / total_v) * 100 if total_v > 0 else 0.0
    print("-" * 70)
    print(f"{'MEAN TOTAL':<25} | {total_v:<10} | {total_n:<12} | {total_savings:>10.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_shootout()
