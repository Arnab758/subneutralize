"""
================================================================================
TASK 1: 1,200-TOKEN BASELINE CEILING STRESS TEST (REVIEWER ATTACK #2 DEFENSE)
MODEL: DeepSeek-R1-Distill-Qwen-7B
DATASET: GSM8K (Exact NEUT_WON Problems from 50-Problem Benchmark)
PURPOSE:
    Directly refute the reviewer critique:
    "Your 450-token ceiling artificially handicapped the vanilla baseline.
    If you gave Vanilla 1,200 tokens, it would have solved the problem."

    We test Vanilla generation with a massive 1,200-token budget on the exact
    subset of problems where Vanilla truncated at 450 tokens and SubNeutralize won.
================================================================================
"""

import gc
import json
import os
import re
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ------------------------------------------------------------------------------
# 1. ROBUST ANSWER EXTRACTION & STOPPING CRITERIA
# ------------------------------------------------------------------------------
def extract_answer(text: str):
    patterns = [
        r"\\boxed\{[^\d]*(-?\d+(?:,\d+)*(?:\.\d+)?)[^\d]*\}",
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*$"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None

class QwenStoppingCriteria(StoppingCriteria):
    """Stops only when </think> is closed AND the final numerical answer is emitted."""
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.complete_box_pattern = re.compile(r"\\boxed\{[^}]+\}")
        self.hash_pattern = re.compile(r"####\s*-?\d+")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_tokens = input_ids[0, self.prompt_len:]
        if len(gen_tokens) < 20:
            return False
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        if "</think>" in text:
            if self.complete_box_pattern.search(text) or self.hash_pattern.search(text) or "<｜end of sentence｜>" in text:
                return True
        return False

# ------------------------------------------------------------------------------
# 2. EXPERIMENTAL EXECUTION
# ------------------------------------------------------------------------------
def run_ceiling_stress_test(max_tokens_vanilla: int = 1200):
    print("=" * 80)
    print("TASK 1: 1,200-TOKEN CEILING STRESS TEST ON DEEPSEEK-R1-DISTILL-QWEN-7B")
    print(f"Vanilla Token Ceiling: {max_tokens_vanilla} tokens (Extended from 450)")
    print("Testing on exact problems where Vanilla truncated and SubNeutralize won.")
    print("=" * 80)

    # 10 Representative NEUT_WON indices from the 50-problem benchmark
    # (Problems 1, 2, 5, 10, 14, 16, 18, 26, 30, 31)
    target_indices = [1, 2, 5, 10, 14, 16, 18, 26, 30, 31]
    
    # Baseline neutralized token counts recorded in the 50-problem benchmark:
    neut_bench_tokens = {
        1: 157, 2: 306, 5: 357, 10: 157, 14: 394,
        16: 450, 18: 199, 26: 243, 30: 395, 31: 312
    }

    gc.collect()
    torch.cuda.empty_cache()

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"\nLoading {model_id} onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("\nLoading dataset: openai/gsm8k (test split)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    print("\n" + "=" * 90)
    print(f"{'#':<4} | {'Van(450)':<9} | {'NeutTk':<8} | {'Van(1200)Tk':<12} | {'Van(1200) Ans':<15} | {'GT':<8} | {'Status'}")
    print("-" * 90)

    results = []
    total_van_1200 = 0
    total_neut = 0
    van_1200_correct = 0

    for prob_idx in target_indices:
        sample = dataset[prob_idx - 1]
        q = sample["question"]
        gt = extract_answer(sample["answer"])
        neut_tk = neut_bench_tokens.get(prob_idx, 0)
        total_neut += neut_tk

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        stop_crit = StoppingCriteriaList([QwenStoppingCriteria(tokenizer, prompt_len)])

        t0 = time.time()
        with torch.no_grad():
            out_van = model.generate(
                **inputs,
                max_new_tokens=max_tokens_vanilla,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                stopping_criteria=stop_crit,
                pad_token_id=tokenizer.eos_token_id
            )
        dt = time.time() - t0
        van_tk = out_van.shape[1] - prompt_len
        total_van_1200 += van_tk

        van_text = tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=True)
        van_ans = extract_answer(van_text)
        is_corr = (van_ans == gt) if (van_ans and gt) else False

        if is_corr:
            van_1200_correct += 1
            savings_vs_neut = ((van_tk - neut_tk) / max(1, van_tk)) * 100.0
            status = f"CORRECT (+{savings_vs_neut:.0f}% waste)"
        else:
            if van_tk >= max_tokens_vanilla:
                status = "FAILED (Still Truncated at 1200!)"
            else:
                status = "FAILED (Wrong Answer / Relapse)"

        print(f"{prob_idx:<4} | {'450':<9} | {neut_tk:<8} | {van_tk:<12} | {str(van_ans):<15} | {str(gt):<8} | {status}")

        # Scan for overthinking relapse markers
        overthinking_markers = len(re.findall(r"(wait|alternatively|let me check|recalculate|double-check|re-evaluate)", van_text, re.IGNORECASE))

        results.append({
            "problem_idx": prob_idx,
            "vanilla_450_tokens": 450,
            "neutralized_tokens": neut_tk,
            "vanilla_1200_tokens": van_tk,
            "vanilla_1200_answer": van_ans,
            "ground_truth": gt,
            "is_correct": is_corr,
            "status": status,
            "overthinking_marker_count": overthinking_markers,
            "latency_seconds": round(dt, 2),
            "reasoning_snippet_tail": van_text[-250:].replace("\n", " ")
        })

    net_waste = ((total_van_1200 - total_neut) / total_van_1200) * 100.0 if total_van_1200 > 0 else 0

    print("\n" + "=" * 90)
    print("SUMMARY OF 1,200-TOKEN BASELINE STRESS TEST")
    print("=" * 90)
    print(f"Total Tokens Used by Vanilla (1200 ceiling): {total_van_1200}")
    print(f"Total Tokens Used by SubNeutralize:          {total_neut}")
    print(f"Tokens Saved by SubNeutralize:               {total_van_1200 - total_neut} ({net_waste:.1f}%)")
    print(f"Vanilla (1200) Accuracy:                     {van_1200_correct}/{len(target_indices)} ({van_1200_correct/len(target_indices)*100:.1f}%)")
    print(f"SubNeutralize (Original) Accuracy:           10/10 (100.0%)")
    print("=" * 90)

    output_path = "experiments/results_ceiling_stress_test_1200.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model_id,
            "target_indices": target_indices,
            "vanilla_1200_accuracy": van_1200_correct / len(target_indices) * 100.0,
            "neutralized_accuracy": 100.0,
            "total_vanilla_1200_tokens": total_van_1200,
            "total_neutralized_tokens": total_neut,
            "net_compute_waste_pct": net_waste,
            "results": results
        }, f, indent=2)
    print(f"\nDetailed audit trail saved to: {output_path}")

if __name__ == "__main__":
    run_ceiling_stress_test(max_tokens_vanilla=1200)
