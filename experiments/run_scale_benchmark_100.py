"""
================================================================================
EXPERIMENT: LARGE-SCALE N=100 BENCHMARK (KILLING ATTACK #1 & ATTACK #8)
MODEL: DeepSeek-R1-Distill-Qwen-7B
HARDWARE: NVIDIA A100-SXM4 GPU
DATASET: openai/gsm8k (test split, N=100 problems)
METRICS MEASURED:
    1. Answer Accuracy (Exact Match vs Ground Truth)
    2. Total Generated Reasoning Tokens
    3. Real Wall-Clock Hardware Time (GPU Seconds)
    4. Hardware Inference Throughput (Tokens / Second)
    5. Cost & Compute Reduction Percentage
    6. McNemar Paired Chi-Squared Test & p-value
================================================================================
"""

import gc
import json
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ------------------------------------------------------------------------------
# 1. ANSWER PARSER
# ------------------------------------------------------------------------------
def extract_answer(text: str):
    patterns = [
        r"\\boxed\{[^\d]*(-?\d+(?:,\d+)*(?:\.\d+)?)[^\d]*\}",
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[-+]?\d*\.\d+|\d+"
    ]
    for p in patterns[:3]:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "").strip()
    nums = re.findall(patterns[3], text.replace(",", ""))
    return nums[-1] if nums else None

# ------------------------------------------------------------------------------
# 2. LATENT TRAJECTORY GOVERNOR
# ------------------------------------------------------------------------------
class LatentVelocityGovernor:
    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = 14,
        velocity_threshold: float = 0.06,
        min_deduction_tokens: int = 100,
        consensus_patience: int = 3
    ):
        self.model = model
        self.target_layer = model.model.layers[target_layer_idx]
        self.velocity_threshold = velocity_threshold
        self.min_deduction_tokens = min_deduction_tokens
        self.consensus_patience = consensus_patience

        self.trajectory = []
        self.low_velocity_streak = 0
        self.stop_triggered = False
        self.tokens_generated = 0
        self.hook_handle = None

    def reset(self):
        self.trajectory.clear()
        self.low_velocity_streak = 0
        self.stop_triggered = False
        self.tokens_generated = 0

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        if hidden.shape[1] > 1:
            return outputs

        self.tokens_generated += 1
        current_h = hidden[:, -1, :].detach().float()

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)

            if self.tokens_generated >= self.min_deduction_tokens:
                if v_t < self.velocity_threshold:
                    self.low_velocity_streak += 1
                    if self.low_velocity_streak >= self.consensus_patience:
                        self.stop_triggered = True
                else:
                    self.low_velocity_streak = 0
        self.trajectory.append(current_h)
        return outputs

    def attach(self):
        self.reset()
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class SubNeutralizeGovernorCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, governor: LatentVelocityGovernor, prompt_len: int, min_tokens: int = 120):
        self.tokenizer = tokenizer
        self.governor = governor
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        curr = input_ids.shape[-1] - self.prompt_len
        if curr < self.min_tokens:
            return False
        recent = self.tokenizer.decode(input_ids[0, -15:], skip_special_tokens=False)
        if "</think>" in recent:
            return True
        if self.governor.stop_triggered:
            return True
        if "Wait" in recent or "Alternatively" in recent:
            return True
        return False

# ------------------------------------------------------------------------------
# 3. BENCHMARK RUNNER (N=100)
# ------------------------------------------------------------------------------
def run_large_scale_benchmark(num_problems: int = 100):
    print("=" * 95)
    print(f"LARGE-SCALE VALIDATION BENCHMARK (N={num_problems} PROBLEMS)")
    print("Model: DeepSeek-R1-Distill-Qwen-7B | Hardware: NVIDIA A100 GPU")
    print("Target: Defeat Reviewer Attack #1 (Scale) & Attack #8 (Wall-Clock GPU Latency)")
    print("=" * 95)

    if 'model' not in globals() or model is None:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
    else:
        print("Reusing active DeepSeek-R1-Distill-Qwen-7B in A100 VRAM.")

    governor = LatentVelocityGovernor(model=model, target_layer_idx=14, velocity_threshold=0.06, min_deduction_tokens=100)
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{num_problems}]")

    print("\n" + "=" * 105)
    print(f"{'#':<4} | {'Vanilla Tk (Time)':<20} | {'Neut Tk (Time)':<20} | {'Tokens Saved':<15} | {'Outcome'}")
    print("-" * 105)

    results = []
    total_van_tokens = 0
    total_neut_tokens = 0
    total_van_sec = 0.0
    total_neut_sec = 0.0
    van_correct = 0
    neut_correct = 0
    flips_prevented = 0
    van_wins = 0

    checkpoint_file = "benchmark_scale_100_checkpoint.json"

    for idx, sample in enumerate(dataset):
        q = sample["question"]
        gt = extract_answer(sample["answer"])

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # -------------------------------------------------------------
        # PASS 1: VANILLA GENERATION
        # -------------------------------------------------------------
        governor.detach()
        torch.cuda.synchronize()
        t0_van = time.time()
        with torch.no_grad():
            out_van = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        dt_van = time.time() - t0_van
        van_tk = out_van.shape[1] - prompt_len
        van_ans = extract_answer(tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=True))
        van_match = (van_ans == gt)

        # -------------------------------------------------------------
        # PASS 2: SUBNEUTRALIZE GENERATION
        # -------------------------------------------------------------
        governor.attach()
        torch.cuda.synchronize()
        t0_neut = time.time()
        stop_crit = StoppingCriteriaList([SubNeutralizeGovernorCriteria(tokenizer, governor, prompt_len, min_tokens=120)])
        with torch.no_grad():
            out_neut = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                stopping_criteria=stop_crit,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        dt_neut = time.time() - t0_neut
        neut_tk = out_neut.shape[1] - prompt_len
        neut_ans = extract_answer(tokenizer.decode(out_neut[0, prompt_len:], skip_special_tokens=True))
        neut_match = (neut_ans == gt)
        governor.detach()

        # Accumulate metrics
        total_van_tokens += van_tk
        total_neut_tokens += neut_tk
        total_van_sec += dt_van
        total_neut_sec += dt_neut

        if van_match: van_correct += 1
        if neut_match: neut_correct += 1

        savings_pct = ((van_tk - neut_tk) / max(1, van_tk)) * 100.0

        if neut_match and not van_match:
            outcome = "NEUT_WON (Flip Prevented!)"
            flips_prevented += 1
        elif van_match and not neut_match:
            outcome = "VAN_WON"
            van_wins += 1
        elif van_match and neut_match:
            outcome = "EQUAL"
        else:
            outcome = "BOTH_FAIL"

        print(
            f"{idx+1:<4} | "
            f"{van_tk} tk ({dt_van:.2f}s, {'OK' if van_match else 'X'}):<20 | "
            f"{neut_tk} tk ({dt_neut:.2f}s, {'OK' if neut_match else 'X'}):<20 | "
            f"{savings_pct:>12.1f}% | "
            f"{outcome}"
        )

        results.append({
            "idx": idx + 1,
            "ground_truth": gt,
            "vanilla": {"tokens": van_tk, "sec": round(dt_van, 3), "ans": van_ans, "correct": van_match},
            "subneutralize": {"tokens": neut_tk, "sec": round(dt_neut, 3), "ans": neut_ans, "correct": neut_match},
            "outcome": outcome
        })

        # Save incremental checkpoint every 10 problems
        if (idx + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({"completed": idx + 1, "results": results}, f, indent=2)

    # Final Compilation
    net_tokens_saved = total_van_tokens - total_neut_tokens
    net_token_savings_pct = (net_tokens_saved / total_van_tokens) * 100.0
    net_time_saved_sec = total_van_sec - total_neut_sec
    net_time_savings_pct = (net_time_saved_sec / total_van_sec) * 100.0

    van_acc = (van_correct / num_problems) * 100.0
    neut_acc = (neut_correct / num_problems) * 100.0

    # McNemar's Paired Chi-Squared Test
    # b = flips prevented (neut won), c = van won
    b = flips_prevented
    c = van_wins
    if (b + c) > 0:
        mcnemar_chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
        # Approximate p-value from chi2 with 1 df
        import math
        p_val = math.erfc(math.sqrt(mcnemar_chi2) / math.sqrt(2))
    else:
        mcnemar_chi2 = 0.0
        p_val = 1.0

    print("\n" + "=" * 95)
    print(f"OFFICIAL N={num_problems} LARGE-SCALE BENCHMARK RESULTS")
    print("=" * 95)
    print(f"Total Vanilla Tokens:             {total_van_tokens:,}")
    print(f"Total SubNeutralize Tokens:       {total_neut_tokens:,}")
    print(f"Tokens Saved:                     {net_tokens_saved:,} ({net_token_savings_pct:.1f}%)")
    print("-" * 95)
    print(f"Total Vanilla GPU Time:           {total_van_sec:.1f} seconds")
    print(f"Total SubNeutralize GPU Time:     {total_neut_sec:.1f} seconds")
    print(f"Wall-Clock Latency Saved:         {net_time_saved_sec:.1f} seconds ({net_time_savings_pct:.1f}%)")
    print("-" * 95)
    print(f"Vanilla Accuracy:                 {van_acc:.1f}% ({van_correct}/{num_problems})")
    print(f"SubNeutralize Accuracy:           {neut_acc:.1f}% ({neut_correct}/{num_problems})")
    print(f"Accuracy Improvement:             {'+' if neut_acc >= van_acc else ''}{neut_acc - van_acc:.1f}% absolute")
    print(f"Cognitive Flips Prevented:        {flips_prevented} problems")
    print(f"McNemar Statistical Test:         chi2 = {mcnemar_chi2:.3f}, p = {p_val:.4e}")
    print("=" * 95)

    final_output_file = "paper/tables/benchmark_scale_100_final.json"
    with open(final_output_file, "w") as f:
        json.dump({
            "num_problems": num_problems,
            "total_vanilla_tokens": total_van_tokens,
            "total_neut_tokens": total_neut_tokens,
            "net_token_savings_pct": net_token_savings_pct,
            "total_vanilla_sec": total_van_sec,
            "total_neut_sec": total_neut_sec,
            "net_time_savings_pct": net_time_savings_pct,
            "vanilla_accuracy_pct": van_acc,
            "neut_accuracy_pct": neut_acc,
            "flips_prevented": flips_prevented,
            "van_wins": van_wins,
            "mcnemar_chi2": mcnemar_chi2,
            "p_value": p_val,
            "results": results
        }, f, indent=2)
    print(f"Full benchmark data saved to: {final_output_file}")

if __name__ == "__main__":
    run_large_scale_benchmark(100)
