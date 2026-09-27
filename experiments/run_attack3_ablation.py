"""
================================================================================
KILLING REVIEWER ATTACK #3: THE MECHANISTIC GOVERNOR ABLATION
MODEL: DeepSeek-R1-Distill-Qwen-7B
PURPOSE:
    Directly refute the critique:
    "You have created an ad-hoc early-stopping heuristic on the word 'Wait'.
    How does your latent trajectory dynamics method differ from naive early stopping
    or fixed token budgets?"

EVALUATION ARMS (5-ARM COMPREHENSIVE ABLATION):
    1. Vanilla Baseline (Unconstrained generation, max 450 tokens)
    2. Fixed Token Budget (Blindly cut off at T = 180 tokens)
    3. Naive Linguistic Early-Exit (Cut off only if text contains "Wait" / "Alternatively")
    4. Pure Latent Velocity Governor (Cut off when cosine velocity v_t < epsilon, NO text check)
    5. Coupled Trajectory Governor (SubNeutralize: Latent velocity stabilization + hesitation gating)
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
# 1. ROBUST REGEX EXTRACTOR
# ------------------------------------------------------------------------------
def extract_answer(text: str):
    patterns = [
        r"\\boxed\{[^\d]*(-?\d+(?:,\d+)*(?:\.\d+)?)[^\d]*\}",
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[-+]?\d*\.\d+|\d+"
    ]
    # Check boxed and #### first
    for p in patterns[:3]:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "").strip()
    # Fallback to last number if stopped at hesitation
    nums = re.findall(patterns[3], text.replace(",", ""))
    return nums[-1] if nums else None

# ------------------------------------------------------------------------------
# 2. STOPPING CONTROLLERS FOR ALL 5 ABLATION ARMS
# ------------------------------------------------------------------------------

# Arm 2: Fixed Token Budget
class FixedBudgetController(StoppingCriteria):
    def __init__(self, prompt_len: int, budget: int = 180):
        self.prompt_len = prompt_len
        self.budget = budget

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        curr = input_ids.shape[-1] - self.prompt_len
        return curr >= self.budget

# Arm 3: Naive Linguistic Early-Exit ("Wait" / "Alternatively")
class NaiveLinguisticController(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int, min_tokens: int = 140):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        curr = input_ids.shape[-1] - self.prompt_len
        if curr < self.min_tokens:
            return False
        recent = self.tokenizer.decode(input_ids[0, -15:], skip_special_tokens=False)
        if "</think>" in recent or "Wait" in recent or "Alternatively" in recent:
            return True
        return False

# Arm 4: Pure Latent Velocity Governor (Zero Text Checks)
class LatentVelocityGovernor:
    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = 14,
        velocity_threshold: float = 0.05,
        min_deduction_tokens: int = 100,
        consensus_patience: int = 3
    ):
        self.model = model
        self.target_layer = model.model.layers[target_layer_idx]
        self.velocity_threshold = velocity_threshold
        self.min_deduction_tokens = min_deduction_tokens
        self.consensus_patience = consensus_patience

        self.trajectory = []
        self.velocities = []
        self.low_velocity_streak = 0
        self.stop_triggered = False
        self.tokens_generated = 0
        self.hook_handle = None

    def reset(self):
        self.trajectory.clear()
        self.velocities.clear()
        self.low_velocity_streak = 0
        self.stop_triggered = False
        self.tokens_generated = 0

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        # Prefill: skip
        if hidden.shape[1] > 1:
            return outputs

        self.tokens_generated += 1
        current_h = hidden[:, -1, :].detach().float()

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
            self.velocities.append(v_t)

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

class LatentStoppingCriteria(StoppingCriteria):
    def __init__(self, governor: LatentVelocityGovernor):
        self.governor = governor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.governor.stop_triggered

# Arm 5: Coupled Trajectory Governor (SubNeutralize: Velocity + Hesitation Confirmation)
class CoupledGovernor(StoppingCriteria):
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
        
        # Stops either if latent velocity confirmed consensus OR hesitation marker confirms transition
        if "</think>" in recent:
            return True
        if self.governor.stop_triggered:
            return True
        if "Wait" in recent or "Alternatively" in recent:
            return True
        return False

# ------------------------------------------------------------------------------
# 3. BENCHMARK EXECUTION (5 ARMS ON A100)
# ------------------------------------------------------------------------------
def run_attack3_ablation(num_problems: int = 10):
    print("=" * 95)
    print("REVIEWER ATTACK #3 DEFENSE: 5-ARM RIGOROUS ABLATION")
    print("Arms: [1. Vanilla | 2. Fixed-180 | 3. Naive Linguistic | 4. Pure Latent Velocity | 5. Coupled]")
    print("=" * 95)

    gc.collect()
    torch.cuda.empty_cache()

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    latent_gov = LatentVelocityGovernor(model=model, target_layer_idx=14, velocity_threshold=0.06, min_deduction_tokens=100)
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{num_problems}]")

    stats = {
        "vanilla": {"tokens": 0, "correct": 0},
        "fixed": {"tokens": 0, "correct": 0},
        "linguistic": {"tokens": 0, "correct": 0},
        "pure_latent": {"tokens": 0, "correct": 0},
        "coupled": {"tokens": 0, "correct": 0},
    }

    results = []

    print(f"\n{'#':<3} | {'Vanilla':<12} | {'Fixed-180':<12} | {'Linguistic':<12} | {'Pure Latent':<14} | {'Coupled':<12} | {'GT'}")
    print("-" * 85)

    for idx, sample in enumerate(dataset):
        q = sample["question"]
        gt = extract_answer(sample["answer"])

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # 1. Vanilla
        latent_gov.detach()
        with torch.no_grad():
            out_van = model.generate(**inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        tk_van = out_van.shape[1] - prompt_len
        ans_van = extract_answer(tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=True))
        c_van = (ans_van == gt)
        stats["vanilla"]["tokens"] += tk_van
        if c_van: stats["vanilla"]["correct"] += 1

        # 2. Fixed Budget (180 tokens)
        with torch.no_grad():
            out_fix = model.generate(**inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True, stopping_criteria=[FixedBudgetController(prompt_len, 180)], pad_token_id=tokenizer.eos_token_id)
        tk_fix = out_fix.shape[1] - prompt_len
        ans_fix = extract_answer(tokenizer.decode(out_fix[0, prompt_len:], skip_special_tokens=True))
        c_fix = (ans_fix == gt)
        stats["fixed"]["tokens"] += tk_fix
        if c_fix: stats["fixed"]["correct"] += 1

        # 3. Naive Linguistic ("Wait" / "Alternatively")
        with torch.no_grad():
            out_ling = model.generate(**inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True, stopping_criteria=[NaiveLinguisticController(tokenizer, prompt_len, 140)], pad_token_id=tokenizer.eos_token_id)
        tk_ling = out_ling.shape[1] - prompt_len
        ans_ling = extract_answer(tokenizer.decode(out_ling[0, prompt_len:], skip_special_tokens=True))
        c_ling = (ans_ling == gt)
        stats["linguistic"]["tokens"] += tk_ling
        if c_ling: stats["linguistic"]["correct"] += 1

        # 4. Pure Latent Velocity Governor
        latent_gov.attach()
        with torch.no_grad():
            out_lat = model.generate(**inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True, stopping_criteria=[LatentStoppingCriteria(latent_gov)], pad_token_id=tokenizer.eos_token_id)
        latent_gov.detach()
        tk_lat = out_lat.shape[1] - prompt_len
        ans_lat = extract_answer(tokenizer.decode(out_lat[0, prompt_len:], skip_special_tokens=True))
        c_lat = (ans_lat == gt)
        stats["pure_latent"]["tokens"] += tk_lat
        if c_lat: stats["pure_latent"]["correct"] += 1

        # 5. Coupled Governor
        latent_gov.attach()
        with torch.no_grad():
            out_coup = model.generate(**inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True, stopping_criteria=[CoupledGovernor(tokenizer, latent_gov, prompt_len, 120)], pad_token_id=tokenizer.eos_token_id)
        latent_gov.detach()
        tk_coup = out_coup.shape[1] - prompt_len
        ans_coup = extract_answer(tokenizer.decode(out_coup[0, prompt_len:], skip_special_tokens=True))
        c_coup = (ans_coup == gt)
        stats["coupled"]["tokens"] += tk_coup
        if c_coup: stats["coupled"]["correct"] += 1

        print(
            f"{idx+1:<3} | "
            f"{tk_van} ({'OK' if c_van else 'X'}):<12 | "
            f"{tk_fix} ({'OK' if c_fix else 'X'}):<12 | "
            f"{tk_ling} ({'OK' if c_ling else 'X'}):<12 | "
            f"{tk_lat} ({'OK' if c_lat else 'X'}):<14 | "
            f"{tk_coup} ({'OK' if c_coup else 'X'}):<12 | "
            f"{gt}"
        )

        results.append({
            "idx": idx + 1, "gt": gt,
            "vanilla": {"tokens": tk_van, "ans": ans_van, "corr": c_van},
            "fixed": {"tokens": tk_fix, "ans": ans_fix, "corr": c_fix},
            "linguistic": {"tokens": tk_ling, "ans": ans_ling, "corr": c_ling},
            "pure_latent": {"tokens": tk_lat, "ans": ans_lat, "corr": c_lat},
            "coupled": {"tokens": tk_coup, "ans": ans_coup, "corr": c_coup},
        })

    print("\n" + "=" * 95)
    print("FINAL 5-ARM ABLATION BENCHMARK RESULTS")
    print("=" * 95)
    for arm, d in stats.items():
        acc = (d["correct"] / num_problems) * 100.0
        print(f"Arm: {arm:<16} | Accuracy: {d['correct']}/{num_problems} ({acc:.1f}%) | Tokens: {d['tokens']}")
    print("=" * 95)

    with open("results_attack3_ablation.json", "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2)

if __name__ == "__main__":
    run_attack3_ablation(10)
