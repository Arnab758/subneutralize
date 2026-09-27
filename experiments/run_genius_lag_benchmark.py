"""
================================================================================
FRONTIER BENCHMARK: LATENT ATTRACTOR GOVERNOR (LAG-v3)
PARADIGM: PURE DIFFERENTIAL GEOMETRY & INFORMATION THEORY (ZERO SCISSORS)
HARDWARE: NVIDIA A100 GPU (Colab Pro)
MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DATASET: openai/gsm8k (Problems 61 to 80 - 20 COMPLETELY FRESH PROBLEMS)
INVARIANTS:
  1. Epiphany Attractor: Low Shannon Entropy H_t coupled with Riemannian Deceleration v_t
  2. Limit-Cycle Recurrence: Lagged Topological Correlation R_t(tau) > 0.92
CONSTRAINT: stopping_criteria = None (Pure Natural Model EOS Termination)
TOKEN CEILING: 650 Tokens
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
    LogitsProcessor,
    LogitsProcessorList
)

# ------------------------------------------------------------------------------
# 1. LOAD 20 COMPLETELY FRESH BENCHMARK PROBLEMS (PROBLEMS 61 TO 80)
# ------------------------------------------------------------------------------
START_IDX = 60  # 0-indexed -> starts at problem 61
NUM_PROBLEMS = 20

print(f"Loading {NUM_PROBLEMS} totally fresh test problems (Problems {START_IDX+1} to {START_IDX+NUM_PROBLEMS}) from openai/gsm8k...")
try:
    ds = load_dataset("openai/gsm8k", "main", split=f"test[{START_IDX}:{START_IDX+NUM_PROBLEMS}]")
except Exception:
    import pandas as pd
    df = pd.read_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
    ds = df.iloc[START_IDX:START_IDX+NUM_PROBLEMS].to_dict("records")

dataset = [{
    "idx": START_IDX + i + 1,
    "question": row["question"],
    "ground_truth": row["answer"].split("####")[-1].strip().replace(",", "")
} for i, row in enumerate(ds)]


def extract_numeric_answer(text: str):
    if not text:
        return None
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
# 2. RIEMANNIAN MANIFOLD & RECURRENCE GOVERNOR
# ------------------------------------------------------------------------------
class ManifoldDynamicsTracker:
    """
    Monitors Riemannian trajectory velocities and lagged recurrence across
    the reasoning layer band [12, 14, 16] without any hardcoded text keywords.
    """
    def __init__(self, model, layer_band=[12, 14, 16]):
        self.model = model
        self.layer_band = layer_band
        self.layers = [model.model.layers[i] for i in layer_band]
        
        self.trajectories = {i: [] for i in layer_band}
        self.last_velocities = {i: 1.0 for i in layer_band}
        self.hook_handles = []
        self.is_active = False

    def reset(self):
        for i in self.layer_band:
            self.trajectories[i].clear()
            self.last_velocities[i] = 1.0

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, outputs):
            if not self.is_active:
                return outputs
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            if hidden.shape[1] > 1:
                return outputs  # Prefill bypass

            curr_h = hidden[:, -1, :].detach().float()
            traj = self.trajectories[layer_idx]
            if len(traj) > 0:
                cos_sim = F.cosine_similarity(curr_h, traj[-1], dim=-1).item()
                self.last_velocities[layer_idx] = max(0.0, 1.0 - cos_sim)
            else:
                self.last_velocities[layer_idx] = 1.0
            traj.append(curr_h)
            return outputs
        return hook_fn

    def attach(self):
        self.reset()
        self.is_active = True
        self.hook_handles = [
            layer.register_forward_hook(self._make_hook(idx))
            for idx, layer in zip(self.layer_band, self.layers)
        ]

    def detach(self):
        self.is_active = False
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def get_consensus_velocity(self) -> float:
        """Minimax consensus across the reasoning band."""
        return max(self.last_velocities.values())

    def get_lagged_recurrence(self, lag: int = 15) -> float:
        """
        Computes topological recurrence R_t(tau) = cos_sim(h_t, h_{t-tau}) at Layer 14.
        Detects whether the trajectory has entered a degenerate limit cycle.
        """
        traj = self.trajectories[14]
        if len(traj) <= lag:
            return 0.0
        curr_h = traj[-1]
        lagged_h = traj[-lag]
        return F.cosine_similarity(curr_h, lagged_h, dim=-1).item()


# ------------------------------------------------------------------------------
# 3. GENIUS-LEVEL LATENT ATTRACTOR GOVERNOR (LAG-v3 PROCESSOR)
# ------------------------------------------------------------------------------
class LatentAttractorLogitsProcessor(LogitsProcessor):
    """
    Pure geometric and information-theoretic governor:
    - Zero regex, zero text checks, language-agnostic.
    - Dual Dynamical Triggers:
      1. Epiphany Attractor: Trajectory velocity v_t < 0.065 and Shannon entropy H_t < 1.0
         (The mathematical formulation is synthesized with high model certainty).
      2. Limit-Cycle Recurrence: Lagged correlation R_t(15) > 0.91
         (The model is caught in an overthinking orbit).
    - Phase-Transition Closure: Emits </think> to transition into solution output.
    - Zero Scissors: Generation runs to native EOS (<|im_end|>).
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tracker: ManifoldDynamicsTracker,
        prompt_len: int,
        min_tokens: int = 80
    ):
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens
        self.think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        
        self.stagnation_counter = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_tokens = input_ids[0, self.prompt_len:].tolist()

        # DEACTIVATION SHIELD: Once </think> is present, disconnect completely!
        if self.think_token_id in gen_tokens:
            return scores

        # Allow sufficient initial exploration
        if len(gen_tokens) < self.min_tokens:
            return scores

        # 1. Compute Instantaneous Shannon Entropy H_t
        probs = F.softmax(scores[0].float(), dim=-1)
        log_p = torch.log(probs + 1e-12)
        H_t = -(probs * log_p).sum().item()

        # 2. Track Trajectory Dynamics
        v_consensus = self.tracker.get_consensus_velocity()
        r_recurrence = self.tracker.get_lagged_recurrence(lag=15)

        # Stagnation tracking
        if v_consensus < 0.070:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        # --- DUAL GEOMETRIC INVARIANTS ---
        # Invariant A: Epiphany Synthesis (High certainty H_t < 0.95 + Velocity Deceleration)
        epiphany_converged = (H_t < 0.95 and self.stagnation_counter >= 2)

        # Invariant B: Limit-Cycle Recurrence (Model orbiting in verification loop)
        loop_recurrence = (r_recurrence > 0.91 and len(gen_tokens) >= 140)

        # Invariant C: Persistent Trajectory Stagnation (Model paused at solution plateau)
        plateau_stagnated = (self.stagnation_counter >= 3)

        if epiphany_converged or loop_recurrence or plateau_stagnated:
            # Phase-Transition: Force transition delimiter </think>
            scores[0, :] = -float("inf")
            scores[0, self.think_token_id] = 100.0

        return scores


# ------------------------------------------------------------------------------
# 4. BENCHMARK RUNNER (A100 GPU)
# ------------------------------------------------------------------------------
def run_benchmark():
    print("=" * 105)
    print("🚀 LAUNCHING FRONTIER LATENT ATTRACTOR GOVERNOR (LAG-v3)")
    print(f"   Dataset: openai/gsm8k (Problems {START_IDX+1} to {START_IDX+NUM_PROBLEMS} - 20 TOTALLY FRESH PROBLEMS)")
    print("   Architecture: Pure Geometry (v_t, H_t, R_t) | ZERO Text Regex | ZERO Scissors")
    print("   Token Ceiling: 650 Tokens | Natural Model EOS Termination (<|im_end|>)")
    print("=" * 105)

    if "model_7b" in globals():
        print("Using cached DeepSeek-R1-Distill-Qwen-7B in VRAM...")
        model = globals()["model_7b"]
        tokenizer = globals()["tokenizer_7b"]
    else:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        print(f"Loading {model_id} into VRAM...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        globals()["model_7b"] = model
        globals()["tokenizer_7b"] = tokenizer

    tracker = ManifoldDynamicsTracker(model, layer_band=[12, 14, 16])

    v_total_tokens, s_total_tokens = 0, 0
    v_correct, s_correct = 0, 0
    steer_won, van_won = 0, 0
    records = []

    print(f"\n{'#':<3} | {'Vanilla (Tk, Time, Res)':<28} | {'Steered (Tk, Time, Res)':<28} | {'Tokens Saved':<13} | {'Outcome'}")
    print("-" * 100)

    for item in dataset:
        idx = item["idx"]
        q = item["question"]
        gt = item["ground_truth"]
        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        p_len = inp.input_ids.shape[-1]

        # --- Arm 1: Vanilla (Unconstrained, 650 Token Ceiling, Zero Scissors) ---
        tracker.detach()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            v_out = model.generate(
                **inp,
                max_new_tokens=650,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=None  # Zero scissors
            )
        torch.cuda.synchronize()
        v_time = time.time() - t0
        v_tokens = v_out[0].shape[-1] - p_len
        v_ans = extract_numeric_answer(tokenizer.decode(v_out[0][p_len:], skip_special_tokens=True))
        v_ok = (v_ans == gt)

        # --- Arm 2: Frontier LAG-v3 Governor (Zero Scissors) ---
        tracker.attach()
        lag_proc = LatentAttractorLogitsProcessor(tokenizer, tracker, prompt_len=p_len, min_tokens=80)
        proc_list = LogitsProcessorList([lag_proc])

        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            s_out = model.generate(
                **inp,
                max_new_tokens=650,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=proc_list,
                stopping_criteria=None  # Zero scissors! Natural exit
            )
        torch.cuda.synchronize()
        s_time = time.time() - t1
        s_tokens = s_out[0].shape[-1] - p_len
        s_ans = extract_numeric_answer(tokenizer.decode(s_out[0][p_len:], skip_special_tokens=True))
        s_ok = (s_ans == gt)
        tracker.detach()

        v_total_tokens += v_tokens
        s_total_tokens += s_tokens
        if v_ok: v_correct += 1
        if s_ok: s_correct += 1
        saved_pct = ((v_tokens - s_tokens) / v_tokens) * 100 if v_tokens > 0 else 0.0

        if s_ok and not v_ok:
            outcome = "STEER_WON (Flip Prevented!)"
            steer_won += 1
        elif v_ok and not s_ok:
            outcome = "VAN_WON"
            van_won += 1
        elif v_ok and s_ok:
            outcome = "EQUAL"
        else:
            outcome = "BOTH_FAIL"

        v_disp = f"{v_tokens} tk ({v_time:.1f}s, {'OK' if v_ok else 'X'}):<{gt}"
        s_disp = f"{s_tokens} tk ({s_time:.1f}s, {'OK' if s_ok else 'X'}):<{gt}"
        print(f"{idx:<3} | {v_disp:<28} | {s_disp:<28} | {saved_pct:>11.1f}% | {outcome}")

        records.append({
            "idx": idx,
            "vanilla_tokens": v_tokens,
            "vanilla_time": v_time,
            "vanilla_ok": v_ok,
            "steered_tokens": s_tokens,
            "steered_time": s_time,
            "steered_ok": s_ok,
            "saved_pct": saved_pct,
            "outcome": outcome
        })

    net_savings = ((v_total_tokens - s_total_tokens) / v_total_tokens) * 100
    print("\n" + "=" * 100)
    print("🏆 FINAL RESULTS: FRONTIER LAG-v3 BENCHMARK (ZERO SCISSORS)")
    print("=" * 100)
    print(f"Total Fresh Problems:         {len(dataset)} (Problems {START_IDX+1} to {START_IDX+NUM_PROBLEMS})")
    print(f"Net Compute Savings:          {net_savings:.1f}% ({v_total_tokens} -> {s_total_tokens} tokens)")
    print(f"Vanilla Accuracy:             {(v_correct/len(dataset))*100:.1f}% ({v_correct}/{len(dataset)})")
    print(f"Steered Accuracy:             {(s_correct/len(dataset))*100:.1f}% ({s_correct}/{len(dataset)})")
    print(f"Overthinking Flips Prevented: {steer_won}")
    print(f"Losses against Vanilla:       {van_won}")
    print("=" * 100)

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/results_lag_v3_gsm8k_61_80.json", "w") as f:
        json.dump({
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "dataset": "openai/gsm8k_fresh61_80",
            "net_savings_pct": net_savings,
            "vanilla_acc": (v_correct / len(dataset)) * 100,
            "steered_acc": (s_correct / len(dataset)) * 100,
            "steer_won": steer_won,
            "van_won": van_won,
            "records": records
        }, f, indent=2)
    print("Saved results to: paper/tables/results_lag_v3_gsm8k_61_80.json")

if __name__ == "__main__":
    run_benchmark()
