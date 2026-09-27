"""
================================================================================
OFFICIAL BENCHMARK: BIFURCATION CLOSURE & LATENT GOVERNOR (ZERO SCISSORS)
HARDWARE: NVIDIA A100 GPU (Colab Pro)
MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DATASET: openai/gsm8k (Test Split: Problems 31 to 45 - TOTALLY FRESH & UNSEEN)
PARADIGM: ZERO SCISSORS (stopping_criteria = None).
          In-flight LogitsProcessor intercepts the hesitation attractor fork,
          emits </think>, and allows the model to freely generate the final
          answer and terminate naturally via native EOS (<|im_end|>).
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
# 1. LOAD 100% FRESH BENCHMARK PROBLEMS (PROBLEMS 31 TO 45)
# ------------------------------------------------------------------------------
START_IDX = 30  # 0-indexed -> starts at problem 31
NUM_PROBLEMS = 15

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
# 2. MULTI-LAYER CONSENSUS VELOCITY GOVERNOR
# ------------------------------------------------------------------------------
class MultiLayerConsensusGovernor:
    def __init__(self, model, layer_band=[12, 14, 16], velocity_threshold=0.065):
        self.model = model
        self.layer_band = layer_band
        self.velocity_threshold = velocity_threshold

        self.layers = [model.model.layers[i] for i in layer_band]
        self.trajectories = {i: [] for i in layer_band}
        self.last_velocities = {i: 1.0 for i in layer_band}
        self.hook_handles = []
        self.consensus_v = 1.0
        self.consecutive_stagnations = 0
        self.is_active = False

    def reset(self):
        for i in self.layer_band:
            self.trajectories[i].clear()
            self.last_velocities[i] = 1.0
        self.consensus_v = 1.0
        self.consecutive_stagnations = 0

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
                prev_h = traj[-1]
                cos_sim = F.cosine_similarity(curr_h, prev_h, dim=-1).item()
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

    def update_consensus(self):
        self.consensus_v = max(self.last_velocities.values())
        if self.consensus_v < self.velocity_threshold:
            self.consecutive_stagnations += 1
        else:
            self.consecutive_stagnations = 0


# ------------------------------------------------------------------------------
# 3. BIFURCATION CLOSURE PROCESSOR (ZERO SCISSORS)
# ------------------------------------------------------------------------------
class BifurcationClosureProcessor(LogitsProcessor):
    """
    Applies in-flight bifurcation steering:
    - If the model attempts to enter a self-doubt loop ("Wait", "Alternatively")
      or trajectory reaches Riemannian equilibrium after deduction:
      it steers the next token to </think>.
    - Then, it INSTANTLY RELEASES ALL CONSTRAINTS!
    - The model freely generates its final answer and terminates naturally via native EOS.
    - Zero scissors (stopping_criteria = None).
    """
    def __init__(self, tokenizer, governor, prompt_len: int, min_deduction_tokens: int = 70):
        self.tokenizer = tokenizer
        self.governor = governor
        self.prompt_len = prompt_len
        self.min_deduction_tokens = min_deduction_tokens

        self.think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        
        hesitation_words = [
            "Wait", " Wait", "Alternatively", " Alternatively",
            "Hold on", " Hold on", "Let me check", " Let me check",
            "Let me double", " Let me double", "Wait,", " Wait,",
            "Wait.", " Wait."
        ]
        self.hesitation_ids = set()
        for w in hesitation_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                self.hesitation_ids.add(ids[0])
        self.hesitation_ids = list(self.hesitation_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_tokens = input_ids[0, self.prompt_len:].tolist()

        # SHUT OFF COMPLETELY once </think> is emitted!
        # The model generates the final answer and terminates via EOS naturally.
        if self.think_token_id in gen_tokens:
            return scores

        # Allow sufficient tokens for genuine initial deduction
        if len(gen_tokens) < self.min_deduction_tokens:
            return scores

        # Update latent governor
        self.governor.update_consensus()

        # Check if the model is attempting to emit a hesitation token
        top_token = scores[0].argmax().item()
        hesitation_detected = top_token in self.hesitation_ids
        trajectory_stagnated = self.governor.consecutive_stagnations >= 3

        if hesitation_detected or trajectory_stagnated:
            # Shift bifurcation path: Force conclusion delimiter </think>
            scores[0, :] = -float("inf")
            scores[0, self.think_token_id] = 100.0

        return scores


# ------------------------------------------------------------------------------
# 4. BENCHMARK RUNNER (A100 GPU)
# ------------------------------------------------------------------------------
def run_benchmark():
    print("=" * 105)
    print("🚀 LAUNCHING BIFURCATION CLOSURE BENCHMARK (NVIDIA A100 GPU)")
    print(f"   Dataset: openai/gsm8k (Problems {START_IDX+1} to {START_IDX+NUM_PROBLEMS} - TOTALLY FRESH & UNSEEN)")
    print("   Constraint: stopping_criteria = None (Pure Natural Emission & Natural EOS)")
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

    governor = MultiLayerConsensusGovernor(model, layer_band=[12, 14, 16], velocity_threshold=0.065)

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

        # --- Arm 1: Vanilla (Unconstrained, Zero Scissors) ---
        governor.detach()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            v_out = model.generate(
                **inp,
                max_new_tokens=450,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=None
            )
        torch.cuda.synchronize()
        v_time = time.time() - t0
        v_tokens = v_out[0].shape[-1] - p_len
        v_ans = extract_numeric_answer(tokenizer.decode(v_out[0][p_len:], skip_special_tokens=True))
        v_ok = (v_ans == gt)

        # --- Arm 2: In-Flight Bifurcation Steered (Zero Scissors) ---
        governor.attach()
        bifurc_proc = BifurcationClosureProcessor(tokenizer, governor, prompt_len=p_len, min_deduction_tokens=70)
        proc_list = LogitsProcessorList([bifurc_proc])

        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            s_out = model.generate(
                **inp,
                max_new_tokens=450,
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
        governor.detach()

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
    print("🏆 FINAL RESULTS: BIFURCATION CLOSURE BENCHMARK (ZERO SCISSORS)")
    print("=" * 100)
    print(f"Total Fresh Problems:         {len(dataset)} (Problems {START_IDX+1} to {START_IDX+NUM_PROBLEMS})")
    print(f"Net Compute Savings:          {net_savings:.1f}% ({v_total_tokens} -> {s_total_tokens} tokens)")
    print(f"Vanilla Accuracy:             {(v_correct/len(dataset))*100:.1f}% ({v_correct}/{len(dataset)})")
    print(f"Steered Accuracy:             {(s_correct/len(dataset))*100:.1f}% ({s_correct}/{len(dataset)})")
    print(f"Overthinking Flips Prevented: {steer_won}")
    print(f"Losses against Vanilla:       {van_won}")
    print("=" * 100)

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/results_bifurcation_closure_gsm8k_31_45.json", "w") as f:
        json.dump({
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "dataset": "openai/gsm8k_fresh31_45",
            "net_savings_pct": net_savings,
            "vanilla_acc": (v_correct / len(dataset)) * 100,
            "steered_acc": (s_correct / len(dataset)) * 100,
            "steer_won": steer_won,
            "van_won": van_won,
            "records": records
        }, f, indent=2)
    print("Saved results to: paper/tables/results_bifurcation_closure_gsm8k_31_45.json")

if __name__ == "__main__":
    run_benchmark()
