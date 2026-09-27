"""
================================================================================
OFFICIAL OLYMPIAD MULTI-SUBJECT BENCHMARK: ZERO-SCISSORS IN-FLIGHT STEERING
HARDWARE: NVIDIA A100 GPU (Colab Pro)
MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DATASET: HuggingFaceH4/MATH-500 (Covering all 7 Competition Subjects, Levels 2-5)
PARADIGM: ZERO SCISSORS (stopping_criteria = None).
          The model derives deep mathematical proofs freely (up to 2,500 tokens)
          and terminates voluntarily via in-flight P_perp and exit steering.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------------------
# 1. LATEX NORMALIZATION & EQUALITY CHECKER
# ------------------------------------------------------------------------------
def clean_latex(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace(" ", "").replace("\n", "")
    m_text = re.search(r"\\text\{([^}]+)\}", s)
    if m_text:
        s = m_text.group(1)
    return s.strip()

def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return clean_latex(matches[-1])
    m = re.findall(r"[Tt]he (?:final )?answer is:?\s*\$?([^\n.$]+)", text)
    if m:
        return clean_latex(m[-1])
    return None

def check_math_equal(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    c_pred = clean_latex(pred)
    c_gt = clean_latex(gt)
    if c_pred == c_gt:
        return True
    try:
        if abs(float(c_pred) - float(c_gt)) < 1e-5:
            return True
    except Exception:
        pass
    return False

# ------------------------------------------------------------------------------
# 2. DIVERSE MULTI-SUBJECT OLYMPIAD LOADER
# ------------------------------------------------------------------------------
def load_diverse_olympiad_probes(n_per_subject: int = 1):
    print("Loading competition math problems across all 7 MATH-500 subjects...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    subjects = [
        "Precalculus",
        "Intermediate Algebra",
        "Number Theory",
        "Geometry",
        "Counting & Probability",
        "Prealgebra",
        "Algebra"
    ]
    
    subject_counts = {s: 0 for s in subjects}
    selected = []
    
    for row in ds:
        subj = row.get("subject", "")
        level = row.get("level", 1)
        # Target challenging competition problems (Level 2 to 5)
        if subj in subject_counts and subject_counts[subj] < n_per_subject and level >= 2:
            selected.append({
                "id": f"math_{subj[:4].lower()}_L{level}",
                "subject": subj,
                "level": level,
                "problem": row["problem"],
                "ground_truth": clean_latex(row["answer"])
            })
            subject_counts[subj] += 1
            if all(c >= n_per_subject for c in subject_counts.values()):
                break
                
    return selected

# ------------------------------------------------------------------------------
# 3. IN-FLIGHT OLYMPIAD STEERING HOOK (ZERO SCISSORS)
# ------------------------------------------------------------------------------
class OlympiadInFlightSteeringHook:
    """
    In-flight Transformer forward hook for Olympiad mathematics.
    Allows long-range proofs (velocity v_t >= epsilon remains untouched).
    When the multi-step derivation reaches representation consensus (v_t < epsilon for k=4 steps),
    it projects out prompt inquiry attractors and aligns with the natural conclusion direction.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        target_layer_idx: int = 14,
        velocity_threshold: float = 0.045,
        consensus_k: int = 4,
        steering_alpha: float = 0.06
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer_idx = target_layer_idx
        self.velocity_threshold = velocity_threshold
        self.consensus_k = consensus_k
        self.steering_alpha = steering_alpha

        self.target_layer = model.model.layers[target_layer_idx]
        self.hook_handle = None
        self.is_active = False

        # Compute exit vector in unembedding space: u_exit = normalize(w_think - w_wait)
        lm_head = model.lm_head
        w_u = lm_head.weight.detach()
        think_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]
        diff = w_u[think_id].float() - w_u[wait_id].float()
        self.u_exit = (diff / torch.linalg.norm(diff)).to(lm_head.weight.device)

        self.P_perp = None
        self.trajectory = []
        self.low_v_streak = 0
        self.steering_engaged = False
        self.engaged_at_token = None

    def set_prompt_bias(self, prompt_hidden: torch.Tensor):
        vec = prompt_hidden.squeeze().float()
        norm = torch.linalg.norm(vec)
        if norm > 1e-8:
            v = (vec / norm).unsqueeze(1)
            D = v.shape[0]
            I = torch.eye(D, device=v.device, dtype=torch.float32)
            self.P_perp = (I - torch.matmul(v, v.T))
        else:
            self.P_perp = None

    def reset(self):
        self.trajectory.clear()
        self.low_v_streak = 0
        self.steering_engaged = False
        self.engaged_at_token = None

    def _hook_fn(self, module, inputs, outputs):
        if not self.is_active:
            return outputs
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.shape[1] > 1:
            return outputs  # Prefill pass, bypass

        current_h = hidden[:, -1, :].detach().float()
        step = len(self.trajectory) + 1

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
            if v_t < self.velocity_threshold:
                self.low_v_streak += 1
                if self.low_v_streak >= self.consensus_k:
                    if not self.steering_engaged:
                        self.steering_engaged = True
                        self.engaged_at_token = step
            else:
                self.low_v_streak = 0
        self.trajectory.append(current_h)

        if self.steering_engaged:
            h_mod = hidden[:, -1, :].clone().float()
            # 1. Deflate prompt bias attractor
            if self.P_perp is not None:
                h_mod = torch.matmul(h_mod, self.P_perp.to(h_mod.device))
            # 2. Directed conclusion alignment
            if self.u_exit is not None and self.steering_alpha > 0:
                norm_h = torch.linalg.norm(h_mod, dim=-1, keepdim=True)
                h_mod = h_mod + self.steering_alpha * norm_h * self.u_exit.unsqueeze(0)
            hidden[:, -1, :] = h_mod.to(hidden.dtype)

        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    def attach(self):
        self.reset()
        self.is_active = True
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def detach(self):
        self.is_active = False
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

# ------------------------------------------------------------------------------
# 4. BENCHMARK EXECUTION
# ------------------------------------------------------------------------------
def run_olympiad_suite():
    print("=" * 105)
    print("🚀 LAUNCHING MULTI-SUBJECT OLYMPIAD BENCHMARK: ZERO-SCISSORS IN-FLIGHT STEERING")
    print("   Dataset: HuggingFaceH4/MATH-500 | Ceiling: 2,500 Tokens | stopping_criteria = None")
    print("=" * 105)

    if "model_7b" in globals():
        print("Using cached DeepSeek-R1-Distill-Qwen-7B in VRAM...")
        model = globals()["model_7b"]
        tokenizer = globals()["tokenizer_7b"]
    else:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        print(f"Loading {model_id} into VRAM...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()
        globals()["model_7b"] = model
        globals()["tokenizer_7b"] = tokenizer

    problems = load_diverse_olympiad_probes(n_per_subject=1)
    hook = OlympiadInFlightSteeringHook(model, tokenizer, target_layer_idx=14, velocity_threshold=0.045, consensus_k=4, steering_alpha=0.06)

    v_total_tokens, s_total_tokens = 0, 0
    v_correct, s_correct = 0, 0
    steer_won, van_won = 0, 0
    records = []

    print(f"\n{'#':<2} | {'Subject / Level':<24} | {'Vanilla (Tk, Time, Res)':<28} | {'Steered (Tk, Time, Res)':<28} | {'Saved':<9} | {'Outcome'}")
    print("-" * 105)

    for i, p in enumerate(problems):
        subj_lvl = f"{p['subject']} L{p['level']}"
        q = p["problem"]
        gt = p["ground_truth"]

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        p_len = inp.input_ids.shape[-1]

        # --- Arm 1: Vanilla (Unconstrained, 2,500 token ceiling, Zero Scissors) ---
        hook.detach()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            v_out = model.generate(**inp, max_new_tokens=2500, temperature=0.6, do_sample=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=None)
        torch.cuda.synchronize()
        v_time = time.time() - t0
        v_tokens = v_out[0].shape[-1] - p_len
        v_pred = extract_boxed_answer(tokenizer.decode(v_out[0][p_len:], skip_special_tokens=True))
        v_ok = check_math_equal(v_pred, gt)

        # --- Arm 2: In-Flight Steered (Layer 14 Hook, 2,500 token ceiling, Zero Scissors) ---
        with torch.no_grad():
            prompt_out = model(**inp, output_hidden_states=True)
            prompt_h14 = prompt_out.hidden_states[14][0, -1, :].clone()
        hook.set_prompt_bias(prompt_h14)
        hook.attach()

        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            s_out = model.generate(**inp, max_new_tokens=2500, temperature=0.6, do_sample=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=None)
        torch.cuda.synchronize()
        s_time = time.time() - t1
        s_tokens = s_out[0].shape[-1] - p_len
        s_pred = extract_boxed_answer(tokenizer.decode(s_out[0][p_len:], skip_special_tokens=True))
        s_ok = check_math_equal(s_pred, gt)
        hook.detach()

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

        v_disp = f"{v_tokens} tk ({v_time:.1f}s, {'OK' if v_ok else 'X'}):<{gt[:6]}"
        s_disp = f"{s_tokens} tk ({s_time:.1f}s, {'OK' if s_ok else 'X'}):<{gt[:6]}"
        print(f"{i+1:<2} | {subj_lvl:<24} | {v_disp:<28} | {s_disp:<28} | {saved_pct:>7.1f}% | {outcome}")

        records.append({
            "idx": i + 1,
            "subject": p["subject"],
            "level": p["level"],
            "ground_truth": gt,
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
    print("\n" + "=" * 105)
    print("🏆 FINAL MULTI-SUBJECT OLYMPIAD RESULTS (ZERO SCISSORS)")
    print("=" * 105)
    print(f"Total Subjects Evaluated:     {len(problems)} (All 7 MATH-500 Disciplines)")
    print(f"Net Compute Savings:          {net_savings:.1f}% ({v_total_tokens} -> {s_total_tokens} tokens)")
    print(f"Vanilla Accuracy:             {(v_correct/len(problems))*100:.1f}% ({v_correct}/{len(problems)})")
    print(f"Steered Accuracy:             {(s_correct/len(problems))*100:.1f}% ({s_correct}/{len(problems)})")
    print(f"Overthinking Flips Prevented: {steer_won}")
    print(f"Losses against Vanilla:       {van_won}")
    print("=" * 105)

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/results_olympiad_no_scissors_multisubject.json", "w") as f:
        json.dump({
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "benchmark": "HuggingFaceH4/MATH-500",
            "net_savings_pct": net_savings,
            "vanilla_acc": (v_correct / len(problems)) * 100,
            "steered_acc": (s_correct / len(problems)) * 100,
            "steer_won": steer_won,
            "van_won": van_won,
            "records": records
        }, f, indent=2)
    print("Saved results to: paper/tables/results_olympiad_no_scissors_multisubject.json")

if __name__ == "__main__":
    run_olympiad_suite()
