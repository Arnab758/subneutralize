"""
================================================================================
OFFICIAL BENCHMARK: THE TRUE NO-SCISSORS IN-FLIGHT NEURAL STEERING ENGINE
HARDWARE: NVIDIA A100 GPU (Colab Pro)
MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DATASET: Official Hugging Face openai/gsm8k (Test Split)
PARADIGM: ZERO SCISSORS (stopping_criteria = None).
          The model concludes reasoning and emits </think> voluntarily via
          in-flight orthogonal projection P_perp and directional exit steering.
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
# 1. ACCREDITED DATASET LOADER (OFFICIAL HF BENCHMARKS ONLY - NO TOY PROBES)
# ------------------------------------------------------------------------------
BENCHMARK_NAME = "gsm8k"  # Switch between "gsm8k" and "math500"
NUM_SAMPLES = 15

def clean_latex(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().replace("$", "").replace("\\$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace(" ", "").replace("\n", "")
    m = re.search(r"\\text\{([^}]+)\}", s)
    if m:
        s = m.group(1)
    return s.strip()

def load_official_dataset(name: str = "gsm8k", n_samples: int = 15):
    print(f"Loading {n_samples} problems from official Hugging Face benchmark '{name}'...")
    problems = []
    if name.lower() == "gsm8k":
        try:
            ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n_samples}]")
        except Exception:
            import pandas as pd
            df = pd.read_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
            ds = df.iloc[:n_samples].to_dict("records")
        for i, row in enumerate(ds):
            gt = row["answer"].split("####")[-1].strip().replace(",", "")
            problems.append({
                "idx": i + 1,
                "question": row["question"],
                "ground_truth": gt,
                "type": "numeric"
            })
    elif name.lower() == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        count = 0
        for i, row in enumerate(ds):
            problems.append({
                "idx": count + 1,
                "question": row["problem"],
                "ground_truth": clean_latex(row["answer"]),
                "type": "latex",
                "subject": row.get("subject", "Math"),
                "level": row.get("level", 1)
            })
            count += 1
            if count >= n_samples:
                break
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return problems


def extract_numeric_answer(text: str):
    if not text:
        return None
    # LaTeX boxed extraction
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return clean_latex(matches[-1])
    patterns = [
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[-+]?\d*\.\d+|\d+"
    ]
    for p in patterns[:2]:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "").strip()
    nums = re.findall(patterns[2], text.replace(",", ""))
    return nums[-1] if nums else None

    return nums[-1] if nums else None


# ------------------------------------------------------------------------------
# 2. IN-FLIGHT NEURAL STEERING HOOK (ZERO SCISSORS)
# ------------------------------------------------------------------------------
class InFlightSteeringHook:
    """
    Applies in-flight orthogonal subspace projection and directional conclusion steering
    inside the Transformer residual stream during forward passes.
    Operates with ZERO external StoppingCriteria.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        target_layer_idx: int = 14,
        velocity_threshold: float = 0.05,
        consensus_k: int = 3,
        steering_alpha: float = 0.08
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

        # Compute conclusion direction u_exit = (w_think - w_wait) / norm
        lm_head = model.lm_head
        w_u = lm_head.weight.detach()
        think_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]

        diff = w_u[think_id].float() - w_u[wait_id].float()
        self.u_exit = (diff / torch.linalg.norm(diff)).to(lm_head.weight.device)

        # Dynamic state
        self.P_perp = None
        self.trajectory = []
        self.low_v_streak = 0
        self.steering_engaged = False
        self.engaged_at_token = None

    def set_prompt_bias(self, prompt_hidden: torch.Tensor):
        vec = prompt_hidden.squeeze().float()
        norm = torch.linalg.norm(vec)
        if norm > 1e-8:
            v = (vec / norm).unsqueeze(1) # [D, 1]
            D = v.shape[0]
            I = torch.eye(D, device=v.device, dtype=torch.float32)
            self.P_perp = (I - torch.matmul(v, v.T)) # [D, D]
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
            return outputs # Prefill pass, bypass

        current_h = hidden[:, -1, :].detach().float()
        step = len(self.trajectory) + 1

        # Track cosine velocity
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

        # In-Flight Steering
        if self.steering_engaged:
            h_mod = hidden[:, -1, :].clone().float()
            
            # Subspace deflation
            if self.P_perp is not None:
                P = self.P_perp.to(h_mod.device)
                h_mod = torch.matmul(h_mod, P)

            # Exit alignment
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
# 3. MAIN BENCHMARK EXECUTION (A100 GPU)
# ------------------------------------------------------------------------------
def run_benchmark():
    print("=" * 100)
    print("🚀 LAUNCHING TRUE NO-SCISSORS IN-FLIGHT NEURAL STEERING BENCHMARK (NVIDIA A100)")
    print("   Dataset: openai/gsm8k (Official Test Split) | Model: DeepSeek-R1-Distill-Qwen-7B")
    print("   Constraint: stopping_criteria = None (Pure voluntary internal conclusion)")
    print("=" * 100)

    # 1. Model Loading
    if "model_7b" in globals():
        print("Using cached DeepSeek-R1-Distill-Qwen-7B in VRAM...")
        model = globals()["model_7b"]
        tokenizer = globals()["tokenizer_7b"]
    else:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        print(f"Loading {model_id} onto GPU...")
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

    # 2. Benchmark Problems
    dataset = load_official_dataset(name=BENCHMARK_NAME, n_samples=NUM_SAMPLES)
    hook = InFlightSteeringHook(model, tokenizer, target_layer_idx=14, velocity_threshold=0.05, consensus_k=3, steering_alpha=0.08)

    records = []
    v_total_tokens = 0
    s_total_tokens = 0
    v_correct = 0
    s_correct = 0
    steer_won = 0
    van_won = 0
    equal = 0
    both_fail = 0

    print(f"\n{'#':<3} | {'Vanilla (Tk, Time, Res)':<27} | {'Steered (Tk, Time, Res)':<27} | {'Tokens Saved':<13} | {'Outcome'}")
    print("-" * 95)

    for item in dataset:
        idx = item["idx"]
        q = item["question"]
        gt = item["ground_truth"]

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        p_len = inputs.input_ids.shape[-1]

        # -------------------------------------------------------------
        # ARM 1: VANILLA (Unconstrained, Zero Scissors)
        # -------------------------------------------------------------
        hook.detach()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            v_out = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=None # ZERO SCISSORS
            )
        torch.cuda.synchronize()
        v_time = time.time() - t0
        v_tokens = v_out[0].shape[-1] - p_len
        v_text = tokenizer.decode(v_out[0][p_len:], skip_special_tokens=True)
        v_ans = extract_numeric_answer(v_text)
        v_ok = (v_ans == gt)
        v_res = "OK" if v_ok else "X"

        # -------------------------------------------------------------
        # ARM 2: IN-FLIGHT STEERED (Forward Hook, Zero Scissors)
        # -------------------------------------------------------------
        # Extract terminal prompt representation at Layer 14 for P_perp
        with torch.no_grad():
            prompt_out = model(**inputs, output_hidden_states=True)
            prompt_h14 = prompt_out.hidden_states[14][0, -1, :].clone()
        hook.set_prompt_bias(prompt_h14)
        hook.attach()

        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            s_out = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=None # ZERO SCISSORS! FREE NATURAL EMISSION
            )
        torch.cuda.synchronize()
        s_time = time.time() - t1
        s_tokens = s_out[0].shape[-1] - p_len
        s_text = tokenizer.decode(s_out[0][p_len:], skip_special_tokens=True)
        s_ans = extract_numeric_answer(s_text)
        s_ok = (s_ans == gt)
        s_res = "OK" if s_ok else "X"
        hook.detach()

        # Metrics
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
            equal += 1
        else:
            outcome = "BOTH_FAIL"
            both_fail += 1

        v_disp = f"{v_tokens} tk ({v_time:.1f}s, {v_res}):<{gt}"
        s_disp = f"{s_tokens} tk ({s_time:.1f}s, {s_res}):<{gt}"
        print(f"{idx:<3} | {v_disp:<27} | {s_disp:<27} | {saved_pct:>11.1f}% | {outcome}")

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

    # Summary
    net_savings = ((v_total_tokens - s_total_tokens) / v_total_tokens) * 100
    v_acc = (v_correct / len(dataset)) * 100
    s_acc = (s_correct / len(dataset)) * 100

    print("\n" + "=" * 95)
    print("🏆 FINAL RESULTS: TRUE NO-SCISSORS IN-FLIGHT NEURAL STEERING")
    print("=" * 95)
    print(f"Total Problems Evaluated:       {len(dataset)}")
    print(f"Vanilla Total Tokens:           {v_total_tokens}")
    print(f"Steered Total Tokens:           {s_total_tokens}")
    print(f"Net Compute Savings:            {net_savings:.1f}%")
    print(f"Vanilla Accuracy:               {v_acc:.1f}% ({v_correct}/{len(dataset)})")
    print(f"Steered Accuracy:               {s_acc:.1f}% ({s_correct}/{len(dataset)})")
    print(f"Overthinking Flips Prevented:   {steer_won}")
    print(f"Losses against Vanilla:         {van_won}")
    print("=" * 95)

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/results_official_no_scissors_gsm8k.json", "w") as f:
        json.dump({
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "dataset": "openai/gsm8k",
            "net_savings_pct": net_savings,
            "vanilla_acc": v_acc,
            "steered_acc": s_acc,
            "steer_won": steer_won,
            "van_won": van_won,
            "records": records
        }, f, indent=2)
    print("Exported results to: paper/tables/results_official_no_scissors_gsm8k.json")


if __name__ == "__main__":
    run_benchmark()
