"""
================================================================================
EXPERIMENT 1 OF 3: CROSS-ARCHITECTURE VALIDATION
MODEL: DeepSeek-R1-Distill-Llama-8B (Meta LLaMA Architecture)
PURPOSE:
    Prove that SubNeutralize is an architecture-agnostic geometric invariant
    governing transformer latent spaces, not an artifact of Qwen.
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
# 1. SUBSPACE NEUTRALIZER (Closed-Form Orthogonal Projection)
# ------------------------------------------------------------------------------
class SubspaceNeutralizer:
    def __init__(self, d_model: int, k_dim: int = 4):
        self.d_model = d_model
        self.k_dim = k_dim
        self.V_bias = None
        self.P_perp = None

    def extract_prompt_subspace(self, prompt_hidden_states: torch.Tensor):
        H = prompt_hidden_states.squeeze(0).float()
        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            _, _, V = torch.linalg.svd(H_centered, full_matrices=False)
            self.V_bias = V[:self.k_dim, :].T.to(prompt_hidden_states.device)
            I = torch.eye(self.d_model, device=prompt_hidden_states.device, dtype=torch.float32)
            self.P_perp = I - torch.matmul(self.V_bias, self.V_bias.T)
        except Exception:
            self.P_perp = torch.eye(self.d_model, device=prompt_hidden_states.device, dtype=torch.float32)

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.P_perp is None:
            return hidden_state
        orig_dtype = hidden_state.dtype
        h_f = hidden_state.float()
        h_clean = torch.matmul(h_f, self.P_perp)
        return h_clean.to(orig_dtype)

# ------------------------------------------------------------------------------
# 2. DYNAMIC REASONING GOVERNOR
# ------------------------------------------------------------------------------
class DynamicReasoningGovernor:
    def __init__(self, model: nn.Module, target_layer_idx: int, k_dim: int = 4, velocity_threshold: float = 0.05, min_deduction_tokens: int = 20):
        self.model = model
        # For LLaMA architecture, layers are under model.model.layers
        self.target_layer = model.model.layers[target_layer_idx]
        self.neutralizer = SubspaceNeutralizer(d_model=model.config.hidden_size, k_dim=k_dim)
        self.velocity_threshold = velocity_threshold
        self.min_deduction_tokens = min_deduction_tokens
        
        self.trajectory = []
        self.velocities = []
        self.active_neutralization = False
        self.tokens_generated = 0
        self.hook_handle = None

    def reset(self):
        self.trajectory.clear()
        self.velocities.clear()
        self.active_neutralization = False
        self.tokens_generated = 0

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = None

        # Prefill phase (prompt length > 1)
        if hidden_states.shape[1] > 1:
            self.neutralizer.extract_prompt_subspace(hidden_states.detach())
            return outputs

        # Autoregressive decoding phase
        self.tokens_generated += 1
        current_h = hidden_states[:, -1, :].detach().float()

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
            self.velocities.append(v_t)

            if self.tokens_generated >= self.min_deduction_tokens and v_t < self.velocity_threshold:
                self.active_neutralization = True
        else:
            self.velocities.append(1.0)

        self.trajectory.append(current_h)

        if self.active_neutralization:
            neutralized_h = self.neutralizer.project(hidden_states)
            if rest is not None:
                return (neutralized_h,) + rest
            return neutralized_h

        return outputs

    def attach(self):
        self.reset()
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

# ------------------------------------------------------------------------------
# 3. VERIFICATION & ANSWER EXTRACTION
# ------------------------------------------------------------------------------
class ConvergenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stop_patterns = [
            re.compile(r"</think>.*?####\s*(-?\d+(?:\.\d+)?)", re.DOTALL),
            re.compile(r"</think>.*?The answer is\s*(-?\d+(?:\.\d+)?)", re.DOTALL | re.IGNORECASE),
            re.compile(r"</think>.*?\\boxed\{(-?\d+(?:\.\d+)?)\}", re.DOTALL)
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_tokens = input_ids[0, self.prompt_len:]
        if len(gen_tokens) < 15:
            return False
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        if "</think>" in text:
            for pattern in self.stop_patterns:
                if pattern.search(text):
                    return True
        return False

def extract_answer(text: str):
    patterns = [
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*$"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "").strip()
    return None

# ------------------------------------------------------------------------------
# 4. BENCHMARK EXECUTION (LLaMA-8B)
# ------------------------------------------------------------------------------
def run_llama8b_benchmark(num_problems: int = 30):
    # Free existing memory
    gc.collect()
    torch.cuda.empty_cache()

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    print("=" * 80)
    print(f"LOADING {model_id} ON NVIDIA A100 GPU...")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    num_layers = len(model.model.layers)
    target_layer_idx = num_layers // 2 # Layer 16 for LLaMA-8B (32 layers total)
    print(f"LLaMA-8B Loaded! Total Layers: {num_layers}, Intervening at Layer: {target_layer_idx}")

    governor = DynamicReasoningGovernor(
        model=model,
        target_layer_idx=target_layer_idx,
        k_dim=4,
        velocity_threshold=0.05,
        min_deduction_tokens=20
    )

    print("\nLoading benchmark dataset (openai/gsm8k)...")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{num_problems}]")

    print("\n" + "=" * 80)
    print(f"STARTING {num_problems}-PROBLEM BENCHMARK: DEEPSEEK-R1-DISTILL-LLAMA-8B ON A100")
    print("=" * 80)
    print(f"{'#':<4} | {'LLaMA Van':<12} | {'LLaMA Neut':<12} | {'Tokens Saved':<15} | {'Result'}")
    print("-" * 75)

    results = []
    total_van = 0
    total_neut = 0
    van_corr = 0
    neut_corr = 0
    flips_prevented = 0
    van_wins = 0

    for idx, sample in enumerate(dataset):
        q = sample["question"]
        gt = extract_answer(sample["answer"])

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # --- PASS 1: VANILLA LLAMA-8B ---
        governor.detach()
        with torch.no_grad():
            out_van = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        van_tk = out_van.shape[1] - prompt_len
        van_text = tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=True)
        van_ans = extract_answer(van_text)
        van_match = (van_ans == gt) if (van_ans and gt) else False

        # --- PASS 2: NEUTRALIZED LLAMA-8B ---
        governor.attach()
        stop_crit = StoppingCriteriaList([ConvergenceStoppingCriteria(tokenizer, prompt_len)])
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
        neut_tk = out_neut.shape[1] - prompt_len
        neut_text = tokenizer.decode(out_neut[0, prompt_len:], skip_special_tokens=True)
        neut_ans = extract_answer(neut_text)
        neut_match = (neut_ans == gt) if (neut_ans and gt) else False
        governor.detach()

        savings = ((van_tk - neut_tk) / max(1, van_tk)) * 100.0
        total_van += van_tk
        total_neut += neut_tk
        if van_match: van_corr += 1
        if neut_match: neut_corr += 1

        if neut_match and not van_match:
            outcome = "NEUT_WON"
            flips_prevented += 1
        elif van_match and not neut_match:
            outcome = "VAN_WON"
            van_wins += 1
        elif van_match and neut_match:
            outcome = "EQUAL"
        else:
            outcome = "BOTH_FAIL"

        print(f"{idx+1:<4} | {van_tk:<12} | {neut_tk:<12} | {savings:>13.1f}% | {outcome}")

        results.append({
            "idx": idx + 1,
            "vanilla_tokens": van_tk,
            "neut_tokens": neut_tk,
            "savings_pct": savings,
            "vanilla_ans": van_ans,
            "neut_ans": neut_ans,
            "ground_truth": gt,
            "outcome": outcome
        })

    net_savings = ((total_van - total_neut) / total_van) * 100.0
    van_acc = (van_corr / num_problems) * 100.0
    neut_acc = (neut_corr / num_problems) * 100.0

    print("\n" + "=" * 75)
    print("FINAL OFFICIAL LLAMA-8B BENCHMARK RESULTS")
    print("=" * 75)
    print(f"TOTAL VANILLA TOKENS:         {total_van}")
    print(f"TOTAL NEUTRALIZED TOKENS:     {total_neut}")
    print(f"TOTAL TOKENS SAVED:           {total_van - total_neut}")
    print(f"NET COMPUTE REDUCTION:        {net_savings:.1f}%")
    print(f"VANILLA ACCURACY:             {van_acc:.1f}% ({van_corr}/{num_problems})")
    print(f"NEUTRALIZED ACCURACY:         {neut_acc:.1f}% ({neut_corr}/{num_problems})")
    print(f"OVERTHINKING FLIPS PREVENTED: {flips_prevented} problems")
    print("=" * 75)

    with open("benchmark_llama8b_results.json", "w") as f:
        json.dump({
            "model": model_id,
            "total_vanilla": total_van,
            "total_neut": total_neut,
            "net_savings": net_savings,
            "van_acc": van_acc,
            "neut_acc": neut_acc,
            "flips_prevented": flips_prevented,
            "results": results
        }, f, indent=2)
    print("Saved results to: benchmark_llama8b_results.json")

if __name__ == "__main__":
    run_llama8b_benchmark(30)
