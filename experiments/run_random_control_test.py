"""
================================================================================
TASK 2: RANDOM-SVD SUBSPACE CONTROL EXPERIMENT (REVIEWER ATTACK #9 DEFENSE)
MODEL: DeepSeek-R1-Distill-Qwen-7B (32 Layers, Hidden Size 3584)
PURPOSE:
    Directly refute the reviewer critique:
    "How do we know any 4-dimensional projection doesn't stabilize reasoning?
    What if removing ANY arbitrary low-rank subspace suppresses second-guessing?"

    We evaluate 3 conditions under identical prompts and hyperparameters:
    1. Vanilla Baseline (No intervention)
    2. SubNeutralize (Projection along V_bias prompt-attractor directions)
    3. Random Subspace Control (Projection along random orthonormal V_rand)
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
# SUBSPACE CONTROLLER (SUPPORTS V_bias vs V_rand)
# ------------------------------------------------------------------------------
class SubspaceControlGovernor:
    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = 14,
        k_dim: int = 4,
        velocity_threshold: float = 0.05,
        min_deduction_tokens: int = 20,
        mode: str = "bias"  # "bias" or "random"
    ):
        self.model = model
        self.target_layer = model.model.layers[target_layer_idx]
        self.d_model = model.config.hidden_size
        self.k_dim = k_dim
        self.velocity_threshold = velocity_threshold
        self.min_deduction_tokens = min_deduction_tokens
        self.mode = mode

        self.P_perp = None
        self.trajectory = []
        self.active_neutralization = False
        self.tokens_generated = 0
        self.hook_handle = None

    def reset(self, mode: str = "bias"):
        self.mode = mode
        self.P_perp = None
        self.trajectory.clear()
        self.active_neutralization = False
        self.tokens_generated = 0

    def compute_projection_matrix(self, prompt_hidden_states: torch.Tensor):
        device = prompt_hidden_states.device
        if self.mode == "bias":
            # True Prompt-Bias Subspace (SVD of centered prompt residual states)
            H = prompt_hidden_states.squeeze(0).float()
            H_centered = H - H.mean(dim=0, keepdim=True)
            try:
                _, _, V = torch.linalg.svd(H_centered, full_matrices=False)
                V_bias = V[:self.k_dim, :].T.to(device)
                I = torch.eye(self.d_model, device=device, dtype=torch.float32)
                self.P_perp = I - torch.matmul(V_bias, V_bias.T)
            except Exception:
                self.P_perp = torch.eye(self.d_model, device=device, dtype=torch.float32)
        elif self.mode == "random":
            # Random Orthogonal Subspace Control (QR decomposition of random Gaussian vectors)
            torch.manual_seed(42)
            G = torch.randn(self.d_model, self.k_dim, device=device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(G)
            I = torch.eye(self.d_model, device=device, dtype=torch.float32)
            self.P_perp = I - torch.matmul(Q, Q.T)
        else:
            self.P_perp = torch.eye(self.d_model, device=device, dtype=torch.float32)

    def _hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = None

        # Prefill phase: compute projection operator
        if hidden_states.shape[1] > 1:
            self.compute_projection_matrix(hidden_states.detach())
            return outputs

        # Autoregressive decoding phase
        self.tokens_generated += 1
        current_h = hidden_states[:, -1, :].detach().float()

        if len(self.trajectory) > 0:
            prev_h = self.trajectory[-1]
            cos_sim = F.cosine_similarity(current_h, prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)

            if self.tokens_generated >= self.min_deduction_tokens and v_t < self.velocity_threshold:
                self.active_neutralization = True
        self.trajectory.append(current_h)

        if self.active_neutralization and self.P_perp is not None:
            orig_dtype = hidden_states.dtype
            h_f = hidden_states.float()
            neutralized_h = torch.matmul(h_f, self.P_perp).to(orig_dtype)
            if rest is not None:
                return (neutralized_h,) + rest
            return neutralized_h

        return outputs

    def attach(self):
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def detach(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

# ------------------------------------------------------------------------------
# BENCHMARK EXECUTION (3-ARM COMPARISON: VANILLA vs V_BIAS vs V_RANDOM)
# ------------------------------------------------------------------------------
def run_random_control_experiment(num_problems: int = 15):
    print("=" * 80)
    print("TASK 2: RANDOM-SVD CONTROL BENCHMARK ON DEEPSEEK-R1-DISTILL-QWEN-7B")
    print(f"Sample Size: {num_problems} problems | Arms: [Vanilla, SubNeutralize (V_bias), Random-SVD (V_rand)]")
    print("=" * 80)

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

    governor = SubspaceControlGovernor(model=model, target_layer_idx=14, k_dim=4, velocity_threshold=0.05)
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{num_problems}]")

    print("\n" + "=" * 90)
    print(f"{'#':<4} | {'Van Tk (Acc)':<15} | {'V_bias Tk (Acc)':<18} | {'V_rand Tk (Acc)':<18} | {'GT':<8}")
    print("-" * 90)

    results = []
    stats = {"van": {"tokens": 0, "correct": 0}, "bias": {"tokens": 0, "correct": 0}, "rand": {"tokens": 0, "correct": 0}}

    for idx, sample in enumerate(dataset):
        q = sample["question"]
        gt = extract_answer(sample["answer"])

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # Arm 1: Vanilla
        governor.detach()
        with torch.no_grad():
            out_van = model.generate(
                **inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True,
                stopping_criteria=StoppingCriteriaList([QwenStoppingCriteria(tokenizer, prompt_len)]),
                pad_token_id=tokenizer.eos_token_id
            )
        van_tk = out_van.shape[1] - prompt_len
        van_ans = extract_answer(tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=True))
        van_corr = (van_ans == gt) if (van_ans and gt) else False
        stats["van"]["tokens"] += van_tk
        if van_corr: stats["van"]["correct"] += 1

        # Arm 2: SubNeutralize (V_bias)
        governor.reset(mode="bias")
        governor.attach()
        with torch.no_grad():
            out_bias = model.generate(
                **inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True,
                stopping_criteria=StoppingCriteriaList([QwenStoppingCriteria(tokenizer, prompt_len)]),
                pad_token_id=tokenizer.eos_token_id
            )
        governor.detach()
        bias_tk = out_bias.shape[1] - prompt_len
        bias_ans = extract_answer(tokenizer.decode(out_bias[0, prompt_len:], skip_special_tokens=True))
        bias_corr = (bias_ans == gt) if (bias_ans and gt) else False
        stats["bias"]["tokens"] += bias_tk
        if bias_corr: stats["bias"]["correct"] += 1

        # Arm 3: Random Control (V_rand)
        governor.reset(mode="random")
        governor.attach()
        with torch.no_grad():
            out_rand = model.generate(
                **inputs, max_new_tokens=450, temperature=0.6, top_p=0.95, do_sample=True,
                stopping_criteria=StoppingCriteriaList([QwenStoppingCriteria(tokenizer, prompt_len)]),
                pad_token_id=tokenizer.eos_token_id
            )
        governor.detach()
        rand_tk = out_rand.shape[1] - prompt_len
        rand_ans = extract_answer(tokenizer.decode(out_rand[0, prompt_len:], skip_special_tokens=True))
        rand_corr = (rand_ans == gt) if (rand_ans and gt) else False
        stats["rand"]["tokens"] += rand_tk
        if rand_corr: stats["rand"]["correct"] += 1

        print(
            f"{idx+1:<4} | "
            f"{van_tk} ({'OK' if van_corr else 'X'}):<15 | "
            f"{bias_tk} ({'OK' if bias_corr else 'X'}):<18 | "
            f"{rand_tk} ({'OK' if rand_corr else 'X'}):<18 | "
            f"{str(gt):<8}"
        )

        results.append({
            "idx": idx + 1,
            "ground_truth": gt,
            "vanilla": {"tokens": van_tk, "answer": van_ans, "correct": van_corr},
            "bias_subneutralize": {"tokens": bias_tk, "answer": bias_ans, "correct": bias_corr},
            "random_control": {"tokens": rand_tk, "answer": rand_ans, "correct": rand_corr}
        })

    print("\n" + "=" * 90)
    print("FINAL RESULTS: CAUSAL SPECIFICITY OF V_BIAS vs RANDOM SUBSPACE")
    print("=" * 90)
    print(f"1. Vanilla Baseline:          Accuracy: {stats['van']['correct']}/{num_problems} ({stats['van']['correct']/num_problems*100:.1f}%) | Tokens: {stats['van']['tokens']}")
    print(f"2. SubNeutralize (V_bias):    Accuracy: {stats['bias']['correct']}/{num_problems} ({stats['bias']['correct']/num_problems*100:.1f}%) | Tokens: {stats['bias']['tokens']}")
    print(f"3. Random Control (V_rand):   Accuracy: {stats['rand']['correct']}/{num_problems} ({stats['rand']['correct']/num_problems*100:.1f}%) | Tokens: {stats['rand']['tokens']}")
    print("=" * 90)

    output_path = "experiments/results_random_control_test.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model_id,
            "num_problems": num_problems,
            "stats": stats,
            "results": results
        }, f, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    run_random_control_experiment(15)
