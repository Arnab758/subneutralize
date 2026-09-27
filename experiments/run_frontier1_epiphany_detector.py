"""
====================================================================================================
FRONTIER 1: ADAPTIVE TRAJECTORY GOVERNOR (EPIPHANY VS. PATHOLOGICAL LIMIT CYCLE DETECTOR)
MODEL: DeepSeek-R1-Distill-Qwen-7B (NVIDIA A100 GPU)
DATASET: GSM8K (Specifically targeting the known Regression/Loss cases + Won cases)

CORE SCIENTIFIC OBJECTIVE:
    Can we eliminate the ~8% Governor Loss Rate by distinguishing:
    1. PATHOLOGICAL LIMIT CYCLE (Second-guessing relapse):
       - Model cycles in previously explored semantic subspace.
       - Effective covariance rank (Participation Ratio) collapses: PR_t < 2.0.
       - Subspace Innovation from history is near zero: Innov_t < 0.15.
       -> ACTION: HALT / NEUTRALIZE IMMEDIATELY (Flips Prevented).

    2. LEGITIMATE ERROR EPIPHANY (Constructive self-correction):
       - Model caught an arithmetic blunder and launched a genuine re-derivation.
       - Effective covariance rank explodes into new dimensions: PR_t >= 3.5.
       - Subspace Innovation surges: Innov_t >= 0.30.
       -> ACTION: EXTEND BUDGET & ALLOW DEDUCTION (Recover Vanilla Win).
====================================================================================================
"""

import gc
import json
import os
import re
import sys
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

# --------------------------------------------------------------------------------------------------
# Regex Extraction & Mathematical Verification
# --------------------------------------------------------------------------------------------------
def extract_answer(text: str):
    patterns = [
        r"\\boxed\{([^{}]+)\}",
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*$"
    ]
    for p in patterns:
        matches = list(re.finditer(p, text))
        if matches:
            ans = matches[-1].group(1).replace(",", "").strip()
            # If boxed contained nested or formula, strip
            num_m = re.search(r"-?\d+(?:\.\d+)?", ans)
            if num_m:
                return num_m.group(0)
            return ans
    return None

def check_correct(pred_str, truth_str):
    if pred_str is None or truth_str is None:
        return False
    try:
        p_clean = re.sub(r"[^\d.-]", "", str(pred_str)).strip()
        t_clean = re.sub(r"[^\d.-]", "", str(truth_str)).strip()
        if not p_clean or not t_clean:
            return str(pred_str).strip() == str(truth_str).strip()
        return abs(float(p_clean) - float(t_clean)) < 1e-4
    except Exception:
        return str(pred_str).strip() == str(truth_str).strip()

# --------------------------------------------------------------------------------------------------
# Geometric Trajectory Metrics (Participation Ratio & Subspace Innovation)
# --------------------------------------------------------------------------------------------------
def compute_participation_ratio(h_window: torch.Tensor) -> float:
    """
    Computes participation ratio (effective dimension) of the sliding window covariance.
    h_window: [W, d] tensor of recent hidden states.
    PR = (Tr(C))^2 / Tr(C^2) = (sum lambda_i)^2 / sum (lambda_i^2)
    """
    W, d = h_window.shape
    if W < 3:
        return 1.0
    centered = h_window - h_window.mean(dim=0, keepdim=True)
    # Gram matrix [W, W] for efficiency (W << d)
    gram = torch.matmul(centered, centered.T) / W
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, min=0.0)
    sum_e = eigvals.sum().item()
    sum_sq = (eigvals ** 2).sum().item()
    if sum_sq < 1e-9:
        return 1.0
    return (sum_e ** 2) / sum_sq

def compute_subspace_innovation(h_t: torch.Tensor, v_hist_basis: torch.Tensor) -> float:
    """
    Measures the fraction of energy in h_t that lies outside the historical deduction basis.
    v_hist_basis: [d, k] orthonormal basis of historical deduction tokens.
    Innovation = 1 - ||V_hist^T h_t||^2 / ||h_t||^2
    """
    if v_hist_basis is None or v_hist_basis.shape[1] == 0:
        return 1.0
    h_norm_sq = torch.sum(h_t ** 2).item()
    if h_norm_sq < 1e-9:
        return 0.0
    proj = torch.matmul(v_hist_basis.T, h_t)
    proj_norm_sq = torch.sum(proj ** 2).item()
    fraction_in_history = min(1.0, max(0.0, proj_norm_sq / h_norm_sq))
    return 1.0 - fraction_in_history

# --------------------------------------------------------------------------------------------------
# Main Experiment Runner
# --------------------------------------------------------------------------------------------------
def main():
    print("=" * 105)
    print("🚀 FRONTIER 1: ADAPTIVE TRAJECTORY GOVERNOR BENCHMARK (A100 GPU)")
    print("   Testing Dynamic Epiphany Detection on Known Loss & Win Problems")
    print("=" * 105)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load GSM8K
    print("\nLoading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Target Problems:
    # 4 Known Vanilla-Won (Governor Loss) Cases: Idx 17, 22, 28, 39 (1-indexed -> 16, 21, 27, 38)
    # 6 Known Neut-Won (Governor Win) Cases: Idx 1, 2, 5, 10, 14, 18 (1-indexed -> 0, 1, 4, 9, 13, 17)
    target_indices = [
        (16, "VAN_WON_LOSS", 17),
        (21, "VAN_WON_LOSS", 22),
        (27, "VAN_WON_LOSS", 28),
        (38, "VAN_WON_LOSS", 39),
        (0,  "NEUT_WON_WIN",  1),
        (1,  "NEUT_WON_WIN",  2),
        (4,  "NEUT_WON_WIN",  5),
        (9,  "NEUT_WON_WIN", 10),
        (13, "NEUT_WON_WIN", 14),
        (17, "NEUT_WON_WIN", 18),
    ]

    target_layer_idx = 14
    W_WINDOW = 10
    k_prompt = 4
    k_hist = 6

    print(f"Target Layer: {target_layer_idx} | Window Size: {W_WINDOW} | k_prompt: {k_prompt}")
    print("\n" + "=" * 105)
    print(f"{'#':<3} | {'Type':<13} | {'Vanilla Tk':<12} | {'Static Gov':<12} | {'Adaptive Gov (F1)':<18} | {'Epiphany Detected?':<20} | {'Outcome'}")
    print("-" * 105)

    results_log = []

    for target_idx, case_type, display_id in target_indices:
        item = ds[target_idx]
        question = item["question"]
        gt_answer = extract_answer(item["answer"])

        prompt_text = (
            f"<｜User｜>{question}"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
        )
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        # ------------------------------------------------------------------------------------------
        # ARM 1: VANILLA GENERATION (max 450 tokens)
        # ------------------------------------------------------------------------------------------
        with torch.no_grad():
            out_van = model.generate(
                **enc,
                max_new_tokens=450,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        van_text = tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=False)
        van_ans = extract_answer(van_text)
        van_correct = check_correct(van_ans, gt_answer)
        van_tokens = len(out_van[0, prompt_len:])

        # ------------------------------------------------------------------------------------------
        # PREFILL & PROMPT SVD FOR GOVERNORS
        # ------------------------------------------------------------------------------------------
        with torch.no_grad():
            base_out = model(**enc, output_hidden_states=True, use_cache=True)
            past_kv = base_out.past_key_values
            h_prompt = base_out.hidden_states[target_layer_idx][0]  # [S, d]
            centered_prompt = h_prompt - h_prompt.mean(dim=0, keepdim=True)
            _, _, V_prompt = torch.linalg.svd(centered_prompt.float(), full_matrices=False)
            V_bias = V_prompt[:k_prompt, :].T.to(device).to(torch.bfloat16)  # [d, k]
            P_perp = torch.eye(V_bias.shape[0], device=device, dtype=torch.bfloat16) - torch.matmul(V_bias, V_bias.T)

        # ------------------------------------------------------------------------------------------
        # ARM 2: STATIC GOVERNOR (Standard SubNeutralize)
        # ------------------------------------------------------------------------------------------
        # Single-token step simulation with static halt on v_t < 0.05
        cur_input = out_van[0, -1:].unsqueeze(0)
        # We run step-by-step with hooks
        curr_kv = past_kv
        curr_token = enc.input_ids[:, -1:]
        gen_tokens_static = []
        static_hidden_history = []
        static_correct = False
        static_ans = None

        hook_handle = None
        phase_neutralized = False

        def static_hook(module, args, output):
            nonlocal phase_neutralized
            if phase_neutralized:
                h = output[0] if isinstance(output, tuple) else output
                h = torch.matmul(h, P_perp)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return output

        hook_handle = model.model.layers[target_layer_idx].register_forward_hook(static_hook)

        curr_kv = past_kv
        curr_token = enc.input_ids[:, -1:]
        all_gen_static = []

        for step in range(450):
            with torch.no_grad():
                out_step = model(
                    input_ids=curr_token,
                    past_key_values=curr_kv,
                    output_hidden_states=True,
                    use_cache=True
                )
            curr_kv = out_step.past_key_values
            h_t = out_step.hidden_states[target_layer_idx][0, -1].float()
            static_hidden_history.append(h_t)

            logits = out_step.logits[0, -1, :] / 0.6
            probs = F.softmax(logits, dim=-1)
            next_tk = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            curr_token = next_tk
            all_gen_static.append(next_tk.item())

            # Velocity check
            if len(static_hidden_history) >= 2 and step >= 20:
                v_t = 1.0 - F.cosine_similarity(static_hidden_history[-1].unsqueeze(0), static_hidden_history[-2].unsqueeze(0)).item()
                if v_t < 0.05:
                    phase_neutralized = True

            gen_text = tokenizer.decode(all_gen_static, skip_special_tokens=False)
            if "</think>" in gen_text and ("\\boxed{" in gen_text or "####" in gen_text):
                break

        hook_handle.remove()
        static_tokens = len(all_gen_static)
        static_ans = extract_answer(gen_text)
        static_correct = check_correct(static_ans, gt_answer)

        # ------------------------------------------------------------------------------------------
        # ARM 3: FRONTIER 1 ADAPTIVE GOVERNOR (Curvature + Covariance Rank + Subspace Innovation)
        # ------------------------------------------------------------------------------------------
        curr_kv = past_kv
        curr_token = enc.input_ids[:, -1:]
        all_gen_adapt = []
        hidden_history_adapt = []
        epiphany_detected = False
        phase_neutralized_adapt = False
        extra_budget_granted = 0

        def adapt_hook(module, args, output):
            nonlocal phase_neutralized_adapt
            if phase_neutralized_adapt:
                h = output[0] if isinstance(output, tuple) else output
                h = torch.matmul(h, P_perp)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return output

        hook_handle_adapt = model.model.layers[target_layer_idx].register_forward_hook(adapt_hook)

        for step in range(450):
            with torch.no_grad():
                out_step = model(
                    input_ids=curr_token,
                    past_key_values=curr_kv,
                    output_hidden_states=True,
                    use_cache=True
                )
            curr_kv = out_step.past_key_values
            h_t = out_step.hidden_states[target_layer_idx][0, -1].float()
            hidden_history_adapt.append(h_t)

            logits = out_step.logits[0, -1, :] / 0.6
            probs = F.softmax(logits, dim=-1)
            next_tk = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            curr_token = next_tk
            all_gen_adapt.append(next_tk.item())

            # Every token, monitor Trajectory Geometry
            if len(hidden_history_adapt) > W_WINDOW and step >= 20:
                h_window = torch.stack(hidden_history_adapt[-W_WINDOW:])  # [W, d]
                pr_t = compute_participation_ratio(h_window)

                # Historical deduction basis from tokens 0 to step - W_WINDOW
                if step > 25:
                    h_hist = torch.stack(hidden_history_adapt[:step - W_WINDOW])
                    _, _, V_h = torch.linalg.svd(h_hist - h_hist.mean(dim=0, keepdim=True), full_matrices=False)
                    v_hist_basis = V_h[:k_hist, :].T  # [d, k_hist]
                    innov_t = compute_subspace_innovation(h_t, v_hist_basis)
                else:
                    innov_t = 0.0

                v_t = 1.0 - F.cosine_similarity(hidden_history_adapt[-1].unsqueeze(0), hidden_history_adapt[-2].unsqueeze(0)).item()

                # CLASSIFY DYNAMICS:
                # Is this a Legitimate Error Epiphany?
                if (pr_t >= 3.2 or innov_t >= 0.28) and v_t > 0.08:
                    epiphany_detected = True
                    extra_budget_granted += 1
                    # Keep neutralizing turned OFF to allow new deduction to freely form!
                    phase_neutralized_adapt = False

                # Is this a Pathological Limit Cycle?
                elif v_t < 0.05 and pr_t < 2.0:
                    phase_neutralized_adapt = True

            gen_text_adapt = tokenizer.decode(all_gen_adapt, skip_special_tokens=False)
            if "</think>" in gen_text_adapt and ("\\boxed{" in gen_text_adapt or "####" in gen_text_adapt):
                # If epiphany in progress, allow it to complete final expression
                if not epiphany_detected or "\\boxed{" in gen_text_adapt:
                    break

        hook_handle_adapt.remove()
        adapt_tokens = len(all_gen_adapt)
        adapt_ans = extract_answer(gen_text_adapt)
        adapt_correct = check_correct(adapt_ans, gt_answer)

        # Outcome Determination
        if adapt_correct and not van_correct:
            outcome_str = "ADAPT_WON (Flip Prevented!)"
        elif adapt_correct and van_correct and not static_correct:
            outcome_str = "RECOVERED_VAN_WIN! (0% Loss Achieved!)"
        elif adapt_correct and van_correct:
            outcome_str = "EQUAL"
        elif not adapt_correct and van_correct:
            outcome_str = "VAN_WON (Remaining Loss)"
        else:
            outcome_str = "BOTH_FAIL"

        epiphany_str = f"YES (PR={pr_t:.2f}, In={innov_t:.2f})" if epiphany_detected else "NO (Cycle Halt)"
        print(f"{display_id:<3} | {case_type:<13} | {van_tokens} tk ({'OK' if van_correct else 'X'}) | {static_tokens} tk ({'OK' if static_correct else 'X'}) | {adapt_tokens} tk ({'OK' if adapt_correct else 'X'}) | {epiphany_str:<20} | {outcome_str}")

        results_log.append({
            "id": display_id,
            "type": case_type,
            "vanilla": {"tokens": van_tokens, "correct": van_correct, "ans": van_ans},
            "static_gov": {"tokens": static_tokens, "correct": static_correct, "ans": static_ans},
            "adaptive_gov": {"tokens": adapt_tokens, "correct": adapt_correct, "ans": adapt_ans, "epiphany": epiphany_detected},
            "outcome": outcome_str
        })

    # Save Results
    os.makedirs("paper/tables", exist_ok=True)
    out_file = "paper/tables/results_frontier1_epiphany.json"
    with open(out_file, "w") as f:
        json.dump(results_log, f, indent=2)

    print("\n" + "=" * 105)
    print(f"✅ Frontier 1 Benchmark Complete! Saved to {out_file}")
    print("=" * 105)

if __name__ == "__main__":
    main()
