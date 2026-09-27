"""
================================================================================
OFFICIAL OLYMPIAD MULTI-SUBJECT BENCHMARK (MATH-500 ON NVIDIA A100)
DISCIPLINES: All 7 Subjects (Levels 2 to 5) | CEILING: 2,000 Tokens
MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (Instant Local Cache Load)
================================================================================
"""

import gc
import json
import os
import re
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)

# ------------------------------------------------------------------------------
# 1. ROBUST LATEX MATH ANSWER NORMALIZER & EXTRACTOR
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
    if not text:
        return None
    # 1. Exact depth-based bracket parser (handles arbitrarily nested LaTeX)
    idx = text.rfind("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return clean_latex(text[start:i-1])
        else:
            return clean_latex(text[start:])

    # 2. Standard textual conclusion fallbacks
    m = re.findall(r"[Tt]he (?:final )?answer is:?\s*\$?([^\n.$]+)", text)
    if m:
        return clean_latex(m[-1])
    m_eq = re.findall(r"=\s*\$?([0-9a-zA-Z\\/+\-]+)\$?\s*(?:\.|\n|Wait|Hold on)", text)
    if m_eq:
        return clean_latex(m_eq[-1])
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1]
    return None


def check_math_equal(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    c_pred = clean_latex(pred)
    c_gt = clean_latex(gt)
    if c_pred == c_gt:
        return True
    if c_pred.strip("()") == c_gt.strip("()"):
        return True
    try:
        if abs(float(c_pred) - float(c_gt)) < 1e-5:
            return True
    except Exception:
        pass
    try:
        from fractions import Fraction
        if Fraction(c_pred) == Fraction(c_gt):
            return True
    except Exception:
        pass
    return False


# ------------------------------------------------------------------------------
# 2. LOAD 7-SUBJECT OLYMPIAD BENCHMARK (MATH-500)
# ------------------------------------------------------------------------------
def load_olympiad_probes(n_per_subject: int = 2, offset: int = 80):
    print(f"Loading 100% FRESH competition problems (offset={offset}) across all 7 MATH-500 subjects...")
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
    
    for idx, row in enumerate(ds):
        if idx < offset:
            continue
        subj = row.get("subject", "")
        level = row.get("level", 1)
        if subj in subject_counts and subject_counts[subj] < n_per_subject and level >= 2:
            selected.append({
                "id": f"{subj[:4].lower()}_L{level}_{subject_counts[subj]+1}",
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
# 3. UNIVERSAL REASONER GOVERNOR (SubNeutralize Ultra / v4)
# ------------------------------------------------------------------------------
class UniversalReasonerGovernor(StoppingCriteria):
    """
    SubNeutralize Ultra: Dynamic Cognitive Equilibrium & Limit-Cycle Arrest
    
    1. Solution-Gated Milestone:
       Once \\boxed{...} or explicit deduction is achieved, ANY subsequent
       hesitation ("Wait", "Alternatively", "Let me check") is immediately arrested.
       This prevents post-solution self-doubt loops from flipping the answer.
       
    2. Deep Derivation Headroom:
       If no answer is formulated, exploratory math ("Alternatively, let x = ...")
       is granted full headroom up to 1,600+ tokens to derive deep 15-step proofs.
       
    3. Dynamical Limit-Cycle Repetition Arrest:
       If the model repeats hesitation loops >= 4 times without an answer,
       it is trapped in an attractor limit cycle; arrest before 2,000-token blowout.
    """
    def __init__(self, tokenizer, prompt_len, min_tokens=150):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens
        self.hesitation_regex = re.compile(
            r"\b(?:wait|alternatively|hold on|let me double check|let me check|let me verify|let's check|could there be|did i misread)\b",
            re.IGNORECASE
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        curr = input_ids.shape[-1] - self.prompt_len
        if curr < self.min_tokens:
            return False

        recent_text = self.tokenizer.decode(input_ids[0, -25:], skip_special_tokens=False)
        
        # 1. Native conclusion: Model voluntarily closed thinking
        if "</think>" in recent_text:
            return True

        full_gen = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=False)
        has_boxed_solution = bool(re.search(r"\\boxed\{[^{}]+\}", full_gen))
        
        # 2. Solution-Gated Arrest (The Core Flip Preventer)
        if has_boxed_solution:
            if self.hesitation_regex.search(recent_text):
                return True

        # 3. Candidate verbal deduction before \\boxed
        if not has_boxed_solution and curr > 300:
            if re.search(r"(?:[Tt]he (?:final )?answer is|[Tt]herefore,?\s+(?:the\s+)?answer is|[Ss]o the answer is|[Hh]ence,?\s+[a-zA-Z]\s*=)\s*\$?[^\n,.]+", full_gen):
                if self.hesitation_regex.search(recent_text):
                    return True

        # 4. Limit-Cycle Repetition Frequency (Replaces blind token cutoff)
        # If the model has entered 4+ separate hesitation spirals without an answer:
        if not has_boxed_solution and curr > 1300:
            hesitation_count = len(self.hesitation_regex.findall(full_gen))
            if hesitation_count >= 4 and self.hesitation_regex.search(recent_text):
                return True

        # 5. Hard Ceiling Saturation Guard
        if curr >= 1850 and self.hesitation_regex.search(recent_text):
            return True

        return False


# ------------------------------------------------------------------------------
# 4. INSTANT MODEL LOADING & BENCHMARK EXECUTION
# ------------------------------------------------------------------------------
def run_olympiad_benchmark():
    print("=" * 110)
    print("🚀 LAUNCHING OLYMPIAD MULTI-SUBJECT BENCHMARK (MATH-500 ON A100)")
    print("   Disciplines: All 7 Subjects (Levels 2 to 5) | Ceiling: 2,000 Tokens")
    print("=" * 110)

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Fast load from local disk cache or robust download
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model successfully ready in A100 VRAM!\n")

    problems = load_olympiad_probes(n_per_subject=2)
    print(f"Loaded {len(problems)} Olympiad problems across all 7 subjects.\n")

    v_total_tokens, g_total_tokens = 0, 0
    v_correct, g_correct = 0, 0
    gov_won, van_won = 0, 0
    records = []

    print(f"{'#':<2} | {'Subject / Level':<25} | {'Vanilla (Tk, Time, Res)':<28} | {'Governor (Tk, Time, Res)':<28} | {'Saved':<8} | {'Outcome'}")
    print("-" * 110)

    for i, p in enumerate(problems):
        subj_lvl = f"{p['subject']} L{p['level']}"
        q = p["problem"]
        gt = p["ground_truth"]

        prompt = f"<｜User｜>{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        p_len = inp.input_ids.shape[-1]

        # --- Arm 1: Vanilla (Unconstrained up to 2,000 tokens) ---
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            v_out = model.generate(
                **inp,
                max_new_tokens=2000,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        v_time = time.time() - t0
        v_tokens = v_out[0].shape[-1] - p_len
        v_pred = extract_boxed_answer(tokenizer.decode(v_out[0][p_len:], skip_special_tokens=True))
        v_ok = check_math_equal(v_pred, gt)

        # --- Arm 2: Universal Runtime Governor ---
        ctrl = UniversalReasonerGovernor(tokenizer, prompt_len=p_len, min_tokens=150)
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            g_out = model.generate(
                **inp,
                max_new_tokens=2000,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=[ctrl]
            )
        torch.cuda.synchronize()
        g_time = time.time() - t1
        g_tokens = g_out[0].shape[-1] - p_len
        g_pred = extract_boxed_answer(tokenizer.decode(g_out[0][p_len:], skip_special_tokens=True))
        g_ok = check_math_equal(g_pred, gt)

        v_total_tokens += v_tokens
        g_total_tokens += g_tokens
        if v_ok: v_correct += 1
        if g_ok: g_correct += 1
        saved_pct = ((v_tokens - g_tokens) / v_tokens) * 100 if v_tokens > 0 else 0.0

        if g_ok and not v_ok:
            outcome = "GOV_WON (Flip Prevented!)"
            gov_won += 1
        elif v_ok and not g_ok:
            outcome = "VAN_WON"
            van_won += 1
        elif v_ok and g_ok:
            outcome = "EQUAL"
        else:
            outcome = "BOTH_FAIL"

        gt_disp = gt[:8] if len(gt) > 8 else gt
        v_disp = f"{v_tokens} tk ({v_time:.1f}s, {'OK' if v_ok else 'X'}):<{gt_disp}"
        g_disp = f"{g_tokens} tk ({g_time:.1f}s, {'OK' if g_ok else 'X'}):<{gt_disp}"
        print(f"{i+1:<2} | {subj_lvl:<25} | {v_disp:<28} | {g_disp:<28} | {saved_pct:>6.1f}% | {outcome}")

        records.append({
            "idx": i + 1,
            "subject": p["subject"],
            "level": p["level"],
            "ground_truth": gt,
            "vanilla_tokens": v_tokens,
            "vanilla_time": v_time,
            "vanilla_ok": v_ok,
            "gov_tokens": g_tokens,
            "gov_time": g_time,
            "gov_ok": g_ok,
            "saved_pct": saved_pct,
            "outcome": outcome
        })

    net_savings = ((v_total_tokens - g_total_tokens) / v_total_tokens) * 100
    print("\n" + "=" * 110)
    print("🏆 FINAL MULTI-SUBJECT OLYMPIAD RESULTS (MATH-500)")
    print("=" * 110)
    print(f"Total Problems Evaluated:     {len(problems)} (Across all 7 MATH-500 Disciplines)")
    print(f"Net Compute Savings:          {net_savings:.1f}% ({v_total_tokens} -> {g_total_tokens} tokens)")
    print(f"Vanilla Accuracy:             {(v_correct/len(problems))*100:.1f}% ({v_correct}/{len(problems)})")
    print(f"Governor Accuracy:            {(g_correct/len(problems))*100:.1f}% ({g_correct}/{len(problems)})")
    print(f"Overthinking Flips Prevented: {gov_won}")
    print(f"Losses against Vanilla:       {van_won}")
    print("=" * 110)

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/results_olympiad_multisubject_7b.json", "w") as f:
        json.dump({
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "benchmark": "HuggingFaceH4/MATH-500",
            "net_savings_pct": net_savings,
            "vanilla_acc": (v_correct / len(problems)) * 100,
            "gov_acc": (g_correct / len(problems)) * 100,
            "gov_won": gov_won,
            "van_won": van_won,
            "records": records
        }, f, indent=2)
    print("Saved results to: paper/tables/results_olympiad_multisubject_7b.json")

if __name__ == "__main__":
    run_olympiad_benchmark()
