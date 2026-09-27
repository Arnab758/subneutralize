"""
================================================================================
CHOICE 2: DEEP OLYMPIAD MATH BENCHMARK (MATH-500 ON NVIDIA A100)
ATTACK #6 (COMPLEX REASONING BEYOND GSM8K) & ATTACK #9 (PRESERVING HARD MATH)
MODEL: DeepSeek-R1-Distill-Qwen-7B
DATASET: HuggingFaceH4/MATH-500 (Competition & Olympiad Problems)
TOKEN CEILING: 3,000 Tokens (Realistic budget for 8-15 step Olympiad proofs)
================================================================================
"""

import gc
import json
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
    # Remove outer text wrapper if present
    m_text = re.search(r"\\text\{([^}]+)\}", s)
    if m_text:
        s = m_text.group(1)
    return s.strip()

def extract_boxed_answer(text: str) -> str:
    # Extracts the content inside the LAST \boxed{...}
    # Handles nested braces up to 2 levels
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return clean_latex(matches[-1])
    # Fallback to "The answer is"
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
    # Try numeric float match if both are numbers
    try:
        if abs(float(c_pred) - float(c_gt)) < 1e-5:
            return True
    except Exception:
        pass
    return False

# ------------------------------------------------------------------------------
# 2. OLYMPIAD TRAJECTORY GOVERNOR (Layer 14)
# ------------------------------------------------------------------------------
class OlympiadVelocityGovernor:
    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = 14,
        velocity_threshold: float = 0.05,
        min_deduction_tokens: int = 400, # Allow at least 400 tokens of genuine algebra
        consensus_patience: int = 4
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
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
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

class OlympiadGovernorCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, governor: OlympiadVelocityGovernor, prompt_len: int, min_tokens: int = 400):
        self.tokenizer = tokenizer
        self.governor = governor
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        curr = input_ids.shape[-1] - self.prompt_len
        if curr < self.min_tokens:
            return False
        
        recent = self.tokenizer.decode(input_ids[0, -20:], skip_special_tokens=False)
        full_gen = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=False)

        # 1. Natural termination if model closed think
        if "</think>" in recent:
            return True

        # 2. If boxed answer has been derived AND model starts second-guessing
        has_boxed = "\\boxed" in full_gen
        if has_boxed and curr >= self.min_tokens:
            if "Wait" in recent or "Alternatively" in recent or "Let me check" in recent:
                return True
            if self.governor.stop_triggered:
                return True

        return False

# ------------------------------------------------------------------------------
# 3. BENCHMARK EXECUTION
# ------------------------------------------------------------------------------
def run_olympiad_math500_benchmark(num_problems: int = 5, max_tokens: int = 3000):
    print("=" * 95)
    print(f"COMMENCING DEEP OLYMPIAD MATH BENCHMARK: MATH-500 (N={num_problems} PROBLEMS)")
    print(f"Token Ceiling: {max_tokens} Tokens | Hardware: NVIDIA A100 GPU")
    print("=" * 95)

    if 'model' not in globals() or model is None:
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
    else:
        print("Reusing active DeepSeek-R1-Distill-Qwen-7B in A100 VRAM.")

    governor = OlympiadVelocityGovernor(model=model, target_layer_idx=14, velocity_threshold=0.05, min_deduction_tokens=400)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    sample_subset = dataset.select(range(num_problems))

    print("\n" + "=" * 110)
    print(f"{'#':<3} | {'Subject':<18} | {'Lvl':<4} | {'Vanilla (Tk, Time, Res)':<26} | {'Governor (Tk, Time, Res)':<26} | {'Tokens Saved':<12} | {'Outcome'}")
    print("-" * 110)

    results = []
    total_van_tokens = 0
    total_neut_tokens = 0
    van_correct = 0
    neut_correct = 0

    for idx, sample in enumerate(sample_subset):
        q = sample["problem"]
        gt = sample["answer"]
        subject = sample.get("subject", "Math")
        level = sample.get("level", "?")

        prompt = f"<｜User｜>{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # --- PASS 1: VANILLA (Max 3,000 tokens) ---
        governor.detach()
        torch.cuda.synchronize()
        t0_van = time.time()
        with torch.no_grad():
            out_van = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        dt_van = time.time() - t0_van
        van_tk = out_van.shape[1] - prompt_len
        van_text = tokenizer.decode(out_van[0, prompt_len:], skip_special_tokens=True)
        van_ans = extract_boxed_answer(van_text)
        van_match = check_math_equal(van_ans, gt)

        # --- PASS 2: OLYMPIAD GOVERNOR (Max 3,000 tokens) ---
        governor.attach()
        torch.cuda.synchronize()
        t0_neut = time.time()
        stop_crit = StoppingCriteriaList([OlympiadGovernorCriteria(tokenizer, governor, prompt_len, min_tokens=400)])
        with torch.no_grad():
            out_neut = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                stopping_criteria=stop_crit,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        dt_neut = time.time() - t0_neut
        neut_tk = out_neut.shape[1] - prompt_len
        neut_text = tokenizer.decode(out_neut[0, prompt_len:], skip_special_tokens=True)
        neut_ans = extract_boxed_answer(neut_text)
        neut_match = check_math_equal(neut_ans, gt)
        governor.detach()

        total_van_tokens += van_tk
        total_neut_tokens += neut_tk
        if van_match: van_correct += 1
        if neut_match: neut_correct += 1

        savings_pct = ((van_tk - neut_tk) / max(1, van_tk)) * 100.0

        if neut_match and not van_match:
            outcome = "NEUT_WON (Flip Saved!)"
        elif van_match and not neut_match:
            outcome = "VAN_WON"
        elif van_match and neut_match:
            outcome = "EQUAL"
        else:
            outcome = "BOTH_FAIL"

        v_str = f"{van_tk} tk ({dt_van:.1f}s, {'OK' if van_match else 'X'})"
        n_str = f"{neut_tk} tk ({dt_neut:.1f}s, {'OK' if neut_match else 'X'})"

        print(
            f"{idx+1:<3} | "
            f"{subject:<18} | "
            f"L{level:<3} | "
            f"{v_str:<26} | "
            f"{n_str:<26} | "
            f"{savings_pct:>10.1f}% | "
            f"{outcome}"
        )
        print(f"    -> Ground Truth: {clean_latex(gt)} | Van Ans: {van_ans} | Neut Ans: {neut_ans}")

        results.append({
            "idx": idx + 1,
            "subject": subject,
            "level": level,
            "ground_truth": gt,
            "vanilla": {"tokens": van_tk, "sec": round(dt_van, 2), "ans": van_ans, "correct": van_match},
            "governor": {"tokens": neut_tk, "sec": round(dt_neut, 2), "ans": neut_ans, "correct": neut_match},
            "outcome": outcome
        })

    net_savings = ((total_van_tokens - total_neut_tokens) / max(1, total_van_tokens)) * 100.0
    print("\n" + "=" * 95)
    print("MATH-500 OLYMPIAD BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"Total Vanilla Tokens:    {total_van_tokens:,}")
    print(f"Total Governor Tokens:   {total_neut_tokens:,}")
    print(f"Net Compute Saved:       {net_savings:.1f}%")
    print(f"Vanilla Accuracy:        {van_correct}/{num_problems} ({van_correct/num_problems*100:.1f}%)")
    print(f"Governor Accuracy:       {neut_correct}/{num_problems} ({neut_correct/num_problems*100:.1f}%)")
    print("=" * 95)

    with open("results_math500_olympiad.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

if __name__ == "__main__":
    run_olympiad_math500_benchmark(num_problems=5, max_tokens=3000)
