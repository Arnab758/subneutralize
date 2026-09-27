"""
================================================================================
EXPERIMENT: Proper Entropy-Adaptive Multi-Layer Consensus Governor
AUTHORS: Arnab Dutta et al.
TARGET: ICLR / NeurIPS / Y-Combinator W27

DESCRIPTION:
Couples Differential Geometry in the residual stream (Riemannian trajectory
velocity v_t across cognitive bottleneck band [12, 14, 16]) with Information
Theory at the language generation head (Instantaneous Shannon Entropy H_t).

THE USER'S ADAPTIVE GOVERNING LAW:
- Low Entropy (Certainty, H_t < 0.8 nats):
  LOWER the threshold (epsilon -> 0.025 - 0.035).
  Strict halt condition protects multi-step arithmetic derivation from
  premature truncation during fluent deduction.
- High Entropy (Confusion / Overthinking Loop, H_t > 1.8 nats):
  WIDEN the threshold (epsilon -> 0.080 - 0.090).
  Relaxed halt condition immediately captures dynamical velocity stagnation
  and halts the model before catastrophic answer flipping occurs.
================================================================================
"""

import os
import re
import sys
import time
import json
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from datasets import load_dataset


# ==============================================================================
# 1. ROBUST ANSWER EXTRACTOR
# ==============================================================================
def extract_answer(text: str) -> Optional[str]:
    """
    Hierarchical answer extractor for GSM8K reasoning completions.
    1. Looks for formal delimiters (####, \\boxed{}, 'The answer is')
    2. Looks for relational equations (equals, =, is)
    3. Fallback: Extracts the trailing numerical entity
    """
    if not text:
        return None
    patterns = [
        r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}",
        r"[Tt]he (?:final )?answer is:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",
        r"(?:equals?|=|\bis\b)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(?:\.|\n|$)",
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            return matches[-1].replace(",", "").strip()
    
    # Fallback to last number in trailing text
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    return nums[-1] if nums else None


# ==============================================================================
# 2. PROPER ENTROPY-ADAPTIVE CONSENSUS GOVERNOR
# ==============================================================================
class ProperEntropyAdaptiveConsensusGovernor:
    def __init__(
        self,
        model: nn.Module,
        layer_band: List[int] = [12, 14, 16],
        base_threshold: float = 0.06,
        ref_entropy: float = 1.2,
        min_threshold: float = 0.025,
        max_threshold: float = 0.090,
    ):
        self.model = model
        self.layer_band = layer_band
        self.base_threshold = base_threshold
        self.ref_entropy = ref_entropy
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.layers = [model.model.layers[idx] for idx in layer_band]
        self.hook_handles = []

        # State buffers
        self.trajectories: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_band}
        self.last_velocities: Dict[int, float] = {idx: 1.0 for idx in layer_band}
        self.step_consensus_v: float = 1.0
        self.step_entropy: float = ref_entropy
        self.step_threshold: float = base_threshold
        self.consecutive_stagnations: int = 0
        self.tokens_generated: int = 0

    def reset(self):
        for idx in self.layer_band:
            self.trajectories[idx].clear()
            self.last_velocities[idx] = 1.0
        self.step_consensus_v = 1.0
        self.step_entropy = self.ref_entropy
        self.step_threshold = self.base_threshold
        self.consecutive_stagnations = 0
        self.tokens_generated = 0

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # Skip prompt prefill
            if hidden.shape[1] > 1:
                return outputs

            curr_h = hidden[:, -1, :].detach().float()
            traj = self.trajectories[layer_idx]
            if len(traj) > 0:
                prev_h = traj[-1]
                cos_sim = F.cosine_similarity(curr_h, prev_h, dim=-1).item()
                v_t = max(0.0, 1.0 - cos_sim)
                self.last_velocities[layer_idx] = v_t
            else:
                self.last_velocities[layer_idx] = 1.0

            traj.append(curr_h)
            return outputs
        return hook_fn

    def attach(self):
        self.reset()
        for idx, layer in zip(self.layer_band, self.layers):
            self.hook_handles.append(layer.register_forward_hook(self._make_hook(idx)))

    def detach(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def update_step(self, scores: Optional[torch.Tensor]):
        """
        Computes instantaneous Shannon Entropy H_t and scales epsilon_t according to
        the user's law:
            Certainty (Low Entropy)  -> LOWER threshold (stricter, protects math)
            Confusion (High Entropy) -> WIDEN threshold (looser, kills loops)
        """
        self.tokens_generated += 1

        # 1. Compute Shannon Entropy H_t
        if scores is not None:
            if isinstance(scores, (list, tuple)) and len(scores) > 0:
                logits = scores[-1][0].float()
            elif isinstance(scores, torch.Tensor):
                logits = scores[-1, 0].float() if scores.ndim == 3 else (scores[0].float() if scores.ndim == 2 else scores.float())
            else:
                logits = None

            if logits is not None:
                probs = F.softmax(logits, dim=-1)
                log_p = torch.log(probs + 1e-12)
                self.step_entropy = -(probs * log_p).sum().item()
            else:
                self.step_entropy = self.ref_entropy
        else:
            self.step_entropy = self.ref_entropy

        # 2. Dynamic Threshold Scaling: epsilon_t = base * (H_t / H_0)
        # Scaled and clamped
        ratio = self.step_entropy / max(0.01, self.ref_entropy)
        raw_threshold = self.base_threshold * ratio
        self.step_threshold = max(self.min_threshold, min(self.max_threshold, raw_threshold))

        # 3. Minimax Multi-Layer Consensus
        self.step_consensus_v = max(self.last_velocities.values())


# ==============================================================================
# 3. DUAL-TRIGGER STOPPING CRITERIA
# ==============================================================================
class EntropyAdaptiveStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        governor: ProperEntropyAdaptiveConsensusGovernor,
        prompt_len: int,
        min_tokens: int = 100,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.governor = governor
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Update governor step with latest logits
        self.governor.update_step(scores)
        gen_tokens = input_ids[0, self.prompt_len:]
        curr_len = len(gen_tokens)

        if curr_len < 20:
            return False

        # Condition 1: Check natural exit after </think> + Answer
        if curr_len % 5 == 0:
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
            if "</think>" in text:
                if extract_answer(text) is not None:
                    return True

        # Condition 2: Latent Dynamical Velocity Governor
        if curr_len >= self.min_tokens:
            v_c = self.governor.step_consensus_v
            eps = self.governor.step_threshold

            # Equilibrium stagnation check
            if v_c < eps:
                self.governor.consecutive_stagnations += 1
                # If high entropy (confused loop), 1 step stagnation is sufficient
                required_stagnations = 1 if self.governor.step_entropy > 1.8 else 2
                if self.governor.consecutive_stagnations >= required_stagnations:
                    return True
            else:
                self.governor.consecutive_stagnations = 0

        return False


# ==============================================================================
# 4. BENCHMARK RUNNER (A100 OPTIMIZED)
# ==============================================================================
def run_benchmark(num_problems: int = 10):
    print("=" * 110)
    print("🚀 INITIALIZING PROPER ENTROPY-ADAPTIVE CONSENSUS GOVERNOR BENCHMARK")
    print("   Architecture: Multi-Layer Consensus [12, 14, 16] + Dynamic Entropy Scaling")
    print("   Governing Law: Low Entropy -> Lower Threshold (Protect Math)")
    print("                  High Entropy -> Widen Threshold (Kill Loops)")
    print("=" * 110)

    # Resolve Model and Tokenizer automatically
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    if "model" in globals():
        m = globals()["model"]
    elif "model_7b" in globals():
        m = globals()["model_7b"]
    else:
        print(f"Loading {model_id} into VRAM...")
        m = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        m.eval()

    if "tokenizer" in globals():
        tok = globals()["tokenizer"]
    elif "tokenizer_7b" in globals():
        tok = globals()["tokenizer_7b"]
    else:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    governor = ProperEntropyAdaptiveConsensusGovernor(
        model=m,
        layer_band=[12, 14, 16],
        base_threshold=0.06,
        ref_entropy=1.2,
        min_threshold=0.025,
        max_threshold=0.090,
    )

    try:
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{num_problems}]")
    except Exception:
        import pandas as pd
        df = pd.read_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
        ds = df.iloc[:num_problems].to_dict("records")

    total_van_tk = 0
    total_gov_tk = 0
    van_corr = 0
    gov_corr = 0
    flips_prevented = 0

    print(f"\n{'#':<4} | {'Vanilla (Tk, Time, Res)':<24} | {'Adaptive Gov (Tk, Time, Res)':<28} | {'Tokens Saved':<14} | {'Outcome'}")
    print("-" * 110)

    for idx, item in enumerate(ds):
        q = item["question"]
        gt = extract_answer(item["answer"])

        prompt = f"<｜User｜>{q}<｜Assistant｜><think>\n"
        inputs = tok(prompt, return_tensors="pt").to(m.device)
        p_len = inputs.input_ids.shape[1]

        # -------------------------------------------------------------
        # PASS 1: VANILLA (Unconstrained DeepSeek-R1)
        # -------------------------------------------------------------
        governor.detach()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0_v = time.time()
        with torch.no_grad():
            out_v = m.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_v = time.time() - t0_v
        v_tk = out_v.shape[1] - p_len
        v_text = tok.decode(out_v[0, p_len:], skip_special_tokens=True)
        v_ans = extract_answer(v_text)
        v_ok = (str(v_ans) == str(gt)) if (v_ans and gt) else False

        # -------------------------------------------------------------
        # PASS 2: ENTROPY-ADAPTIVE CONSENSUS GOVERNOR
        # -------------------------------------------------------------
        governor.attach()
        stop_crit = StoppingCriteriaList([
            EntropyAdaptiveStoppingCriteria(tok, governor, p_len, min_tokens=100)
        ])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0_g = time.time()
        with torch.no_grad():
            out_g = m.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                stopping_criteria=stop_crit,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_g = time.time() - t0_g
        g_seq = out_g.sequences
        g_tk = g_seq.shape[1] - p_len
        g_text = tok.decode(g_seq[0, p_len:], skip_special_tokens=True)
        g_ans = extract_answer(g_text)
        g_ok = (str(g_ans) == str(gt)) if (g_ans and gt) else False
        governor.detach()

        # Metrics
        total_van_tk += v_tk
        total_gov_tk += g_tk
        if v_ok:
            van_corr += 1
        if g_ok:
            gov_corr += 1
        saved_pct = ((v_tk - g_tk) / max(1, v_tk)) * 100.0

        if g_ok and not v_ok:
            outcome = "GOV_WON (Flip Prevented!)"
            flips_prevented += 1
        elif v_ok and not g_ok:
            outcome = "VAN_WON"
        elif v_ok and g_ok:
            outcome = "EQUAL"
        else:
            outcome = "BOTH_FAIL"

        print(f"{idx+1:<4} | {v_tk} tk ({dt_v:.1f}s, {'OK' if v_ok else 'X'}):<24 | {g_tk} tk ({dt_g:.1f}s, {'OK' if g_ok else 'X'}):<28 | {saved_pct:>13.1f}% | {outcome}")

    net_compute_saved = ((total_van_tk - total_gov_tk) / max(1, total_van_tk)) * 100.0
    van_acc = (van_corr / num_problems) * 100.0
    gov_acc = (gov_corr / num_problems) * 100.0

    print("\n" + "=" * 110)
    print("PROPER ENTROPY-ADAPTIVE CONSENSUS GOVERNOR: FINAL STATISTICAL SUMMARY")
    print("=" * 110)
    print(f"Vanilla Accuracy:        {van_acc:.1f}% ({van_corr}/{num_problems})")
    print(f"Adaptive Gov Accuracy:   {gov_acc:.1f}% ({gov_corr}/{num_problems})")
    print(f"Accuracy Delta:          {'+' if gov_acc >= van_acc else ''}{gov_acc - van_acc:.1f}%")
    print(f"Catastrophic Flips Saved:{flips_prevented} queries")
    print(f"Net Tokens Saved:        {net_compute_saved:.1f}% ({total_van_tk - total_gov_tk:,} tokens)")
    print("=" * 110)


if __name__ == "__main__":
    run_benchmark(num_problems=10)
