"""
================================================================================
SUBNEUTRALIZE (32B FRONTIER) — ENTERPRISE & CYBERSECURITY CODING BENCHMARK
TARGET PARTNER: FreeBuf / Enterprise Inference Infrastructure
MODEL: unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit (NF4, ~19GB VRAM)
ARCHITECTURE: 64 Transformer Layers | Cognitive Bottleneck: [30, 32, 34]
HARDWARE: NVIDIA A100-SXM4-40GB

THE SUBNEUTRALIZE INVENTION:
1. Residual Stream Riemannian Velocity:
   v_t^{(l)} = 1.0 - cos(h_t^{(l)}, h_{t-1}^{(l)})
2. Minimax Bottleneck Consensus:
   v_t^{consensus} = max_{l in [30, 32, 34]} v_t^{(l)}
3. Dynamic Shannon Entropy Scaling:
   epsilon_t = clamp(base_eps * (H_t / H_ref), min_eps, max_eps)
   - Low Entropy (Certainty): Stricter threshold protects multi-step logical deduction.
   - High Entropy (Confusion): Relaxed threshold halts overthinking limit cycles.
4. Two-Stage Governed Decoupling:
   - Stage 1: Latent-governed thinking. Halts at dynamical equilibrium (v_c < eps).
   - Stage 2: Direct greedy code synthesis with prompt continuity (</think>\n```python).
================================================================================
"""

import os
import re
import sys
import time
import textwrap
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ------------------------------------------------------------------------------
# 1. HARDWARE & MODEL CONFIGURATION
# ------------------------------------------------------------------------------
MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"
LAYER_BAND = [30, 32, 34]  # 64-layer 32B cognitive bottleneck band

print("=" * 115)
print(f"[*] INITIALIZING SUBNEUTRALIZE 32B BENCHMARK ON: {torch.cuda.get_device_name(0)}")
print(f"[*] Model: {MODEL_ID}")
print(f"[*] Layer Band: {LAYER_BAND} (64-layer mid-network cognitive core)")
print("=" * 115)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

vram_alloc = torch.cuda.memory_allocated() / 1e9
vram_res = torch.cuda.memory_reserved() / 1e9
print(f"[✓] 32B Weights resident in VRAM: {vram_alloc:.2f} GB (Allocated) | {vram_res:.2f} GB (Reserved)")
print(f"[✓] Free VRAM for KV-Cache: ~{40.0 - vram_res:.1f} GB (Generous headroom on A100)\n")


# ------------------------------------------------------------------------------
# 2. THE SUBNEUTRALIZE CONSENSUS-ENTROPY GOVERNOR CORE
# ------------------------------------------------------------------------------
class ConsensusEntropyGovernor:
    def __init__(
        self,
        model,
        layer_band: List[int] = LAYER_BAND,
        base_threshold: float = 0.055,
        ref_entropy: float = 1.2,
        min_threshold: float = 0.020,
        max_threshold: float = 0.095,
    ):
        self.layer_band = layer_band
        self.base_threshold = base_threshold
        self.ref_entropy = ref_entropy
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.layers = [model.model.layers[i] for i in layer_band]
        self.hook_handles = []

        self.last_h: Dict[int, Optional[torch.Tensor]] = {i: None for i in layer_band}
        self.last_v: Dict[int, float] = {i: 1.0 for i in layer_band}
        self.consensus_v: float = 1.0
        self.entropy: float = ref_entropy
        self.threshold: float = base_threshold
        self.stagnations: int = 0
        self.tokens_generated: int = 0

    def _make_hook(self, idx: int):
        def hook_fn(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # Skip prompt prefill pass (seq_len > 1)
            if hidden.shape[1] > 1:
                return outputs
            curr = hidden[:, -1, :].detach().float()
            if self.last_h[idx] is not None:
                cos = F.cosine_similarity(curr, self.last_h[idx], dim=-1).item()
                self.last_v[idx] = max(0.0, 1.0 - cos)
            else:
                self.last_v[idx] = 1.0
            self.last_h[idx] = curr
            return outputs
        return hook_fn

    def attach(self):
        self.reset()
        for idx, layer in zip(self.layer_band, self.layers):
            self.hook_handles.append(layer.register_forward_hook(self._make_hook(idx)))

    def detach(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def reset(self):
        for i in self.layer_band:
            self.last_h[i] = None
            self.last_v[i] = 1.0
        self.consensus_v = 1.0
        self.entropy = self.ref_entropy
        self.threshold = self.base_threshold
        self.stagnations = 0
        self.tokens_generated = 0

    def step(self, scores):
        self.tokens_generated += 1
        # Instantaneous Shannon Entropy Calculation
        if scores is not None and len(scores) > 0:
            try:
                logits = scores[-1] if (isinstance(scores, tuple) or scores.ndim > 2) else scores
                if logits.ndim == 2:
                    logits = logits[0]
                probs = F.softmax(logits.float(), dim=-1)
                self.entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
            except Exception:
                self.entropy = self.ref_entropy

        # Dynamic Threshold Modulation
        ratio = self.entropy / max(1e-3, self.ref_entropy)
        self.threshold = max(self.min_threshold, min(self.max_threshold, self.base_threshold * ratio))
        self.consensus_v = max(self.last_v.values())


# ------------------------------------------------------------------------------
# 3. DUAL STOPPING CRITERIA (NO CONTRADICTIONS)
# ------------------------------------------------------------------------------
class ThinkingPhaseStopping(StoppingCriteria):
    def __init__(self, governor: ConsensusEntropyGovernor, prompt_len: int, min_tokens: int = 80):
        super().__init__()
        self.governor = governor
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.governor.step(scores)
        gen_len = input_ids.shape[1] - self.prompt_len

        # Natural exit: if model outputs </think>, stop thinking phase cleanly
        tail = tokenizer.decode(input_ids[0, -10:], skip_special_tokens=False)
        if "</think>" in tail:
            return True

        # Warmup protection: protect initial problem representation
        if gen_len < self.min_tokens:
            return False

        # Latent Dynamical Equilibrium Trigger
        if self.governor.consensus_v < self.governor.threshold:
            self.governor.stagnations += 1
            required = 1 if self.governor.entropy > 1.8 else 2
            if self.governor.stagnations >= required:
                return True
        else:
            self.governor.stagnations = 0

        return False


class CodeBlockStopping(StoppingCriteria):
    def __init__(self, prompt_len: int):
        super().__init__()
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < 10:
            return False
        tail = tokenizer.decode(input_ids[0, -4:], skip_special_tokens=False)
        return "```" in tail


# ------------------------------------------------------------------------------
# 4. FREEBUF-GRADE CYBERSECURITY & BACKEND CODING PROBLEMS
# ------------------------------------------------------------------------------
BENCHMARK_SUITE = [
    {
        "id": "SEC-01",
        "title": "Path Traversal & Jailbreak Sanitizer",
        "prompt": (
            "Write a Python function `sanitize_path(base_dir: str, user_input: str) -> str` "
            "that safely resolves user_input within base_dir. It must prevent directory traversal "
            "(e.g., '../', relative escapes, absolute path overrides). "
            "If the resolved path attempts to escape base_dir, raise ValueError. "
            "Otherwise, return the canonical resolved absolute path as a string."
        ),
        "test": textwrap.dedent("""\
            import os, tempfile
            with tempfile.TemporaryDirectory() as base:
                safe = sanitize_path(base, "uploads/avatar.png")
                expected = os.path.realpath(os.path.join(base, "uploads/avatar.png"))
                assert os.path.realpath(safe) == expected, f"Safe path mismatch: {safe} vs {expected}"
                try:
                    sanitize_path(base, "../../etc/shadow")
                    assert False, "Failed to trap ../ traversal attack"
                except ValueError:
                    pass
                try:
                    sanitize_path(base, "/etc/passwd")
                    assert False, "Failed to trap absolute escape"
                except ValueError:
                    pass
        """),
    },
    {
        "id": "SEC-02",
        "title": "Constant-Time HMAC Replay Verifier",
        "prompt": (
            "Write a Python function `verify_signed_request(secret_key: bytes, message: bytes, signature_hex: str, timestamp: float, max_age_seconds: float = 60.0) -> bool` "
            "that verifies an HMAC-SHA256 signature using constant-time comparison to prevent timing attacks. "
            "It must also verify that abs(time.time() - timestamp) <= max_age_seconds to prevent replay attacks. "
            "Return True if both the timestamp is fresh and the signature is valid; otherwise return False."
        ),
        "test": textwrap.dedent("""\
            import hmac, hashlib, time
            secret = b"k_sec_99428"
            msg = b"action=transfer&amount=500"
            now = time.time()
            sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
            
            # Valid request
            assert verify_signed_request(secret, msg, sig, now) is True, "Valid signature failed"
            # Tampered message
            assert verify_signed_request(secret, b"action=transfer&amount=9999", sig, now) is False, "Tampered message accepted"
            # Expired timestamp (replay attack)
            assert verify_signed_request(secret, msg, sig, now - 120.0) is False, "Replay attack accepted"
            # Future timestamp skew
            assert verify_signed_request(secret, msg, sig, now + 120.0) is False, "Future replay accepted"
        """),
    },
    {
        "id": "SEC-03",
        "title": "Sliding-Window IP Rate Limiter",
        "prompt": (
            "Implement a Python class `SlidingWindowRateLimiter(max_requests: int, window_seconds: float)` "
            "that tracks request timestamps per IP address. Implement:\n"
            "- `allow_request(ip: str, timestamp: float) -> bool`: returns True and logs the request if the IP has made fewer "
            "than max_requests in the trailing window_seconds (i.e. strictly greater than timestamp - window_seconds). "
            "Otherwise, returns False without adding the request.\n"
            "- `cleanup(current_time: float)`: purges timestamps older than current_time - window_seconds across all tracked IPs.\n"
            "Do not use threading. Keep it pure Python with dicts/lists."
        ),
        "test": textwrap.dedent("""\
            limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10.0)
            ip = "192.168.1.100"
            assert limiter.allow_request(ip, 100.0) is True
            assert limiter.allow_request(ip, 102.0) is True
            assert limiter.allow_request(ip, 105.0) is True
            # 4th request within 10s should be blocked
            assert limiter.allow_request(ip, 108.0) is False
            # Request at 111.0: timestamp 100.0 is expired (> 10s ago), so only 102 and 105 remain -> allow!
            assert limiter.allow_request(ip, 111.0) is True
            limiter.cleanup(120.0)
        """),
    },
    {
        "id": "SEC-04",
        "title": "Merchant Currency Ledger Aggregator",
        "prompt": (
            "Write a Python function `aggregate_transactions(records: list) -> dict` "
            "where each record is a dict with keys: merchant_id (str), amount (float), currency (str). "
            "Standardize all amounts to USD using fixed rates: USD: 1.0, EUR: 1.08, GBP: 1.27. "
            "Return a dict mapping merchant_id -> {'total_usd': float, 'count': int}, "
            "where total_usd is rounded to 2 decimal places."
        ),
        "test": textwrap.dedent("""\
            records = [
                {"merchant_id": "FREEBUF_CORP", "amount": 100.0, "currency": "USD"},
                {"merchant_id": "FREEBUF_CORP", "amount": 50.0,  "currency": "EUR"},
                {"merchant_id": "SEC_LABS",     "amount": 200.0, "currency": "GBP"},
                {"merchant_id": "FREEBUF_CORP", "amount": 25.0,  "currency": "GBP"},
            ]
            res = aggregate_transactions(records)
            assert res["FREEBUF_CORP"]["count"] == 3, f"Count error: {res['FREEBUF_CORP']['count']}"
            exp_freebuf = round(100.0 + 50.0*1.08 + 25.0*1.27, 2)
            assert abs(res["FREEBUF_CORP"]["total_usd"] - exp_freebuf) < 1e-3, f"Total mismatch: {res['FREEBUF_CORP']['total_usd']} vs {exp_freebuf}"
            assert abs(res["SEC_LABS"]["total_usd"] - round(200.0*1.27, 2)) < 1e-3
        """),
    },
]


# ------------------------------------------------------------------------------
# 5. EXECUTION & CODE VALIDATION ENGINE
# ------------------------------------------------------------------------------
def extract_clean_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    for kw in ["def ", "class ", "import "]:
        idx = text.find(kw)
        if idx != -1:
            code = text[idx:]
            return code.split("```")[0].strip() if "```" in code else code.strip()
    return text.strip()


def run_code_test(code_text: str, test_snippet: str) -> Tuple[bool, str]:
    code = extract_clean_code(code_text)
    env = {}
    try:
        exec(compile(code, "<agent_solution>", "exec"), env)
        exec(compile(test_snippet, "<unit_test>", "exec"), env)
        return True, "PASSED ✓"
    except AssertionError as e:
        return False, f"ASSERTION FAIL: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


# ------------------------------------------------------------------------------
# 6. INFERENCE PASSES: VANILLA VS SUBNEUTRALIZE
# ------------------------------------------------------------------------------
def generate_vanilla(prompt: str, max_tokens: int = 1600) -> Tuple[int, str, float]:
    p = f"<｜User｜>{prompt}\nWrite the complete implementation wrapped in ```python ... ``` fences.<｜Assistant｜><think>\n"
    inp = tokenizer(p, return_tensors="pt").to(model.device)
    plen = inp.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    gen_tokens = out.shape[1] - plen
    text = tokenizer.decode(out[0, plen:], skip_special_tokens=True)
    return gen_tokens, text, dt


def generate_subneutralize(
    prompt: str, governor: ConsensusEntropyGovernor, think_max: int = 900, code_max: int = 700
) -> Tuple[int, int, str, float]:
    # Stage 1: Latent-governed thinking
    think_p = f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
    inp1 = tokenizer(think_p, return_tensors="pt").to(model.device)
    plen1 = inp1.input_ids.shape[1]

    governor.attach()
    stop_crit = StoppingCriteriaList([ThinkingPhaseStopping(governor, plen1, min_tokens=80)])

    t0 = time.time()
    with torch.no_grad():
        out1 = model.generate(
            **inp1,
            max_new_tokens=think_max,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            stopping_criteria=stop_crit,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    governor.detach()
    think_tokens = out1.sequences.shape[1] - plen1
    raw_think = tokenizer.decode(out1.sequences[0, plen1:], skip_special_tokens=True)
    clean_think = raw_think.split("</think>")[0].strip()

    # Stage 2: Direct continuation into greedy code block
    code_p = f"<｜User｜>{prompt}<｜Assistant｜><think>\n{clean_think}\n</think>\n```python\n"
    inp2 = tokenizer(code_p, return_tensors="pt").to(model.device)
    plen2 = inp2.input_ids.shape[1]

    code_stop = StoppingCriteriaList([CodeBlockStopping(plen2)])
    with torch.no_grad():
        out2 = model.generate(
            **inp2,
            max_new_tokens=code_max,
            temperature=0.0,
            do_sample=False,
            stopping_criteria=code_stop,
            pad_token_id=tokenizer.eos_token_id,
        )
    total_dt = time.time() - t0
    code_tokens = out2.shape[1] - plen2
    gen_code = tokenizer.decode(out2[0, plen2:], skip_special_tokens=True)

    return think_tokens, code_tokens, gen_code, total_dt


# ------------------------------------------------------------------------------
# 7. MAIN BENCHMARK EXECUTION
# ------------------------------------------------------------------------------
def run_benchmark():
    gov = ConsensusEntropyGovernor(model)

    print("=" * 115)
    print("  SUBNEUTRALIZE 32B ENTERPRISE CODING BENCHMARK (NVIDIA A100-SXM4-40GB)")
    print(f"  Architecture: 64 Layers | Layer Band: {LAYER_BAND} | Dynamic Entropy Scaling")
    print("=" * 115)
    print(f"{'ID':<8} | {'PROBLEM':<35} | {'VANILLA':<18} | {'SUBNEUTRALIZE (Th+Co)':<22} | {'SAVED':>7} | CODE TEST")
    print("-" * 115)

    tot_van = 0
    tot_sub = 0
    pass_cnt = 0

    for prob in BENCHMARK_SUITE:
        pid, title, p_text, t_code = prob["id"], prob["title"], prob["prompt"], prob["test"]

        # Run Vanilla
        torch.cuda.synchronize()
        v_tok, v_text, v_dt = generate_vanilla(p_text)
        v_pass, _ = run_code_test(v_text, t_code)

        # Run SubNeutralize
        torch.cuda.synchronize()
        th_tok, co_tok, s_code, s_dt = generate_subneutralize(p_text, gov)
        s_tok = th_tok + co_tok
        s_pass, s_msg = run_code_test(s_code, t_code)
        torch.cuda.synchronize()

        tot_van += v_tok
        tot_sub += s_tok
        if s_pass:
            pass_cnt += 1

        saved_pct = ((v_tok - s_tok) / max(1, v_tok)) * 100.0
        v_str = f"{v_tok}t ({v_dt:.1f}s)"
        s_str = f"{th_tok}t+{co_tok}t ({s_dt:.1f}s)"
        res_str = "PASS ✓" if s_pass else f"FAIL ✗ ({s_msg[:25]})"

        print(f"{pid:<8} | {title:<35} | {v_str:<18} | {s_str:<22} | {saved_pct:>6.1f}%  | {res_str}")

    net_savings = ((tot_van - tot_sub) / max(1, tot_van)) * 100.0
    print("=" * 115)
    print(f"  TOTALS  |  Vanilla: {tot_van} tokens  |  SubNeutralize: {tot_sub} tokens")
    print(f"  NET COMPUTE REDUCTION: {net_savings:.1f}%  |  VERIFIED TESTS PASSED: {pass_cnt}/{len(BENCHMARK_SUITE)}")
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
