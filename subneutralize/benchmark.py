"""
Official Enterprise Coding Benchmark Suite for SubNeutralize.
Evaluates accuracy and token efficiency with matched-budget sampling parity.
"""

import sys
import time
import textwrap
from typing import List, Dict, Tuple, Any

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .governor import extract_clean_code, extract_clean_answer


BENCHMARK_SUITE: List[Dict[str, Any]] = [
    {
        "id": "PROB-01",
        "title": "SQL Injection Sanitizer & Query Builder",
        "prompt": textwrap.dedent("""\
            Write a Python function `build_secure_query(table: str, filters: dict, allowed_columns: list) -> str`
            that builds a parameterized SQL query. It must:
            1. Validate that the table name contains only alphanumeric characters and underscores.
            2. Validate that all filter keys are in `allowed_columns`.
            3. Sanitize filter values: strip any tautology attacks (e.g. `' OR '1'='1`, `1=1`), UNION SELECT, and comment delimiters (`--`, `/*`).
            4. Return the query string using `%s` placeholders and sort the WHERE conditions alphabetically by column name.
            Raise ValueError on any invalid or injection-laden input.
            Write only the code inside ```python ... ``` fences.
        """),
        "test": textwrap.dedent("""\
            # Unit Test 1: Valid inputs
            q = build_secure_query("users", {"age": 25, "status": "active"}, ["age", "status", "name"])
            assert "SELECT * FROM users WHERE age = %s AND status = %s" in q or "status = %s AND age = %s" in q

            # Unit Test 2: Invalid table name
            try:
                build_secure_query("users; DROP TABLE users;", {"age": 25}, ["age"])
                assert False, "Failed to reject injection table"
            except ValueError:
                pass

            # Unit Test 3: Disallowed column
            try:
                build_secure_query("users", {"password": "secret"}, ["age", "status"])
                assert False, "Failed to reject unapproved column"
            except ValueError:
                pass

            # Unit Test 4: Tautology injection in value
            try:
                build_secure_query("users", {"status": "' OR '1'='1"}, ["status"])
                assert False, "Failed to reject SQL tautology"
            except ValueError:
                pass
        """),
    },
    {
        "id": "PROB-02",
        "title": "Token Bucket Rate Limiter with Refill",
        "prompt": textwrap.dedent("""\
            Implement a thread-safe Python class `TokenBucketLimiter`:
            - `__init__(self, capacity: int, refill_rate: float)`: capacity in tokens, refill_rate in tokens per second.
            - `allow_request(self, tokens: int = 1) -> bool`: returns True and consumes tokens if available, else False.
            Tokens refill smoothly based on elapsed time since the last request.
            Include a method `get_available_tokens(self) -> float` returning current available tokens (capped at capacity).
            Write only the code inside ```python ... ``` fences.
        """),
        "test": textwrap.dedent("""\
            import time
            limiter = TokenBucketLimiter(capacity=10, refill_rate=5.0)
            assert limiter.allow_request(5) is True
            assert abs(limiter.get_available_tokens() - 5.0) < 0.5
            assert limiter.allow_request(6) is False
            time.sleep(1.0) # Refills ~5 tokens
            assert limiter.allow_request(6) is True
        """),
    },
    {
        "id": "PROB-03",
        "title": "JWT Claims & Expiry Validator",
        "prompt": textwrap.dedent("""\
            Write a Python function `validate_jwt_claims(payload: dict, required_claims: list, leeway_seconds: int = 60, current_time: int = None) -> bool`
            that checks:
            1. All `required_claims` exist in `payload`.
            2. If 'exp' exists, verify current_time < exp + leeway_seconds.
            3. If 'nbf' exists, verify current_time >= nbf - leeway_seconds.
            4. If 'iat' exists, verify iat <= current_time + leeway_seconds.
            Return True if completely valid, otherwise raise ValueError with descriptive error message.
            Write only the code inside ```python ... ``` fences.
        """),
        "test": textwrap.dedent("""\
            now = 1700000000
            p = {"sub": "12345", "exp": now + 300, "nbf": now - 10, "iat": now}
            assert validate_jwt_claims(p, ["sub"], current_time=now) is True

            # Expired beyond leeway
            try:
                validate_jwt_claims({"sub": "12345", "exp": now - 120}, ["sub"], leeway_seconds=60, current_time=now)
                assert False, "Failed to reject expired token"
            except ValueError:
                pass

            # Missing claim
            try:
                validate_jwt_claims({"exp": now + 300}, ["sub"], current_time=now)
                assert False, "Failed to reject missing claim"
            except ValueError:
                pass
        """),
    },
    {
        "id": "PROB-04",
        "title": "LRU Cache with Time-To-Live (TTL)",
        "prompt": textwrap.dedent("""\
            Implement a Python class `TTLCache`:
            - `__init__(self, maxsize: int, default_ttl_seconds: float)`
            - `set(self, key, value, ttl: float = None) -> None`: store key/value with custom or default TTL. If cache exceeds maxsize, evict least recently used non-expired entry.
            - `get(self, key, default=None) -> Any`: return value if key exists and not expired, update LRU order; else default.
            - `cleanup(self) -> int`: purge all currently expired entries, returning the count of purged items.
            Write only the code inside ```python ... ``` fences.
        """),
        "test": textwrap.dedent("""\
            import time
            cache = TTLCache(maxsize=2, default_ttl_seconds=1.0)
            cache.set("a", 1)
            cache.set("b", 2)
            assert cache.get("a") == 1
            cache.set("c", 3) # Should evict "b" since "a" was recently read
            assert cache.get("b") is None
            assert cache.get("a") == 1
            assert cache.get("c") == 3
            time.sleep(1.1)
            assert cache.get("a") is None
            purged = cache.cleanup()
            assert purged >= 1
        """),
    }
]


def run_code_test(code_snippet: str, test_code: str) -> Tuple[bool, str]:
    """Safely executes candidate code against problem unit tests in an isolated namespace."""
    exec_globals: Dict[str, Any] = {}
    try:
        exec(code_snippet, exec_globals)
        exec(test_code, exec_globals)
        return True, "PASSED"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


REFERENCE_SOLUTIONS: Dict[str, str] = {
    "PROB-01": textwrap.dedent(r"""
        import re

        def build_secure_query(table: str, filters: dict, allowed_columns: list) -> str:
            if not re.match(r'^[A-Za-z0-9_]+$', table):
                raise ValueError("Invalid table name")
            for col in filters.keys():
                if col not in allowed_columns:
                    raise ValueError(f"Disallowed column: {col}")
            clauses = []
            for col in sorted(filters.keys()):
                val = str(filters[col])
                if re.search(r"('|--|/\*|1=1|UNION\s+SELECT)", val, re.IGNORECASE):
                    raise ValueError(f"Injection detected in column {col}")
                clauses.append(f"{col} = %s")
            where = " AND ".join(clauses)
            return f"SELECT * FROM {table} WHERE {where}" if where else f"SELECT * FROM {table}"
    """).strip(),
    "PROB-02": textwrap.dedent("""
        import time
        import threading

        class TokenBucketLimiter:
            def __init__(self, capacity: int, refill_rate: float):
                self.capacity = float(capacity)
                self.refill_rate = float(refill_rate)
                self.tokens = self.capacity
                self.last_refill = time.time()
                self.lock = threading.Lock()

            def _refill(self):
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now

            def allow_request(self, tokens: int = 1) -> bool:
                with self.lock:
                    self._refill()
                    if self.tokens >= tokens:
                        self.tokens -= tokens
                        return True
                    return False

            def get_available_tokens(self) -> float:
                with self.lock:
                    self._refill()
                    return self.tokens
    """).strip(),
    "PROB-03": textwrap.dedent("""
        import time

        def validate_jwt_claims(payload: dict, required_claims: list, leeway_seconds: int = 60, current_time: int = None) -> bool:
            now = int(time.time()) if current_time is None else current_time
            for claim in required_claims:
                if claim not in payload:
                    raise ValueError(f"Missing required claim: {claim}")
            if "exp" in payload:
                if now >= payload["exp"] + leeway_seconds:
                    raise ValueError("Token has expired")
            if "nbf" in payload:
                if now < payload["nbf"] - leeway_seconds:
                    raise ValueError("Token not yet valid (nbf)")
            if "iat" in payload:
                if payload["iat"] > now + leeway_seconds:
                    raise ValueError("Token issued in future (iat)")
            return True
    """).strip(),
    "PROB-04": textwrap.dedent("""
        import time
        from collections import OrderedDict

        class TTLCache:
            def __init__(self, maxsize: int, default_ttl_seconds: float):
                self.maxsize = maxsize
                self.default_ttl = default_ttl_seconds
                self.cache = OrderedDict()
                self.expirations = {}

            def _is_expired(self, key, now):
                return key in self.expirations and now >= self.expirations[key]

            def set(self, key, value, ttl: float = None):
                now = time.time()
                ttl_val = self.default_ttl if ttl is None else ttl
                if key in self.cache:
                    del self.cache[key]
                elif len(self.cache) >= self.maxsize:
                    for k in list(self.cache.keys()):
                        if self._is_expired(k, now):
                            del self.cache[k]
                            self.expirations.pop(k, None)
                    if len(self.cache) >= self.maxsize:
                        oldest_key, _ = self.cache.popitem(last=False)
                        self.expirations.pop(oldest_key, None)
                self.cache[key] = value
                self.expirations[key] = now + ttl_val

            def get(self, key, default=None):
                now = time.time()
                if key not in self.cache:
                    return default
                if self._is_expired(key, now):
                    del self.cache[key]
                    self.expirations.pop(key, None)
                    return default
                self.cache.move_to_end(key)
                return self.cache[key]

            def cleanup(self) -> int:
                now = time.time()
                expired = [k for k in list(self.cache.keys()) if self._is_expired(k, now)]
                for k in expired:
                    del self.cache[k]
                    self.expirations.pop(k, None)
                return len(expired)
    """).strip()
}


def main():
    import argparse
    default_dev = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    parser = argparse.ArgumentParser(description="SubNeutralize Enterprise Benchmark Suite")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model ID to evaluate (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    parser.add_argument("--device", type=str, default=default_dev, help="Device for execution")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum token ceiling per task")
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("      SUBNEUTRALIZE ENTERPRISE BENCHMARK SUITE (MATCHED-BUDGET PROTOCOL)      ")
    print("=" * 90)

    if args.model is None:
        print("\n[*] Running self-validation on reference solution unit test suites...")
        print("-" * 90)
        all_passed = True
        for prob in BENCHMARK_SUITE:
            pid = prob["id"]
            title = prob["title"]
            ref_code = REFERENCE_SOLUTIONS[pid]
            passed, msg = run_code_test(ref_code, prob["test"])
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {pid}: {title} -> {msg}")
            if not passed:
                all_passed = False
        print("-" * 90)
        if all_passed:
            print("[SUCCESS] All reference unit test suites verified with 100% correctness.")
            print("\nTo benchmark a live model with SubNeutralize vs Vanilla, run:")
            print("  python -m subneutralize.benchmark --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\n")
        else:
            sys.exit(1)
    else:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to evaluate models. Please install torch: pip install torch")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .engine import SubNeutralize

        print(f"\n[*] Loading model: {args.model} on {args.device}...")
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device, torch_dtype=dtype)
        governor = SubNeutralize(model, tokenizer)

        print("\n" + "=" * 105)
        print(f"{'ID':<8} | {'VANILLA TOK (TEST)':<22} | {'GOV TOK (TEST)':<20} | {'SAVINGS':<10} | {'HALT STEP':<12} | {'SPEEDUP':<10}")
        print("=" * 105)

        total_v_tokens = 0
        total_g_tokens = 0
        total_v_time = 0.0
        total_g_time = 0.0

        for prob in BENCHMARK_SUITE:
            pid = prob["id"]
            prompt = prob["prompt"]
            formatted_prompt = governor._format_prompt(prompt)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(args.device)
            prompt_len = inputs.input_ids.shape[-1]

            # 1. Fair Vanilla Baseline (matched sampling and chat template)
            t0 = time.time()
            with torch.no_grad():
                out_v = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
            t_v = time.time() - t0
            v_tok = out_v.shape[-1] - prompt_len
            v_text = tokenizer.decode(out_v[0, prompt_len:], skip_special_tokens=True)
            v_passed, _ = run_code_test(extract_clean_code(v_text), prob["test"])
            v_status = "PASS" if v_passed else "FAIL"

            # 2. SubNeutralize Dynamical Governor
            t0 = time.time()
            out_g = governor.generate(prompt, max_new_tokens=args.max_tokens, temperature=0.6, top_p=0.95)
            t_g = time.time() - t0
            g_tok = out_g.total_tokens
            g_passed, _ = run_code_test(out_g.clean_code, prob["test"])
            g_status = "PASS" if g_passed else "FAIL"

            savings = (1.0 - g_tok / max(1, v_tok)) * 100.0
            speedup = t_v / max(1e-4, t_g)
            halt_str = f"Tok {out_g.consensus_step}" if out_g.consensus_reached else "Natural"

            total_v_tokens += v_tok
            total_g_tokens += g_tok
            total_v_time += t_v
            total_g_time += t_g

            v_col = f"{v_tok}t ({v_status})"
            g_col = f"{g_tok}t ({g_status})"

            print(f"{pid:<8} | {v_col:<22} | {g_col:<20} | {savings:>7.1f}% | {halt_str:<12} | {speedup:>7.2f}x")

        print("=" * 105)
        overall_savings = (1.0 - total_g_tokens / max(1, total_v_tokens)) * 100.0
        overall_speedup = total_v_time / max(1e-4, total_g_time)
        print(f"OVERALL SUMMARY: {total_v_tokens}t -> {total_g_tokens}t ({overall_savings:.1f}% saved) | {overall_speedup:.2f}x average speedup\n")


if __name__ == "__main__":
    main()
