"""
Official Enterprise Coding Benchmark Suite for SubNeutralize.
Replicates the 32B empirical evaluation on NVIDIA A100 SXM4.
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

from .governor import extract_clean_code


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
                validate_jwt_claims({"sub": "u", "exp": now - 100}, ["sub"], leeway_seconds=30, current_time=now)
                assert False, "Failed to catch expired token"
            except ValueError:
                pass

            # Missing required claim
            try:
                validate_jwt_claims({"exp": now + 300}, ["sub", "role"], current_time=now)
                assert False, "Failed to catch missing claim"
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
            - `set(self, key, value, ttl: float = None)`: stores key-value pair with TTL. If size exceeds maxsize, evicts least recently accessed unexpired item.
            - `get(self, key, default = None)`: returns value if key exists and has not expired, and updates its LRU access order. Returns default if expired or missing.
            - `cleanup(self) -> int`: removes all expired keys and returns count of removed keys.
            Write only the code inside ```python ... ``` fences.
        """),
        "test": textwrap.dedent("""\
            import time
            cache = TTLCache(maxsize=2, default_ttl_seconds=1.0)
            cache.set("a", 100)
            cache.set("b", 200)
            assert cache.get("a") == 100
            cache.set("c", 300) # Evicts 'b' because 'a' was recently accessed
            assert cache.get("b") is None
            assert cache.get("c") == 300

            time.sleep(1.1) # Wait for expiration
            assert cache.get("a") is None
            cleaned = cache.cleanup()
            assert cleaned >= 1
        """),
    }
]


def run_code_test(code_text: str, test_snippet: str) -> Tuple[bool, str]:
    """Executes generated code in an isolated environment against test assertions."""
    code = extract_clean_code(code_text)
    env: Dict[str, Any] = {}
    try:
        exec(compile(code, "<agent_solution>", "exec"), env)
        exec(compile(test_snippet, "<unit_test>", "exec"), env)
        return True, "PASSED"
    except AssertionError as e:
        return False, f"ASSERTION FAIL: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


REFERENCE_SOLUTIONS = {
    "PROB-01": textwrap.dedent("""
        import re

        def build_secure_query(table: str, filters: dict, allowed_columns: list) -> str:
            if not re.match(r"^[A-Za-z0-9_]+$", table):
                raise ValueError("Invalid table name")
            for col in filters.keys():
                if col not in allowed_columns:
                    raise ValueError(f"Column {col} not permitted")
            
            danger_patterns = ["' OR '1'='1", "1=1", "--", "/*", "UNION SELECT"]
            for val in filters.values():
                val_str = str(val).upper()
                for pat in danger_patterns:
                    if pat in val_str:
                        raise ValueError(f"Malicious pattern detected: {pat}")
            
            sorted_cols = sorted(filters.keys())
            clauses = [f"{col} = %s" for col in sorted_cols]
            where_sql = " AND ".join(clauses)
            return f"SELECT * FROM {table} WHERE {where_sql}"
    """).strip(),
    "PROB-02": textwrap.dedent("""
        import time
        import threading

        class TokenBucketLimiter:
            def __init__(self, capacity: int, refill_rate: float):
                self.capacity = capacity
                self.refill_rate = refill_rate
                self.tokens = float(capacity)
                self.last_refill = time.time()
                self.lock = threading.Lock()

            def _refill(self):
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
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
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      SUBNEUTRALIZE ENTERPRISE REASONING BENCHMARK SUITE (A100 SPEC)      ")
    print("=" * 80)

    if args.model is None:
        print("\n[*] Running self-validation on 4 Enterprise Problem Test Suites...")
        print("-" * 80)
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
        print("-" * 80)
        if all_passed:
            print("[SUCCESS] All 4 enterprise unit test suites verified with 100% correctness.")
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
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
        )
        governor = SubNeutralize(model, tokenizer)

        print("\n" + "=" * 90)
        print(f"{'ID':<8} | {'VANILLA TOKENS':<16} | {'SUBNEUTRALIZE':<16} | {'SAVINGS':<10} | {'UNIT TEST':<12}")
        print("=" * 90)

        for prob in BENCHMARK_SUITE:
            pid = prob["id"]
            prompt = prob["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            prompt_len = inputs.input_ids.shape[-1]

            # 1. Vanilla
            with torch.no_grad():
                out_v = model.generate(**inputs, max_new_tokens=1024)
            v_tok = out_v.shape[-1] - prompt_len

            # 2. SubNeutralize
            out_g = governor.generate(prompt, max_new_tokens=1024)
            g_tok = out_g.total_tokens

            passed, _ = run_code_test(out_g.clean_code, prob["test"])
            status = "PASS" if passed else "FAIL"
            savings = (1.0 - g_tok / max(1, v_tok)) * 100.0

            print(f"{pid:<8} | {v_tok:<16} | {g_tok:<16} | {savings:>7.1f}% | {status:<12}")

        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
