"""
Official Enterprise Coding Benchmark Suite for SubNeutralize.
Replicates the 32B empirical evaluation on NVIDIA A100 SXM4.
"""

import sys
import time
import textwrap
from typing import List, Dict, Tuple, Any
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
        return True, "PASSED ✓"
    except AssertionError as e:
        return False, f"ASSERTION FAIL: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"
