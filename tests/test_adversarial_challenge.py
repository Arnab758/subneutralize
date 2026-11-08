"""
================================================================================
   SUBNEUTRALIZE v0.2.0: EMPIRICAL ADVERSARIAL CHALLENGE & STRESS TEST SUITE
================================================================================
Author: Challenger 1 (Archetype: teamwork_challenger)
Purpose:
  Adversarially stress-test the benchmark implementation, edge cases,
  traceback extraction under failure modes, Static Truncation (Arm 3) code
  fence mutilation, and token accounting mathematical integrity.

Test Modules:
  1. TestEdgeCaseCodeHandling:
     - Malformed code generation (unclosed code fences, missing markers)
     - SyntaxError in candidate patches
     - Empty candidate patches ("" or whitespace)
     - Infinite loops in candidate patches (subprocess timeout 15s)
     - Unhandled exceptions (ZeroDivisionError, RecursionError)
     - Multiple code blocks and preamble text
  2. TestPytestFailureModesAndTracebackExtraction:
     - Assertion failure (Exit code 1, passes count, failed count)
     - Runtime error (Exit code 1, stderr/stdout capture)
     - Syntax/collection error (Exit code 2, SyntaxError detected)
     - Subprocess timeout (Exit code 124, timeout message in error_traceback)
     - Feedback injection prompt verification
  3. TestStaticTruncationArm3Corruption:
     - Fixed cut at step 65 without manifold consensus
     - Code fence corruption & unexpected EOF detection
     - Syntax integrity flag validation (syntax_intact == False)
     - Negative control contrast vs SubNeutralize clean boundary exit
  4. TestTelemetryMathematicalIntegrity:
     - Per-turn prompt token continuity and accumulation
     - KV eviction arithmetic: prompt_tokens(Turn N) = raw - evicted
     - Prefill growth ratio: last_turn_prompt / first_turn_prompt
     - Token compute savings % relative to Vanilla reference
     - Summary aggregation math (pass rates, compute saved, mean latency, growth)
  5. TestAdversarialThinkingTagEdgeCases:
     - 9 adversarial unclosed/nested/orphan <think> tag edge cases
     - AST invariance preservation across code fences and comments
     - Pytest failure traceback preservation under unclosed <think>
     - compact_messages AST invariance safety fallback
     - Standalone server StatefulMultiTurnEvictor parity
================================================================================
"""

import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import pytest
import torch
from transformers import LogitsProcessorList

# Setup workspace paths
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import modules under test
from benchmark_multiturn_agent_loop import (
    TASKS,
    ARMS,
    PytestResult,
    execute_pytest_in_sandbox,
    StaticTruncationLogitsProcessor,
    compile_benchmark_report,
    DEFAULT_OUTPUT_JSON,
)
from subneutralize.governor import extract_clean_code, extract_reasoning_and_answer
from subneutralize.evictor import StatefulMultiTurnEvictor


# ==============================================================================
# 1. TEST SUITE: CANDIDATE PATCH EDGE CASES & EXTRACTION ROBUSTNESS
# ==============================================================================
class TestEdgeCaseCodeHandling:
    """Stress-tests code extraction and execution against adversarial model outputs."""

    @pytest.fixture(autouse=True)
    def setup_sandbox(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="adv_test_sandbox_"))
        yield
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unclosed_code_fence(self):
        """Model generates an unclosed code block (e.g. output cut off mid-way)."""
        malformed_output = (
            "<think>Drafting code...</think>\n"
            "```python\n"
            "def leap_year(year: int) -> bool:\n"
            "    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)\n"
        )  # No closing ```
        extracted = extract_clean_code(malformed_output)
        assert "def leap_year" in extracted
        # Verify it still compiles or runs through pytest
        res = execute_pytest_in_sandbox(
            self.tmp_dir, extracted, TASKS["leap"]["test_suite"], timeout=10
        )
        assert res.passed is True
        assert res.passed_count == 9

    def test_syntax_error_in_patch(self):
        """Model generates code with blatant syntax errors."""
        bad_code = "def leap_year(year:\n    return True\n"  # missing closing paren
        extracted = extract_clean_code(bad_code)
        res = execute_pytest_in_sandbox(
            self.tmp_dir, extracted, TASKS["leap"]["test_suite"], timeout=10
        )
        assert res.passed is False
        assert res.exit_code == 2  # Pytest collection error exit code
        assert "SyntaxError" in res.error_traceback or "SyntaxError" in res.stdout or "SyntaxError" in res.stderr
        assert res.total_count == 0  # No tests could be collected

    def test_empty_string_patch(self):
        """Model generates empty string or whitespace."""
        empty_code = "   \n\t  "
        extracted = extract_clean_code(empty_code)
        res = execute_pytest_in_sandbox(
            self.tmp_dir, extracted, TASKS["leap"]["test_suite"], timeout=10
        )
        assert res.passed is False
        assert res.exit_code == 2  # ImportError during collection
        assert "ImportError" in res.error_traceback or "cannot import name" in res.error_traceback

    def test_infinite_loop_timeout_handling(self):
        """Model generates code that enters an infinite while loop."""
        loop_code = (
            "def leap_year(year: int) -> bool:\n"
            "    while True:\n"
            "        pass\n"
            "    return True\n"
        )
        t0 = time.perf_counter()
        # Set a short 3-second timeout to stress-test timeout interception
        res = execute_pytest_in_sandbox(
            self.tmp_dir, loop_code, TASKS["leap"]["test_suite"], timeout=3
        )
        duration = time.perf_counter() - t0
        assert res.passed is False
        assert res.exit_code == 124, f"Expected timeout exit code 124, got {res.exit_code}"
        assert "timed out after 3 seconds" in res.stderr
        assert duration >= 1.5 and duration < 15.0, f"Execution hung or timeout failed: {duration}s"

    def test_infinite_recursion_stack_overflow(self):
        """Model generates recursive code that exhausts call stack."""
        rec_code = (
            "def leap_year(year: int) -> bool:\n"
            "    return leap_year(year)\n"
        )
        res = execute_pytest_in_sandbox(
            self.tmp_dir, rec_code, TASKS["leap"]["test_suite"], timeout=10
        )
        assert res.passed is False
        assert res.exit_code == 1
        assert "RecursionError" in res.error_traceback or "maximum recursion depth exceeded" in res.error_traceback

    def test_multiple_code_blocks(self):
        """Model generates multiple markdown blocks (e.g. reasoning explanation + final code)."""
        multi_output = (
            "Here is an example snippet:\n"
            "```python\n"
            "x = 42\n"
            "```\n\n"
            "And here is the complete solution:\n"
            "```python\n"
            "def leap_year(year: int) -> bool:\n"
            "    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)\n"
            "```\n"
        )
        extracted = extract_clean_code(multi_output)
        # extract_clean_code extracts the code block
        assert "def leap_year" in extracted or "x = 42" in extracted


# ==============================================================================
# 2. TEST SUITE: PYTEST FAILURE MODES & TRACEBACK EXTRACTION
# ==============================================================================
class TestPytestFailureModesAndTracebackExtraction:
    """Verifies that all pytest execution outcomes produce accurate telemetry & tracebacks."""

    @pytest.fixture(autouse=True)
    def setup_sandbox(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="adv_pytest_mode_"))
        yield
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_assertion_failure_mode(self):
        """Test with flawed logic (buggy leap starter code): 6 passed, 3 failed."""
        buggy_code = TASKS["leap"]["buggy_starter_code"]
        test_suite = TASKS["leap"]["test_suite"]
        res = execute_pytest_in_sandbox(self.tmp_dir, buggy_code, test_suite, timeout=10)

        assert res.passed is False
        assert res.exit_code == 1
        assert res.passed_count == 6
        assert res.failed_count == 3
        assert res.total_count == 9
        assert "FAILED test_solution.py::test_year_divisible_by_100_not_divisible_by_400" in res.error_traceback
        assert "FAILED test_solution.py::test_year_divisible_by_100_not_divisible_by_400_in_future" in res.error_traceback
        assert "FAILED test_solution.py::test_year_divisible_by_200_not_divisible_by_400" in res.error_traceback
        assert "assert True is False" in res.error_traceback or "where True = leap_year" in res.error_traceback

    def test_unhandled_runtime_exception_mode(self):
        """Test with code that raises ZeroDivisionError inside the function."""
        div_code = (
            "def leap_year(year: int) -> bool:\n"
            "    return 1 / (year - 2000) == 0\n"
        )
        res = execute_pytest_in_sandbox(self.tmp_dir, div_code, TASKS["leap"]["test_suite"], timeout=10)
        assert res.passed is False
        assert res.exit_code == 1
        assert "ZeroDivisionError: division by zero" in res.error_traceback

    def test_collection_syntax_error_mode(self):
        """Test with code containing unmatched parenthesis at module level."""
        syntax_err_code = "def leap_year(year: int) -> bool:\n    return ((year % 4 == 0)\n"
        res = execute_pytest_in_sandbox(self.tmp_dir, syntax_err_code, TASKS["leap"]["test_suite"], timeout=10)
        assert res.passed is False
        assert res.exit_code == 2
        assert res.passed_count == 0
        assert res.failed_count == 0
        assert res.total_count == 0
        assert "SyntaxError" in res.error_traceback

    def test_traceback_prompt_formatting_integrity(self):
        """Verifies traceback prompt construction reproduces failure information without truncation."""
        buggy_code = TASKS["leap"]["buggy_starter_code"]
        res = execute_pytest_in_sandbox(self.tmp_dir, buggy_code, TASKS["leap"]["test_suite"], timeout=10)

        traceback_prompt = (
            f"Automated test execution FAILED with exit code {res.exit_code}.\n\n"
            f"Pytest Output:\n{res.stdout}\n{res.stderr}\n\n"
            f"Please analyze the failure traceback above, locate the exact defect in your implementation, "
            f"and output a corrected complete python implementation inside a single ```python ... ``` block."
        )

        assert f"exit code {res.exit_code}" in traceback_prompt
        assert "FAILED test_solution.py" in traceback_prompt
        assert "assert leap_year(1900) is False" in traceback_prompt or "where True = leap_year(1900)" in traceback_prompt


# ==============================================================================
# 3. TEST SUITE: STATIC TRUNCATION (ARM 3) GENUINE CORRUPTION
# ==============================================================================
class TestStaticTruncationArm3Corruption:
    """Verifies that Static Truncation (Scissors control) genuinely mutilates code fences without geometric consensus."""

    def test_static_truncation_processor_injects_think_end(self):
        """Verifies StaticTruncationLogitsProcessor forces think_end token at cut_step."""
        cut_step = 10
        think_end_id = 9999
        proc = StaticTruncationLogitsProcessor(think_end_token_id=think_end_id, cut_step=cut_step)

        # Step 0: prompt length = 5
        input_ids = torch.zeros((1, 5), dtype=torch.long)
        scores = torch.zeros((1, 10000), dtype=torch.float)
        out_scores = proc(input_ids, scores)
        assert proc.prompt_len == 5
        assert proc.cut_injected is False
        assert out_scores[0, think_end_id] == 0.0

        # Step 9: 14 total tokens (9 generated)
        input_ids = torch.zeros((1, 14), dtype=torch.long)
        scores = torch.zeros((1, 10000), dtype=torch.float)
        out_scores = proc(input_ids, scores)
        assert proc.cut_injected is False

        # Step 10: 15 total tokens (10 generated) -> Trigger!
        input_ids = torch.zeros((1, 15), dtype=torch.long)
        scores = torch.zeros((1, 10000), dtype=torch.float)
        out_scores = proc(input_ids, scores)
        assert proc.cut_injected is True
        assert out_scores[0, think_end_id] >= 100.0

    def test_mutilated_code_fence_fails_pytest(self):
        """Verifies that scissors-truncated code fails pytest with SyntaxError / Exit code 2."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="adv_scissors_"))
        try:
            # Replicate the exact mock truncation emitted by Arm 3
            truncated_response = (
                "<think>\n"
                "Analyzing requirement and starting code generation...\n"
                "def leap_year(year:\n"
                "    # Truncated by scissors control: missing closing parenthesis, body and code fence!\n"
            )
            extracted = extract_clean_code(truncated_response)
            res = execute_pytest_in_sandbox(tmp_dir, extracted, TASKS["leap"]["test_suite"], timeout=10)

            assert res.passed is False
            assert res.exit_code == 2
            assert "SyntaxError" in res.error_traceback or "SyntaxError" in res.stdout or "SyntaxError" in res.stderr
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_arm3_telemetry_records_corruption(self):
        """Verifies that in results_multiturn_simulation.json, Arm 3 correctly records syntax failure and 0% pass rate."""
        assert DEFAULT_OUTPUT_JSON.exists(), f"Missing telemetry file: {DEFAULT_OUTPUT_JSON}"
        data = json.loads(DEFAULT_OUTPUT_JSON.read_text(encoding="utf-8"))

        for task_id in ["leap", "luhn", "circular-buffer"]:
            arm3 = data["tasks"][task_id]["arms"]["static_truncation"]
            assert arm3["solved"] is False, f"Arm 3 unexpectedly marked solved on {task_id}"
            assert arm3["syntax_intact"] is False, f"Arm 3 syntax marked intact on {task_id}"
            assert arm3["turns_to_solve"] is None, f"Arm 3 turns_to_solve must be null on {task_id}"
            assert arm3["final_pytest"]["passed"] == 0, f"Arm 3 unexpectedly passed tests on {task_id}"
            assert arm3["final_pytest"]["exit_code"] != 0, f"Arm 3 final pytest exit code should be non-zero on {task_id}"

        # Global summary for Arm 3
        assert data["summary"]["pass_rates"]["static_truncation"] == 0.0


# ==============================================================================
# 4. TEST SUITE: TELEMETRY DATA INTEGRITY & MATHEMATICAL INVARIANTS
# ==============================================================================
class TestTelemetryMathematicalIntegrity:
    """Verifies that all numbers in results_multiturn_simulation.json are mathematically sound."""

    @pytest.fixture(scope="class")
    def telemetry_data(self):
        assert DEFAULT_OUTPUT_JSON.exists()
        return json.loads(DEFAULT_OUTPUT_JSON.read_text(encoding="utf-8"))

    def test_schema_completeness(self, telemetry_data):
        """Verifies 3 tasks and 4 arms exist."""
        assert "tasks" in telemetry_data
        assert "summary" in telemetry_data
        tasks = telemetry_data["tasks"]
        for expected_task in ["leap", "luhn", "circular-buffer"]:
            assert expected_task in tasks
            arms = tasks[expected_task]["arms"]
            for expected_arm in ["vanilla", "subneutralize_full", "static_truncation", "governor_only"]:
                assert expected_arm in arms

    def test_turn_token_sum_invariants(self, telemetry_data):
        """Verifies total_prompt_tokens, total_generated_tokens, total_thinking_tokens, and total_evicted_tokens match sum of turns."""
        for task_id, task_obj in telemetry_data["tasks"].items():
            for arm_id, arm_obj in task_obj["arms"].items():
                turns = arm_obj["turns"]
                calc_prompt = sum(t["prompt_tokens"] for t in turns)
                calc_gen = sum(t["gen_tokens"] for t in turns)
                calc_think = sum(t["think_tokens"] for t in turns)
                calc_evicted = sum(t["evicted_tokens"] for t in turns)

                assert arm_obj["total_prompt_tokens"] == calc_prompt, (
                    f"{task_id} {arm_id}: prompt tokens mismatch: {arm_obj['total_prompt_tokens']} vs sum {calc_prompt}"
                )
                assert arm_obj["total_generated_tokens"] == calc_gen, (
                    f"{task_id} {arm_id}: gen tokens mismatch: {arm_obj['total_generated_tokens']} vs sum {calc_gen}"
                )
                assert arm_obj["total_thinking_tokens"] == calc_think, (
                    f"{task_id} {arm_id}: think tokens mismatch: {arm_obj['total_thinking_tokens']} vs sum {calc_think}"
                )
                assert arm_obj["total_evicted_tokens"] == calc_evicted, (
                    f"{task_id} {arm_id}: evicted tokens mismatch: {arm_obj['total_evicted_tokens']} vs sum {calc_evicted}"
                )

    def test_prefill_growth_ratio_math(self, telemetry_data):
        """Verifies prefill_growth_ratio == round(last_prompt / first_prompt, 2)."""
        for task_id, task_obj in telemetry_data["tasks"].items():
            for arm_id, arm_obj in task_obj["arms"].items():
                turns = arm_obj["turns"]
                first_prompt = turns[0]["prompt_tokens"]
                last_prompt = turns[-1]["prompt_tokens"]
                expected_ratio = round(last_prompt / max(1, first_prompt), 2)
                assert arm_obj["prefill_growth_ratio"] == expected_ratio, (
                    f"{task_id} {arm_id}: prefill growth ratio mismatch: {arm_obj['prefill_growth_ratio']} vs {expected_ratio}"
                )

    def test_subneutralize_eviction_reduces_prefill_growth(self, telemetry_data):
        """Verifies SubNeutralize Full prefill growth ratio is strictly less than Vanilla baseline."""
        for task_id, task_obj in telemetry_data["tasks"].items():
            vanilla_ratio = task_obj["arms"]["vanilla"]["prefill_growth_ratio"]
            subn_ratio = task_obj["arms"]["subneutralize_full"]["prefill_growth_ratio"]
            assert subn_ratio < vanilla_ratio, (
                f"{task_id}: SubNeutralize prefill ratio ({subn_ratio}) must be lower than Vanilla ({vanilla_ratio})"
            )
            # Verify evicted tokens > 0 in SubNeutralize
            assert task_obj["arms"]["subneutralize_full"]["total_evicted_tokens"] > 0, (
                f"{task_id}: SubNeutralize evicted tokens must be > 0"
            )

    def test_token_savings_pct_calculation(self, telemetry_data):
        """Verifies token_savings_pct calculation against vanilla reference."""
        for task_id, task_obj in telemetry_data["tasks"].items():
            vanilla_arm = task_obj["arms"]["vanilla"]
            vanilla_total = vanilla_arm["total_prompt_tokens"] + vanilla_arm["total_generated_tokens"]
            assert vanilla_arm["token_savings_pct"] == 0.0

            for arm_id, arm_obj in task_obj["arms"].items():
                if arm_id == "vanilla":
                    continue
                curr_total = arm_obj["total_prompt_tokens"] + arm_obj["total_generated_tokens"]
                if curr_total <= vanilla_total:
                    expected_savings = round(((vanilla_total - curr_total) / vanilla_total) * 100.0, 1)
                else:
                    expected_savings = 0.0
                assert arm_obj["token_savings_pct"] == expected_savings, (
                    f"{task_id} {arm_id}: savings pct mismatch: {arm_obj['token_savings_pct']} vs {expected_savings}"
                )

    def test_summary_aggregates_match_task_averages(self, telemetry_data):
        """Verifies summary aggregates are exact arithmetic means of the 3 tasks."""
        summary = telemetry_data["summary"]
        tasks = telemetry_data["tasks"]
        task_keys = list(tasks.keys())
        assert len(task_keys) == 3

        for arm_id in ["vanilla", "subneutralize_full", "static_truncation", "governor_only"]:
            # Pass rates
            expected_pr = round(sum(1 for k in task_keys if tasks[k]["arms"][arm_id]["solved"]) / 3.0 * 100.0, 1)
            assert summary["pass_rates"][arm_id] == expected_pr

            # Compute saved pct
            expected_saved = round(sum(tasks[k]["arms"][arm_id]["token_savings_pct"] for k in task_keys) / 3.0, 1)
            assert summary["total_compute_saved_pct"][arm_id] == expected_saved

            # Mean latency
            expected_lat = round(sum(tasks[k]["arms"][arm_id]["total_wall_clock_s"] for k in task_keys) / 3.0, 2)
            assert summary["mean_wall_latency_s"][arm_id] == expected_lat

            # Mean prefill growth
            expected_growth = round(sum(tasks[k]["arms"][arm_id]["prefill_growth_ratio"] for k in task_keys) / 3.0, 2)
            assert summary["mean_prefill_growth_ratio"][arm_id] == expected_growth


# ==============================================================================
# 5. TEST SUITE: ADVERSARIAL THINKING TAG EDGE CASES & AST PRESERVATION
# ==============================================================================
class TestAdversarialThinkingTagEdgeCases:
    """Verifies that StatefulMultiTurnEvictor maintains 100% AST invariance under adversarial scratchpad patterns."""

    @pytest.fixture
    def evictor(self):
        return StatefulMultiTurnEvictor(preserve_code=True, preserve_errors=True)

    @pytest.mark.parametrize("case_name,raw_text", [
        ("Case A: Unclosed <think> at start with subsequent code",
         "<think>Thinking...\n```python\ndef add(a, b):\n    return a + b\n```"),
        ("Case B: Unclosed <think> between two code blocks",
         "```python\ndef first(): return 1\n```\n<think>thinking about second\n```python\ndef second(): return 2\n```"),
        ("Case C: Unclosed code fence containing <think>",
         "```python\ndef foo():\n    # <think> comment\n    return 42"),
        ("Case D: Closed <think> followed by code followed by unclosed <think>",
         "<think>thought 1</think>\n```python\ndef bar(): return 3\n```\n<think>unclosed trailing thought"),
        ("Case E: Multiple closed <think> blocks with code in between",
         "<think>t1</think>\n```python\ndef a(): pass\n```\n<think>t2</think>\n```python\ndef b(): pass\n```"),
        ("Case F: Unclosed code fence at end of message (clean code)",
         "<think>thought</think>\n```python\ndef c():\n    return 4"),
        ("Case G: <think> inside string and comment in closed fence",
         "<think>thought</think>\n```python\n# <think> in comment\ns = '<think>in str</think>'\n```"),
        ("Case H: Pytest failure traceback followed by unclosed <think>",
         "FAILED test_foo.py - AssertionError\nTraceback (most recent call last):\n  File 'test.py', line 1\n<think>Analyzing failure..."),
        ("Case I: Unclosed <think> at start with pytest failure traceback",
         "<think>Analyzing...\nFAILED test_foo.py - AssertionError\nTraceback (most recent call last):\n  File 'test.py', line 1\nAssertionError\n```python\ndef fix(): pass\n```"),
    ])
    def test_evictor_ast_invariance_adversarial_cases(self, evictor, case_name, raw_text):
        cleaned = evictor._strip_scratchpad_ast_safe(raw_text)
        assert evictor.validate_ast_invariants(raw_text, cleaned), (
            f"Failed AST invariance on {case_name}. Cleaned output:\n{cleaned}"
        )
        if "FAILED" in raw_text:
            assert "FAILED" in cleaned, f"Traceback lost on {case_name}"

    def test_compact_messages_safety_fallback(self, evictor):
        messages = [
            {"role": "user", "content": "Write add function"},
            {"role": "assistant", "content": "<think>Drafting...\n```python\ndef add(a, b):\n    return a + b\n```"},
            {"role": "user", "content": "Now add tests"},
        ]
        compacted, clean_tokens, evicted = evictor.compact_messages(messages)
        assert len(compacted) == 3
        # Historical assistant turn must retain valid code
        hist_assistant = compacted[1]["content"]
        assert "def add(a, b):" in hist_assistant
        assert evictor.validate_ast_invariants(messages[1]["content"], hist_assistant)

    @pytest.mark.parametrize("case_name,raw_text", [
        ("Case A: Unclosed <think> at start with subsequent code",
         "<think>Thinking...\n```python\ndef add(a, b):\n    return a + b\n```"),
        ("Case B: Unclosed <think> between two code blocks",
         "```python\ndef first(): return 1\n```\n<think>thinking about second\n```python\ndef second(): return 2\n```"),
        ("Case C: Unclosed code fence containing <think>",
         "```python\ndef foo():\n    # <think> comment\n    return 42"),
        ("Case D: Closed <think> followed by code followed by unclosed <think>",
         "<think>thought 1</think>\n```python\ndef bar(): return 3\n```\n<think>unclosed trailing thought"),
        ("Case E: Multiple closed <think> blocks with code in between",
         "<think>t1</think>\n```python\ndef a(): pass\n```\n<think>t2</think>\n```python\ndef b(): pass\n```"),
        ("Case F: Unclosed code fence at end of message (clean code)",
         "<think>thought</think>\n```python\ndef c():\n    return 4"),
        ("Case G: <think> inside string and comment in closed fence",
         "<think>thought</think>\n```python\n# <think> in comment\ns = '<think>in str</think>'\n```"),
        ("Case H: Pytest failure traceback followed by unclosed <think>",
         "FAILED test_foo.py - AssertionError\nTraceback (most recent call last):\n  File 'test.py', line 1\n<think>Analyzing failure..."),
        ("Case I: Unclosed <think> at start with pytest failure traceback",
         "<think>Analyzing...\nFAILED test_foo.py - AssertionError\nTraceback (most recent call last):\n  File 'test.py', line 1\nAssertionError\n```python\ndef fix(): pass\n```"),
    ])
    def test_standalone_server_evictor_parity(self, case_name, raw_text):
        import subneutralize_server
        server_evictor = subneutralize_server.StatefulMultiTurnEvictor(preserve_code=True, preserve_errors=True)
        cleaned = server_evictor._strip_scratchpad_ast_safe(raw_text)
        assert server_evictor.validate_ast_invariants(raw_text, cleaned), (
            f"Server evictor failed AST invariance on {case_name}. Cleaned:\n{cleaned}"
        )
        if "FAILED" in raw_text:
            assert "FAILED" in cleaned, f"Server evictor lost traceback on {case_name}"

