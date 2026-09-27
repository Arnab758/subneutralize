"""
Unit tests for the enterprise benchmark suite and code runner.
"""

from subneutralize.benchmark import BENCHMARK_SUITE, REFERENCE_SOLUTIONS, run_code_test


def test_benchmark_suite_integrity():
    assert len(BENCHMARK_SUITE) == 4
    for prob in BENCHMARK_SUITE:
        assert "id" in prob
        assert "title" in prob
        assert "prompt" in prob
        assert "test" in prob
        assert prob["id"] in REFERENCE_SOLUTIONS


def test_all_reference_solutions_pass():
    for prob in BENCHMARK_SUITE:
        pid = prob["id"]
        ref_code = REFERENCE_SOLUTIONS[pid]
        passed, msg = run_code_test(ref_code, prob["test"])
        assert passed is True, f"Reference solution for {pid} failed: {msg}"
