"""
Unit Tests for SubNeutralize Terminal Telemetry Monitor (subneutralize.monitor).
"""
import pytest
from subneutralize.monitor import (
    render_bar,
    format_number,
    generate_dashboard_text,
    run_terminal_monitor,
)


def test_render_bar():
    # 0%
    bar0 = render_bar(0.0, width=20)
    assert "[░░░░░░░░░░░░░░░░░░░░]   0.0%" in bar0

    # 50%
    bar50 = render_bar(50.0, width=20)
    assert "[██████████░░░░░░░░░░]  50.0%" in bar50

    # 100%
    bar100 = render_bar(100.0, width=20)
    assert "[████████████████████] 100.0%" in bar100

    # Clamp bounds (>100% and <0%)
    assert "100.0%" in render_bar(125.0, width=10)
    assert "0.0%" in render_bar(-15.0, width=10)


def test_format_number():
    assert format_number(42) == "42"
    assert format_number(1500) == "1.5k"
    assert format_number(3_500_000) == "3.50M"


def test_generate_dashboard_text_offline():
    text = generate_dashboard_text(None, host="127.0.0.1", port=8000)
    assert "Offline / Disconnected" in text
    assert "http://127.0.0.1:8000/v1/telemetry" in text


def test_generate_dashboard_text_online():
    telemetry_mock = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "engine_version": "SubNeutralize v0.2.0 (Layer L-2 Manifold Governor + Stateful KV Evictor)",
        "total_requests": 142,
        "total_prompt_tokens": 124500,
        "total_completion_tokens": 48200,
        "total_prefill_tokens_evicted": 58300,
        "compacted_multi_turn_requests": 95,
        "manifold_interventions": 88,
    }

    text = generate_dashboard_text(telemetry_mock, host="127.0.0.1", port=8000)
    assert "ONLINE" in text
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in text
    assert "142" in text
    assert "58.3k" in text
    assert "31.9%" in text  # 58300 / (124500 + 58300) = 31.89%


def test_run_terminal_monitor_max_cycles():
    # Verify that run_terminal_monitor executes with max_cycles without hanging
    run_terminal_monitor(host="127.0.0.1", port=9999, interval=0.01, max_cycles=2)
