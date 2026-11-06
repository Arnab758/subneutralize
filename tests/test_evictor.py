"""
Unit tests for StatefulMultiTurnEvictor.
Verifies historical <think> scratchpad eviction while preserving latest turn and canonical code.
"""

import pytest
from subneutralize.evictor import StatefulMultiTurnEvictor


class MockTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        # Approximate tokens by splitting on whitespace
        return text.split()


def test_evictor_single_turn_preservation():
    tokenizer = MockTokenizer()
    evictor = StatefulMultiTurnEvictor(tokenizer)

    messages = [{"role": "user", "content": "Write a python function."}]
    compacted, tokens, evicted = evictor.compact_messages(messages, evict_history_scratchpads=True)

    assert len(compacted) == 1
    assert compacted[0]["content"] == "Write a python function."
    assert evicted == 0


def test_evictor_multi_turn_scratchpad_stripping():
    tokenizer = MockTokenizer()
    evictor = StatefulMultiTurnEvictor(tokenizer)

    messages = [
        {"role": "user", "content": "Implement binary search in Python."},
        {
            "role": "assistant",
            "content": "<think>\nLet me think about binary search.\nFirst check left <= right.\nMidpoint calculation.\n</think>\n```python\ndef binary_search(arr, x): return 0\n```"
        },
        {"role": "user", "content": "Now add unit tests for it."}
    ]

    compacted, tokens, evicted = evictor.compact_messages(messages, evict_history_scratchpads=True)

    assert len(compacted) == 3
    # First user message untouched
    assert compacted[0]["content"] == messages[0]["content"]

    # Assistant turn should have <think>...</think> evicted
    assistant_content = compacted[1]["content"]
    assert "<think>" not in assistant_content
    assert "</think>" not in assistant_content
    assert "def binary_search(arr, x)" in assistant_content

    # Latest user turn completely untouched
    assert compacted[2]["content"] == "Now add unit tests for it."

    # Tokens should be evicted
    assert evicted > 0
    assert evictor.total_prefill_tokens_evicted > 0
    assert evictor.compacted_turns_count == 1


def test_evictor_disabled_flag():
    tokenizer = MockTokenizer()
    evictor = StatefulMultiTurnEvictor(tokenizer)

    messages = [
        {"role": "user", "content": "Test prompt."},
        {"role": "assistant", "content": "<think>some internal thought</think>result"},
        {"role": "user", "content": "Follow up."}
    ]

    compacted, tokens, evicted = evictor.compact_messages(messages, evict_history_scratchpads=False)
    assert evicted == 0
    assert "<think>" in compacted[1]["content"]
