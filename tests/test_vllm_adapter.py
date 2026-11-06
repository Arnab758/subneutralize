"""
Unit Tests for SubNeutralize vLLM Adapter (VLLMSubNeutralizeLogitsProcessor & VLLMMultiTurnMiddleware).
"""
import pytest
import torch
from subneutralize.vllm_adapter import (
    VLLMSubNeutralizeLogitsProcessor,
    VLLMMultiTurnMiddleware,
    create_vllm_sampling_params,
)


class MockTokenizer:
    """Mock tokenizer for vLLM adapter tests."""
    def __init__(self):
        self.eos_token_id = 999
        self.vocab = {
            "</think>": 100,
            "```": 101,
            "def": 201,
            "(": 202,
            ")": 203,
            "foo": 204,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        tokens = []
        for word in text.split():
            tokens.append(self.vocab.get(word, 500))
        return tokens

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.inv_vocab.get(t, "x") for t in token_ids)


def test_vllm_logits_processor_initialization():
    tokenizer = MockTokenizer()
    processor = VLLMSubNeutralizeLogitsProcessor(
        tokenizer=tokenizer,
        think_end_token_id=100,
        min_reasoning_tokens=5,
        max_reasoning_tokens=50,
        entropy_threshold=1.5,
    )
    assert processor.think_end_token_id == 100
    assert 999 in processor.eos_token_ids


def test_vllm_logits_processor_early_generation_pass_through():
    tokenizer = MockTokenizer()
    processor = VLLMSubNeutralizeLogitsProcessor(
        tokenizer=tokenizer,
        think_end_token_id=100,
        min_reasoning_tokens=10,
        max_reasoning_tokens=50,
    )

    logits = torch.randn(1000)
    prompt_ids = [10, 20, 30]
    past_ids = [201, 204]  # len = 2 < min_reasoning_tokens

    # Early in generation, logits must be returned unchanged
    modified_logits = processor(prompt_ids, past_ids, logits.clone())
    assert torch.allclose(modified_logits, logits)


def test_vllm_logits_processor_consensus_transition():
    tokenizer = MockTokenizer()
    processor = VLLMSubNeutralizeLogitsProcessor(
        tokenizer=tokenizer,
        think_end_token_id=100,
        min_reasoning_tokens=5,
        max_reasoning_tokens=20,
        entropy_threshold=0.1,  # Low threshold to trigger hesitation
        stagnation_patience=2,
    )

    prompt_ids = [1, 2, 3]
    past_ids = [201] * 10  # Repetitive tokens, gen_len = 10 > min_reasoning_tokens

    # Step 1: Delimiter balanced, hesitation triggers stagnation
    logits = torch.ones(1000) * 0.001
    processor(prompt_ids, past_ids, logits.clone())

    # Step 2: Trigger transition
    final_logits = processor(prompt_ids, past_ids, logits.clone())

    # After transition, token 100 (</think>) must have overwhelming probability
    assert final_logits[100] == 100.0
    assert torch.isneginf(final_logits[0])


def test_create_vllm_sampling_params():
    tokenizer = MockTokenizer()
    params = create_vllm_sampling_params(
        tokenizer=tokenizer,
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
    )
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.9
    assert params["max_tokens"] == 500
    assert len(params["logits_processors"]) == 1
    assert isinstance(params["logits_processors"][0], VLLMSubNeutralizeLogitsProcessor)


def test_vllm_multi_turn_middleware():
    import asyncio

    async def run_test():
        async def mock_downstream_app(scope, receive, send):
            body_event = await receive()
            import json
            payload = json.loads(body_event["body"].decode("utf-8"))
            # Verify compacted scratchpad
            assert len(payload["messages"]) == 2
            assert "<think>" not in payload["messages"][0]["content"]

        middleware = VLLMMultiTurnMiddleware(app=mock_downstream_app, tokenizer=MockTokenizer())

        # Mock multi-turn request
        request_data = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [
                {
                    "role": "assistant",
                    "content": "<think>Drafting loop logic...</think>```python\ndef solution():\n    return True\n```"
                },
                {
                    "role": "user",
                    "content": "Can you optimize this?"
                }
            ]
        }
        import json
        body_bytes = json.dumps(request_data).encode("utf-8")

        async def mock_receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        async def mock_send(event):
            pass

        scope = {
            "type": "http",
            "path": "/v1/chat/completions",
            "method": "POST",
        }

        await middleware(scope, mock_receive, mock_send)
        assert middleware.requests_intercepted == 1

    asyncio.run(run_test())
