"""
Unit tests for the OpenAI-compatible SubNeutralize server.
Verifies compatibility with Cursor, VS Code, and standard OpenAI API clients.
"""

import pytest
from starlette.testclient import TestClient
from subneutralize.server import create_app


class MockOutput:
    clean_code = "def add(a, b): return a + b"
    answer = "Here is the function to add two numbers:\n```python\ndef add(a, b): return a + b\n```\nIt returns the sum."
    reasoning = "The user wants an addition function. def add(a, b): return a + b is optimal."
    code_tokens = 12
    answer_tokens = 25
    thinking_tokens = 45
    total_tokens = 70
    wall_clock_seconds = 0.85
    consensus_reached = True
    consensus_step = 45


class MockEngine:
    def generate(self, prompt, **kwargs):
        return MockOutput()


def test_server_health_and_models():
    engine = MockEngine()
    app = create_app(engine, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    client = TestClient(app)

    # 1. Test /health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # 2. Test /v1/models (required by Cursor / OpenAI SDK)
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def test_chat_completions_non_streaming():
    engine = MockEngine()
    app = create_app(engine, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    client = TestClient(app)

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": [
            {"role": "user", "content": "Write a python function to add two numbers"}
        ],
        "stream": False
    }

    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    resp_json = res.json()

    assert resp_json["object"] == "chat.completion"
    # Verify that the full answer text (including explanation) is preserved
    assert "Here is the function to add two numbers:" in resp_json["choices"][0]["message"]["content"]
    assert "def add(a, b): return a + b" in resp_json["choices"][0]["message"]["content"]
    assert resp_json["choices"][0]["finish_reason"] == "stop"
    assert resp_json["usage"]["completion_tokens"] == 25
    assert resp_json["usage"]["subneutralize_telemetry"]["consensus_reached"] is True


def test_chat_completions_multiturn():
    engine = MockEngine()
    app = create_app(engine, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    client = TestClient(app)

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
            {"role": "user", "content": "Write an addition function."}
        ],
        "stream": False
    }

    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    resp_json = res.json()
    assert "def add(a, b): return a + b" in resp_json["choices"][0]["message"]["content"]


def test_chat_completions_streaming():
    engine = MockEngine()
    app = create_app(engine, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    client = TestClient(app)

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": [
            {"role": "user", "content": "Write a python function to add two numbers"}
        ],
        "stream": True
    }

    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "data: [DONE]" in res.text

