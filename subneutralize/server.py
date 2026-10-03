"""
OpenAI-Compatible Local & Cloud Inference Server for SubNeutralize.
Enables drop-in integration with Cursor, VS Code (Continue/Cline), Antigravity, and Aider.
"""

import time
import uuid
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "subneutralize-default"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1500
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False


def create_app(engine: Any, model_id: str) -> "FastAPI":
    """Creates a FastAPI application configured with SubNeutralize."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI and Uvicorn are required to run the server. Install via: pip install fastapi uvicorn")

    app = FastAPI(
        title="SubNeutralize Inference Gateway",
        description="OpenAI-compatible inference server with runtime overthinking interception.",
        version="0.1.3"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": model_id,
            "governor": "SubNeutralize Scale-Free Consensus Governor"
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "subneutralize"
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # 1. Reconstruct prompt from messages
        prompt_parts = []
        for msg in req.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

        # 2. Execute SubNeutralize governed generation
        try:
            out = engine.generate(
                prompt=prompt,
                max_new_tokens=req.max_tokens or 1500,
                temperature=req.temperature or 0.6,
                top_p=req.top_p or 0.95
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # Select content: code if available, otherwise answer or full text
        response_content = getattr(out, "clean_code", None) or getattr(out, "answer", None) or getattr(out, "text", "")
        reasoning_content = getattr(out, "reasoning", "")
        code_toks = getattr(out, "code_tokens", getattr(out, "answer_tokens", 0))

        # 3. Handle Streaming response for Cursor/IDE real-time typing
        if req.stream:
            async def event_generator():
                words = response_content.split(" ")
                for i, word in enumerate(words):
                    chunk_text = word if i == len(words) - 1 else word + " "
                    chunk_payload = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"

                # Send final stop chunk
                stop_payload = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(stop_payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # 4. Standard Non-Streaming JSON Response
        message_data = {
            "role": "assistant",
            "content": response_content
        }
        if reasoning_content:
            message_data["reasoning_content"] = reasoning_content

        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message_data,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": code_toks,
                "total_tokens": out.total_tokens,
                "subneutralize_telemetry": {
                    "thinking_tokens": out.thinking_tokens,
                    "consensus_reached": out.consensus_reached,
                    "consensus_token": out.consensus_step,
                    "latency_seconds": round(out.wall_clock_seconds, 2)
                }
            }
        }

    return app


def start_server(model_id: str, host: str = "0.0.0.0", port: int = 8000, device: Optional[str] = None):
    """Initializes model, attaches SubNeutralize, and starts the HTTP server."""
    if not _HAS_FASTAPI:
        print("[ERROR] FastAPI and Uvicorn are required. Run: pip install fastapi uvicorn")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import SubNeutralize

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("      SUBNEUTRALIZE OPENAI-COMPATIBLE IDE INFERENCE GATEWAY      ")
    print("=" * 80)
    print(f"[*] Target Model:    {model_id}")
    print(f"[*] Compute Device:  {resolved_device}")
    print(f"[*] Gateway Host:    http://{host}:{port}/v1")
    print("-" * 80)
    print("  IDE CONFIGURATION (Cursor / VS Code Continue / Antigravity / Aider):")
    print(f"  * Base URL:        http://localhost:{port}/v1")
    print(f"  * Model Name:      {model_id}")
    print("  * API Key:         any (e.g. 'subneutralize')")
    print("-" * 80)

    print(f"[*] Loading {model_id} into memory...")
    dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=resolved_device, torch_dtype=dtype)
    governor = SubNeutralize(model, tokenizer)

    app = create_app(governor, model_id)
    print(f"[READY] Server running on http://{host}:{port}/v1. You can now code in your IDE!")
    uvicorn.run(app, host=host, port=port)
