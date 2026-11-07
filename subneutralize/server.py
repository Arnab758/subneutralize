"""
================================================================================
   🚀 SubNeutralize v0.2.0: OpenAI-Compatible Inference Server & Gateway
   Author: Arnab Dutta | Apache-2.0 License
================================================================================
Enables drop-in integration with Cursor, VS Code (Continue/Cline), Antigravity, and Aider.
Features:
  1. Stateful Multi-Turn KV Scratchpad Eviction (55%-75% prefill memory reduction)
  2. Live CFO & Hardware Telemetry (/v1/telemetry)
  3. Real-Time SSE Streaming & Non-Streaming Completions
================================================================================
"""

import time
import uuid
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from subneutralize.evictor import StatefulMultiTurnEvictor


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
    enable_eviction: Optional[bool] = True


def create_app(engine: Any, model_id: str) -> "FastAPI":
    """Creates a FastAPI application configured with SubNeutralize v0.2.0."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI and Uvicorn are required to run the server. Install via: pip install fastapi uvicorn")

    app = FastAPI(
        title="SubNeutralize Inference Gateway",
        description="OpenAI-compatible inference server with runtime overthinking interception & KV eviction.",
        version="0.2.1"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Multi-Turn Evictor & Cumulative Telemetry State
    tokenizer = getattr(engine, "tokenizer", None)
    evictor = StatefulMultiTurnEvictor(tokenizer)

    telemetry_state = {
        "total_requests": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_prefill_tokens_evicted": 0,
        "compacted_turns_count": 0,
        "manifold_interventions": 0,
    }

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SubNeutralize v0.2.1 Gateway</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #090d16; color: #e2e8f0; margin: 0; padding: 24px; }}
    .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .stat-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; margin-top: 4px; }}
    .stat-lbl {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .btn {{ background: #0284c7; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
    .btn:hover {{ background: #0369a1; }}
    textarea {{ width: 100%; box-sizing: border-box; background: #030712; border: 1px solid #374151; color: #f8fafc; padding: 12px; border-radius: 8px; font-family: monospace; font-size: 14px; resize: vertical; }}
    pre {{ background: #030712; border: 1px solid #1e293b; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; color: #a5f3fc; white-space: pre-wrap; min-height: 80px; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: bold; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
  </style>
</head>
<body>
  <div style="max-width: 900px; margin: 0 auto;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <div>
        <h1 style="margin: 0; font-size: 24px; font-weight: 800; color: #f8fafc;">SubNeutralize Gateway</h1>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px;">In-Flight Latent Manifold Inference Governor • v0.2.1</p>
      </div>
      <div>
        <span class="badge">ONLINE</span>
        <a href="/docs" style="color: #38bdf8; font-size: 13px; text-decoration: none; margin-left: 12px;">API Docs &rarr;</a>
      </div>
    </div>

    <div class="grid">
      <div class="card"><div class="stat-lbl">Model</div><div class="stat-val" style="font-size: 15px; word-break: break-all;">{model_id}</div></div>
      <div class="card"><div class="stat-lbl">Total Requests</div><div class="stat-val" id="stat-req">0</div></div>
      <div class="card"><div class="stat-lbl">Tokens Evicted</div><div class="stat-val" id="stat-evicted" style="color: #c084fc;">0</div></div>
      <div class="card"><div class="stat-lbl">Manifold Interventions</div><div class="stat-val" id="stat-int" style="color: #34d399;">0</div></div>
    </div>

    <div class="card">
      <h3 style="margin-top: 0; font-size: 16px;">Interactive Inference Playground</h3>
      <textarea id="prompt-input" rows="3" placeholder="Enter coding or reasoning prompt... (e.g. Write a thread-safe circular buffer in Python)"></textarea>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
        <label style="font-size: 13px; color: #94a3b8;"><input type="checkbox" id="chk-evict" checked> Stateful KV-Cache Scratchpad Eviction</label>
        <button class="btn" id="btn-submit" onclick="runInference()">Generate with Governor</button>
      </div>
      <h4 style="margin: 16px 0 8px 0; font-size: 13px; color: #94a3b8;">Response:</h4>
      <pre id="output-box">Output will appear here...</pre>
    </div>
  </div>

  <script>
    async function updateTelemetry() {{
      try {{
        const res = await fetch('/v1/telemetry');
        const data = await res.json();
        document.getElementById('stat-req').textContent = data.total_requests;
        document.getElementById('stat-evicted').textContent = data.total_prefill_tokens_evicted;
        document.getElementById('stat-int').textContent = data.manifold_interventions;
      }} catch(e) {{}}
    }}
    setInterval(updateTelemetry, 2500);
    updateTelemetry();

    async function runInference() {{
      const prompt = document.getElementById('prompt-input').value.trim();
      if (!prompt) return;
      const btn = document.getElementById('btn-submit');
      const out = document.getElementById('output-box');
      btn.disabled = true;
      btn.textContent = 'Generating...';
      out.textContent = 'Computing trajectory through latent manifold...';

      try {{
        const res = await fetch('/v1/chat/completions', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            messages: [{{ role: 'user', content: prompt }}],
            enable_eviction: document.getElementById('chk-evict').checked
          }})
        }});
        const data = await res.json();
        out.textContent = data.choices[0].message.content;
      }} catch(e) {{
        out.textContent = 'Error: ' + e;
      }} finally {{
        btn.disabled = false;
        btn.textContent = 'Generate with Governor';
        updateTelemetry();
      }}
    }}
  </script>
</body>
</html>"""

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": model_id,
            "version": "0.2.1",
            "governor": "SubNeutralize Manifold Consensus Governor (v0.2.1)"
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

    @app.get("/v1/telemetry")
    def get_telemetry():
        return {
            "engine_version": "SubNeutralize v0.2.1 (Layer L-2 Manifold Governor + Stateful KV Evictor)",
            "model": model_id,
            "total_requests": telemetry_state["total_requests"],
            "total_prompt_tokens": telemetry_state["total_prompt_tokens"],
            "total_completion_tokens": telemetry_state["total_completion_tokens"],
            "total_prefill_tokens_evicted": evictor.total_prefill_tokens_evicted,
            "compacted_multi_turn_requests": evictor.compacted_turns_count,
            "manifold_interventions": telemetry_state["manifold_interventions"],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        telemetry_state["total_requests"] += 1

        # 1. Prepare structured messages
        messages_payload = [{"role": msg.role, "content": msg.content} for msg in req.messages]

        # 2. Stateful Multi-Turn KV Scratchpad Eviction
        compacted_messages, prompt_tokens, evicted = evictor.compact_messages(
            messages_payload, evict_history_scratchpads=bool(req.enable_eviction)
        )
        telemetry_state["total_prompt_tokens"] += prompt_tokens
        telemetry_state["total_prefill_tokens_evicted"] += evicted

        # 3. Execute SubNeutralize governed generation
        try:
            out = engine.generate(
                prompt=compacted_messages,
                max_new_tokens=req.max_tokens or 1500,
                temperature=req.temperature or 0.6,
                top_p=req.top_p or 0.95
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # Select content: preserve full synthesized answer/explanation in IDE gateway
        response_content = getattr(out, "answer", None) or getattr(out, "clean_code", None) or getattr(out, "text", "")
        reasoning_content = getattr(out, "reasoning", "")
        ans_toks = getattr(out, "answer_tokens", getattr(out, "code_tokens", 0))
        tot_toks = getattr(out, "total_tokens", prompt_tokens + ans_toks)
        telemetry_state["total_completion_tokens"] += ans_toks

        consensus = getattr(out, "consensus_reached", False)
        if consensus:
            telemetry_state["manifold_interventions"] += 1

        # 4. Handle Streaming response for Cursor/IDE real-time typing
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

        # 5. Standard Non-Streaming JSON Response
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
                "prompt_tokens": prompt_tokens,
                "prefill_tokens_evicted": evicted,
                "completion_tokens": ans_toks,
                "total_tokens": tot_toks,
                "subneutralize_telemetry": {
                    "thinking_tokens": getattr(out, "thinking_tokens", 0),
                    "consensus_reached": consensus,
                    "consensus_token": getattr(out, "consensus_step", None),
                    "latency_seconds": round(getattr(out, "wall_clock_seconds", 0.0), 2)
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
    print("      SUBNEUTRALIZE v0.2.0 OPENAI-COMPATIBLE INFERENCE GATEWAY     ")
    print("=" * 80)
    print(f"[*] Target Model:       {model_id}")
    print(f"[*] Compute Device:     {resolved_device}")
    print(f"[*] Gateway Host:       http://{host}:{port}/v1")
    print(f"[*] Telemetry Endpoint: http://{host}:{port}/v1/telemetry")
    print("-" * 80)
    print("  IDE CONFIGURATION (Cursor / VS Code Continue / Antigravity / Aider):")
    print(f"  * Base URL:           http://localhost:{port}/v1")
    print(f"  * Model Name:         {model_id}")
    print("  * API Key:            any (e.g. 'subneutralize')")
    print("-" * 80)

    print(f"[*] Loading {model_id} into memory...")
    dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=resolved_device, torch_dtype=dtype)
    governor = SubNeutralize(model, tokenizer)

    app = create_app(governor, model_id)
    print(f"[READY] Server running on http://{host}:{port}/v1. You can now code in your IDE!")
    uvicorn.run(app, host=host, port=port)
