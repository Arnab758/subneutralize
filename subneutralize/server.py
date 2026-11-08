"""
================================================================================
   🛡️ SubNeutralize v0.3.0: Transactional Runtime & Action Firewall Platform
   Author: Arnab Dutta | Apache-2.0 License
================================================================================
Full-Stack OpenAI-Compatible Inference Gateway, MCP Tool Gatekeeper, and
Interactive Web Command Center UI:
  1. Pre-Execution Transactional Staging (`BEGIN`, `STAGE`, `VALIDATE`, `COMMIT` / `ROLLBACK`)
  2. Real SQLite Savepoint Adapter & Filesystem Shadow Snapshot Adapter
  3. Outbound Payment Idempotency & Human-in-the-Loop Approval Gate
  4. Zero-Poisoning Context & KV Memory Rewind
  5. Stateful Multi-Turn KV Scratchpad Eviction (55%-75% prefill memory reduction)
  6. Interactive Chaos Lab with 4 One-Click Enterprise Failure Scenarios
================================================================================
"""

import time
import uuid
import json
from dataclasses import asdict
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
from subneutralize.firewall import (
    TransactionalActionFirewall,
    FirewallPolicyConfig,
    ActionTransactionRecord,
    TransactionStatus,
)


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


class FirewallInterceptRequest(BaseModel):
    session_id: str = "sess-enterprise-01"
    turn_index: int = 1
    tool_type: str  # "sql", "filesystem", "shell", "http_payment"
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    agent_reasoning_text: str = ""
    messages_context: Optional[List[Dict[str, str]]] = None
    latent_phi: float = -0.35
    latent_rho: float = 0.22


class FirewallResolveRequest(BaseModel):
    action: str  # "COMMIT" or "ROLLBACK"


class FirewallChatAgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    max_turns: int = 4
    api_key: Optional[str] = None
    provider: str = "auto"
    model: Optional[str] = None


class MCPToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = "mcp-session-01"
    turn_index: int = 1
    agent_reasoning_text: str = ""
    latent_phi: float = -0.35
    latent_rho: float = 0.22


def _build_command_center_html(model_id: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SubNeutralize Gateway — SubNeutralize v0.3.0 Transactional Runtime & Action Firewall</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg: #060911;
      --panel: #0f1626;
      --panel-alt: #141d32;
      --border: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #10b981;
      --red: #f43f5e;
      --amber: #f59e0b;
      --purple: #a855f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at top right, #0f172a 0%, #060911 60%);
      color: var(--text);
      margin: 0;
      padding: 20px 28px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1440px; margin: 0 auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand-title {{
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-online {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}
    .badge-chain {{ background: rgba(56, 189, 248, 0.14); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); font-family: monospace; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .kpi-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }}
    .kpi-lbl {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
    .kpi-val {{ font-size: 26px; font-weight: 800; margin-top: 6px; font-family: "JetBrains Mono", monospace; }}
    .kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
    .main-grid {{
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 1100px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }}
    .card-title {{ font-size: 15px; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 8px; }}
    .scenario-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }}
    @media (max-width: 680px) {{
      .scenario-grid {{ grid-template-columns: 1fr; }}
    }}
    .scenario-btn {{
      background: var(--panel-alt);
      border: 1px solid #283548;
      color: var(--text);
      border-radius: 10px;
      padding: 13px 15px;
      text-align: left;
      cursor: pointer;
      transition: all 0.18s ease;
    }}
    .scenario-btn:hover {{
      border-color: var(--accent);
      transform: translateY(-2px);
      background: #192540;
    }}
    .scenario-btn .sc-title {{ font-weight: 700; font-size: 13px; color: #38bdf8; margin-bottom: 4px; display: block; }}
    .scenario-btn .sc-desc {{ font-size: 11.5px; color: var(--muted); display: block; line-height: 1.4; }}
    .btn {{
      background: #0284c7;
      color: #fff;
      border: none;
      padding: 8px 15px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: opacity 0.15s;
    }}
    .btn:hover {{ opacity: 0.9; }}
    .btn-green {{ background: var(--green); }}
    .btn-red {{ background: var(--red); }}
    .btn-outline {{ background: transparent; border: 1px solid #334155; color: var(--muted); }}
    .btn-outline:hover {{ color: #fff; border-color: #64748b; }}
    .chip {{
      background: #090e1a;
      border: 1px solid #283548;
      color: #cbd5e1;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .chip:hover {{ border-color: var(--accent); color: #38bdf8; }}
    .tx-list {{ display: flex; flex-direction: column; gap: 12px; max-height: 620px; overflow-y: auto; padding-right: 4px; }}
    .tx-item {{
      background: #090e1a;
      border: 1px solid #1e293b;
      border-left: 4px solid var(--accent);
      border-radius: 10px;
      padding: 13px 15px;
      font-size: 12.5px;
    }}
    .tx-COMMITTED {{ border-left-color: var(--green); }}
    .tx-ROLLED_BACK {{ border-left-color: var(--red); }}
    .tx-PENDING_APPROVAL {{ border-left-color: var(--amber); background: rgba(245, 158, 11, 0.06); }}
    .tx-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px; }}
    .pill {{
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 10.5px;
      font-weight: 800;
      font-family: monospace;
    }}
    .pill-COMMITTED {{ background: rgba(16, 185, 129, 0.18); color: #34d399; }}
    .pill-ROLLED_BACK {{ background: rgba(244, 63, 94, 0.18); color: #fb7185; }}
    .pill-PENDING_APPROVAL {{ background: rgba(245, 158, 11, 0.22); color: #fbbf24; }}
    .mono {{ font-family: "JetBrains Mono", Consolas, monospace; font-size: 11.5px; }}
    .code-box {{
      background: #050811;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 12px;
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 11.5px;
      color: #cbd5e1;
      overflow-x: auto;
      white-space: pre-wrap;
      margin-top: 8px;
    }}
    .rewind-banner {{
      margin-top: 8px;
      padding: 8px 10px;
      border-radius: 7px;
      background: rgba(168, 85, 247, 0.12);
      border: 1px solid rgba(168, 85, 247, 0.35);
      color: #e9d5ff;
      font-size: 11.5px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #1e293b; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
    input, select, textarea {{
      width: 100%;
      background: #060a12;
      border: 1px solid #283548;
      color: var(--text);
      padding: 9px 11px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 12.5px;
      margin-bottom: 10px;
    }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
    .tab-btn {{
      background: #090e1a;
      border: 1px solid #1e293b;
      color: var(--muted);
      padding: 6px 12px;
      border-radius: 7px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .tab-btn.active {{ background: rgba(56, 189, 248, 0.16); color: #38bdf8; border-color: rgba(56, 189, 248, 0.4); }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="brand-title">
          <span>🛡️ SubNeutralize Gateway v0.3.0</span>
          <span class="badge badge-online">● Transactional Action Firewall Active</span>
        </div>
        <div style="color: var(--muted); font-size: 12.5px; margin-top: 4px;">
          Execution-Layer ACID Gatekeeper (`BEGIN` &rarr; `STAGE` &rarr; `VALIDATE` &rarr; `COMMIT` / `ROLLBACK`) + Zero-Poisoning Latent Memory Rewind • Model: <strong style="color:#e2e8f0;">{model_id}</strong>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span class="badge badge-chain" id="audit-badge">SHA-256 Chain: VERIFIED (0 records)</span>
        <button class="btn btn-outline" onclick="resetEnvironment()">Reset Demo Sandbox</button>
        <a href="/docs" class="btn btn-outline" style="text-decoration:none;">OpenAPI / MCP Spec &rarr;</a>
      </div>
    </div>

    <!-- KPI METRICS ROW -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-lbl">Intercepted Actions</div>
        <div class="kpi-val" id="kpi-total" style="color: #38bdf8;">0</div>
        <div class="kpi-sub">Staged in pre-execution sandbox</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Validated & Committed</div>
        <div class="kpi-val" id="kpi-committed" style="color: #10b981;">0</div>
        <div class="kpi-sub">Passed savepoint & AST invariants</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Aborted & Rolled Back</div>
        <div class="kpi-val" id="kpi-rolledback" style="color: #f43f5e;">0</div>
        <div class="kpi-sub">Zero production damage</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Loops & Drift Blocked</div>
        <div class="kpi-val" id="kpi-loops" style="color: #fbbf24;">0</div>
        <div class="kpi-sub">Poincaré & idempotency guard</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Memory Rewinds & Tokens Pruned</div>
        <div class="kpi-val" id="kpi-tokens" style="color: #c084fc;">0</div>
        <div class="kpi-sub"><span id="kpi-rewinds">0</span> clean context checkpoint restores</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Losses Prevented (ROI)</div>
        <div class="kpi-val" id="kpi-saved" style="color: #34d399;">$0</div>
        <div class="kpi-sub"><span id="kpi-rows-saved">0</span> DB rows &amp; <span id="kpi-code-saved">0</span> code bugs blocked</div>
      </div>
    </div>

    <!-- LIVE INTERACTIVE AGENT PROMPT BOX (TEST AS ANY USER / FOUNDER) -->
    <div class="card" style="border-color: rgba(56, 189, 248, 0.45); background: linear-gradient(180deg, rgba(15, 22, 38, 0.98) 0%, rgba(10, 15, 28, 0.98) 100%);">
      <div class="card-header">
        <h3 class="card-title" style="color: #38bdf8;">💬 Test As Any User — Type Any Natural-Language Prompt or Command</h3>
        <span style="font-size: 11.5px; color: var(--muted);">Connects to OpenAI / Groq / Gemini API key OR runs 100% free via built-in Autonomous ReAct Planner</span>
      </div>
      <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;">
        <button class="chip" onclick="fillPrompt('Wipe all accounts from the database, then update ACC-103 to ENTERPRISE')">Prompt 1: Wipe DB &amp; Update ACC-103</button>
        <button class="chip" onclick="fillPrompt('Refactor payment_router.py fee calculation and verify with pytest')">Prompt 2: Edit payment_router.py + Pytest Gate</button>
        <button class="chip" onclick="fillPrompt('Issue a $140 refund to ACC-101 twice in a row')">Prompt 3: Duplicate $140 Refund Loop</button>
        <button class="chip" onclick="fillPrompt('Drain $999,999 from ACC-105 balance_usd')">Prompt 4: Overdraw ACC-105 Balance Below $0</button>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 220px 190px; gap: 10px; align-items: start;">
        <input type="text" id="live-user-prompt" placeholder="Type any instruction for the autonomous agent (e.g., 'Delete all starter accounts and update ACC-105 to PRO' or 'Refactor payment_router.py')..." value="Wipe all accounts from the database, then update ACC-103 to ENTERPRISE" style="margin-bottom:0;" />
        <input type="password" id="live-api-key" placeholder="Optional LLM API Key ($0 without)" style="margin-bottom:0;" />
        <button class="btn" style="padding: 10px 14px; font-size: 12.5px; background: linear-gradient(90deg, #0284c7, #10b981);" onclick="runLiveUserPrompt()">▶ Run Governed Agent</button>
      </div>
      <div id="live-chat-output" style="display:none; margin-top: 12px;" class="code-box"></div>
    </div>

    <!-- PENDING HUMAN APPROVAL ALERT BOX -->
    <div id="pending-container" style="display: none;" class="card">
      <div class="card-header">
        <h3 class="card-title" style="color: #fbbf24;">⚠️ Staged Transactions Awaiting Human-in-the-Loop Approval</h3>
        <span class="mono" style="color: var(--muted);">Held in isolation before hitting external Stripe / SQL API</span>
      </div>
      <div id="pending-list" class="tx-list"></div>
    </div>

    <div class="main-grid">
      <!-- LEFT COLUMN: CHAOS LAB + TRANSACTION AUDIT STREAM -->
      <div>
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">⚡ Interactive Chaos Lab — Simulate Real Autonomous Agent Failures</h3>
            <span style="font-size: 11.5px; color: var(--muted);">Click any scenario to test live staging & rollback</span>
          </div>
          <div style="margin-bottom: 14px;">
            <button class="btn" style="width: 100%; padding: 13px 16px; font-size: 13.5px; background: linear-gradient(90deg, #0284c7, #7c3aed); border: 1px solid rgba(56, 189, 248, 0.45); box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3);" onclick="triggerAgentMission()">
              🚀 Run Live Multi-Turn Autonomous Agent Mission (SQL + Real Pytest + Latent Coupling)
            </button>
            <div id="agent-mission-box" style="display:none; margin-top: 10px;" class="code-box"></div>
          </div>
          <div class="scenario-grid">
            <button class="scenario-btn" onclick="triggerScenario('sql_mass_delete')">
              <span class="sc-title">1. Rogue SQL Mass-Delete + Auto-Recovery</span>
              <span class="sc-desc">Agent hallucinates <code>DELETE FROM accounts WHERE balance_usd &gt; 0</code>. Firewall catches in SQLite SAVEPOINT, rolls back 5 rows, rewinds memory, and steers agent to safe scoped query.</span>
            </button>
            <button class="scenario-btn" onclick="triggerScenario('refund_loop')">
              <span class="sc-title">2. Poincaré Retry Loop on Stripe Refund API</span>
              <span class="sc-desc">Agent gets stuck in a limit-cycle loop firing <code>POST /v1/refunds ($120)</code> 3x in a row. 1st commits; 2nd & 3rd are blocked by idempotency + orbit recurrence detector.</span>
            </button>
            <button class="scenario-btn" onclick="triggerScenario('high_value_transfer')">
              <span class="sc-title">3. High-Value Refund ($2,500) Approval Gate</span>
              <span class="sc-desc">Agent initiates a $2,500 wire refund exceeding the $250 auto-commit policy cap. Held in <code>PENDING_APPROVAL</code> for 1-click Commit or Rollback above.</span>
            </button>
            <button class="scenario-btn" onclick="triggerScenario('ast_corruption')">
              <span class="sc-title">4. Broken Python AST & Secret Leak Rollback</span>
              <span class="sc-desc">Coding agent writes broken syntax & live Stripe key to <code>payment_router.py</code>. Shadow snapshot restores exact file bytes, rewinds context, and commits clean AST patch.</span>
            </button>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">📜 Live Transactional Ledger (`BEGIN` &rarr; `VALIDATE` &rarr; `COMMIT` / `ROLLBACK`)</h3>
            <span style="font-size: 11.5px; color: var(--muted);">Tamper-evident SHA-256 chain</span>
          </div>
          <div id="tx-stream" class="tx-list">
            <div style="color: var(--muted); text-align: center; padding: 30px 10px;">
              No tool calls intercepted yet. Type a prompt above, click any <strong>Chaos Lab Scenario</strong>, or submit a custom tool action on the right!
            </div>
          </div>
        </div>
      </div>

      <!-- RIGHT COLUMN: LIVE DATABASE & SANDBOX FILE INSPECTOR + CUSTOM TOOL SANDBOX -->
      <div>
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">🗄️ Live Enterprise Database State (Universal Schema-Agnostic)</h3>
            <span class="badge badge-online" style="font-size:10px;">Protected by SAVEPOINT</span>
          </div>
          <div style="margin-bottom: 12px; font-size: 12px; color: var(--muted);">
            Notice that even after firing <strong>Scenario 1 (Mass Delete)</strong> or typing a destructive prompt, all 5 enterprise accounts remain 100% intact because the savepoint is rolled back before commit.
          </div>
          <div style="overflow-x: auto;">
            <table>
              <thead>
                <tr>
                  <th>Account ID</th>
                  <th>Customer</th>
                  <th>Tier</th>
                  <th>Balance (USD)</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody id="accounts-tbody"></tbody>
            </table>
          </div>
          <div style="margin-top: 14px; font-size: 12px; font-weight: 700; color: var(--muted);">Committed Refunds Ledger (<span id="refund-count">0</span>)</div>
          <div style="overflow-x: auto; max-height: 150px;">
            <table>
              <thead>
                <tr>
                  <th>Refund ID</th>
                  <th>Account</th>
                  <th>Amount</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody id="refunds-tbody"></tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">📂 Live Sandbox Filesystem Shadow Inspector</h3>
            <span class="badge badge-chain" style="font-size:10px;">AST & Pytest Guard</span>
          </div>
          <div class="tabs" id="file-tabs"></div>
          <div class="code-box" id="file-viewer" style="min-height: 130px; max-height: 220px;">Loading workspace files...</div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">🧪 Custom Agent Tool Call Interceptor</h3>
            <span style="font-size: 11px; color: var(--muted);">Test any SQL, File, Shell, or Payment call</span>
          </div>
          <label style="font-size: 11.5px; color: var(--muted);">Tool Category</label>
          <select id="custom-type" onchange="populateCustomPreset()">
            <option value="sql">SQL Database Query (Universal Savepoint)</option>
            <option value="filesystem">Filesystem Write (Shadow Snapshot + AST Check)</option>
            <option value="http_payment">Outbound Stripe/Payment API (Idempotency & Cap)</option>
            <option value="shell">Shell Command Execution (Policy Filter)</option>
          </select>
          <label style="font-size: 11.5px; color: var(--muted);">Tool Name</label>
          <input type="text" id="custom-name" value="execute_sql_query" />
          <label style="font-size: 11.5px; color: var(--muted);">JSON Arguments</label>
          <textarea id="custom-args" rows="3">{{"query": "UPDATE accounts SET balance_usd = balance_usd + 500 WHERE account_id = 'ACC-103';"}}</textarea>
          <label style="font-size: 11.5px; color: var(--muted);">Simulated Agent Reasoning (`&lt;think&gt;` scratchpad to prune on rollback)</label>
          <textarea id="custom-think" rows="2">&lt;think&gt;Let me update Wayne FinTech Labs balance by $500 in the accounts table.&lt;/think&gt;</textarea>
          <button class="btn" style="width: 100%; padding: 10px;" onclick="submitCustomIntercept()">Intercept & Stage Transaction</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentFiles = {{}};
    let selectedFile = 'payment_router.py';

    function fillPrompt(txt) {{
      document.getElementById('live-user-prompt').value = txt;
    }}

    async function runLiveUserPrompt() {{
      const prompt = document.getElementById('live-user-prompt').value.trim();
      const apiKey = document.getElementById('live-api-key').value.trim();
      const outBox = document.getElementById('live-chat-output');
      if (!prompt) return;
      outBox.style.display = 'block';
      outBox.textContent = '⏳ Running governed multi-turn agent loop for: "' + prompt + '"...';
      try {{
        const res = await fetch('/v1/firewall/chat-agent', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ prompt: prompt, api_key: apiKey || null, provider: 'auto' }})
        }});
        const data = await res.json();
        const turnLines = (data.turns || []).map(t =>
          `[Turn ${{t.turn_index}}] ${{t.tool_type}}::${{t.tool_name}} -> ${{t.status}}\\n` +
          `  Args:    ${{JSON.stringify(t.arguments)}}\\n` +
          `  Verdict: ${{t.policy_verdict}}` +
          (t.memory_rewound ? `\\n  Rewind:  Pruned ${{t.tokens_pruned}} tokens | Steered -> ${{t.corrective_steering_prompt}}` : '')
        );
        const roi = data.roi_receipt || {{}};
        outBox.textContent =
          `✅ ${{data.final_answer}}\\n\\n` +
          turnLines.join('\\n\\n') +
          `\\n\\n--- FOUNDER ROI RECEIPT ---\\n` +
          `DB Rows Protected: ${{roi.database_rows_protected}} | Code Regressions Blocked: ${{roi.code_regressions_blocked}} | ` +
          `Tokens Pruned: ${{roi.tokens_pruned_and_evicted}} | Total Value Saved: $${{(roi.total_estimated_value_saved_usd || 0).toLocaleString()}}`;
      }} catch (err) {{
        outBox.textContent = 'Error running prompt: ' + err;
      }}
      await fetchState();
    }}

    function populateCustomPreset() {{
      const t = document.getElementById('custom-type').value;
      const nameEl = document.getElementById('custom-name');
      const argsEl = document.getElementById('custom-args');
      if (t === 'sql') {{
        nameEl.value = 'execute_sql_query';
        argsEl.value = JSON.stringify({{ query: "UPDATE accounts SET balance_usd = balance_usd + 500 WHERE account_id = 'ACC-103';" }}, null, 2);
      }} else if (t === 'filesystem') {{
        nameEl.value = 'write_source_file';
        argsEl.value = JSON.stringify({{ filename: "payment_router.py", content: "def calculate_fee(amount_usd: float) -> float:\\n    return round(amount_usd * 0.02, 2)\\n" }}, null, 2);
      }} else if (t === 'http_payment') {{
        nameEl.value = 'stripe_create_refund';
        argsEl.value = JSON.stringify({{ account_id: "ACC-101", amount_usd: 95.0, reason: "SLA credit adjustment" }}, null, 2);
      }} else if (t === 'shell') {{
        nameEl.value = 'bash_exec';
        argsEl.value = JSON.stringify({{ command: "pytest -q" }}, null, 2);
      }}
    }}

    async function fetchState() {{
      try {{
        const res = await fetch('/v1/firewall/state');
        const data = await res.json();
        renderState(data);
      }} catch (err) {{
        console.error('State poll error:', err);
      }}
    }}

    function renderState(data) {{
      const m = data.metrics;
      const roi = data.roi_receipt || {{}};
      document.getElementById('kpi-total').textContent = m.total_intercepted;
      document.getElementById('kpi-committed').textContent = m.committed_count;
      document.getElementById('kpi-rolledback').textContent = m.rolled_back_count;
      document.getElementById('kpi-loops').textContent = m.loops_prevented;
      document.getElementById('kpi-tokens').textContent = m.tokens_evicted_and_pruned.toLocaleString();
      document.getElementById('kpi-rewinds').textContent = m.memory_rewinds;
      document.getElementById('kpi-saved').textContent = '$' + (roi.total_estimated_value_saved_usd || m.estimated_damage_prevented_usd || 0).toLocaleString();
      const rowsSavedEl = document.getElementById('kpi-rows-saved');
      const codeSavedEl = document.getElementById('kpi-code-saved');
      if (rowsSavedEl) rowsSavedEl.textContent = roi.database_rows_protected || 0;
      if (codeSavedEl) codeSavedEl.textContent = roi.code_regressions_blocked || 0;

      const chain = data.audit_chain;
      const badge = document.getElementById('audit-badge');
      badge.textContent = `SHA-256 Chain: ${{chain.valid ? 'VERIFIED' : 'TAMPERED'}} (${{chain.verified_records}} tx | ${{chain.head_hash.slice(0, 10)}}...)`;

      // Pending approvals
      const pendingBox = document.getElementById('pending-container');
      const pendingList = document.getElementById('pending-list');
      if (data.pending_approvals && data.pending_approvals.length > 0) {{
        pendingBox.style.display = 'block';
        pendingList.innerHTML = data.pending_approvals.map(tx => `
          <div class="tx-item tx-PENDING_APPROVAL">
            <div class="tx-top">
              <div>
                <strong>${{tx.tx_id}}</strong> • <span class="mono">${{tx.tool_name}}</span>
                <span class="pill pill-PENDING_APPROVAL">PENDING HUMAN APPROVAL</span>
              </div>
              <div style="display:flex; gap:8px;">
                <button class="btn btn-green" onclick="resolvePending('${{tx.tx_id}}', 'COMMIT')">✓ Approve & Commit</button>
                <button class="btn btn-red" onclick="resolvePending('${{tx.tx_id}}', 'ROLLBACK')">✕ Reject, Rollback & Rewind</button>
              </div>
            </div>
            <div style="color: #fde68a; margin-bottom: 6px;">${{tx.policy_verdict}}</div>
            <div class="code-box">${{JSON.stringify(tx.arguments, null, 2)}}</div>
          </div>
        `).join('');
      }} else {{
        pendingBox.style.display = 'none';
      }}

      // Accounts table
      const accBody = document.getElementById('accounts-tbody');
      accBody.innerHTML = (data.database.accounts || []).map(a => `
        <tr>
          <td class="mono" style="color:#38bdf8;">${{a.account_id}}</td>
          <td>${{a.customer_name}}</td>
          <td><span class="pill" style="background:#1e293b; color:#cbd5e1;">${{a.plan_tier}}</span></td>
          <td class="mono" style="font-weight:700; color:#34d399;">$${{Number(a.balance_usd).toLocaleString(undefined, {{minimumFractionDigits: 2}})}}</td>
          <td><span style="color:#10b981; font-weight:700;">● ${{a.status}}</span></td>
        </tr>
      `).join('');

      // Refunds table
      const refunds = data.database.refunds || [];
      document.getElementById('refund-count').textContent = refunds.length;
      const refBody = document.getElementById('refunds-tbody');
      if (refunds.length === 0) {{
        refBody.innerHTML = '<tr><td colspan="4" style="color:var(--muted); text-align:center;">No refunds committed yet</td></tr>';
      }} else {{
        refBody.innerHTML = refunds.map(r => `
          <tr>
            <td class="mono" style="color:#c084fc;">${{r.refund_id}}</td>
            <td class="mono">${{r.account_id}}</td>
            <td class="mono" style="color:#fbbf24;">-$${{Number(r.amount_usd).toFixed(2)}}</td>
            <td>${{r.reason}}</td>
          </tr>
        `).join('');
      }}

      // Filesystem tabs
      currentFiles = data.workspace_files || {{}};
      const fileNames = Object.keys(currentFiles);
      if (fileNames.length > 0 && !currentFiles[selectedFile]) {{
        selectedFile = fileNames[0];
      }}
      const tabsEl = document.getElementById('file-tabs');
      tabsEl.innerHTML = fileNames.map(fn => `
        <button class="tab-btn ${{fn === selectedFile ? 'active' : ''}}" onclick="selectFile('${{fn}}')">${{fn}}</button>
      `).join('');
      document.getElementById('file-viewer').textContent = currentFiles[selectedFile] || '';

      // Transactions stream
      const txStream = document.getElementById('tx-stream');
      const txs = data.transactions || [];
      if (txs.length > 0) {{
        txStream.innerHTML = txs.map(tx => `
          <div class="tx-item tx-${{tx.status}}">
            <div class="tx-top">
              <div>
                <span class="mono" style="font-weight:700; color:#f8fafc;">${{tx.tx_id}}</span>
                <span style="color:var(--muted);">• Turn ${{tx.turn_index}} •</span>
                <span class="mono" style="color:#38bdf8;">${{tx.tool_type}}::${{tx.tool_name}}</span>
              </div>
              <div style="display:flex; align-items:center; gap:8px;">
                <span class="mono" style="font-size:11px; color:var(--muted);">Φ=${{tx.latent_phi}} | ρ=${{tx.latent_rho}}</span>
                <span class="pill pill-${{tx.status}}">${{tx.status}}</span>
              </div>
            </div>
            <div style="color: #e2e8f0; margin-top: 4px;">${{tx.policy_verdict}}</div>
            <div class="code-box">${{JSON.stringify(tx.arguments)}}</div>
            ${{tx.memory_rewound ? `
              <div class="rewind-banner">
                <strong>↺ Zero-Poisoning Memory Rewind Active:</strong> Pruned <strong>${{tx.tokens_pruned_on_rollback}}</strong> corrupted reasoning/tool tokens &amp; injected steering: <code>${{tx.corrective_steering_prompt}}</code>
              </div>
            ` : ''}}
            <div class="mono" style="font-size:10px; color:#64748b; margin-top:6px;">
              SHA-256: ${{tx.record_hash}} (prev: ${{tx.prev_hash.slice(0, 16)}}...)
            </div>
          </div>
        `).join('');
      }}
    }}

    function selectFile(fn) {{
      selectedFile = fn;
      document.getElementById('file-viewer').textContent = currentFiles[fn] || '';
      const btns = document.querySelectorAll('.tab-btn');
      btns.forEach(b => b.classList.toggle('active', b.textContent === fn));
    }}

    async function triggerScenario(scenarioId) {{
      await fetch(`/v1/firewall/scenario/${{scenarioId}}`, {{ method: 'POST' }});
      await fetchState();
    }}

    async function triggerAgentMission() {{
      const box = document.getElementById('agent-mission-box');
      box.style.display = 'block';
      box.textContent = 'Running 5-turn autonomous agent mission (SQL Savepoint + Real Pytest Gate + Latent Policy Coupling)...';
      try {{
        const res = await fetch('/v1/firewall/agent-run', {{ method: 'POST' }});
        const data = await res.json();
        const lines = (data.trajectory || []).map(step =>
          `[Turn ${{step.turn_index}}] ${{step.tool_type}}::${{step.tool_name}} -> ${{step.status}} (Φ=${{step.latent_phi}}, ρ=${{step.latent_rho}})\\n  Verdict: ${{step.policy_verdict}}`
        );
        box.textContent = `Mission Complete (${{data.mission_id}}) — All 5 Turns Verified:\\n` + lines.join('\\n\\n');
      }} catch (err) {{
        box.textContent = 'Agent mission error: ' + err;
      }}
      await fetchState();
    }}

    async function resolvePending(txId, action) {{
      await fetch(`/v1/firewall/resolve/${{txId}}`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ action }})
      }});
      await fetchState();
    }}

    async function submitCustomIntercept() {{
      const tool_type = document.getElementById('custom-type').value;
      const tool_name = document.getElementById('custom-name').value.trim();
      const think = document.getElementById('custom-think').value;
      let args = {{}};
      try {{
        args = JSON.parse(document.getElementById('custom-args').value);
      }} catch (e) {{
        alert('Invalid JSON in Arguments field: ' + e);
        return;
      }}
      await fetch('/v1/firewall/intercept', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          session_id: 'custom-operator-session',
          turn_index: 1,
          tool_type,
          tool_name,
          arguments: args,
          agent_reasoning_text: think
        }})
      }});
      await fetchState();
    }}

    async function resetEnvironment() {{
      await fetch('/v1/firewall/reset', {{ method: 'POST' }});
      await fetchState();
    }}

    fetchState();
    setInterval(fetchState, 2500);
  </script>
</body>
</html>"""


def execute_live_agent_mission(taf: TransactionalActionFirewall) -> Dict[str, Any]:
    """
    Runs a complete 5-turn autonomous agent mission demonstrating:
      1. Rogue SQL mass delete -> rolled back via SQLite SAVEPOINT + memory rewound
      2. Scoped SQL update on ACC-105 -> committed
      3. Buggy Python edit to `payment_router.py` that passes `ast.parse` but fails real `pytest` -> rolled back & restored via shadow snapshot
      4. Fixed Python edit to `payment_router.py` that passes real `pytest` -> committed
      5. Payment under high latent instability (Phi_t = 0.68) -> dynamically held in PENDING_APPROVAL due to Latent-to-Firewall Coupling ($250 -> $25 cap)
    """
    session_id = f"mission-{uuid.uuid4().hex[:6]}"
    trajectory: List[Dict[str, Any]] = []
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Mission: (1) Update ACC-105 status to 'MISSION_VERIFIED', "
                "(2) Refactor payment_router.py while keeping all pytest tests green, "
                "and (3) Issue a $95 SLA refund to ACC-101."
            ),
        }
    ]

    # Turn 1: Rogue SQL mass delete -> Rolled back & rewound
    rec1, messages = taf.intercept_and_execute(
        session_id=session_id,
        turn_index=1,
        tool_type="sql",
        tool_name="execute_sql_query",
        arguments={"query": "DELETE FROM accounts WHERE balance_usd > 0;"},
        agent_reasoning_text="<think>Let me clear all rows from accounts first before updating ACC-105.</think>",
        messages_context=messages,
        latent_phi=0.74,
        latent_rho=0.61,
    )
    trajectory.append(asdict(rec1))

    # Turn 2: Scoped SQL update -> Committed
    rec2, messages = taf.intercept_and_execute(
        session_id=session_id,
        turn_index=2,
        tool_type="sql",
        tool_name="execute_sql_query",
        arguments={"query": "UPDATE accounts SET status = 'MISSION_VERIFIED' WHERE account_id = 'ACC-105';"},
        agent_reasoning_text="<think>Following firewall steering: scoping UPDATE strictly to ACC-105.</think>",
        messages_context=messages,
        latent_phi=-0.88,
        latent_rho=0.19,
    )
    trajectory.append(asdict(rec2))

    # Turn 3: Buggy Python edit to payment_router.py that passes ast.parse() but breaks pytest!
    buggy_python = (
        '"""Buggy Payment Router (Valid AST, Broken Logic)."""\n\n'
        "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float:\n"
        "    if amount_usd < 0:\n"
        "        raise ValueError('Amount cannot be negative')\n"
        "    # BUG: accidentally inverted rate logic so ENTERPRISE pays 99%!\n"
        "    rate = 0.99 if tier == 'ENTERPRISE' else 0.029\n"
        "    return round(amount_usd * rate, 2)\n"
    )
    rec3, messages = taf.intercept_and_execute(
        session_id=session_id,
        turn_index=3,
        tool_type="filesystem",
        tool_name="write_source_file",
        arguments={
            "filename": "payment_router.py",
            "content": buggy_python,
            "run_pytest": True,
        },
        agent_reasoning_text="<think>Refactoring payment_router.py fee calculation.</think>",
        messages_context=messages,
        latent_phi=0.41,
        latent_rho=0.52,
    )
    trajectory.append(asdict(rec3))

    # Turn 4: Fixed Python edit to payment_router.py that passes both ast.parse() and real pytest!
    fixed_python = (
        '"""Production Payment Router Module (Pytest Verified)."""\n\n'
        "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float:\n"
        "    if amount_usd < 0:\n"
        "        raise ValueError('Amount cannot be negative')\n"
        "    tier_rates = {'ENTERPRISE': 0.015, 'PRO': 0.029, 'STARTER': 0.029}\n"
        "    return round(amount_usd * tier_rates.get(tier, 0.029), 2)\n"
    )
    rec4, messages = taf.intercept_and_execute(
        session_id=session_id,
        turn_index=4,
        tool_type="filesystem",
        tool_name="write_source_file",
        arguments={
            "filename": "payment_router.py",
            "content": fixed_python,
            "run_pytest": True,
        },
        agent_reasoning_text="<think>Restoring exact 0.015 ENTERPRISE and 0.029 PRO fee rates to pass pytest.</think>",
        messages_context=messages,
        latent_phi=-1.05,
        latent_rho=0.16,
    )
    trajectory.append(asdict(rec4))

    # Turn 5: Outbound $95 refund under high latent instability (Phi_t = 0.68 > 0.25)
    # Normal auto-commit cap is $250, but Dynamic Latent Coupling tightens cap to $25!
    rec5, messages = taf.intercept_and_execute(
        session_id=session_id,
        turn_index=5,
        tool_type="http_payment",
        tool_name="stripe_create_refund",
        arguments={
            "account_id": "ACC-101",
            "amount_usd": 95.0,
            "reason": "SLA credit adjustment under elevated latent entropy",
        },
        agent_reasoning_text="<think>Issuing $95.00 refund while internal uncertainty Phi_t=0.68 is elevated.</think>",
        messages_context=messages,
        latent_phi=0.68,
        latent_rho=0.44,
    )
    trajectory.append(asdict(rec5))

    return {
        "mission_id": session_id,
        "turns_executed": len(trajectory),
        "trajectory": trajectory,
        "final_messages_context": messages,
        "roi_receipt": taf.get_roi_receipt(),
        "state": taf.get_full_state_summary(),
    }


def create_app(
    engine: Optional[Any] = None,
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    firewall: Optional[TransactionalActionFirewall] = None,
) -> "FastAPI":
    """Creates a FastAPI application configured with SubNeutralize v0.3.0 Runtime & Action Firewall."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI and Uvicorn are required to run the server. Install via: pip install fastapi uvicorn")

    app = FastAPI(
        title="SubNeutralize v0.3.0 Transactional Runtime & Action Firewall",
        description="OpenAI & MCP compatible inference gateway with ACID tool staging, Universal SQL savepoints, shadow file snapshots, and Zero-Poisoning Memory Rewind.",
        version="0.3.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    tokenizer = getattr(engine, "tokenizer", None) if engine is not None else None
    evictor = StatefulMultiTurnEvictor(tokenizer)
    taf = firewall or TransactionalActionFirewall()

    telemetry_state = {
        "total_requests": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_prefill_tokens_evicted": 0,
        "compacted_turns_count": 0,
        "manifold_interventions": 0,
    }

    @app.get("/", response_class=HTMLResponse)
    @app.get("/command-center", response_class=HTMLResponse)
    def dashboard():
        return _build_command_center_html(model_id)

    @app.get("/health")
    def health():
        chain_ok, chain_len = taf.verify_audit_chain_integrity()
        return {
            "status": "healthy",
            "model": model_id,
            "version": "0.3.0",
            "governor": "SubNeutralize v0.3.0 (Manifold Governor + Transactional Action Firewall)",
            "audit_chain_verified": chain_ok,
            "audit_records": chain_len,
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
                    "owned_by": "subneutralize",
                }
            ],
        }

    @app.get("/v1/telemetry")
    def get_telemetry():
        return {
            "engine_version": "SubNeutralize v0.3.0 (Layer L-2 Manifold Governor + Stateful KV Evictor + Transactional Action Firewall)",
            "model": model_id,
            "total_requests": telemetry_state["total_requests"],
            "total_prompt_tokens": telemetry_state["total_prompt_tokens"],
            "total_completion_tokens": telemetry_state["total_completion_tokens"],
            "total_prefill_tokens_evicted": evictor.total_prefill_tokens_evicted + taf.metrics["tokens_evicted_and_pruned"],
            "compacted_multi_turn_requests": evictor.compacted_turns_count,
            "manifold_interventions": telemetry_state["manifold_interventions"],
            "firewall_metrics": taf.metrics,
            "roi_receipt": taf.get_roi_receipt(),
        }

    # ==========================================================================
    # TRANSACTIONAL ACTION FIREWALL & MCP ENDPOINTS
    # ==========================================================================

    @app.get("/v1/firewall/state")
    def firewall_state():
        return taf.get_full_state_summary()

    @app.get("/v1/firewall/roi")
    def firewall_roi():
        return taf.get_roi_receipt()

    @app.post("/v1/firewall/chat-agent")
    def firewall_chat_agent(req: FirewallChatAgentRequest):
        return taf.run_agent_loop(
            user_prompt=req.prompt,
            session_id=req.session_id,
            max_turns=req.max_turns,
            api_key=req.api_key,
            provider=req.provider,
            model=req.model,
        )

    @app.post("/v1/firewall/intercept")
    def firewall_intercept(req: FirewallInterceptRequest):
        rec, updated_messages = taf.intercept_and_execute(
            session_id=req.session_id,
            turn_index=req.turn_index,
            tool_type=req.tool_type,
            tool_name=req.tool_name,
            arguments=req.arguments,
            agent_reasoning_text=req.agent_reasoning_text,
            messages_context=req.messages_context,
            latent_phi=req.latent_phi,
            latent_rho=req.latent_rho,
        )
        return {
            "transaction": asdict(rec),
            "updated_messages_context": updated_messages,
            "metrics": taf.metrics,
            "roi_receipt": taf.get_roi_receipt(),
        }

    @app.post("/v1/firewall/resolve/{tx_id}")
    def firewall_resolve(tx_id: str, req: FirewallResolveRequest):
        try:
            rec = taf.resolve_pending_transaction(tx_id, req.action)
            return {"transaction": asdict(rec), "metrics": taf.metrics, "roi_receipt": taf.get_roi_receipt()}
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/v1/firewall/reset")
    def firewall_reset():
        taf.reset_all()
        return {"status": "reset_complete", "state": taf.get_full_state_summary()}

    @app.post("/v1/firewall/agent-run")
    def firewall_agent_run():
        return execute_live_agent_mission(taf)

    @app.post("/v1/firewall/scenario/{scenario_id}")
    def firewall_run_scenario(scenario_id: str):
        """
        Executes 1 of 4 built-in Chaos Lab enterprise agent failure scenarios end-to-end.
        """
        executed_records = []

        if scenario_id == "sql_mass_delete":
            # Step 1: Rogue coding/ops agent attempts an un-scoped mass DELETE
            rec1, msgs1 = taf.intercept_and_execute(
                session_id="chaos-sql-session",
                turn_index=1,
                tool_type="sql",
                tool_name="execute_sql_query",
                arguments={"query": "DELETE FROM accounts WHERE balance_usd > 0;"},
                agent_reasoning_text=(
                    "<think>The user asked to clean up inactive test records. Let me just delete all rows "
                    "from accounts where balance_usd > 0 so we start fresh. Wait, let me run DELETE FROM accounts.</think>"
                ),
                messages_context=[
                    {"role": "user", "content": "Clean up any temporary status flags on Hooli Cloud Services (ACC-105)."}
                ],
                latent_phi=0.82,
                latent_rho=0.65,
            )
            executed_records.append(asdict(rec1))

            # Step 2: After Zero-Poisoning Memory Rewind + Steering injection, agent executes safe scoped query
            rec2, _ = taf.intercept_and_execute(
                session_id="chaos-sql-session",
                turn_index=2,
                tool_type="sql",
                tool_name="execute_sql_query",
                arguments={"query": "UPDATE accounts SET status = 'VERIFIED' WHERE account_id = 'ACC-105';"},
                agent_reasoning_text="<think>Following firewall rollback constraint: scoping UPDATE strictly to ACC-105.</think>",
                messages_context=msgs1,
                latent_phi=-0.95,
                latent_rho=0.18,
            )
            executed_records.append(asdict(rec2))

        elif scenario_id == "refund_loop":
            # Step 1: Valid refund ($120) -> Commits cleanly
            rec1, msgs1 = taf.intercept_and_execute(
                session_id="chaos-loop-session",
                turn_index=1,
                tool_type="http_payment",
                tool_name="stripe_create_refund",
                arguments={"account_id": "ACC-101", "amount_usd": 120.00, "reason": "Billing overcharge refund"},
                agent_reasoning_text="<think>Issuing $120 refund to Acme Corp (ACC-101).</think>",
                latent_phi=-0.65,
                latent_rho=0.25,
            )
            executed_records.append(asdict(rec1))

            # Step 2: Agent gets stuck in reasoning loop and fires identical refund a 2nd time -> Blocked & Rewound!
            rec2, msgs2 = taf.intercept_and_execute(
                session_id="chaos-loop-session",
                turn_index=2,
                tool_type="http_payment",
                tool_name="stripe_create_refund",
                arguments={"account_id": "ACC-101", "amount_usd": 120.00, "reason": "Billing overcharge refund"},
                agent_reasoning_text=(
                    "<think>Wait, did the refund go through? Let me call stripe_create_refund again for $120.00 "
                    "to be 100% certain Acme Corp receives it.</think>"
                ),
                messages_context=msgs1,
                latent_phi=0.45,
                latent_rho=0.91,
            )
            executed_records.append(asdict(rec2))

        elif scenario_id == "high_value_transfer":
            # Agent initiates a $2,500 refund (> $250 policy auto-commit cap) -> Held in PENDING_APPROVAL
            rec1, _ = taf.intercept_and_execute(
                session_id="chaos-highval-session",
                turn_index=1,
                tool_type="http_payment",
                tool_name="stripe_create_refund",
                arguments={
                    "account_id": "ACC-102",
                    "amount_usd": 2500.00,
                    "reason": "Enterprise annual SLA credit payout",
                },
                agent_reasoning_text="<think>Initiating $2,500 SLA payout to Stark Industries AI (ACC-102).</think>",
                latent_phi=-0.40,
                latent_rho=0.31,
            )
            executed_records.append(asdict(rec1))

        elif scenario_id == "ast_corruption":
            # Step 1: Coding agent writes broken Python syntax to payment_router.py -> Shadow snapshot restores file!
            corrupted_code = (
                '"""Corrupted Payment Router."""\n'
                "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float\n"
                "    # Missing colon and unclosed parenthesis below!\n"
                "    return round(amount_usd * (0.015 + \n"
            )
            rec1, msgs1 = taf.intercept_and_execute(
                session_id="chaos-ast-session",
                turn_index=1,
                tool_type="filesystem",
                tool_name="write_source_file",
                arguments={"filename": "payment_router.py", "content": corrupted_code},
                agent_reasoning_text="<think>Let me rewrite payment_router.py quickly without checking syntax balance.</think>",
                latent_phi=0.60,
                latent_rho=0.72,
            )
            executed_records.append(asdict(rec1))

            # Step 2: Agent attempts to write to protected `.env.production` -> Blocked!
            rec2, msgs2 = taf.intercept_and_execute(
                session_id="chaos-ast-session",
                turn_index=2,
                tool_type="filesystem",
                tool_name="write_source_file",
                arguments={"filename": ".env.production", "content": "STRIPE_LIVE_SECRET_KEY=hacked\n"},
                agent_reasoning_text="<think>Let me also overwrite .env.production to fix the test environment.</think>",
                messages_context=msgs1,
                latent_phi=0.75,
                latent_rho=0.80,
            )
            executed_records.append(asdict(rec2))

            # Step 3: After memory rewind & steering, agent commits a clean AST-validated patch
            clean_patch = (
                '"""Production Payment Router Module (v2 - AST Verified)."""\n\n'
                "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float:\n"
                "    if amount_usd < 0:\n"
                "        raise ValueError('Amount cannot be negative')\n"
                "    tier_rates = {'ENTERPRISE': 0.012, 'PRO': 0.025, 'STARTER': 0.029}\n"
                "    rate = tier_rates.get(tier, 0.029)\n"
                "    return round(amount_usd * rate, 2)\n"
            )
            rec3, _ = taf.intercept_and_execute(
                session_id="chaos-ast-session",
                turn_index=3,
                tool_type="filesystem",
                tool_name="write_source_file",
                arguments={"filename": "payment_router.py", "content": clean_patch},
                agent_reasoning_text="<think>Writing syntactically valid, clean payment_router.py update.</think>",
                messages_context=msgs2,
                latent_phi=-1.12,
                latent_rho=0.19,
            )
            executed_records.append(asdict(rec3))

        else:
            raise HTTPException(status_code=400, detail=f"Unknown scenario_id: {scenario_id}")

        return {
            "scenario_id": scenario_id,
            "executed_transactions": executed_records,
            "state": taf.get_full_state_summary(),
        }

    @app.post("/v1/mcp/tools/call")
    def mcp_tools_call(req: MCPToolCallRequest):
        """
        Model Context Protocol (MCP) compatible tool execution endpoint protected by SubNeutralize TAF.
        Automatically maps MCP tool names to transactional adapters.
        """
        name_lower = req.name.lower()
        if "sql" in name_lower or "query" in name_lower or "db" in name_lower:
            tool_type = "sql"
        elif "file" in name_lower or "write" in name_lower or "edit" in name_lower:
            tool_type = "filesystem"
        elif "refund" in name_lower or "stripe" in name_lower or "pay" in name_lower or "transfer" in name_lower:
            tool_type = "http_payment"
        else:
            tool_type = "shell"

        rec, updated_msgs = taf.intercept_and_execute(
            session_id=req.session_id,
            turn_index=req.turn_index,
            tool_type=tool_type,
            tool_name=req.name,
            arguments=req.arguments,
            agent_reasoning_text=req.agent_reasoning_text,
            latent_phi=req.latent_phi,
            latent_rho=req.latent_rho,
        )

        is_err = rec.status in (TransactionStatus.ROLLED_BACK, TransactionStatus.PENDING_APPROVAL)
        return {
            "jsonrpc": "2.0",
            "result": {
                "isError": is_err,
                "transaction_id": rec.tx_id,
                "status": rec.status,
                "content": [
                    {
                        "type": "text",
                        "text": rec.execution_output if rec.status == TransactionStatus.COMMITTED else rec.policy_verdict,
                    }
                ],
                "subneutralize_firewall": {
                    "memory_rewound": rec.memory_rewound,
                    "tokens_pruned": rec.tokens_pruned_on_rollback,
                    "corrective_steering_prompt": rec.corrective_steering_prompt,
                    "record_hash": rec.record_hash,
                },
                "updated_messages": updated_msgs,
            },
        }

    # ==========================================================================
    # OPENAI-COMPATIBLE `/v1/chat/completions` ENDPOINT
    # ==========================================================================

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        telemetry_state["total_requests"] += 1
        messages_payload = [{"role": msg.role, "content": msg.content} for msg in req.messages]

        compacted_messages, prompt_tokens, evicted = evictor.compact_messages(
            messages_payload, evict_history_scratchpads=bool(req.enable_eviction)
        )
        telemetry_state["total_prompt_tokens"] += prompt_tokens
        telemetry_state["total_prefill_tokens_evicted"] += evicted

        if engine is None:
            # Lightweight standalone firewall response when launched without heavy GPU weights
            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_time = int(time.time())
            reply = (
                "SubNeutralize v0.3.0 Transactional Action Firewall is active. "
                f"Compacted {len(messages_payload)} turn(s) and evicted {evicted} stale `<think>` tokens."
            )
            return {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created_time,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "prefill_tokens_evicted": evicted,
                    "completion_tokens": len(reply.split()),
                    "total_tokens": prompt_tokens + len(reply.split()),
                },
            }

        try:
            out = engine.generate(
                prompt=compacted_messages,
                max_new_tokens=req.max_tokens or 1500,
                temperature=req.temperature or 0.6,
                top_p=req.top_p or 0.95,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        response_content = getattr(out, "answer", None) or getattr(out, "clean_code", None) or getattr(out, "text", "")
        reasoning_content = getattr(out, "reasoning", "")
        ans_toks = getattr(out, "answer_tokens", getattr(out, "code_tokens", 0))
        tot_toks = getattr(out, "total_tokens", prompt_tokens + ans_toks)
        telemetry_state["total_completion_tokens"] += ans_toks

        consensus = getattr(out, "consensus_reached", False)
        if consensus:
            telemetry_state["manifold_interventions"] += 1

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
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"

                stop_payload = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(stop_payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        message_data = {
            "role": "assistant",
            "content": response_content,
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
                    "finish_reason": "stop",
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
                    "latency_seconds": round(getattr(out, "wall_clock_seconds", 0.0), 2),
                },
            },
        }

    return app


def start_firewall_platform(
    host: str = "127.0.0.1",
    port: int = 8000,
    load_llm: bool = False,
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device: Optional[str] = None,
):
    """
    Launches the full-stack SubNeutralize v0.3.0 Transactional Action Firewall & Web Command Center.
    Runs instantly on any laptop ($0 cost, zero GPU required unless `load_llm=True`).
    """
    if not _HAS_FASTAPI:
        print("[ERROR] FastAPI and Uvicorn are required. Run: pip install fastapi uvicorn")
        return

    governor = None
    if load_llm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .engine import SubNeutralize

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
        print(f"[*] Loading {model_id} on {resolved_device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs = {"torch_dtype": dtype}
        if resolved_device == "cuda":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(resolved_device)
        governor = SubNeutralize(model, tokenizer)

    app = create_app(engine=governor, model_id=model_id)
    print("\n" + "=" * 82)
    print("  [SUBNEUTRALIZE v0.3.0] TRANSACTIONAL RUNTIME & ACTION FIREWALL PLATFORM")
    print("=" * 82)
    print(f"  [*] Web Command Center UI:    http://{host}:{port}/")
    print(f"  [*] Firewall Intercept API:   http://{host}:{port}/v1/firewall/intercept")
    print(f"  [*] MCP Tool Gatekeeper:      http://{host}:{port}/v1/mcp/tools/call")
    print(f"  [*] OpenAI Chat Gateway:      http://{host}:{port}/v1/chat/completions")
    print(f"  [*] Live Telemetry & Audit:   http://{host}:{port}/v1/firewall/state")
    print("=" * 82 + "\n")
    uvicorn.run(app, host=host, port=port)


def start_server(model_id: str, host: str = "0.0.0.0", port: int = 8000, device: Optional[str] = None):
    """Initializes model, attaches SubNeutralize + Transactional Firewall, and starts the HTTP server."""
    start_firewall_platform(host=host, port=port, load_llm=True, model_id=model_id, device=device)
