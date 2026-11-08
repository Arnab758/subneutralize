# 🛡️ SubNeutralize: Latent Dynamical Inference Governor & Transactional Action Firewall for Autonomous Reasoning Agents

[![PyPI version](https://badge.fury.io/py/subneutralize.svg)](https://pypi.org/project/subneutralize/)
[![CI](https://github.com/Arnab758/subneutralize/actions/workflows/ci.yml/badge.svg)](https://github.com/Arnab758/subneutralize/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22941619.svg)](https://doi.org/10.5281/zenodo.22941619)
[![Paper PDF](https://img.shields.io/badge/Research%20Paper-PDF-red.svg)](paper/SubNeutralize_Paper.pdf)
[![Research Paper](https://img.shields.io/badge/Research%20Paper-Markdown-blue.svg)](paper/RESEARCH_PAPER.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

> **SubNeutralize** is an open-source, parameter-free runtime inference governor and transactional execution firewall for autoregressive reasoning models (DeepSeek-R1, QwQ, Llama, and agentic LLMs).  
> It unites **In-Flight Latent Manifold Governance** (cutting reasoning overthinking by 35%–50% without weight retraining) with an **ACID Execution Firewall** (`BEGIN` $\to$ `STAGE` $\to$ `VALIDATE` $\to$ `COMMIT` / `ROLLBACK`) that prevents autonomous agents from corrupting databases or repositories and performs **surgical memory rewind** to kill multi-turn retry loops.

---

## 📑 Table of Contents
1. [The Double Crisis of Autonomous Reasoning Agents](#-the-double-crisis-of-autonomous-reasoning-agents)
2. [Architecture & System Flow](#-architecture--system-flow)
3. [The Mathematical Formalism (Research Foundation)](#-the-mathematical-formalism-research-foundation)
4. [Empirical Benchmarks & Research Findings](#-empirical-benchmarks--research-findings)
5. [5 Ways to Deploy SubNeutralize](#-5-ways-to-deploy-subneutralize-choose-your-architecture)
6. [Universal Schema-Agnostic SQL & Filesystem Sandboxing](#-universal-schema-agnostic-sql--filesystem-sandboxing)
7. [Competitive Comparison Matrix](#-competitive-comparison-matrix)
8. [Comprehensive Test Suite & Verification](#-comprehensive-test-suite--verification)
9. [Academic Citation & License](#-academic-citation--license)

---

## 💡 The Double Crisis of Autonomous Reasoning Agents

Modern autonomous AI agents (coding assistants, Text-to-SQL agents, operations bots) suffer from two distinct failure modes that break production systems:

### 1. The Test-Time "Overthinking Trap"
Large Reasoning Models (LRMs) such as DeepSeek-R1 and QwQ frequently solve the foundational algorithmic deduction within the first 80–150 tokens. However, lacking internal termination dynamics, they enter extensive cyclic verification loops (*"Wait, let me double-check... but what if..."*), consuming 1,500+ redundant tokens, inflating time-to-first-answer (TTFA), and blowing up API bills.

### 2. The Uncontrolled Blast Radius & Context Poisoning Crisis
Giving an LLM write permissions to databases, repositories, or external APIs carries catastrophic risk. A single hallucinated query can execute `DELETE FROM users;` or commit code that breaks existing unit tests. Furthermore, when a tool call fails, standard frameworks dump a 100-line stack trace into the prompt history. The model gets confused, apologizes, repeats the exact same mistake, and enters a **10-turn retry death loop** ($20–$50 wasted per run).

---

## 🏗️ Architecture & System Flow

SubNeutralize couples **Inference-Layer Hidden State Monitoring** with an **Execution-Layer ACID Action Firewall**:

```
[ LLM Agent (LangGraph / CrewAI / OpenAI / Claude / DeepSeek) ]
                          │
                 1. Tool Call Emitted
                          ▼
┌─────────────────────────────────────────────────────────────┐
│          SUBNEUTRALIZE TRANSACTIONAL ACTION FIREWALL        │
│                                                             │
│  [1] Latent & Policy Guard:                                 │
│      • Evaluates Layer L-2 Lyapunov stability (Φ_t)         │
│      • Computes Poincaré limit-cycle recurrence (ρ_orbit)   │
│      • Enforces idempotency (blocks duplicate retry loops)  │
│                                                             │
│  [2] ACID Staging Sandbox:                                  │
│      • SQL: Opens database `SAVEPOINT sp_<tx>`              │
│      • Code: Captures byte-exact shadow RAM snapshot        │
│      • Shell: Sandboxed subprocess execution                │
│                                                             │
│  [3] Deterministic Validation Gate:                         │
│      • SQL: Modifies > N rows? Violates non-negative stock? │
│      • Code: Did `ast.parse` or real `pytest -q` fail?      │
│      • Shell: Non-zero exit code or timeout?                │
│                                                             │
│  [4] Resolution & Context Rewind:                           │
│      • Pass: `RELEASE SAVEPOINT` ──► Commit State           │
│      • Fail: `ROLLBACK TO SAVEPOINT` ──► Restore Bytes      │
│              + Surgically prunes failed <think> tokens      │
│              + Slices physical GPU KV-cache blocks          │
│              + Injects 1-line corrective steering signal    │
└─────────────────────────────────────────────────────────────┘
                          │
                 2. Safe State Only
                          ▼
[ Real Production Infrastructure: PostgreSQL / Codebase / APIs ]
```

---

## 📐 The Mathematical Formalism (Research Foundation)

SubNeutralize operates directly on the internal geometry of reasoning without altering model weights.

### 1. Thin-SVD Prompt Bias Subspace Neutralization
Autoregressive decoding is heavily skewed by static prompt bias. At Decoder Layer $L-2$, we capture prefill activations $H_{\text{prompt}} \in \mathbb{R}^{S \times D}$, compute its thin Singular Value Decomposition:
$$\widetilde{H}_{\text{prompt}} = U \Sigma V^T$$
and extract the top-$k$ principal bias directions $V_{\text{bias}} \in \mathbb{R}^{D \times k}$. The orthogonal projection operator:
$$P_\perp = I - V_{\text{bias}} V_{\text{bias}}^T$$
neutralizes prompt bias, projecting decoding hidden states into the pure deduction manifold:
$$h^\perp_t = P_\perp h_t$$

### 2. Self-Sizing Roy-Vetterli Coupon Collector Horizons
Rather than using arbitrary hyperparameter thresholds, SubNeutralize dynamically sizes the decoding burn-in ($T_{\text{burn}}$) and covering bound ($T_{\text{cover}}$) using the Renyi entropy effective ranks of the singular values:
$$r_{L1} = \exp\left(-\sum p_i \ln p_i\right), \quad r_{L2} = \exp\left(-\sum q_i \ln q_i\right)$$
$$H_{\text{coupon}} = \left\lceil r_{L1} \cdot \ln(r_{L2}) \right\rceil$$

### 3. Poincaré Limit-Cycle Orbit Recurrence
To catch repetitive reasoning loops in-flight, SubNeutralize computes the cosine similarity between the current trajectory centroid $\bar{h}_t$ and historical trajectory states with a lag $\tau \ge 20$:
$$\rho_{\text{orbit}}(t) = \max_{20 \le \tau \le 96} \cos(\bar{h}_t, \bar{h}_{t-\tau})$$
When $\rho_{\text{orbit}} \ge 0.85$, the model is mathematically certified to be trapped in a periodic limit-cycle attractor.

### 4. Dimensionless Lyapunov Potential ($\Phi_t$)
We aggregate normalized geometric metrics using online Welford statistics ($z$-scores):
$$\Phi_t = z(r_{\text{erank}}) + z(v_t) + z(H_{\text{entropy}}) - z(C_{\text{deer}})$$
Convergence into an attractor basin occurs when the exponential moving average $\text{EMA}(\Phi_t) < -1.05$ with high confidence ($C_{\text{deer}} \ge 0.86$) and low entropy ($H \le 0.48$).

### 5. Delimiter-Depth Syntactic Gating ($D_t = 0$)
To guarantee 0% syntax mutilation, SubNeutralize tracks bracket/parenthesis balancing:
$$D_t = d_{\text{paren}} + d_{\text{bracket}} + d_{\text{brace}}$$
The reasoning phase is only terminated when $D_t = 0$ at natural structural boundaries (`\n`, `.\n`, `:\n`), or under emergency attractor saturation ($t \ge 12$).

### 6. MemDecay Region-Aware Multi-Turn Retention
Historical context is categorized into semantic regions with exponential attention decay:
$$w(r, \Delta t) = 2^{-\Delta t / t_{1/2}(r)}$$
* `ASSISTANT_THINK`: $t_{1/2} = 15.0$ tokens (rapid purge of dead reasoning)
* `SYSTEM` / `CANONICAL_CODE` / `PYTEST_TRACEBACK`: $t_{1/2} \to \infty$ (100% permanence)

---

## 📊 Empirical Benchmarks & Research Findings

SubNeutralize has been validated across NVIDIA A100 GPU clusters and local hardware on reasoning models (DeepSeek-R1-Distill-Qwen 1.5B/7B/14B, QwQ-32B).

### Benchmark 1: Multi-Agent Collaborative Coding Workflow (A100)
Evaluated across a 3-agent autonomous software engineering pipeline:

| Metric | Vanilla Baseline | SubNeutralize v0.3.0 | Improvement |
| :--- | :--- | :--- | :--- |
| **Unit Test Pass Rate** | **0 / 5 (0%)** | **5 / 5 (100%)** | **+100% Success** (eliminated context rot) |
| **Total Workflow Tokens** | 8,269 tokens | 4,024 tokens | **-51.3% Token Waste** |
| **Agent 3 Prefill Memory** | 3,377 tokens | 1,210 tokens | **-64.2% Prefill Memory** |
| **KV Cache Footprint** | Unbounded expansion | Constant bounded | **-60.8% VRAM Peak** |

### Benchmark 2: SWE-bench Lite (5 Real-World Repositories)
Tested on real Python issue resolution tasks:

| Metric | Vanilla Baseline | SubNeutralize v0.3.0 | Improvement |
| :--- | :--- | :--- | :--- |
| **Resolved Issues** | 2 / 5 (40%) | **3 / 5 (60%)** | **+50% Relative Accuracy** |
| **Total Consumed Tokens** | 14,194 tokens | 8,610 tokens | **-39.3% Net Compute** |
| **Prefill Context Retention**| Full Raw History | Compact Canonical | **-61.2% Context Bloat** |

### Benchmark 3: 5-Turn Stratified Agent Coding Loop
| Metric | Vanilla Baseline | SubNeutralize v0.3.0 | Improvement |
| :--- | :--- | :--- | :--- |
| **Tier C Wall-Clock Latency** | 571.85s (9.53m) | **462.68s (7.71m)** | **-19.1% Faster (-109.17s)** |
| **Tier C Prefill Ratio** | 4.16× (3,192 tok) | **3.42× (2,941 tok)** | **-323 tok / turn bloat** |
| **Tier B Self-Correction** | 0 / 12 (Answer Flip)| **7 / 12 (+6 passes)**| **Monotonic Learning** |
| **Syntactic Code Integrity** | 100% | **100%** | **0% AST Mutilation** |

---

## 🛠️ 5 Ways to Deploy SubNeutralize (Choose Your Architecture)

### Option 1: As 2-Line Agent Middleware (LangChain / LangGraph / OpenAI)
Wrap your existing tools in two lines of code without altering prompts or models:

```python
from subneutralize import TransactionalActionFirewall, FirewallPolicyConfig

fw = TransactionalActionFirewall(
    db_url="postgresql://user:pass@localhost:5432/app_db", # or sqlite:///app.db
    workspace_dir="./my_app_repo",
    policy=FirewallPolicyConfig(
        max_sql_affected_rows=1,              # Block any SQL touching > 1 row
        run_pytest_on_python_edits=True,      # Block any code patch that breaks pytest
        max_auto_payment_amount_usd=250.0     # Hold transfers > $250 in PENDING_APPROVAL
    )
)

# Wrap existing functions or LangChain tools:
safe_query_db, safe_edit_code, safe_payout = fw.wrap_tools([
    query_database,
    edit_source_file,
    stripe_refund
])

# Pass wrapped tools directly to LangGraph / CrewAI / OpenAI
agent = create_agent(tools=[safe_query_db, safe_edit_code, safe_payout])
```

---

### Option 2: As an OpenAI-Compatible AI Gateway (Cursor, VS Code, Cline, Aider)
Run SubNeutralize as an inference proxy. It exposes a standard `/v1/chat/completions` endpoint with SSE streaming that intercepts reasoning loops before streaming back to your IDE:

```bash
# Launch OpenAI gateway with in-flight governor:
subneutralize serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8000

# Open a side terminal for real-time telemetry:
subneutralize top

# Or launch Aider with automatic 1-click SubNeutralize routing:
subneutralize aider
```

**IDE Configuration:**
* **Cursor:** Set OpenAI API Base URL to `http://localhost:8000/v1` (API key: `subneutralize`).
* **VS Code (Continue / Cline):** Set model endpoint to `http://localhost:8000/v1`.
* **Aider:** Run `subneutralize aider`.

---

### Option 3: Full-Stack Web Command Center & Chaos Lab UI
Launch the self-hosted glassmorphism Command Center UI locally on any laptop ($0 cost, zero GPU required):

```bash
subneutralize up --port 8000
```
Open **`http://localhost:8000`** in your browser:
* **Interactive Chaos Lab:** 1-click simulations of Rogue SQL Mass-Deletes, Stripe Refund Loops, and Python AST Corruptions.
* **Live Interactive Agent Prompt Box:** Type any free-form natural language prompt (or paste an OpenAI/Gemini/Groq API key) and watch the multi-turn agent stage tools, roll back errors, and recover live.
* **Founder ROI & Audit Receipt:** Real-time metrics tracking database rows saved, test regressions blocked, and dollar savings.

---

### Option 4: High-Throughput Serving (Native vLLM PagedAttention Integration)
Attach `VLLMMultiTurnMiddleware` to your `vllm serve` gateway to strip transient historical `<think>` scratchpads, slashing PagedAttention KV-cache occupancy by **35%–55%**:

```python
from vllm.entrypoints.openai.api_server import app
from subneutralize import VLLMMultiTurnMiddleware

app.add_middleware(VLLMMultiTurnMiddleware)
```

Or attach the native `SubNeutralizeManifoldProcessor` logits processor to any vLLM V1 inference loop:

```python
from vllm import LLM, SamplingParams
from subneutralize import create_vllm_sampling_params

llm = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct", tensor_parallel_size=1)
params = create_vllm_sampling_params(
    tokenizer=llm.get_tokenizer(),
    temperature=0.6,
    top_p=0.95,
)
sampling_params = SamplingParams(**params)
outputs = llm.generate(["Implement a lock-free concurrent queue in Python:"], sampling_params)
```

---

### Option 5: As a Python SDK Wrapper or Hugging Face `StoppingCriteria`
Wrap any open-weights model in 2 lines of Python for backend batch processing:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import SubNeutralize

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

governor = SubNeutralize(model, tokenizer)
output = governor.generate("Write a thread-safe circular buffer in Python:")

print(output.clean_code if output.clean_code else output.answer)
print(f"Thinking Tokens: {output.thinking_tokens} | Answer Tokens: {output.answer_tokens}")
print(f"Consensus Reached: {output.consensus_reached} at token {output.consensus_step}")
```

---

## 🗄️ Universal Schema-Agnostic SQL & Filesystem Sandboxing

SubNeutralize v0.3.0 is built for **any real-world database schema**:

```bash
# Test SubNeutralize against a custom E-Commerce startup backend:
# Schema: `customers`, `inventory`, `orders` (Zero hardcoded demo tables!)
python try_your_own_app.py --interactive
```

1. **PostgreSQL & SQLite URL Binding:** Supports `db_url="postgresql://..."` (`psycopg`/`psycopg2`) or `db_url="sqlite:///..."`.
2. **Dynamic Schema Introspection:** Auto-discovers any tables and enforces non-negative constraints on any column matching `balance`, `amount`, `credit`, `stock`, `quantity`, or `inventory`.
3. **Automated `pytest` Gate:** Runs `pytest -q` in a subprocess on shadow copies before code commits. If tests fail, it automatically restores the exact pre-edit file bytes.

---

## 📊 Competitive Comparison Matrix

| Capability | Standard Agent Frameworks (LangChain, CrewAI) | Text Guardrails (Guardrails AI, Lakera) | Model Distillation (Sparset Thinklite-27B) | **SubNeutralize v0.3.0** |
| :--- | :---: | :---: | :---: | :---: |
| **Approach Layer** | Application Orchestration | Surface Text / Regex | Offline Weight Retraining | **In-Flight Runtime + ACID Execution** |
| **Model Portability** | Universal | Universal | Locked to 1 Model (27B) | **Universal (Zero Retraining)** |
| **SQL State Rollback** | ❌ No | ❌ No (Regex only) | ❌ No | **✅ Real `SAVEPOINT` Undo** |
| **Code Regression Guard** | ❌ No | ❌ No | ❌ No | **✅ Auto `pytest` Shadow Restore** |
| **Context Error Recovery** | ❌ Appends stack trace | ❌ Appends blocked string | ❌ No | **✅ Surgical Memory Rewind** |
| **Multi-Turn KV Eviction** | ❌ Unbounded bloat | ❌ No | ❌ No | **✅ 50%–65% Context Reduction** |
| **Latent Dynamic Coupling**| ❌ No | ❌ No | ❌ Static dataset | **✅ Layer L-2 Lyapunov Tracking** |
| **2-Line Drop-in SDK** | N/A | ⚠️ Complex schemas | ❌ Swap entire model | **✅ 2 Lines (`wrap_tools`)** |

---

## 🧪 Comprehensive Test Suite & Verification

SubNeutralize is backed by **90 automated unit and integration tests** passing 100% green:

```bash
# 1. Run the entire unit & integration test suite (90/90 passing):
python -m pytest tests/ -v

# 2. Run the 6-part core runtime E2E proof:
python live_agent_e2e_proof.py

# 3. Test on a fresh, realistic custom E-Commerce startup backend:
python try_your_own_app.py --non-interactive
```

```text
========================================================================================
  EXECUTIVE VERIFICATION SCORECARD (SUBNEUTRALIZE v0.3.0)
========================================================================================
  [PASS]  1. SQLite & PostgreSQL SAVEPOINT Rollback & Autonomous Recovery
  [PASS]  2. Automated pytest Regression Gate & Shadow File Byte Restore
  [PASS]  3. Subprocess Shell Snapshot & Non-Zero Exit Rollback
  [PASS]  4. Latent-to-Firewall Dynamic Policy Coupling (Φ_t & ρ_orbit)
  [PASS]  5. Physical KV-Cache Tensor Slicing on Rollback
  [PASS]  6. 1-Line @firewall.transactional_tool SDK & wrap_tools() Middleware
  [PASS]  7. Dynamic Schema Introspection (Custom orders/inventory/customers tables)
  [PASS]  8. Cryptographic SHA-256 Audit Ledger Verification
========================================================================================
  ALL VERIFICATION PROOFS VERIFIED 100% GREEN.
```

---

## 📄 Academic Citation & Research Paper

SubNeutralize is licensed under the [Apache-2.0 License](LICENSE).

If you use SubNeutralize in your research or production systems, please cite our technical paper:

```bibtex
@article{dutta2026subneutralize,
  title={SubNeutralize: Parameter-Free Latent Dynamical Inference Governor and Transactional Action Firewall for Reasoning Models},
  author={Dutta, Arnab},
  journal={arXiv preprint},
  year={2026},
  doi={10.5281/zenodo.22941619},
  url={https://doi.org/10.5281/zenodo.22941619}
}
```
