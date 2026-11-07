# SubNeutralize: Latent Dynamical Inference Governor for Reasoning Models

[![PyPI version](https://badge.fury.io/py/subneutralize.svg)](https://pypi.org/project/subneutralize/)
[![CI](https://github.com/Arnab758/subneutralize/actions/workflows/ci.yml/badge.svg)](https://github.com/Arnab758/subneutralize/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22941619.svg)](https://doi.org/10.5281/zenodo.22941619)
[![Paper PDF](https://img.shields.io/badge/Research%20Paper-PDF-red.svg)](paper/SubNeutralize_Paper.pdf)
[![Research Paper](https://img.shields.io/badge/Research%20Paper-Markdown-blue.svg)](paper/RESEARCH_PAPER.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

> **SubNeutralize** is a runtime inference governor for autoregressive reasoning models (DeepSeek-R1, Qwen-QwQ, and open reasoning LLMs). By tracking latent representation velocity and attractor dispersion in the mid-layer residual stream, it detects when algorithmic deduction is complete and transitions directly to answer emission with **100% KV-cache preservation**.

---

## 💡 The Core Problem: The Test-Time "Overthinking Trap"

Autoregressive reasoning models often solve the core algorithmic deduction within the first 80–150 tokens. However, without external calibration, they frequently enter extensive verification loops (*"Wait, let me double-check... but what if..."*), consuming hundreds of redundant tokens before generating the final answer.

SubNeutralize introduces a **non-destructive dynamical governor** that monitors the internal geometry of reasoning:
1. **Attractor Basin Detection:** Monitors directional velocity $v_t = 1 - \cos(h_t, h_{t-1})$ and windowed trajectory dispersion at the cognitive midpoint layer (~50% model depth).
2. **In-Flight KV Transition:** Upon dynamical consensus, it closes the reasoning phase (`</think>`) and primes the existing KV cache, continuing generation without costly re-tokenization.
3. **Task-Agnostic:** Operates seamlessly across code generation, mathematical reasoning, and general prompt answering.

---

## 🛠️ 3 Ways to Use SubNeutralize (Choose Your Architecture)

SubNeutralize can be deployed in three distinct modes depending on your workflow and stack:

### Option 1: As an OpenAI-Compatible AI Gateway (Cursor, VS Code, Cline, Aider)
Run SubNeutralize on your GPU server hosting an open-source model. It exposes a standard `/v1/chat/completions` endpoint with SSE streaming that intercepts reasoning loops before streaming back to your IDE:

```bash
# Launch directly via Python CLI:
subneutralize serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8000

# Or launch 1-click GPU container via Docker Compose:
docker compose up -d

# Open a side terminal for real-time telemetry:
subneutralize top

# Or launch Aider with automatic 1-click SubNeutralize routing:
subneutralize aider
```

#### IDE Configuration:
* **Cursor:** Set OpenAI API Base URL to `http://localhost:8000/v1` (API key: `subneutralize`).
* **VS Code (Continue / Cline):** Add model endpoint pointing to `http://localhost:8000/v1`.
* **Aider (1-Click Drop-in):** `subneutralize aider` (automatically detects the running gateway, routes traffic to `http://localhost:8000/v1`, and attaches optimal model flags).
* **Interactive Web Dashboard:** Open `http://localhost:8000/` in any browser to inspect real-time intervention counters and test prompts.

---

### Option 2: As a Python SDK Wrapper (for Backends & AI Agents)
Wrap any open-weights reasoning model in 2 lines of Python for backend microservices, batch processing, or agentic frameworks (LangChain, DSPy, CrewAI):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import SubNeutralize

# 1. Load any open-weight reasoning model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Or 7B / 14B / 32B
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Wrap with SubNeutralize (automatically resolves cognitive bottleneck layer)
governor = SubNeutralize(model, tokenizer)

# 3. Generate with real-time dynamical equilibrium tracking
output = governor.generate("Write a thread-safe token bucket rate limiter in Python:")

print("--- Final Answer ---")
print(output.clean_code if output.clean_code else output.answer)
print(f"\nThinking Tokens: {output.thinking_tokens} | Answer Tokens: {output.answer_tokens}")
print(f"Consensus Reached: {output.consensus_reached} at token {output.consensus_step}")
```

---

### Option 3: As a Drop-in Hugging Face `StoppingCriteria`
If you already have existing `model.generate()` pipelines with custom temperature, top-p, or speculative decoding, drop in the governor without altering your architecture:

```python
governor = SubNeutralize(model, tokenizer)

outputs = model.generate(
    **inputs,
    max_new_tokens=1500,
    stopping_criteria=[governor.as_stopping_criteria()]
)
```

---

### ⚡ Matched-Budget Side-by-Side Shootout

Run a fair, side-by-side comparison of **Vanilla Unconstrained Reasoning** vs. **SubNeutralize Governed Reasoning** directly in your terminal:

```bash
# Fair matched-budget comparison vs. unconstrained baseline
subneutralize --compare "Write a thread-safe token bucket rate limiter with microsecond refills in Python"

# Single prompt execution with live telemetry
subneutralize "Explain why the square root of 2 is irrational"

# Interactive REPL session
subneutralize -i

# Real-time terminal telemetry monitor (live KV savings & Lyapunov potential)
subneutralize top

# Or launch Aider with automatic 1-click SubNeutralize routing:
subneutralize aider
```

---

## 🔬 Mathematical Formulation

### 1. Directional Cosine Velocity
At decoding step $t$, the directional displacement of the hidden state $h_t$ at target layer $L_{\text{mid}}$ is computed via cosine distance:
$$v_t = 1 - \frac{h_t \cdot h_{t-1}}{\|h_t\|_2 \|h_{t-1}\|_2}$$

### 2. Dimension-Invariant Momentum Ratio
Because latent norms scale across model parameter dimensions (e.g. $d=1536$ vs $d=5120$), instantaneous velocity is normalized against an Exponential Moving Average (EMA):
$$\bar{v}_t = \alpha v_t + (1 - \alpha) \bar{v}_{t-1}, \quad \alpha = 0.12$$
$$R_t = \frac{v_t}{\max(\bar{v}_t, 10^{-6})}$$

### 3. Windowed Attractor Dispersion
Over a sliding window $W = \{h_{t-K+1}, \dots, h_t\}$ with centroid $\bar{h}_W = \frac{1}{K}\sum_{i} h_i$, the geometric dispersion is:
$$\rho_t = \frac{1}{K} \sum_{i=1}^{K} \left(1 - \frac{h_{t-i+1} \cdot \bar{h}_W}{\|h_{t-i+1}\|_2 \|\bar{h}_W\|_2}\right)$$

### 4. Dynamical Consensus Criterion
A consensus state is certified when:
1. **Warmup Satisfied:** $t \ge T_{\text{warmup}}$ (allowing foundational reasoning deduction).
2. **Trajectory Convergence:** $(v_t \le v_{\text{ceiling}} \land R_t \le R_{\text{threshold}}) \lor (\rho_t \le \rho_{\text{ceiling}} \land v_t \le 1.5 v_{\text{ceiling}})$.
3. **Entropy Safety:** Shannon entropy $H(p_t) \le H_{\text{threshold}}$, ensuring the model is in a high-confidence prediction state.
4. **Persistence Debounce:** The condition remains satisfied across $N_{\text{debounce}}$ consecutive tokens (default 3 tokens).

---

## 📊 Verification & Benchmarks

Run the benchmark suite locally to evaluate unit test correctness and token savings:

```bash
# Verify reference test suites
python -m subneutralize.benchmark

# Run live model evaluation against matched baseline
python -m subneutralize.benchmark --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### 🚀 v0.2.0 Empirical Benchmarks (NVIDIA A100 Validation)

SubNeutralize v0.2.0 introduces **Layer $L-2$ Thin-SVD Prompt Subspace Neutralization**, **Lyapunov Dissipation Potential ($\Phi_t$)**, and **Stateful Multi-Turn KV Scratchpad Eviction**.

| Benchmark / Task | Metric | Vanilla Baseline | SubNeutralize v0.2.0 | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Agent Coding**<br>(3-Agent Collaborative Workflow) | **Unit Test Pass Rate** | **0 / 5 (0%)** | **5 / 5 (100%)** | **+100% Success** (eliminated context rot) |
| | **Total Workflow Tokens** | 8,269 tokens | 4,024 tokens | **-51.3% Token Waste** |
| | **Agent 3 Prefill Memory** | 3,377 tokens | 1,210 tokens | **-64.2% Prefill Memory** |
| **SWE-bench Lite**<br>(5 Real-World Repositories) | **Resolved Issues** | 2 / 5 (40%) | **3 / 5 (60%)** | **+50% Relative Accuracy** |
| | **Total Consumed Tokens** | 14,194 tokens | 8,610 tokens | **-39.3% Net Compute** |
| | **Prefill Context Retention** | Full Raw History | Compact Canonical | **-61.2% Context Bloat** |
| **5-Turn Agent Coding Loop**<br>(Stratified Tiers A, B, C) | **Tier C Wall-Clock Latency** | 571.85s (9.53m) | **462.68s (7.71m)** | **-19.1% Faster (-109.17s)** |
| | **Tier C Prefill Ratio** | 4.16× (3,192 tok) | **3.42× (2,941 tok)** | **-323 tok / turn bloat** |
| | **Tier B Self-Correction** | 0 / 12 (Answer Flip) | **7 / 12 (+6 passes)** | **Monotonic Learning** |
| | **Syntactic Code Integrity** | 100% | **100%** | **0% AST Mutilation (vs 0% Scissors)** |

To reproduce these benchmarks:
```bash
# Multi-Agent Collaborative Coding Workflow Benchmark
python experiments/benchmark_multiagent_coding.py

# 5-Repo SWE-bench Lite Real-World Benchmark
python experiments/benchmark_swebench_lite.py
```

### Protocol:
* **Sampling Parity:** Both Vanilla and SubNeutralize are evaluated with identical chat templates, `temperature=0.6`, `top_p=0.95`, and equal generation ceilings.
* **Non-Scissoring:** SubNeutralize transitions naturally to code synthesis upon dynamical consensus rather than imposing an artificial hard cutoff.

---

## 🌐 High-Throughput Serving (Native vLLM PagedAttention Integration)

SubNeutralize natively bridges latent geometric steering and high-throughput continuous-batching engines:

### 1. Native vLLM LogitsProcessor & SamplingParams
Wrap vLLM requests with in-flight Shannon entropy and syntactic delimiter consensus tracking:

```python
from vllm import LLM, SamplingParams
from subneutralize import create_vllm_sampling_params

llm = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct", tensor_parallel_size=1)
params = create_vllm_sampling_params(
    tokenizer=llm.get_tokenizer(),
    temperature=0.6,
    top_p=0.95,
    entropy_threshold=1.85,
)
sampling_params = SamplingParams(**params)
outputs = llm.generate(["Implement a lock-free concurrent queue in Python:"], sampling_params)
```

### 2. Multi-Turn PagedAttention Middleware (`vllm serve`)
Attach `VLLMMultiTurnMiddleware` to your `vllm serve` API gateway to strip transient historical `<think>` scratchpads, slashing PagedAttention KV-cache occupancy by **35%–55%** per multi-turn coding iteration:

```python
from vllm.entrypoints.openai.api_server import app
from subneutralize import VLLMMultiTurnMiddleware

app.add_middleware(VLLMMultiTurnMiddleware)
```

---

## 📜 Citation

```bibtex
@article{dutta2026subneutralize,
  title={SubNeutralize: The Geometry of Reasoning and the Elimination of the Overthinking Trap},
  author={Dutta, Arnab},
  year={2026},
  doi={10.5281/zenodo.22941619},
  url={https://doi.org/10.5281/zenodo.22941619}
}
```

## 📄 License
Apache-2.0 License. Free for research and commercial use.
