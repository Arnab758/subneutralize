# SubNeutralize: Latent Dynamical Inference Governor for Reasoning Models

[![PyPI version](https://badge.fury.io/py/subneutralize.svg)](https://pypi.org/project/subneutralize/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22941619.svg)](https://doi.org/10.5281/zenodo.22941619)
[![Paper PDF](https://img.shields.io/badge/Research%20Paper-PDF-red.svg)](paper/SubNeutralize_Paper.pdf)
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

## 🚀 Quickstart

### 1. Installation

```bash
pip install subneutralize
```

For IDE gateway support (Cursor / VS Code Continue):
```bash
pip install "subneutralize[serve]"
```

---

### 2. Python SDK: 2-Line Integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import SubNeutralize

# 1. Load any open-weight reasoning model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # Or 7B / 14B / 32B
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

### 3. Drop-in Hugging Face `StoppingCriteria`

Integrates directly with native `model.generate()` pipelines:

```python
governor = SubNeutralize(model, tokenizer)

outputs = model.generate(
    **inputs,
    max_new_tokens=1500,
    stopping_criteria=[governor.as_stopping_criteria()]
)
```

---

### 4. Interactive Terminal & Side-by-Side Comparison

```bash
# Single prompt execution with telemetry
subneutralize "Explain why the square root of 2 is irrational"

# Fair matched-budget comparison vs. unconstrained baseline
subneutralize --compare "Write a thread-safe token bucket rate limiter in Python"

# Interactive REPL
subneutralize -i
```

---

## 💻 IDE Gateway: Connect to Cursor, VS Code, or Antigravity

SubNeutralize includes an **OpenAI-compatible inference gateway** with Server-Sent Events (SSE) streaming, allowing you to use governed reasoning directly inside Cursor, VS Code, and other developer tools:

```bash
subneutralize serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8000
```

### IDE Configuration:
* **Cursor:** Set OpenAI API Base URL to `http://localhost:8000/v1` (API key: `subneutralize`).
* **VS Code (Continue / Cline):** Add model endpoint pointing to `http://localhost:8000/v1`.
* **Aider:** `aider --openai-api-base http://localhost:8000/v1 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

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

### Protocol:
* **Sampling Parity:** Both Vanilla and SubNeutralize are evaluated with identical chat templates, `temperature=0.6`, `top_p=0.95`, and equal generation ceilings.
* **Non-Scissoring:** SubNeutralize transitions naturally to code synthesis upon dynamical consensus rather than imposing an artificial hard cutoff.

---

## 🌐 High-Throughput Serving (vLLM / SGLang Roadmap)

SubNeutralize is designed to bridge mechanistic interpretability and high-throughput inference engines:
* **Hugging Face / PyTorch:** Fully supported via `SubNeutralize` wrapper and `as_stopping_criteria()`.
* **vLLM / SGLang:** Mid-layer hidden states can be accessed via custom worker hooks or logits processors. Upstream RFCs and custom runner integrations are under active development.

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
