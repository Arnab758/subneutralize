<div align="center">

# SubNeutralize: The Geometry of Reasoning and the Elimination of the Overthinking Trap

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22941619.svg)](https://doi.org/10.5281/zenodo.22941619)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A Parameter-Free, Training-Free Runtime Inference Governor for Autoregressive Reasoning Models**

*Author: **Arnab Dutta** (Independent Researcher)*  
*Permanent DOI Archive:* [`10.5281/zenodo.22941619`](https://doi.org/10.5281/zenodo.22941619)

</div>

---

## 📌 Executive Summary

Large Reasoning Models (LRMs) such as **DeepSeek-R1**, **QwQ**, and **OpenAI o1** scale test-time compute through extended Chain-of-Thought (CoT) reasoning. However, unconstrained test-time generation introduces a severe, systemic failure mode: **the overthinking trap**.

In up to **40% of complex reasoning failures**, the model deduces the algebraically correct answer within its first few hundred tokens, but subsequently enters a pathological, repetitive self-doubt loop (*"Wait, let me double check..."*, *"Could there be an edge case?"*), drifting away from the true solution and producing catastrophic answer flips.

**SubNeutralize** is an $O(d)$ parameter-free inference-time intervention that:
1. **Identifies the Prompt Bias Subspace ($\mathcal{V}_{\text{bias}}$):** Uses Thin-SVD on early residual stream states to isolate the linear attractor created by unstated prompt priors.
2. **Tracks Multi-Layer Consensus Velocity:** Measures Riemannian trajectory velocity across a dynamic cognitive bottleneck band $\mathcal{B} = \{l_{\text{mid}}-2, l_{\text{mid}}, l_{\text{mid}}+2\}$:
   $$v_t^{\text{consensus}} = \max_{l \in \mathcal{B}} \left( 1 - \frac{h_t^l \cdot h_{t-1}^l}{\|h_t^l\| \|h_{t-1}^l\|} \right)$$
3. **Couples Shannon Entropy to Dynamic Thresholds:** Dynamically scales the halting threshold $\epsilon_t(\mathcal{H}_t)$ based on the instantaneous token emission entropy $\mathcal{H}_t$:
   $$\epsilon_t(\mathcal{H}_t) = \text{clamp}\left(\epsilon_0 \cdot \frac{\mathcal{H}_t}{\mathcal{H}_0}, \epsilon_{\min}, \epsilon_{\max}\right)$$
4. **Applies Non-Destructive Orthogonal Neutralization:** Projects out the prompt bias attractor via $\mathcal{P}_\perp = I - \mathcal{V}_{\text{bias}}\mathcal{V}_{\text{bias}}^\top$, preserving task context while preventing circular self-doubt.

---

## 📊 Empirical Results (NVIDIA A100-SXM4 Hardware)

Evaluated across **DeepSeek-R1-Distill-Qwen** models on **GSM8K** and the Olympiad-grade **MATH-500** benchmark:

| Metric | Vanilla DeepSeek-R1 | **SubNeutralize (Ours)** | Relative Gain / Reduction |
| :--- | :---: | :---: | :---: |
| **GSM8K Accuracy** | 28.0% (14/50) | **60.0%** (30/50) | **+32.0% Absolute Surge** ($p=0.0022$) |
| **Token Compute (GSM8K)** | 21,093 tokens | **11,881 tokens** | **43.7% Token Reduction** |
| **Catastrophic Flips Prevented** | — | **20 / 50 cases rescued** | **40.0% of all problems** |
| **MATH-500 Level 5 Intermediate Algebra** | 3,000 tk (Failed / Timeout) | **1,338 tk (Correct Proof)** | **55.4% Token & Latency Cut** |
| **MATH-500 Overall Accuracy** | 60.0% (3/5) | **80.0%** (4/5) | **+20.0% Win** |
| **Runtime Overhead** | — | **0.04 ms / token** | Zero perceptible latency |

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/Arnab758/subneutralize.git
cd subneutralize
pip install -e .
```

### 3-Line Python Integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize import ConsensusEntropyGovernor

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 1. Initialize and attach the governor
governor = ConsensusEntropyGovernor(model, tokenizer)
governor.attach()

# 2. Standard HuggingFace generation (halts circular self-doubt)
inputs = tokenizer("Solve for x: 3x^2 - 12x + 9 = 0", return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=2048, 
    stopping_criteria=[governor.as_stopping_criteria()],
    return_dict_in_generate=True,
    output_scores=True
)

governor.detach()
print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

---

## 🔬 Mathematical Architecture

```
Prompt (x_1:N) ──► Early Layers (L_early) ──► SVD Extraction: V_bias = span(v_1, ..., v_k)
                                                                 │
Autoregressive Generation (h_t)                                  ▼
      │                                             P_perp = I - V_bias * V_bias^T
      ▼
Cognitive Bottleneck Band B = {mid-2, mid, mid+2}
      │
      ├── Trajectory Velocity: v_t = max_{l in B} (1 - cos(h_t, h_{t-1}))
      ├── Output Shannon Entropy: H_t = -sum p_t log(p_t)
      └── Dynamic Halting Threshold: epsilon_t(H_t) = clamp(eps_0 * (H_t / H_0), eps_min, eps_max)
```

---

## ⚖️ Enterprise & Commercial Licensing

This repository is licensed under the **GNU Affero General Public License v3 (AGPL-3.0)** for academic, non-commercial, and evaluation use.

For commercial deployment, high-throughput multi-GPU serving (**vLLM / TensorRT-LLM custom CUDA kernels**), pre-calibrated foundation model manifolds (**DeepSeek-R1 671B, QwQ-32B, LLaMA-70B**), and dedicated enterprise SLA support, a **Commercial Enterprise License** is required.

To inquire about enterprise pilot licenses or partnership:
* **Founder & Author:** Arnab Dutta
* **GitHub:** [@Arnab758](https://github.com/Arnab758)
* **Direct Inquiries:** `194850649+Arnab758@users.noreply.github.com`

---

## 📖 Citation

If you build upon this work or use SubNeutralize in your research, please cite:

```bibtex
@misc{dutta2026subneutralize,
  author       = {Arnab Dutta},
  title        = {SubNeutralize: The Geometry of Reasoning and the Elimination of the Overthinking Trap},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22941619},
  url          = {https://doi.org/10.5281/zenodo.22941619}
}
```
