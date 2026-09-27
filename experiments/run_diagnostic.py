"""
Phase 1 Diagnostic Probe:
Runs a reasoning model on curated benchmark questions, extracts residual stream
hidden states, fits the First Impression prompt subspace V_bias, and plots
the kinematic trajectory (velocity, curvature, bias re-activation).
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.hooks import ResidualStreamProbe
from src.bias_subspace import BiasSubspaceAnalyzer
from src.benchmark_loader import load_diagnostic_probes, format_r1_prompt


def run_diagnostic_experiment(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    output_dir: str = "paper/figures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 400
):
    print(f"=== Starting Phase 1 Diagnostic Experiment ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "paper", "tables"), exist_ok=True)

    # 1. Load Tokenizer and Model
    print("Loading model and tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with fp16 or bf16 if GPU, float32 if CPU
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    num_layers = len(model.model.layers)
    target_layer = num_layers // 2  # Monitor middle-to-late layer (e.g. layer 14 of 28)
    print(f"Probing decoder layer: {target_layer} of {num_layers}")

    # 2. Select Probe Problem
    probes = load_diagnostic_probes()
    test_case = probes[0]  # The classic Bat and Ball cognitive bias problem
    print(f"\nQuestion: {test_case['question']}")
    print(f"Documented Intuitive Bias: {test_case['intuitive_bias']}")
    print(f"Ground Truth: {test_case['ground_truth']}")

    formatted_prompt = format_r1_prompt(test_case["question"])
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    # 3. Attach Probe Hook
    probe = ResidualStreamProbe(model, target_layers=[target_layer])
    probe.attach()

    # 4. Generate Output with Hidden State Tracking
    print("\nGenerating reasoning trajectory...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    probe.detach()
    full_output_ids = outputs[0].tolist()
    reasoning_ids = full_output_ids[prompt_len:]
    full_text = tokenizer.decode(full_output_ids, skip_special_tokens=False)
    reasoning_text = tokenizer.decode(reasoning_ids, skip_special_tokens=True)

    print("\n--- Generated Reasoning Excerpt ---")
    print(reasoning_text[:600] + "...\n")

    # 5. Extract Trajectory and Fit Bias Subspace
    trajectory = probe.get_layer_trajectory(target_layer)[0]  # [total_tokens, hidden_dim]
    prompt_hidden = trajectory[:prompt_len]
    reasoning_hidden = trajectory[prompt_len:]

    # Prompt terminal token represents initial bias state
    v_prompt_terminal = prompt_hidden[-1]

    analyzer = BiasSubspaceAnalyzer(subspace_dim=1)
    analyzer.set_bias_vector(v_prompt_terminal)
    metrics = analyzer.compute_trajectory_metrics(reasoning_hidden)

    # 6. Plot Diagnostic Figures
    print("Generating publication-grade diagnostic plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    token_steps = np.arange(len(metrics["velocity"]))

    # Subplot 1: Trajectory Velocity (Identifies solution stabilization)
    ax1.plot(token_steps, metrics["velocity"], color="#1f77b4", lw=2, label="Cosine Velocity ($v_t$)")
    ax1.axhline(y=0.05, color="gray", linestyle="--", alpha=0.7, label="Candidate Stabilization Threshold ($\epsilon=0.05$)")
    ax1.set_ylabel("Trajectory Velocity ($1 - \cos$)", fontsize=12)
    ax1.set_title(f"Residual Stream Dynamics: {test_case['id']} (Layer {target_layer})", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Subplot 2: Bias Subspace Alignment (Measures prompt bias reactivation)
    ax2.plot(token_steps + 1, metrics["bias_alignment"][1:], color="#d62728", lw=2, label="Prompt Bias Alignment ($\|h_t V_{\\text{bias}}\|^2$)")
    ax2.set_xlabel("Reasoning Token Steps ($t$)", fontsize=12)
    ax2.set_ylabel("Bias Projection Energy", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(PROJECT_ROOT, output_dir, "diagnostic_trajectory.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic vector plot to: {plot_path}")

    # 7. Save JSON Metrics Summary
    results = {
        "probe_id": test_case["id"],
        "prompt_tokens": prompt_len,
        "reasoning_tokens": len(reasoning_ids),
        "mean_velocity": float(np.mean(metrics["velocity"])),
        "max_bias_alignment": float(np.max(metrics["bias_alignment"])),
        "min_bias_alignment": float(np.min(metrics["bias_alignment"])),
        "figure_path": plot_path
    }
    json_path = os.path.join(PROJECT_ROOT, "paper", "tables", "diagnostic_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics summary to: {json_path}")
    print("=== Phase 1 Diagnostic Run Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--tokens", type=int, default=400)
    args = parser.parse_args()

    run_diagnostic_experiment(model_name=args.model, max_new_tokens=args.tokens)
