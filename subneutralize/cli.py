"""
Command-Line Interface (CLI) for SubNeutralize.
Provides interactive governed generation, side-by-side comparison, and benchmarking.
"""

import sys
import time
import argparse
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def print_banner():
    banner = """
================================================================================
  SubNeutralize: Scale-Free Runtime Inference Governor for Reasoning Models
  Eliminating the Overthinking Crisis | 77.9% Compute Reduction
================================================================================
"""
    print(banner.strip() + "\n")


def run_prompt_comparison(model_id: str, prompt: str, device: str, max_tokens: int = 1024):
    """Runs a live side-by-side comparison of Vanilla reasoning vs SubNeutralize."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import SubNeutralize

    print(f"[*] Loading model: {model_id} on {device}...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    governor = SubNeutralize(model, tokenizer)

    print(f"\n[PROMPT]: {prompt}\n")
    print("-" * 80)
    print("[1/2] Running Vanilla Baseline (Unconstrained Reasoning)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    t0 = time.time()
    with torch.no_grad():
        out_v = model.generate(**inputs, max_new_tokens=max_tokens)
    t_vanilla = time.time() - t0
    v_tokens = out_v.shape[-1] - prompt_len
    print(f"  -> Vanilla generated {v_tokens} tokens in {t_vanilla:.2f}s")

    print("\n[2/2] Running SubNeutralize (Dynamical Consensus Governor)...")
    t0 = time.time()
    out_gov = governor.generate(prompt, max_new_tokens=max_tokens)
    t_gov = time.time() - t0

    savings = (1.0 - out_gov.total_tokens / max(1, v_tokens)) * 100.0
    speedup = t_vanilla / max(1e-4, t_gov)

    print("-" * 80)
    print("                      HEAD-TO-HEAD COMPARISON RESULTS                           ")
    print("-" * 80)
    print(f"  {'Metric':<25} | {'Vanilla Baseline':<18} | {'SubNeutralize':<18} | {'Improvement':<15}")
    print("-" * 80)
    print(f"  {'Tokens Consumed':<25} | {v_tokens:<18} | {out_gov.total_tokens:<18} | {savings:>6.1f}% Saved")
    print(f"  {'Inference Latency':<25} | {t_vanilla:<16.2f}s | {t_gov:<16.2f}s | {speedup:>6.2f}x Faster")
    print(f"  {'Consensus Detected':<25} | {'N/A (Cycled)':<18} | {f'Token {out_gov.consensus_step}':<18} | {'Early Halt':<15}")
    print("-" * 80)

    print("\n--- [Clean Extracted Code / Solution] ---")
    print(out_gov.clean_code)
    print("-" * 80 + "\n")


def run_governed_prompt(model_id: str, prompt: str, device: str, max_tokens: int = 1024):
    """Runs single governed prompt and prints clean code with telemetry."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import SubNeutralize

    print(f"[*] Loading model: {model_id} on {device}...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    governor = SubNeutralize(model, tokenizer)

    print(f"\n[PROMPT]: {prompt}\n")
    print("[*] Generating with dynamical consensus equilibrium tracking...")

    out = governor.generate(prompt, max_new_tokens=max_tokens)

    print("\n" + "=" * 80)
    print("                     SUBNEUTRALIZE EXECUTION SUMMARY                     ")
    print("=" * 80)
    print(f"  * Thinking Tokens:      {out.thinking_tokens}")
    print(f"  * Code/Answer Tokens:   {out.code_tokens}")
    print(f"  * Total Tokens:         {out.total_tokens}")
    print(f"  * Consensus Reached:    {out.consensus_reached} (at token {out.consensus_step})")
    print(f"  * Latency:              {out.wall_clock_seconds:.2f}s")
    print("=" * 80)
    print("\n--- Output ---\n")
    print(out.clean_code)
    print("\n" + "=" * 80 + "\n")


def run_interactive_repl(model_id: str, device: str):
    """Interactive terminal shell for prompt-by-prompt experimentation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import SubNeutralize

    print_banner()
    print(f"[*] Initializing SubNeutralize REPL with {model_id} on {device}...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    governor = SubNeutralize(model, tokenizer)

    print("\nReady! Enter your prompt below (or type 'exit' or 'quit' to stop).\n")
    while True:
        try:
            prompt = input("SubNeutralize> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Exiting SubNeutralize. Goodbye!")
                break

            print("\nThinking with real-time dynamical governor...")
            out = governor.generate(prompt)

            print(f"\n[Consensus at token {out.consensus_step} | Total: {out.total_tokens}t | {out.wall_clock_seconds:.2f}s]")
            print("-" * 60)
            print(out.clean_code)
            print("-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    default_dev = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    parser = argparse.ArgumentParser(
        prog="subneutralize",
        description="SubNeutralize: Scale-Free Runtime Inference Governor for Reasoning Models"
    )
    parser.add_argument("command_or_prompt", nargs="?", default=None, help="'serve' to launch OpenAI gateway for Cursor/VS Code, or prompt to govern")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model ID (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    parser.add_argument("--device", type=str, default=default_dev, help=f"Execution device (default: {default_dev})")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gateway host for serve mode (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Gateway port for serve mode (default: 8000)")
    parser.add_argument("--serve", action="store_true", help="Launch OpenAI-compatible local proxy for Cursor / IDEs")
    parser.add_argument("--compare", action="store_true", help="Run head-to-head comparison vs. unconstrained Vanilla baseline")
    parser.add_argument("--benchmark", action="store_true", help="Run the official enterprise 4-problem test suite")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive REPL session")

    args = parser.parse_args()

    if args.benchmark:
        from .benchmark import main as run_benchmark
        run_benchmark()
        return

    if args.serve or (args.command_or_prompt and args.command_or_prompt.lower() == "serve"):
        from .server import start_server
        start_server(model_id=args.model, host=args.host, port=args.port, device=args.device)
        return

    if not _HAS_TORCH:
        print("[ERROR] PyTorch is required to run inference. Please install torch: pip install torch")
        sys.exit(1)

    prompt = args.command_or_prompt
    if args.interactive or (prompt is None and not args.benchmark):
        run_interactive_repl(args.model, args.device)
    elif args.compare:
        print_banner()
        run_prompt_comparison(args.model, prompt, args.device)
    else:
        print_banner()
        run_governed_prompt(args.model, prompt, args.device)


if __name__ == "__main__":
    main()
