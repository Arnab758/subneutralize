"""
Command-line interface for SubNeutralize.
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subneutralize.governor import ConsensusEntropyGovernor


def main():
    parser = argparse.ArgumentParser(
        description="SubNeutralize: Runtime Inference Governor for Reasoning Models"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Input reasoning prompt"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=2048,
        help="Maximum generation budget"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )

    args = parser.parse_args()

    print(f"[*] Loading model {args.model} onto {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device
    )

    governor = ConsensusEntropyGovernor(model, tokenizer)
    governor.attach()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print("[*] Generating with SubNeutralize active...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            stopping_criteria=[governor.as_stopping_criteria()],
            return_dict_in_generate=True,
            output_scores=True
        )

    governor.detach()
    tokens = outputs.sequences[0].shape[-1] - inputs.input_ids.shape[-1]
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    print(f"[✓] Generation completed in {tokens} tokens.")
    print("\n--- Output ---")
    print(decoded)


if __name__ == "__main__":
    main()
