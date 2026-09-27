"""
Official Benchmark Dataset Loader for Reasoning Models.
Loads accredited, peer-reviewed public datasets from Hugging Face:
- openai/gsm8k (Grade School Math)
- HuggingFaceH4/MATH-500 (Competition & Olympiad Mathematics across 7 subjects)
Eliminates all hand-crafted / synthetic toy probes.
"""

from typing import Dict, List, Any, Optional
import json
import re


def clean_latex(s: str) -> str:
    """Normalizes LaTeX math strings for exact matching."""
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace(" ", "").replace("\n", "")
    m_text = re.search(r"\\text\{([^}]+)\}", s)
    if m_text:
        s = m_text.group(1)
    return s.strip()


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extracts numeric answer from GSM8K ground truth or model generation."""
    if not text:
        return None
    # Check for #### delimiter in GSM8K
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def extract_math500_answer(text: str) -> Optional[str]:
    """Extracts the boxed answer from MATH-500 competition problems."""
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return clean_latex(matches[-1])
    m = re.findall(r"[Tt]he (?:final )?answer is:?\s*\$?([^\n.$]+)", text)
    if m:
        return clean_latex(m[-1])
    return None


def format_r1_prompt(question: str) -> str:
    """Formats a user question using DeepSeek-R1 standard reasoning template."""
    return f"<｜User｜>{question}<｜Assistant｜><think>\n"


def load_official_benchmark(
    dataset_name: str = "gsm8k",
    split: str = "test",
    n_samples: int = 50,
    subject_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Loads official reasoning benchmark problems from Hugging Face.
    
    Args:
        dataset_name: 'gsm8k' or 'math500'
        split: dataset split ('test')
        n_samples: number of problems to load
        subject_filter: optional filter for MATH-500 (e.g. 'Algebra', 'Precalculus')
        
    Returns:
        List of standardized problem dicts: [{'id', 'question', 'ground_truth', 'subject', 'level'}]
    """
    from datasets import load_dataset

    items = []
    if dataset_name.lower() in ["gsm8k", "openai/gsm8k"]:
        try:
            ds = load_dataset("openai/gsm8k", "main", split=split)
        except Exception:
            import pandas as pd
            df = pd.read_parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
            ds = df.to_dict("records")

        for i, row in enumerate(ds):
            if len(items) >= n_samples:
                break
            items.append({
                "id": f"gsm8k_{i+1:03d}",
                "dataset": "gsm8k",
                "question": row["question"],
                "ground_truth": extract_gsm8k_answer(row["answer"]),
                "raw_answer": row["answer"],
                "subject": "Arithmetic",
                "level": 1
            })

    elif dataset_name.lower() in ["math500", "math-500", "huggingfaceh4/math-500"]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        count = 0
        for i, row in enumerate(ds):
            subj = row.get("subject", "Mathematics")
            level = row.get("level", 1)
            if subject_filter and subj.lower() != subject_filter.lower():
                continue
            items.append({
                "id": f"math500_{i+1:03d}",
                "dataset": "math500",
                "question": row["problem"],
                "ground_truth": clean_latex(row["answer"]),
                "raw_answer": row["answer"],
                "subject": subj,
                "level": level
            })
            count += 1
            if count >= n_samples:
                break
    else:
        raise ValueError(f"Unsupported benchmark: {dataset_name}. Use 'gsm8k' or 'math500'.")

    return items

