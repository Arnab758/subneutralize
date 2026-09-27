"""
Publication Plotting Engine for SubNeutralize
Generates camera-ready 300-DPI vector plots (PDF + PNG) from benchmark_7b_gsm8k_50.json
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Apply clean, modern scientific aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14

JSON_PATH = r"C:\Users\ARNAB DUTTA\.gemini\antigravity-ide\scratch\latent-reasoning-dynamics\paper\tables\benchmark_7b_gsm8k_50.json"
OUTPUT_DIR = r"C:\Users\ARNAB DUTTA\.gemini\antigravity-ide\scratch\latent-reasoning-dynamics\paper\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

results = data["problem_results"]
agg = data["aggregate_metrics"]

van_tokens = [r["vanilla_tokens"] for r in results]
neut_tokens = [r["neut_tokens"] for r in results]
savings = [r["savings_pct"] for r in results]
indices = np.arange(1, len(results) + 1)

# ==============================================================================
# FIGURE 1: ACCURACY & CUMULATIVE COMPUTE PARETO FRONTIER
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# (A) Cumulative Tokens
cum_van = np.cumsum(van_tokens)
cum_neut = np.cumsum(neut_tokens)

ax1.plot(indices, cum_van, color="#e63946", lw=2.5, label="Vanilla DeepSeek-R1-7B (21,093 tokens)")
ax1.plot(indices, cum_neut, color="#1d3557", lw=2.5, label="SubNeutralize (Ours) (11,881 tokens)")
ax1.fill_between(indices, cum_neut, cum_van, color="#a8dadc", alpha=0.4, label="Compute Saved: 43.7% (9,212 tokens)")
ax1.set_xlabel("GSM8K Problem Index (1–50)")
ax1.set_ylabel("Cumulative Inference Tokens")
ax1.set_title("A. Cumulative Test-Time Compute on NVIDIA A100", fontweight="bold")
ax1.legend(loc="upper left", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.5)

# (B) Accuracy & Flips Bar
categories = ["Vanilla DeepSeek-R1-7B", "SubNeutralize (Ours)"]
accuracies = [agg["vanilla_accuracy_pct"], agg["neutralized_accuracy_pct"]]
colors = ["#e63946", "#2a9d8f"]

bars = ax2.bar(categories, accuracies, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 80)
ax2.set_title("B. Accuracy Gain via Overthinking Suppression", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

# Annotations on bars
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

ax2.annotate(
    "+32.0% Accuracy Gain\n(p = 0.0022 via McNemar)",
    xy=(0.5, 45), xycoords="data",
    xytext=(0.5, 68), textcoords="data",
    ha="center", fontsize=11, fontweight="bold", color="#1d3557",
    bbox=dict(boxstyle="round,pad=0.5", fc="#f1faee", ec="#1d3557", lw=1.5),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#1d3557", lw=1.5)
)

plt.tight_layout()
fig1_path_png = os.path.join(OUTPUT_DIR, "figure1_pareto_accuracy.png")
fig1_path_pdf = os.path.join(OUTPUT_DIR, "figure1_pareto_accuracy.pdf")
plt.savefig(fig1_path_png, dpi=300)
plt.savefig(fig1_path_pdf)
plt.close()
print(f"Saved: {fig1_path_png}")

# ==============================================================================
# FIGURE 2: OUTCOMES & MC-NEMAR CONTINGENCY BREAKDOWN
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# (A) Outcome Counts
outcomes = ["NEUT_WON\n(Flips Prevented)", "EQUAL\n(Both Correct)", "VAN_WON\n(Regression)", "BOTH_FAIL\n(Hard Problems)"]
counts = [agg["flips_prevented_neut_won"], agg["both_equal"], agg["vanilla_won"], agg["both_fail"]]
bar_colors = ["#2a9d8f", "#457b9d", "#e76f51", "#6c757d"]

bars = ax1.bar(outcomes, counts, color=bar_colors, edgecolor="black", linewidth=1.2)
ax1.set_ylabel("Number of Problems (N=50)")
ax1.set_title("A. Head-to-Head Outcome Breakdown", fontweight="bold")
ax1.set_ylim(0, 25)
ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2.0, count + 0.6, f"{count} ({count/50*100:.0f}%)", ha='center', va='bottom', fontweight='bold')

# (B) Per-Problem Token Savings Distribution
ax2.hist(savings, bins=12, range=(0, 80), color="#1d3557", edgecolor="white", alpha=0.85)
ax2.axvline(agg["net_compute_reduction_pct"], color="#e63946", linestyle="dashed", linewidth=2.5, label=f"Net Mean Savings: {agg['net_compute_reduction_pct']:.1f}%")
ax2.set_xlabel("Compute Saved per Problem (%)")
ax2.set_ylabel("Frequency")
ax2.set_title("B. Distribution of Compute Reduction Across GSM8K", fontweight="bold")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig2_path_png = os.path.join(OUTPUT_DIR, "figure2_outcomes_distribution.png")
fig2_path_pdf = os.path.join(OUTPUT_DIR, "figure2_outcomes_distribution.pdf")
plt.savefig(fig2_path_png, dpi=300)
plt.savefig(fig2_path_pdf)
plt.close()
print(f"Saved: {fig2_path_png}")

print("ALL PUBLICATION-GRADE PLOTS SUCCESSFULLY COMPILED!")
