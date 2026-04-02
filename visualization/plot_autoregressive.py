"""
Autoregressive Experiment Figure
=================================
Bar chart: ρ(prefill, auto) per model × bitwidth.

Usage:
  python -m visualization.plot_autoregressive --results-dir results/ddt --output-dir figures/ddt
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_SHORT = {
    "Llama-3.1-8B": "Llama-8B",
    "Qwen2.5-7B": "Qwen-7B",
    "Mistral-7B-v0.3": "Mistral-7B",
    "Qwen2.5-32B": "Qwen-32B",
}

MODEL_ORDER = ["Llama-3.1-8B", "Mistral-7B-v0.3", "Qwen2.5-32B", "Qwen2.5-7B"]

MODEL_COLORS = {
    "Llama-3.1-8B": "#0072B2",
    "Qwen2.5-7B": "#D55E00",
    "Mistral-7B-v0.3": "#009E73",
    "Qwen2.5-32B": "#CC79A7",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_auto_results(results_dir: Path) -> Dict:
    all_data = {}
    for f in sorted(results_dir.glob("autoregressive_experiment_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tag = data["model_tag"]
        all_data[tag] = data
        print(f"  Loaded {tag}")
    return all_data


def fig_auto_transfer(all_data: Dict, output_dir: Path):
    """Two-panel figure:
    Left: ρ(prefill, auto) per model × bit — the transfer question
    Right: ρ(tr(MΣ), auto) vs ρ(MSE, auto) — DDT vs MSE in auto setting
    """
    models = [m for m in MODEL_ORDER if m in all_data]
    bits_list = [3, 4]
    n_models = len(models)
    bit_colors = {"3b": "#56B4E9", "4b": "#009E73"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_width = 0.35
    x = np.arange(n_models)

    # ---- Left: ρ(prefill, auto) ----
    for i, bits in enumerate(bits_list):
        rhos = []
        for model_tag in models:
            data = all_data[model_tag]
            bit_key = f"{bits}b"
            r = data.get("results", {}).get(bit_key, {}).get("ranking", {})
            rho = r.get("rho_prefill_auto", float("nan"))
            rhos.append(rho)

        offset = (i - 0.5) * bar_width
        color = bit_colors[f"{bits}b"]
        ax1.bar(x + offset, rhos, bar_width, alpha=0.85,
                color=color, label=f"{bits}b",
                edgecolor="white", linewidth=0.5)

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("ρ(prefill ΔL, auto ΔL)", fontsize=11)
    ax1.set_title("Prefill → autoregressive transfer", fontsize=12,
                  fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=9)
    ax1.legend(fontsize=9, framealpha=0.8)

    # Auto y-range with padding
    all_rhos_left = []
    for bits in bits_list:
        for model_tag in models:
            r = all_data[model_tag].get("results", {}).get(f"{bits}b", {}).get("ranking", {})
            v = r.get("rho_prefill_auto", 0)
            if not np.isnan(v):
                all_rhos_left.append(v)
    if all_rhos_left:
        ax1.set_ylim(min(min(all_rhos_left) - 0.1, -0.1),
                      max(all_rhos_left) + 0.15)

    # Annotate Qwen-7B
    qwen7b_idx = [i for i, m in enumerate(models) if m == "Qwen2.5-7B"]
    if qwen7b_idx:
        idx = qwen7b_idx[0]
        # Find the max rho for Qwen to place annotation
        qwen_rhos = []
        for bits in bits_list:
            r = all_data["Qwen2.5-7B"].get("results", {}).get(f"{bits}b", {}).get("ranking", {})
            v = r.get("rho_prefill_auto", 0)
            if not np.isnan(v):
                qwen_rhos.append(v)
        if qwen_rhos:
            top = max(qwen_rhos)
            ax1.annotate("strong\ntransfer", xy=(idx, top + 0.03), fontsize=8,
                         ha="center", va="bottom", color="#D55E00",
                         fontweight="bold")

    # ---- Right: DDT vs MSE in auto setting ----
    # Average ρ across bitwidths per model — the question is DDT vs MSE,
    # not 3b vs 4b, so collapsing bits makes the message clearer.
    col_ddt = "#0072B2"   # blue — DDT
    col_mse = "#D55E00"   # vermillion — MSE

    avg_trms = []
    avg_mse = []
    for model_tag in models:
        trms_vals = []
        mse_vals = []
        for bits in bits_list:
            r = all_data[model_tag].get("results", {}).get(f"{bits}b", {}).get("ranking", {})
            v_t = r.get("rho_trms_auto", float("nan"))
            v_m = r.get("rho_mse_auto", float("nan"))
            if not np.isnan(v_t):
                trms_vals.append(v_t)
            if not np.isnan(v_m):
                mse_vals.append(v_m)
        avg_trms.append(np.mean(trms_vals) if trms_vals else 0)
        avg_mse.append(np.mean(mse_vals) if mse_vals else 0)

    ax2.bar(x - bar_width / 2, avg_trms, bar_width, alpha=0.85,
            color=col_ddt, edgecolor="white", linewidth=0.5, label="tr(MΣ)")
    ax2.bar(x + bar_width / 2, avg_mse, bar_width, alpha=0.7,
            color=col_mse, edgecolor="white", linewidth=0.5,
            hatch="//", label="MSE")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("ρ(metric, auto ΔL)", fontsize=11)
    ax2.set_title("DDT vs MSE in autoregressive setting", fontsize=12,
                  fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=9)
    ax2.legend(fontsize=9, framealpha=0.8)

    all_rhos_right = avg_trms + avg_mse
    if all_rhos_right:
        ax2.set_ylim(min(min(all_rhos_right) - 0.1, -0.1),
                      max(all_rhos_right) + 0.15)

    fig.tight_layout()

    out = output_dir / "fig_autoregressive.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig_autoregressive.png")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/ddt")
    parser.add_argument("--output-dir", type=str, default="figures/ddt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading autoregressive results...")
    all_data = load_auto_results(Path(args.results_dir))

    print("\nGenerating figure...")
    fig_auto_transfer(all_data, output_dir)


if __name__ == "__main__":
    main()