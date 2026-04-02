"""
Hadamard Experiment Figure
===========================
Strip plot: No Hadamard vs Hadamard ΔL per model at fixed bitwidth.

Usage:
  python -m visualization.plot_hadamard --results-dir results/ddt --output-dir figures/ddt
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

# Consistent with plot_figures.py
COL_NO_HAD = "#D55E00"   # vermillion — no Hadamard
COL_HAD = "#0072B2"      # blue — with Hadamard

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_hadamard_results(results_dir: Path) -> Dict:
    all_data = {}
    for f in sorted(results_dir.glob("hadamard_experiment_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tag = data["model_tag"]
        all_data[tag] = data
        print(f"  Loaded {tag}")
    return all_data


def fig_hadamard_strip(all_data: Dict, output_dir: Path, bits: int = 4):
    """Strip plot with mean/IQR: No Hadamard vs Hadamard, per model."""
    models = [m for m in MODEL_ORDER if m in all_data]
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(3.0 * n_models, 4.5),
                              sharey=False)
    if n_models == 1:
        axes = [axes]

    for col, model_tag in enumerate(models):
        ax = axes[col]
        short = MODEL_SHORT.get(model_tag, model_tag)
        data = all_data[model_tag]
        bit_key = f"{bits}b"

        if bit_key not in data["results"]:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(short)
            continue

        configs = data["results"][bit_key]["configs"]
        dls_no = np.array([c["dl_no_hadamard"] for c in configs])
        dls_had = np.array([c["dl_hadamard"] for c in configs])

        # Light background bands for each group
        ax.axvspan(-0.3, 0.3, color=COL_NO_HAD, alpha=0.06, zorder=0)
        ax.axvspan(0.7, 1.3, color=COL_HAD, alpha=0.06, zorder=0)

        # Jittered scatter
        np.random.seed(42)
        no_x = np.random.normal(0, 0.07, len(dls_no))
        had_x = 1 + np.random.normal(0, 0.07, len(dls_had))

        ax.scatter(no_x, dls_no, s=24, alpha=0.5, c=COL_NO_HAD,
                   edgecolors="none", zorder=3)
        ax.scatter(had_x, dls_had, s=24, alpha=0.5, c=COL_HAD,
                   edgecolors="none", zorder=3)

        # Mean + IQR error bars
        for pos, vals, color in [(0, dls_no, COL_NO_HAD),
                                  (1, dls_had, COL_HAD)]:
            mean = np.mean(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            ax.plot(pos, mean, marker="D", color="white", markeredgecolor=color,
                    markeredgewidth=1.8, markersize=8, zorder=5)
            ax.vlines(pos, q25, q75, color=color, linewidth=2.5, zorder=4,
                      alpha=0.8)

        # Connect means
        mean_no = np.mean(dls_no)
        mean_had = np.mean(dls_had)
        ax.plot([0, 1], [mean_no, mean_had], color="black", linewidth=1.2,
                linestyle="--", alpha=0.4, zorder=2)

        # Mean reduction annotation
        if abs(mean_no) > 1e-6:
            pct = (mean_no - mean_had) / abs(mean_no) * 100
            ann_color = "#009E73" if pct > 0 else "#D55E00"
            ax.text(0.5, 0.02, f"mean: {pct:+.0f}%",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    color=ann_color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Had", "Had"], fontsize=10)
        ax.set_title(short, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("ΔL", fontsize=11)
        ax.set_xlim(-0.4, 1.4)

    fig.suptitle(f"Hadamard pre-rotation effect on ΔL ({bits}-bit)",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = output_dir / f"fig_hadamard_{bits}b.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / f"fig_hadamard_{bits}b.png")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_hadamard_summary(all_data: Dict, output_dir: Path):
    """Two-panel: mean ΔL reduction (%) + CV reduction (%)."""
    models = [m for m in MODEL_ORDER if m in all_data]
    bits_list = [3, 4]
    n_models = len(models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Left: mean ΔL change (%) ----
    bar_width = 0.35
    x = np.arange(n_models)
    bit_colors = {"3b": "#56B4E9", "4b": "#009E73"}

    for i, bits in enumerate(bits_list):
        pcts = []
        for model_tag in models:
            data = all_data[model_tag]
            bit_key = f"{bits}b"
            if bit_key in data["results"]:
                s = data["results"][bit_key]["summary"]
                mean_no = s["mean_no_had"]
                mean_had = s["mean_had"]
                pct = (mean_no - mean_had) / abs(mean_no) * 100 if abs(mean_no) > 1e-6 else 0
            else:
                pct = 0
            pcts.append(pct)

        offset = (i - 0.5) * bar_width
        color = bit_colors.get(f"{bits}b", "gray")
        bars = ax1.bar(x + offset, pcts, bar_width, alpha=0.85,
                       label=f"{bits}b", color=color,
                       edgecolor="white", linewidth=0.5)
        # Hatch negative bars (Hadamard hurts)
        for bar, pct in zip(bars, pcts):
            if pct < 0:
                bar.set_hatch("//")

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Mean ΔL reduction (%)", fontsize=11)
    ax1.set_title("Hadamard effect on mean ΔL", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=9)
    ax1.legend(fontsize=9, framealpha=0.8)
    ax1.text(0.01, 0.97, "↑ Hadamard helps", transform=ax1.transAxes,
             fontsize=8, color="#009E73", fontweight="bold", va="top")

    # ---- Right: CV reduction (%) per model, color-coded ----
    model_bar_colors = [MODEL_COLORS.get(m, "gray") for m in models]
    cvs = []
    for model_tag in models:
        data = all_data[model_tag]
        ap = data.get("anisotropy_proxy", {})
        cv_pct = ap.get("cv_reduction_pct", 0)
        cvs.append(cv_pct)

    bars = ax2.bar(x, cvs, 0.5, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, model_bar_colors):
        bar.set_color(c)

    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("Block RMS CV reduction (%)", fontsize=11)
    ax2.set_title("Hadamard effect on blockwise heterogeneity",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=9)
    ax2.set_ylim(0, max(cvs) * 1.15 if cvs and max(cvs) > 0 else 50)

    fig.tight_layout()

    out = output_dir / "fig_hadamard_summary.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig_hadamard_summary.png")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/ddt")
    parser.add_argument("--output-dir", type=str, default="figures/ddt")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Hadamard results...")
    all_data = load_hadamard_results(Path(args.results_dir))

    for bits in args.bits:
        print(f"\nGenerating {bits}b strip plot...")
        fig_hadamard_strip(all_data, output_dir, bits=bits)

    print("\nGenerating summary figure...")
    fig_hadamard_summary(all_data, output_dir)


if __name__ == "__main__":
    main()