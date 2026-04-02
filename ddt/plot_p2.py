"""
P2 Figure: Dithered vs Deterministic ΔL spread
================================================
Strip plot showing that dithering collapses permutation-induced variation.

Usage:
  python -m ddt.plot_p2 --results-dir results/ddt --output-dir figures/ddt
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

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_p2_results(results_dir: Path) -> Dict:
    all_data = {}
    for f in sorted(results_dir.glob("dithered_comparison_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tag = data["model_tag"]
        all_data[tag] = data
        print(f"  Loaded {tag}")
    return all_data


def fig_p2_strip(all_data: Dict, output_dir: Path, bits: int = 4):
    """
    Strip plot: deterministic vs dithered ΔL per model at fixed bitwidth.
    Each dot = one permutation config's ΔL.
    """
    models = [m for m in MODEL_ORDER if m in all_data]
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(2.8 * n_models, 4.5),
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

        det_dls = [c["det_dl_fresh"] for c in configs
                   if c["det_dl_fresh"] is not None]
        dith_means = [c["dith_mean"] for c in configs
                      if c["dith_mean"] is not None and not np.isnan(c["dith_mean"])]

        # Jitter for visibility
        np.random.seed(42)
        det_x = np.zeros(len(det_dls)) + 0.0 + np.random.normal(0, 0.06, len(det_dls))
        dith_x = np.ones(len(dith_means)) + np.random.normal(0, 0.06, len(dith_means))

        ax.scatter(det_x, det_dls, s=18, alpha=0.6, c="#D55E00",
                   edgecolors="none", zorder=3)
        ax.scatter(dith_x, dith_means, s=18, alpha=0.6, c="#0072B2",
                   edgecolors="none", zorder=3)

        # Range annotations
        det_range = max(det_dls) - min(det_dls) if det_dls else 0
        dith_range = max(dith_means) - min(dith_means) if dith_means else 0
        if det_range > 0:
            ratio = det_range / dith_range if dith_range > 1e-8 else float("inf")
            ax.text(0.5, 0.02, f"range ratio: {ratio:.1f}×",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color="#333333", style="italic")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Det", "Dith"], fontsize=10)
        ax.set_title(short, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("ΔL", fontsize=11)

    fig.suptitle(f"Dithering collapses permutation-induced ΔL variation ({bits}-bit)",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = output_dir / f"fig_p2_strip_{bits}b.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / f"fig_p2_strip_{bits}b.png")
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

    print("Loading P2 results...")
    all_data = load_p2_results(Path(args.results_dir))

    for bits in args.bits:
        print(f"\nGenerating {bits}b strip plot...")
        fig_p2_strip(all_data, output_dir, bits=bits)


if __name__ == "__main__":
    main()