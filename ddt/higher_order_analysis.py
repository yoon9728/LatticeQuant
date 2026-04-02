"""
DDT — Higher-Order Analysis (P4)
=================================
Diagnoses the empirical contribution of first-order vs residual terms,
motivated by the Taylor expansion structure in Theorem A.

Key questions:
  1. What fraction of |ΔL| is accounted for by |Q1| (first-order magnitude)?
  2. Does |ΔL - Q1| (absolute residual) increase with error magnitude tr(Σ)?
  3. How does the Q1 magnitude share change across regimes?

Theory (Theorem A):
  ΔL = Q1 + Q2 + R3
  Q1 = Σ s^T e           (first-order, signed)
  Q2 = ½ Σ e^T H e       (second-order, positive)
  |R3| ≤ (B3/6)||e||^3   (bounded remainder)

For deterministic quantization:
  - Q1 predicts RANKING (which config is better within a bitwidth)
  - Q2 predicts LEVEL (overall magnitude of ΔL)
  - Residual = ΔL - Q1 ≈ Q2 + R3

No new model runs — reads P0 JSONs only.

Usage:
  python -m ddt.higher_order_analysis --results-dir results/ddt --output-dir figures/ddt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import spearmanr as scipy_spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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

MODEL_MARKERS = {
    "Llama-3.1-8B": "o",
    "Qwen2.5-7B": "s",
    "Mistral-7B-v0.3": "^",
    "Qwen2.5-32B": "D",
}

COLORS = {
    "2b": "#E69F00",
    "3b": "#56B4E9",
    "4b": "#009E73",
    "5b": "#CC79A7",
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


def spearman_rho(x, y):
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(xa) < 3:
        return float("nan")
    if HAS_SCIPY:
        rho, _ = scipy_spearmanr(xa, ya)
        return float(rho)
    n = len(xa)
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def load_all_p0(results_dir: Path) -> Dict:
    results = {}
    for f in sorted(results_dir.glob("caba_explain_v2_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tag = data["model_tag"]
        results[tag] = data
        print(f"  Loaded {tag}: {len(data['config_list'])} configs")
    return results


def fig_p4_taylor(all_data: Dict, output_dir: Path):
    """
    Two-panel figure:

    Left: Q1 magnitude share — median |Q1| / |ΔL| per model × bitwidth.
      Line plot (one line per model) instead of grouped bars for clarity.

    Right: Residual vs error magnitude — |ΔL - Q1| vs tr(Σ).
      Shows residual scales with error power (Q2 dominance).
      Includes log-log trend line to highlight scaling relationship.
    """
    models = [m for m in MODEL_ORDER if m in all_data]
    bits_list = [2, 3, 4, 5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ---- Left panel: Q1 fraction as line plot ----
    for model_tag in models:
        short = MODEL_SHORT.get(model_tag, model_tag)
        color = MODEL_COLORS.get(model_tag, "gray")
        marker = MODEL_MARKERS.get(model_tag, "o")
        data = all_data[model_tag]
        configs = data["config_list"]

        fracs = []
        for bits in bits_list:
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None]
            if not items:
                fracs.append(float("nan"))
                continue
            ratios = []
            for c in items:
                dl = abs(c["delta_loss"])
                q1 = abs(c["linear_pred"])
                if dl > 1e-6:
                    ratios.append(q1 / dl)
            fracs.append(np.median(ratios) if ratios else float("nan"))

        ax1.plot(bits_list, fracs, marker=marker, color=color, label=short,
                 linewidth=1.8, markersize=7)

    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="|Q₁| = |ΔL|")
    ax1.set_xlabel("Bitwidth", fontsize=11)
    ax1.set_ylabel("Median |Q₁| / |ΔL|", fontsize=11)
    ax1.set_title("First-order magnitude share", fontsize=12, fontweight="bold")
    ax1.set_xticks(bits_list)
    ax1.set_xticklabels([f"{b}b" for b in bits_list])
    ax1.legend(fontsize=8, loc="best", framealpha=0.8)
    ax1.set_ylim(0, None)

    # ---- Right panel: |Residual| vs tr(Σ) ----
    all_tr_sigma = []
    all_abs_resid = []

    for model_tag in models:
        color = MODEL_COLORS.get(model_tag, "gray")
        marker = MODEL_MARKERS.get(model_tag, "o")
        data = all_data[model_tag]
        configs = data["config_list"]

        for bits_str, bit_color in COLORS.items():
            bits = int(bits_str[0])
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None]
            if not items:
                continue

            tr_sigma = [c["tr_Sigma"] for c in items]
            abs_resid = [abs(c["delta_loss"] - c["linear_pred"]) for c in items]

            ax2.scatter(tr_sigma, abs_resid, c=bit_color, marker=marker,
                       s=20, alpha=0.5, edgecolors="none")
            all_tr_sigma.extend(tr_sigma)
            all_abs_resid.extend(abs_resid)

    # Log-log trend line
    xs = np.array(all_tr_sigma)
    ys = np.array(all_abs_resid)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() > 10:
        log_x, log_y = np.log10(xs[mask]), np.log10(ys[mask])
        slope, intercept = np.polyfit(log_x, log_y, 1)
        x_fit = np.linspace(log_x.min(), log_x.max(), 100)
        ax2.plot(10 ** x_fit, 10 ** (slope * x_fit + intercept),
                 color="black", linewidth=1.2, linestyle="--", alpha=0.7,
                 label=f"slope = {slope:.2f}")

    rho_overall = spearman_rho(all_tr_sigma, all_abs_resid)
    ax2.text(0.03, 0.95, f"ρ(tr(Σ), |residual|) = {rho_overall:.2f}",
             transform=ax2.transAxes, fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax2.set_xlabel("tr(Σ) — error magnitude", fontsize=11)
    ax2.set_ylabel("|ΔL − Q₁| (absolute residual)", fontsize=11)
    ax2.set_title("Residual vs error magnitude", fontsize=12, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Legend: bits (colors) + models (markers with actual colors)
    bit_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=7, label=b) for b, c in COLORS.items()]
    model_handles = [Line2D([0], [0], marker=MODEL_MARKERS.get(t, "o"), color="w",
                            markerfacecolor=MODEL_COLORS.get(t, "gray"),
                            markeredgecolor=MODEL_COLORS.get(t, "gray"),
                            markersize=7, label=MODEL_SHORT.get(t, t))
                     for t in models]
    # Include trend line in legend
    existing_handles, existing_labels = ax2.get_legend_handles_labels()
    trend_handles = [h for h, l in zip(existing_handles, existing_labels)
                     if l.startswith("slope")]
    ax2.legend(handles=bit_handles + model_handles + trend_handles, fontsize=7,
               loc="upper left", ncol=2, framealpha=0.8)

    fig.tight_layout()
    out = output_dir / "fig_p4_taylor.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig_p4_taylor.png")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_stats(all_data: Dict):
    """Print per-model per-bit statistics for paper text."""
    models = [m for m in MODEL_ORDER if m in all_data]
    bits_list = [2, 3, 4, 5]

    print("\n" + "=" * 70)
    print("P4: Higher-Order Statistics")
    print("=" * 70)

    for model_tag in models:
        short = MODEL_SHORT.get(model_tag, model_tag)
        data = all_data[model_tag]
        configs = data["config_list"]
        print(f"\n--- {short} ---")

        for bits in bits_list:
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None]
            if not items:
                continue

            dls = np.array([c["delta_loss"] for c in items])
            q1s = np.array([c["linear_pred"] for c in items])
            residuals = dls - q1s
            tr_sigmas = np.array([c["tr_Sigma"] for c in items])

            # Q1 fraction
            q1_fracs = np.abs(q1s) / np.maximum(np.abs(dls), 1e-8)
            median_frac = np.median(q1_fracs)

            # Residual correlation with tr(Σ) — use |residual| for Q2 dominance
            rho_resid = spearman_rho(tr_sigmas, np.abs(residuals))

            # Same-sign fraction: Q1 and ΔL agree in sign
            same_sign = np.mean((q1s > 0) == (dls > 0))

            print(f"  {bits}b: median|Q1|/|ΔL|={median_frac:.3f}  "
                  f"ρ(tr(Σ),|resid|)={rho_resid:+.3f}  "
                  f"same_sign={same_sign:.0%}  "
                  f"mean residual={np.mean(residuals):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="DDT P4: Higher-Order Analysis"
    )
    parser.add_argument("--results-dir", type=str, default="results/ddt")
    parser.add_argument("--output-dir", type=str, default="figures/ddt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading P0 results...")
    all_data = load_all_p0(Path(args.results_dir))

    print_stats(all_data)

    print("\nGenerating figure...")
    fig_p4_taylor(all_data, output_dir)


if __name__ == "__main__":
    main()