"""
DDT Paper v2 — Figure Generation (P1)
=======================================
Reads JSON results from P0 experiments, generates publication-quality figures.

Required files in results/ddt/:
  caba_explain_v2_Llama-3.1-8B.json
  caba_explain_v2_Qwen2.5-7B.json
  caba_explain_v2_Mistral-7B-v0.3.json
  caba_explain_v2_Qwen2.5-32B.json

Usage:
  python -m visualization.plot_figures --results-dir results/ddt --output-dir figures/ddt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ============================================================
# Style
# ============================================================

# Color-blind-friendly palette (Okabe-Ito)
COLORS = {
    "2b": "#E69F00",   # orange
    "3b": "#56B4E9",   # sky blue
    "4b": "#009E73",   # green
    "5b": "#CC79A7",   # pink
}

MODEL_MARKERS = {
    "Llama-3.1-8B": "o",
    "Qwen2.5-7B": "s",
    "Mistral-7B-v0.3": "^",
    "Qwen2.5-32B": "D",
}

MODEL_COLORS = {
    "Llama-3.1-8B": "#0072B2",    # blue
    "Qwen2.5-7B": "#D55E00",      # vermillion
    "Mistral-7B-v0.3": "#009E73",  # green
    "Qwen2.5-32B": "#CC79A7",     # pink
}

MODEL_SHORT = {
    "Llama-3.1-8B": "Llama-8B",
    "Qwen2.5-7B": "Qwen-7B",
    "Mistral-7B-v0.3": "Mistral-7B",
    "Qwen2.5-32B": "Qwen-32B",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ============================================================
# Data loading
# ============================================================

def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """Load all v2 JSON results."""
    results = {}
    rdir = Path(results_dir)
    for f in sorted(rdir.glob("caba_explain_v2_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tag = data["model_tag"]
        results[tag] = data
        print(f"  Loaded {tag}: {len(data['config_list'])} configs")
    return results


def get_configs(data: Dict, bits: int = None) -> List[Dict]:
    """Get config list, optionally filtered by bits."""
    configs = data["config_list"]
    if bits is not None:
        configs = [c for c in configs if c["bits"] == bits]
    return configs


# ============================================================
# Correlation helpers
# ============================================================

def spearman_rho(x, y):
    """Manual Spearman for annotation (no scipy dependency in plotting)."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def pearson_r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# Fig 1: Central Claim — MSE vs tr(MΣ) as ΔL predictors
#   Two-panel scatter, all models, bit-colored
# ============================================================

def fig1_central_claim(all_data: Dict, output_dir: Path):
    """
    Left: MSE (tr(Σ)) vs ΔL — poor within-bitwidth predictor
    Right: tr(MΣ) vs ΔL — good within-bitwidth predictor

    Key fix: annotate with WITHIN-BITWIDTH ρ (averaged across models)
    to show that pooled ρ (both high due to cross-bit variation)
    masks the real story: MSE fails within a bitwidth, tr(MΣ) doesn't.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Collect within-bitwidth ρ for annotation
    within_rho_mse = {}   # bits -> list of per-model ρ
    within_rho_trms = {}

    for model_tag, data in all_data.items():
        marker = MODEL_MARKERS.get(model_tag, "o")

        for bits_str, color in COLORS.items():
            bits = int(bits_str[0])
            configs = get_configs(data, bits)
            configs = [c for c in configs if c["delta_loss"] is not None]
            if len(configs) < 5:
                continue

            mse = [c["tr_Sigma"] for c in configs]
            trms = [c["tr_M_Sigma"] for c in configs]
            dl = [c["delta_loss"] for c in configs]

            ax1.scatter(mse, dl, c=color, marker=marker, s=15, alpha=0.6,
                       edgecolors="none")
            ax2.scatter(trms, dl, c=color, marker=marker, s=15, alpha=0.6,
                       edgecolors="none")

            # Track within-bitwidth ρ
            within_rho_mse.setdefault(bits, []).append(spearman_rho(mse, dl))
            within_rho_trms.setdefault(bits, []).append(spearman_rho(trms, dl))

    # Pooled ρ
    all_mse, all_trms, all_dl = [], [], []
    for data in all_data.values():
        for c in data["config_list"]:
            if c["delta_loss"] is not None:
                all_mse.append(c["tr_Sigma"])
                all_trms.append(c["tr_M_Sigma"])
                all_dl.append(c["delta_loss"])

    rho_mse = spearman_rho(all_mse, all_dl)
    rho_trms = spearman_rho(all_trms, all_dl)

    ax1.set_xlabel("MSE = tr(Σ)", fontsize=11)
    ax1.set_ylabel("ΔL (measured loss change)", fontsize=11)
    ax1.set_title(f"MSE vs ΔL", fontsize=12, fontweight="bold")
    ax1.set_xscale("log")

    ax2.set_xlabel("tr(MΣ) — directional risk", fontsize=11)
    ax2.set_ylabel("ΔL (measured loss change)", fontsize=11)
    ax2.set_title(f"tr(MΣ) vs ΔL", fontsize=12, fontweight="bold")
    ax2.set_xscale("log")

    # Annotation box: pooled + within-bitwidth ρ (mean across models)
    # This is the key: shows MSE's pooled ρ is inflated by cross-bit
    ann_mse = f"Pooled ρ = {rho_mse:.2f}\n"
    ann_trms = f"Pooled ρ = {rho_trms:.2f}\n"
    ann_mse += "─── Within-bitwidth ───\n"
    ann_trms += "─── Within-bitwidth ───\n"
    for bits in [2, 3, 4, 5]:
        if bits in within_rho_mse:
            avg_mse = np.mean(within_rho_mse[bits])
            avg_trms = np.mean(within_rho_trms[bits])
            ann_mse += f"  {bits}b: ρ = {avg_mse:+.2f}\n"
            ann_trms += f"  {bits}b: ρ = {avg_trms:+.2f}\n"

    ax1.text(0.03, 0.97, ann_mse.strip(), transform=ax1.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    ax2.text(0.03, 0.97, ann_trms.strip(), transform=ax2.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # Legend: bits (colors) + models (markers)
    bit_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=7, label=b) for b, c in COLORS.items()]
    model_handles = [Line2D([0], [0], marker=m, color="w",
                            markerfacecolor=MODEL_COLORS.get(t, "gray"),
                            markeredgecolor=MODEL_COLORS.get(t, "gray"),
                            markersize=7, label=MODEL_SHORT.get(t, t))
                     for t, m in MODEL_MARKERS.items() if t in all_data]

    ax1.legend(handles=bit_handles + model_handles, loc="lower right",
               ncol=2, fontsize=7, framealpha=0.8)

    fig.tight_layout()

    out = output_dir / "fig1_central_claim.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig1_central_claim.png")
    plt.close(fig)
    print(f"  Fig 1 saved: {out}")


# ============================================================
# Fig 2: Within-bitwidth scatter (strongest cases)
#   Per-model, per-bit, best metric vs ΔL
# ============================================================

def fig2_within_bitwidth(all_data: Dict, output_dir: Path):
    """
    Direct contrast: MSE vs DDT at 4-bit, per model.

    Top row: MSE (tr(Σ)) vs ΔL — showing no within-bitwidth correlation
    Bottom row: Best DDT metric vs ΔL — showing positive correlation

    This is the visual proof that error direction matters within a fixed
    bitwidth, not just across bitwidths.
    """
    models = list(all_data.keys())
    n_models = len(models)
    target_bits = 4  # the most important single bitwidth

    fig, axes = plt.subplots(2, n_models, figsize=(3.2 * n_models, 6))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, model_tag in enumerate(models):
        data = all_data[model_tag]
        short = MODEL_SHORT.get(model_tag, model_tag)
        configs = get_configs(data, target_bits)
        configs = [c for c in configs if c["delta_loss"] is not None]

        if len(configs) < 5:
            axes[0, col].text(0.5, 0.5, "insufficient data",
                             transform=axes[0, col].transAxes, ha="center")
            axes[1, col].text(0.5, 0.5, "insufficient data",
                             transform=axes[1, col].transAxes, ha="center")
            continue

        dl = np.array([c["delta_loss"] for c in configs])
        mse = np.array([c["tr_Sigma"] for c in configs])

        # Find best DDT metric
        q1 = np.array([c["linear_pred"] for c in configs])
        trms = np.array([c["tr_M_Sigma"] for c in configs])
        rho_q1 = abs(spearman_rho(q1, dl))
        rho_trms = abs(spearman_rho(trms, dl))
        if rho_q1 >= rho_trms:
            ddt_vals = q1
            ddt_name = "Q₁"
            rho_ddt = spearman_rho(q1, dl)
        else:
            ddt_vals = trms
            ddt_name = "tr(MΣ)"
            rho_ddt = spearman_rho(trms, dl)

        rho_mse = spearman_rho(mse, dl)

        # Top row: MSE vs ΔL
        ax_top = axes[0, col]
        ax_top.scatter(mse, dl, c=COLORS["4b"], s=15, alpha=0.6, edgecolors="none")
        ax_top.set_title(short, fontsize=11, fontweight="bold")
        ax_top.annotate(f"ρ = {rho_mse:.2f}", xy=(0.05, 0.92),
                       xycoords="axes fraction", fontsize=9,
                       color="#D55E00", fontweight="bold")
        if col == 0:
            ax_top.set_ylabel("ΔL", fontsize=10)

        # Bottom row: DDT vs ΔL
        ax_bot = axes[1, col]
        ax_bot.scatter(ddt_vals, dl, c=COLORS["4b"], s=15, alpha=0.6, edgecolors="none")
        ax_bot.set_xlabel(ddt_name, fontsize=10)
        ax_bot.annotate(f"ρ = {rho_ddt:.2f}", xy=(0.05, 0.92),
                       xycoords="axes fraction", fontsize=9,
                       color="#0072B2", fontweight="bold")
        if col == 0:
            ax_bot.set_ylabel("ΔL", fontsize=10)

    # Row labels on the left
    axes[0, 0].annotate("MSE →", xy=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=11, fontweight="bold", color="#D55E00",
                        rotation=90, ha="center", va="center")
    axes[1, 0].annotate("DDT →", xy=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=11, fontweight="bold", color="#0072B2",
                        rotation=90, ha="center", va="center")

    fig.suptitle(f"Within {target_bits}-bit: MSE fails, DDT predicts ΔL ranking",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = output_dir / "fig2_within_bitwidth.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig2_within_bitwidth.png")
    plt.close(fig)
    print(f"  Fig 2 saved: {out}")


# ============================================================
# Fig 3: K/V sensitivity breakdown across models
# ============================================================

def fig3_kv_breakdown(all_data: Dict, output_dir: Path):
    """
    Grouped bar chart: K-path fraction per model per bitwidth.
    Shows Qwen-7B ≈ 97% K vs Llama ≈ 32%.
    """
    models = list(all_data.keys())
    bits_list = [2, 3, 4, 5]
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(8, 4))

    bar_width = 0.18
    x = np.arange(len(bits_list))

    for i, model_tag in enumerate(models):
        short = MODEL_SHORT.get(model_tag, model_tag)
        data = all_data[model_tag]

        k_fracs = []
        for bits in bits_list:
            if f"{bits}b" in data.get("kv_breakdown", {}):
                k_fracs.append(data["kv_breakdown"][f"{bits}b"]["K_frac_mean"] * 100)
            else:
                # Compute from config_list
                configs = get_configs(data, bits)
                fracs = []
                for c in configs:
                    total = c["tr_M_Sigma_K"] + c["tr_M_Sigma_V"]
                    if total > 1e-20:
                        fracs.append(c["tr_M_Sigma_K"] / total * 100)
                k_fracs.append(np.mean(fracs) if fracs else 50)

        offset = (i - n_models / 2 + 0.5) * bar_width
        color = MODEL_COLORS.get(model_tag, None)
        ax.bar(x + offset, k_fracs, bar_width, label=short, alpha=0.85,
               color=color)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="K=V")
    ax.set_xlabel("Bitwidth")
    ax.set_ylabel("K-path fraction (%)")
    ax.set_title("K vs V sensitivity concentration")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}b" for b in bits_list])
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    out = output_dir / "fig3_kv_breakdown.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig3_kv_breakdown.png")
    plt.close(fig)
    print(f"  Fig 3 saved: {out}")


# ============================================================
# Fig 4: Bias drift across bitwidths
# ============================================================

def fig4_bias_drift(all_data: Dict, output_dir: Path):
    """
    Line plot: bias drift (centered/uncentered gap %) vs bitwidth, per model.
    Shows Qwen-7B has massive bias drift (94% at 2b) vs Llama (34%).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    bits_list = [2, 3, 4, 5]

    for model_tag, data in all_data.items():
        short = MODEL_SHORT.get(model_tag, model_tag)
        marker = MODEL_MARKERS.get(model_tag, "o")

        gaps = []
        for bits in bits_list:
            if f"{bits}b" in data.get("bias_drift", {}):
                gaps.append(data["bias_drift"][f"{bits}b"]["gap_pct_mean"])
            else:
                configs = get_configs(data, bits)
                g = []
                for c in configs:
                    u = c["tr_M_Sigma"]
                    cent = c["tr_M_Sigma_c"]
                    if abs(u) > 1e-20:
                        g.append(abs(u - cent) / abs(u) * 100)
                gaps.append(np.mean(g) if g else 0)

        color = MODEL_COLORS.get(model_tag, None)
        ax.plot(bits_list, gaps, marker=marker, label=short, linewidth=1.5,
                markersize=6, color=color)

    ax.set_xlabel("Bitwidth")
    ax.set_ylabel("Bias drift gap (%)")
    ax.set_title("D2(a) violation: centered vs uncentered tr(MΣ)")
    ax.set_xticks(bits_list)
    ax.set_xticklabels([f"{b}b" for b in bits_list])
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = output_dir / "fig4_bias_drift.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig4_bias_drift.png")
    plt.close(fig)
    print(f"  Fig 4 saved: {out}")


# ============================================================
# Fig 5: End-to-end evidence summary
# ============================================================

def fig5_e2e_evidence(all_data: Dict, output_dir: Path):
    """
    Per-model panels showing absolute ΔL of MSE-best vs DDT-best config.

    Each model gets its own y-axis scale (sharey=False), so models with
    tiny ΔL (e.g. Mistral 4b/5b ≈ 0.005) show bars near zero naturally
    rather than being invisible or producing misleading percentages.
    """
    models = [t for t in all_data.keys()]
    bits_list = [2, 3, 4, 5]
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(3.2 * n_models, 4),
                             sharey=False)
    if n_models == 1:
        axes = [axes]

    bar_width = 0.35

    for col, model_tag in enumerate(models):
        ax = axes[col]
        data = all_data[model_tag]
        short = MODEL_SHORT.get(model_tag, model_tag)
        e2e = data.get("end_to_end_evidence", {})

        mse_vals = []
        ddt_vals = []
        valid_bits = []

        for bits in bits_list:
            key = f"{bits}b"
            if key not in e2e or "error" in e2e[key]:
                mse_vals.append(0)
                ddt_vals.append(0)
                valid_bits.append(False)
                continue

            trms_dl = e2e[key]["tr_M_Sigma"]["best_actual_delta_loss"]
            mse_dl = e2e[key]["MSE_tr_Sigma"]["best_actual_delta_loss"]
            ddt_best = trms_dl

            mse_vals.append(mse_dl)
            ddt_vals.append(ddt_best)
            valid_bits.append(True)

        x = np.arange(len(bits_list))
        bars_mse = ax.bar(x - bar_width / 2, mse_vals, bar_width,
                          label="MSE-best", color="#D55E00", alpha=0.85)
        bars_ddt = ax.bar(x + bar_width / 2, ddt_vals, bar_width,
                          label="DDT-best", color="#0072B2", alpha=0.85)

        for bar, valid in zip(list(bars_mse) + list(bars_ddt),
                              valid_bits + valid_bits):
            if not valid:
                bar.set_color("lightgray")
                bar.set_alpha(0.3)

        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}b" for b in bits_list])
        ax.set_xlabel("Bitwidth", fontsize=10)
        if col == 0:
            ax.set_ylabel("ΔL (lower = better)", fontsize=10)
        ax.set_ylim(bottom=0)

        if col == n_models - 1:
            ax.legend(fontsize=7, framealpha=0.8)

    fig.suptitle("End-to-end: ΔL of MSE-best vs DDT-best config",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = output_dir / "fig5_e2e_evidence.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig5_e2e_evidence.png")
    plt.close(fig)
    print(f"  Fig 5 saved: {out}")


# ============================================================
# Fig 6: Correlation heatmap — metric × model × bitwidth
# ============================================================

def fig6_correlation_heatmap(all_data: Dict, output_dir: Path):
    """
    Heatmap: Spearman ρ for each (model, bitwidth, metric) combination.
    Shows which metric wins where.
    """
    models = list(all_data.keys())
    bits_list = [2, 3, 4, 5]
    metrics = ["Q1_linear_pred", "tr_M_Sigma", "MSE_tr_Sigma"]
    metric_labels = ["Q₁", "tr(MΣ)", "MSE"]

    # Build matrix: rows = model×bits, cols = metrics
    row_labels = []
    matrix = []

    for model_tag in models:
        short = MODEL_SHORT.get(model_tag, model_tag)
        data = all_data[model_tag]
        corr_data = data.get("correlations", {})

        for bits in bits_list:
            key = f"{bits}b"
            if key not in corr_data:
                continue

            row_labels.append(f"{short} {bits}b")
            row = []
            for metric in metrics:
                rho = corr_data[key]["correlations"].get(metric, {}).get("spearman_rho", 0)
                row.append(rho)
            matrix.append(row)

    if not matrix:
        print("  Fig 6 skipped: no correlation data")
        return

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(5, 0.4 * len(row_labels) + 1.5))

    vmax = max(0.6, np.nanmax(matrix) + 0.05)
    vmin = min(-0.3, np.nanmin(matrix) - 0.05)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.35 else "black"
            weight = "bold" if abs(val) > 0.3 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   color=color, fontsize=8, fontweight=weight)

    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Within-bitwidth Spearman ρ(metric, ΔL)")

    fig.tight_layout()
    out = output_dir / "fig6_correlation_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(output_dir / "fig6_correlation_heatmap.png")
    plt.close(fig)
    print(f"  Fig 6 saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DDT Paper v2 Figure Generation")
    parser.add_argument("--results-dir", type=str, default="results/ddt")
    parser.add_argument("--output-dir", type=str, default="figures/ddt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_data = load_all_results(args.results_dir)

    if not all_data:
        print("ERROR: No result files found.")
        return

    print(f"\nGenerating figures ({len(all_data)} models)...")
    fig1_central_claim(all_data, output_dir)
    fig2_within_bitwidth(all_data, output_dir)
    fig3_kv_breakdown(all_data, output_dir)
    fig4_bias_drift(all_data, output_dir)
    fig5_e2e_evidence(all_data, output_dir)
    fig6_correlation_heatmap(all_data, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()