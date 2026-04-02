"""
DDT — Paper Statistics (S1-S3)
================================
Computes statistics needed for paper text:
  S1: DDT-best vs MSE-best win-rate (how many cells DDT wins)
  S2: Median relative improvement of DDT-best over MSE-best
  S3: Bonferroni-surviving cell count

Usage:
  python -m ddt.paper_stats --results-dir results/ddt
"""

import json
import argparse
from pathlib import Path
import numpy as np

MODEL_ORDER = ["Llama-3.1-8B", "Mistral-7B-v0.3", "Qwen2.5-32B", "Qwen2.5-7B"]
MODEL_SHORT = {
    "Llama-3.1-8B": "Llama-8B",
    "Qwen2.5-7B": "Qwen-7B",
    "Mistral-7B-v0.3": "Mistral-7B",
    "Qwen2.5-32B": "Qwen-32B",
}


def load_all_p0(results_dir: Path):
    all_data = {}
    for f in sorted(results_dir.glob("caba_explain_v2_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        tag = data["model_tag"]
        all_data[tag] = data
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    all_data = load_all_p0(Path(args.results_dir))
    print(f"Loaded {len(all_data)} models\n")

    bits_list = [2, 3, 4, 5]

    # =========================================================
    # S1 & S2: DDT-best vs MSE-best
    # =========================================================
    print("=" * 60)
    print("S1/S2: DDT-best vs MSE-best comparison")
    print("=" * 60)

    wins_ddt = 0
    wins_mse = 0
    ties = 0
    improvements = []
    nontrivial_improvements = []

    for model_tag in MODEL_ORDER:
        if model_tag not in all_data:
            continue
        short = MODEL_SHORT.get(model_tag, model_tag)
        configs = all_data[model_tag]["config_list"]

        for bits in bits_list:
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None
                     and c["tr_M_Sigma"] is not None and c["tr_Sigma"] is not None]
            if not items:
                continue

            # DDT-best: lowest tr(MΣ)
            ddt_best = min(items, key=lambda c: c["tr_M_Sigma"])
            # MSE-best: lowest tr(Σ)
            mse_best = min(items, key=lambda c: c["tr_Sigma"])

            dl_ddt = ddt_best["delta_loss"]
            dl_mse = mse_best["delta_loss"]

            if abs(dl_mse) > 1e-6:
                rel_imp = (dl_mse - dl_ddt) / abs(dl_mse) * 100
            else:
                rel_imp = 0

            # Nontrivial: ΔL > 0.01
            nontrivial = abs(dl_mse) > 0.01 or abs(dl_ddt) > 0.01

            if dl_ddt < dl_mse - 1e-6:
                wins_ddt += 1
                result = "DDT"
            elif dl_mse < dl_ddt - 1e-6:
                wins_mse += 1
                result = "MSE"
            else:
                ties += 1
                result = "TIE"

            improvements.append(rel_imp)
            if nontrivial:
                nontrivial_improvements.append(rel_imp)

            print(f"  {short:12s} {bits}b: "
                  f"DDT={dl_ddt:+.4f}  MSE={dl_mse:+.4f}  "
                  f"imp={rel_imp:+.1f}%  [{result}]"
                  f"{'  (nontrivial)' if nontrivial else ''}")

    total = wins_ddt + wins_mse + ties
    print(f"\n  --- Overall ---")
    print(f"  S1 Win-rate: DDT wins {wins_ddt}/{total} "
          f"({wins_ddt/total*100:.0f}%), "
          f"MSE wins {wins_mse}/{total} ({wins_mse/total*100:.0f}%), "
          f"ties {ties}/{total}")
    print(f"  S2 All cells: mean={np.mean(improvements):.1f}%, "
          f"median={np.median(improvements):.1f}%")
    if nontrivial_improvements:
        print(f"  S2 Nontrivial cells (ΔL > 0.01): "
              f"mean={np.mean(nontrivial_improvements):.1f}%, "
              f"median={np.median(nontrivial_improvements):.1f}%")

    # ---- Per-bitwidth breakdown (paper uses 3b stats) ----
    print(f"\n  --- Per-bitwidth breakdown ---")
    for bits in bits_list:
        bit_imps = []
        bit_wins_ddt = 0
        bit_total = 0
        for model_tag in MODEL_ORDER:
            if model_tag not in all_data:
                continue
            configs = all_data[model_tag]["config_list"]
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None
                     and c["tr_M_Sigma"] is not None and c["tr_Sigma"] is not None]
            if not items:
                continue
            ddt_best = min(items, key=lambda c: c["tr_M_Sigma"])
            mse_best = min(items, key=lambda c: c["tr_Sigma"])
            dl_ddt = ddt_best["delta_loss"]
            dl_mse = mse_best["delta_loss"]
            rel = (dl_mse - dl_ddt) / abs(dl_mse) * 100 if abs(dl_mse) > 1e-6 else 0
            bit_imps.append(rel)
            bit_total += 1
            if dl_ddt < dl_mse - 1e-6:
                bit_wins_ddt += 1
        if bit_imps:
            print(f"  {bits}b: DDT wins {bit_wins_ddt}/{bit_total}, "
                  f"median imp={np.median(bit_imps):+.1f}%, "
                  f"range=[{min(bit_imps):+.1f}%, {max(bit_imps):+.1f}%]")

    # ---- Qwen-7B specific (catastrophic regime) ----
    print(f"\n  --- Qwen-7B (catastrophic regime) ---")
    if "Qwen2.5-7B" in all_data:
        configs = all_data["Qwen2.5-7B"]["config_list"]
        for bits in bits_list:
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None
                     and c["tr_M_Sigma"] is not None and c["tr_Sigma"] is not None]
            if not items:
                continue
            ddt_best = min(items, key=lambda c: c["tr_M_Sigma"])
            mse_best = min(items, key=lambda c: c["tr_Sigma"])
            dl_ddt = ddt_best["delta_loss"]
            dl_mse = mse_best["delta_loss"]
            rel = (dl_mse - dl_ddt) / abs(dl_mse) * 100 if abs(dl_mse) > 1e-6 else 0
            winner = "DDT" if dl_ddt < dl_mse - 1e-6 else "MSE"
            print(f"  {bits}b: DDT={dl_ddt:+.2f}  MSE={dl_mse:+.2f}  "
                  f"imp={rel:+.1f}%  [{winner}]")

    # =========================================================
    # S3: Bonferroni-surviving cells
    # =========================================================
    print(f"\n{'=' * 60}")
    print("S3: Bonferroni correction (α = 0.05/48 ≈ 0.001)")
    print("=" * 60)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy required for p-values")
        return

    bonf_threshold = 0.05 / 48
    surviving = []

    for model_tag in MODEL_ORDER:
        if model_tag not in all_data:
            continue
        short = MODEL_SHORT.get(model_tag, model_tag)
        configs = all_data[model_tag]["config_list"]

        for bits in bits_list:
            items = [c for c in configs
                     if c["bits"] == bits and c["delta_loss"] is not None]
            if len(items) < 5:
                continue

            dls = [c["delta_loss"] for c in items]

            for metric_name, metric_key in [
                ("Q1", "linear_pred"),
                ("tr(MΣ)", "tr_M_Sigma"),
                ("MSE", "tr_Sigma"),
            ]:
                vals = [c.get(metric_key) for c in items]
                if any(v is None for v in vals):
                    continue

                rho, p = spearmanr(vals, dls)
                survives = p < bonf_threshold
                if survives:
                    surviving.append(
                        f"{short} {bits}b {metric_name}: "
                        f"ρ={rho:+.3f}, p={p:.2e} ✅"
                    )
                    print(f"  {short:12s} {bits}b {metric_name:8s}: "
                          f"ρ={rho:+.3f}  p={p:.2e}  ✅ SURVIVES")

    print(f"\n  S3: {len(surviving)} cells survive Bonferroni (out of ~48)")
    for s in surviving:
        print(f"    {s}")


if __name__ == "__main__":
    main()