"""
DDT — Theorem C Validation: Isotropic Safety
===============================================
Validates Theorem C's two claims:

(i) Uniqueness: Σ = σ²I is the ONLY error covariance making tr(MΣ)
    independent of M's eigenbasis.  Demonstrated by rotating Σ with
    random orthogonal matrices and showing risk variation for
    anisotropic Σ vs constant risk for isotropic Σ.

(ii) HLP spectral bounds:
    Σ_i a_i b_{d+1-i}  ≤  tr(MΣ)  ≤  Σ_i a_i b_i
    Verified per-head for deterministic quantizer errors under
    different permutations.

Central message:
    Under the causal constraint (M unknown at quantization time),
    isotropic error is the unique basis-invariant strategy.
    Anisotropic error couples with M's spectral structure,
    creating alignment-dependent risk.

Experiment:
  1. Measure M per head (reuse SensitivityMeasurer).
  2. For each permutation mode, quantize deterministically → Σ.
  3. Compute tr(MΣ), HLP bounds, alignment position.
  4. Uniqueness test: rotate Σ by K random orthogonal matrices,
     show tr(M · UΣU^T) varies for anisotropic Σ, constant for isotropic.
  5. Llama vs Qwen contrast: M anisotropy → alignment sensitivity.

Usage:
  python -m ddt.isotropic_safety \\
      --model meta-llama/Llama-3.1-8B \\
      --caba results/caba_llama_3.1_8b.json

  python -m ddt.isotropic_safety \\
      --model Qwen/Qwen2.5-7B \\
      --caba results/caba_qwen2.5_7b.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import load_dataset

from ddt.caba_explain import (
    SensitivityMeasurer,
    load_model,
    load_sorted_permutations,
    make_identity_permutations,
    make_random_permutations,
    quantize_uniform_blocks,
)


# ============================================================
# Per-head spectral analysis
# ============================================================

def compute_spectral_metrics(
    M_data: Dict,
    perms: Dict,
    head_dim: int,
    block_size: int = 8,
) -> Dict:
    """Compute HLP bounds and alignment for each head under a permutation.

    For each (l, comp, h):
      1. Quantize v_h with permutation (deterministic, no dither)
      2. Compute error covariance Σ = (1/T) e^T e
      3. Get eigenvalues of M (descending) and Σ (descending)
      4. Compute:
         - actual = tr(MΣ)
         - upper = Σ a_i b_i  (co-aligned eigenvalues)
         - lower = Σ a_i b_{d+1-i}  (counter-aligned)
         - position = (actual - lower) / (upper - lower)
    """
    per_head = {}

    for (l, comp, h), data in M_data.items():
        M = data["M"]
        v_h = data["tensor"]
        M_eigvals = data["M_eigenvalues"]  # descending
        T = v_h.shape[0]

        perm = perms[l][comp][h]
        inv_perm = torch.argsort(perm)

        # Deterministic quantize (matches caba_eval)
        v_perm = v_h[:, perm]
        n_blocks = head_dim // block_size
        blocks = v_perm.reshape(T, n_blocks, block_size)
        blocks_qd = quantize_uniform_blocks(blocks, bits=4)
        v_hat_perm = blocks_qd.reshape(T, head_dim)
        v_hat = v_hat_perm[:, inv_perm]

        e = v_hat - v_h
        Sigma = (e.T @ e) / T

        # Eigenvalues of Σ (descending)
        Sigma_eigvals = torch.linalg.eigvalsh(Sigma).flip(0)

        # Actual risk
        actual = torch.trace(M @ Sigma).item()

        # HLP bounds (Theorem C(ii))
        a = M_eigvals      # descending
        b = Sigma_eigvals   # descending
        upper = (a * b).sum().item()
        lower = (a * b.flip(0)).sum().item()
        span = upper - lower
        position = (actual - lower) / span if span > 1e-20 else 0.5

        # M anisotropy: top eigenvalue fraction
        M_top1 = (a[0] / a.sum()).item() if a.sum() > 0 else 0
        # Σ anisotropy
        Sigma_top1 = (b[0] / b.sum()).item() if b.sum() > 0 else 0

        per_head[(l, comp, h)] = {
            "actual": actual,
            "upper": upper,
            "lower": lower,
            "position": position,
            "M_top1_frac": M_top1,
            "Sigma_top1_frac": Sigma_top1,
            "M_trace": a.sum().item(),
            "Sigma_trace": b.sum().item(),
        }

    return per_head


# ============================================================
# Uniqueness test: rotate Σ, observe risk variation
# ============================================================

def uniqueness_test(
    M: torch.Tensor,
    Sigma_aniso: torch.Tensor,
    n_rotations: int = 200,
) -> Dict:
    """Theorem C(i) uniqueness test on a single (M, Σ) pair.

    For K random orthogonal U:
      - Compute tr(M · U Σ_aniso U^T) → varies with U
      - Compute tr(M · σ²I) = σ² tr(M) → constant

    Returns distribution statistics.
    """
    d = M.shape[0]
    sigma2 = Sigma_aniso.trace().item() / d  # isotropic equivalent: same total energy

    iso_risk = sigma2 * M.trace().item()  # constant for all rotations

    aniso_risks = []
    for _ in range(n_rotations):
        U, _ = torch.linalg.qr(torch.randn(d, d))  # random orthogonal
        Sigma_rotated = U @ Sigma_aniso @ U.T
        risk = torch.trace(M @ Sigma_rotated).item()
        aniso_risks.append(risk)

    aniso_risks = np.array(aniso_risks)

    return {
        "iso_risk": iso_risk,
        "aniso_mean": float(np.mean(aniso_risks)),
        "aniso_std": float(np.std(aniso_risks)),
        "aniso_min": float(np.min(aniso_risks)),
        "aniso_max": float(np.max(aniso_risks)),
        "aniso_range_ratio": float((np.max(aniso_risks) - np.min(aniso_risks)) / iso_risk)
            if iso_risk > 1e-20 else 0.0,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT Theorem C: Isotropic Safety Validation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--caba", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--n-rotations", type=int, default=200,
                        help="Rotations for uniqueness test (default: 200)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"isotropic_safety_{model_tag}.json")

    print(f"Model:       {args.model}")
    print(f"Rotations:   {args.n_rotations}")
    print(f"Output:      {args.output}")

    # ---- Load model ----
    print("\nLoading model...")
    model, tokenizer = load_model(args.model, load_in_8bit=not args.no_8bit)

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Calibration data + M measurement ----
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids

    print(f"\n{'='*60}")
    print("Phase 1: Measuring M")
    print(f"{'='*60}")
    measurer = SensitivityMeasurer(model, num_kv_heads, head_dim)
    M_data, _ = measurer.measure(
        all_input_ids, seq_len=args.seq_len, n_chunks=args.n_chunks
    )
    print(f"  {len(M_data)} entries")

    del model
    torch.cuda.empty_cache()
    print("  GPU freed")

    # ---- Permutations ----
    print(f"\n{'='*60}")
    print("Phase 2: Permutation sets")
    print(f"{'='*60}")
    perm_sets = {"baseline": make_identity_permutations(num_layers, num_kv_heads, head_dim)}
    if args.caba is not None:
        perm_sets["sorted"] = load_sorted_permutations(args.caba)
    perm_sets["random_s42"] = make_random_permutations(num_layers, num_kv_heads, head_dim, seed=42)
    print(f"  Modes: {list(perm_sets.keys())}")

    # ---- Phase 3: HLP bounds per permutation ----
    print(f"\n{'='*60}")
    print("Phase 3: Spectral bounds (Theorem C(ii))")
    print(f"{'='*60}")
    all_spectral = {}
    for mode, perms in perm_sets.items():
        t0 = time.time()
        per_head = compute_spectral_metrics(M_data, perms, head_dim)

        positions = [v["position"] for v in per_head.values()]
        M_top1s = [v["M_top1_frac"] for v in per_head.values()]
        Sigma_top1s = [v["Sigma_top1_frac"] for v in per_head.values()]

        all_spectral[mode] = {
            "per_head": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in per_head.items()},
            "position_mean": float(np.mean(positions)),
            "position_std": float(np.std(positions)),
            "M_top1_mean": float(np.mean(M_top1s)),
            "Sigma_top1_mean": float(np.mean(Sigma_top1s)),
        }

        elapsed = time.time() - t0
        print(f"  {mode:16s}: HLP position={np.mean(positions):.3f}±{np.std(positions):.3f}  "
              f"M_top1={np.mean(M_top1s):.3f}  Σ_top1={np.mean(Sigma_top1s):.3f}  "
              f"({elapsed:.1f}s)")

    # ---- Phase 4: Uniqueness test (Theorem C(i)) ----
    # Design: for one representative head, test ALL combinations of:
    #   - Σ from each permutation (baseline, sorted, random) → "not one Σ's accident"
    #   - M types: actual, rank-1, near-isotropic → "not one M's accident"
    # This demonstrates that basis-dependence is structural.
    print(f"\n{'='*60}")
    print(f"Phase 4: Uniqueness test (Theorem C(i), {args.n_rotations} rotations)")
    print(f"{'='*60}")

    # Pick head with most anisotropic M
    M_aniso_scores = {
        key: data["M_eigenvalues"][0].item() / data["M_eigenvalues"].sum().item()
        for key, data in M_data.items()
        if data["M_eigenvalues"].sum().item() > 0
    }
    best_key = max(M_aniso_scores, key=M_aniso_scores.get)
    l, comp, h = best_key
    data = M_data[best_key]
    M_actual = data["M"]
    v_h = data["tensor"]
    T = v_h.shape[0]
    d = head_dim
    print(f"  Test head: L{l}_{comp}_H{h} (M_top1={M_aniso_scores[best_key]:.3f})")

    # Build Σ for each permutation mode
    sigmas = {}
    for mode, perms in perm_sets.items():
        perm = perms[l][comp][h]
        inv_perm = torch.argsort(perm)
        v_perm = v_h[:, perm]
        n_blocks = d // 8
        blocks = v_perm.reshape(T, n_blocks, 8)
        blocks_qd = quantize_uniform_blocks(blocks, bits=4)
        v_hat_perm = blocks_qd.reshape(T, d)
        v_hat = v_hat_perm[:, inv_perm]
        e = v_hat - v_h
        sigmas[mode] = (e.T @ e) / T

    # Build M variants
    M_variants = {"actual": M_actual}

    # Rank-1: project onto top eigenvector of M_actual
    eigvals, eigvecs = torch.linalg.eigh(M_actual)
    top_vec = eigvecs[:, -1]
    M_variants["rank1"] = torch.outer(top_vec, top_vec) * M_actual.trace()

    # Near-isotropic: M ≈ (tr(M)/d) I + small perturbation
    M_variants["near_iso"] = (M_actual.trace() / d) * torch.eye(d) * 0.95 + M_actual * 0.05

    uniqueness_results = {}
    print(f"\n  {'M_type':<12s} {'Σ_source':<12s} {'iso_risk':>12s} "
          f"{'aniso_std':>12s} {'range/iso':>10s}")
    print(f"  {'-'*62}")

    for m_label, M_test in M_variants.items():
        for s_label, Sigma in sigmas.items():
            result = uniqueness_test(M_test, Sigma, n_rotations=args.n_rotations)
            key = f"{m_label}_{s_label}"
            uniqueness_results[key] = {
                **result,
                "M_type": m_label,
                "Sigma_source": s_label,
            }
            print(f"  {m_label:<12s} {s_label:<12s} {result['iso_risk']:>12.4e} "
                  f"{result['aniso_std']:>12.4e} {result['aniso_range_ratio']:>10.2f}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f"THEOREM C VALIDATION — {model_tag}")
    print(f"{'='*80}")

    print(f"\n(a) HLP Bound Position: where does actual tr(MΣ) fall in [lower, upper]?")
    print(f"{'Mode':<20s} {'mean':>8s} {'std':>8s}  (0=counter-aligned, 1=co-aligned)")
    print("-" * 50)
    for mode, data in sorted(all_spectral.items()):
        print(f"{mode:<20s} {data['position_mean']:>8.3f} {data['position_std']:>8.3f}")

    print(f"\n(b) Uniqueness: risk variation under random rotation of Σ")
    print(f"    For each (M_type, Σ_source), range/iso shows how much risk varies")
    print(f"    with Σ orientation.  range/iso > 0 for ALL anisotropic Σ confirms C(i).")
    print(f"\n  {'M_type':<12s} {'Σ_source':<12s} {'range/iso':>10s}")
    print(f"  {'-'*38}")
    for key, res in sorted(uniqueness_results.items()):
        print(f"  {res['M_type']:<12s} {res['Sigma_source']:<12s} {res['aniso_range_ratio']:>10.2f}")
    print(f"\n    Theorem C(i): ONLY Σ=σ²I gives range/iso=0 for ALL M."
          f"\n    If range/iso > 0 for all rows → anisotropic Σ is basis-dependent.")

    print(f"\n(c) M anisotropy (eigenvalue concentration):")
    print(f"  Mean top-1 eigenvalue fraction: {all_spectral['baseline']['M_top1_mean']:.3f}")
    print(f"  (1/d={1/head_dim:.4f} would be perfectly isotropic)")

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "config": {
            "seq_len": args.seq_len,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "n_rotations": args.n_rotations,
        },
        "spectral": {k: {kk: vv for kk, vv in v.items() if kk != "per_head"}
                     for k, v in all_spectral.items()},
        "uniqueness": uniqueness_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()