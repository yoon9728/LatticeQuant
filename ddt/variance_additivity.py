"""
DDT — Theorem B Validation: Variance Additivity
==================================================
Validates two claims of DDT Theorem B(ii):

(a) Variance additivity (cross-term vanishing):
    Under independent dithering (D1) + mean-zero error (D2a),
    the MDS structure yields:

      Var(Σ_{l,j} X_{l,j})  =  Σ_{l,j} Var(X_{l,j})

    where X_{l,j} = s^l_j · e^l_j is the exact Theorem B object.
    Measured by running N dithered quantization trials.
    Position-level additivity is the direct operational validation of
    the theorem's MDS structure; head-level additivity is a stronger
    empirical check (finer decomposition than theorem requires).

(b) Analytical variance prediction (Crypto Lemma):
    On the integer lattice Z with step Δ = rms (per-block):

      Var(X_total) = Σ_{l,h,j,d} s_{j,d}² × rms²(block) / 12

    Computed analytically under the dithered Z-lattice model.
    The Crypto Lemma gives error variance per dimension as rms²/12
    (Z is unbounded, so no overload by construction).

Dithering:
    Subtractive dither on Z lattice with step = rms:
      x_scaled = x / rms
      x_hat_scaled = round(x_scaled + U) - U,  U ~ Uniform(-0.5, 0.5)
    The Crypto Lemma holds exactly for this dithered Z-lattice
    surrogate (no overload by construction).  This validates the
    theorem's dithered model; deployed quantizers with finite-bit
    clipping are a separate question.

Usage:
  python -m ddt.variance_additivity \\
      --model meta-llama/Llama-3.1-8B \\
      --caba results/caba_llama_3.1_8b.json

  python -m ddt.variance_additivity \\
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
)


# ============================================================
# Dithered block quantizer (subtractive dither)
# ============================================================

def quantize_dithered_blocks(
    x: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Subtractive dither on the integer lattice Z with step size Δ = rms.

    Construction:
      1. Normalize: x_scaled = x / rms
      2. Subtractive dither:
           x_dithered = x_scaled + U,   U ~ Uniform(-0.5, 0.5)
           x_quant = round(x_dithered)  ∈ Z  (unbounded integers)
           x_hat_scaled = x_quant - U
      3. Invert: x_hat = x_hat_scaled × rms

    Error per dim: U(-0.5, 0.5) × rms.
    Variance per dim: rms² / 12.
    The Crypto Lemma holds exactly on the infinite integer lattice
    — no overload by construction (Z is unbounded).

    Returns:
        quantized_dequantized tensor, same shape as x.
    """
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    x_scaled = x / rms

    dither = torch.rand(x_scaled.shape, generator=generator, device=x.device) - 0.5
    x_dithered = x_scaled + dither
    x_quant = torch.round(x_dithered)
    x_hat_scaled = x_quant - dither

    return x_hat_scaled * rms


# ============================================================
# Single dithered trial
# ============================================================

def run_trial(
    M_data: Dict,
    perms: Dict,
    head_dim: int,
    seed: int,
    block_size: int = 8,
) -> Tuple[Dict, Dict, float]:
    """One dithered quantization trial on Z lattice (step=rms).

    Returns:
        per_head_X:  {(l, comp, h): float}  — head-level X_{l,h} = Σ_j s_j·e_j
        per_pos_X:   {l: np.array[T]}       — position-level X_{l,j} = Σ_{comp,h} s·e
                     This is the exact object in Theorem B(ii).
        total_X:     float
    """
    g = torch.Generator()
    g.manual_seed(seed)

    per_head_X = {}
    per_pos_X = {}
    total_X = 0.0

    for (l, comp, h), data in M_data.items():
        assert "grad" in data, f"M_data missing 'grad' for ({l},{comp},{h}). Update caba_explain.py."
        v_h = data["tensor"]
        g_h = data["grad"]
        T = v_h.shape[0]

        perm = perms[l][comp][h]
        inv_perm = torch.argsort(perm)

        v_perm = v_h[:, perm]
        n_blocks = head_dim // block_size
        blocks = v_perm.reshape(-1, n_blocks, block_size)
        blocks_qd = quantize_dithered_blocks(blocks, g)
        v_hat_perm = blocks_qd.reshape(T, head_dim)
        v_hat = v_hat_perm[:, inv_perm]

        e = v_hat - v_h

        X_lh = (g_h * e).sum().item()
        per_head_X[(l, comp, h)] = X_lh

        pos_contrib = (g_h * e).sum(dim=1).cpu().numpy()
        if l not in per_pos_X:
            per_pos_X[l] = np.zeros(T)
        per_pos_X[l] += pos_contrib

        total_X += X_lh

    return per_head_X, per_pos_X, total_X


# ============================================================
# Analytical Var prediction via Crypto Lemma (exact, no Monte Carlo)
# ============================================================

def compute_analytical_predicted_var(
    M_data: Dict,
    perms: Dict,
    head_dim: int,
    block_size: int = 8,
) -> float:
    """Theorem B(ii) predicted variance, analytic under dithered Z-lattice model.

    On Z lattice with step Δ=rms:
      Error variance per dim = rms²(block) / 12

    Predicted variance:
      Σ_{l,h,j,d} s_{j,d}² × rms²(block of d at j) / 12
    """
    total_predicted = 0.0

    for (l, comp, h), data in M_data.items():
        v_h = data["tensor"]
        g_h = data["grad"]
        T = v_h.shape[0]

        perm = perms[l][comp][h]
        inv_perm = torch.argsort(perm)

        v_perm = v_h[:, perm]
        n_blocks = head_dim // block_size
        blocks = v_perm.reshape(T, n_blocks, block_size)
        rms2 = (blocks ** 2).mean(dim=-1)  # [T, n_blocks]

        # σ²_e per dim = rms²/12
        var_perm = (rms2 / 12.0).unsqueeze(-1).expand_as(blocks)
        var_perm = var_perm.reshape(T, head_dim)
        var_orig = var_perm[:, inv_perm]

        contribution = (g_h ** 2 * var_orig).sum().item()
        total_predicted += contribution

    return total_predicted


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT Theorem B: Variance Additivity Validation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--caba", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Chunks for M (default: 1 — Theorem B needs "
                             "fixed source, not averaged M)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of dithered trials (default: 100)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"variance_additivity_{model_tag}.json")

    print(f"Model:    {args.model}")
    print(f"Trials:   {args.n_trials}")
    print(f"Output:   {args.output}")

    # ---- Load model ----
    print("\nLoading model...")
    model, tokenizer = load_model(args.model, load_in_8bit=not args.no_8bit)

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Calibration data ----
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids

    # ---- Measure M + grads (single chunk: fixed source, varied dither) ----
    # Theorem B validation needs: same source data → repeated dither trials.
    # M averaging helps stability in caba_explain, but here we want M, grad,
    # and tensor all from the SAME chunk for exact consistency.
    print(f"\n{'='*60}")
    print(f"Phase 1: Measuring M + grads (single chunk, fixed source)")
    print(f"{'='*60}")
    measurer = SensitivityMeasurer(model, num_kv_heads, head_dim)
    M_data, _ = measurer.measure(
        all_input_ids, seq_len=args.seq_len, n_chunks=args.n_chunks
    )
    n_heads_total = len(M_data)
    T = next(iter(M_data.values()))["tensor"].shape[0]  # actual sequence length from data
    print(f"  {n_heads_total} (layer, comp, head) entries, T={T}")

    # Free GPU memory — all remaining work is CPU
    del model
    torch.cuda.empty_cache()
    print("  GPU freed — remaining computation is CPU only")

    # ---- Build permutation sets ----
    print(f"\n{'='*60}")
    print("Phase 2: Permutation sets")
    print(f"{'='*60}")
    perm_sets = {"baseline": make_identity_permutations(num_layers, num_kv_heads, head_dim)}
    if args.caba is not None:
        perm_sets["sorted"] = load_sorted_permutations(args.caba)
    for rs in range(3):
        perm_sets[f"random_s{42+rs}"] = make_random_permutations(
            num_layers, num_kv_heads, head_dim, seed=42 + rs
        )
    print(f"  Modes: {list(perm_sets.keys())}")

    # ---- Run dithered trials ----
    all_results = {}
    print(f"\n{'='*60}")
    print(f"Phase 3: Dithered trials — {args.n_trials} trials, step=rms")
    print(f"{'='*60}")

    for mode, perms in perm_sets.items():
        t0 = time.time()

        trial_totals = []
        trial_per_head = {}
        trial_per_pos = {}

        for trial_idx in range(args.n_trials):
            seed = trial_idx * 7919 + 31
            per_head_X, per_pos_X, total_X = run_trial(
                M_data, perms, head_dim, seed
            )
            trial_totals.append(total_X)
            for key, val in per_head_X.items():
                if key not in trial_per_head:
                    trial_per_head[key] = []
                trial_per_head[key].append(val)
            for l, pos_arr in per_pos_X.items():
                if l not in trial_per_pos:
                    trial_per_pos[l] = []
                trial_per_pos[l].append(pos_arr)

        trial_totals = np.array(trial_totals)

        for l in trial_per_pos:
            trial_per_pos[l] = np.array(trial_per_pos[l])

        # (a1) HEAD-level additivity
        var_total = np.var(trial_totals, ddof=1)
        sum_of_head_vars = 0.0
        per_head_vars = {}
        for key, vals in trial_per_head.items():
            v = np.var(vals, ddof=1)
            per_head_vars[key] = v
            sum_of_head_vars += v
        head_additivity = var_total / sum_of_head_vars if sum_of_head_vars > 0 else float("nan")

        # (a2) POSITION-level additivity — exact Theorem B(ii)
        sum_of_pos_vars = 0.0
        for l, arr in trial_per_pos.items():
            pos_vars = np.var(arr, axis=0, ddof=1)
            sum_of_pos_vars += pos_vars.sum()
        pos_additivity = var_total / sum_of_pos_vars if sum_of_pos_vars > 0 else float("nan")

        # (b) Analytical prediction (Crypto Lemma, exact)
        predicted_var = compute_analytical_predicted_var(M_data, perms, head_dim)
        prediction_ratio = var_total / predicted_var if predicted_var > 0 else float("nan")

        elapsed = time.time() - t0
        mean_total = np.mean(trial_totals)

        var_K = sum(v for (l, c, h), v in per_head_vars.items() if c == "K")
        var_V = sum(v for (l, c, h), v in per_head_vars.items() if c == "V")

        all_results[mode] = {
            "mode": mode,
            "n_trials": args.n_trials,
            "mean_X_total": mean_total,
            "var_total": var_total,
            "pos_additivity": pos_additivity,
            "head_additivity": head_additivity,
            "sum_of_pos_vars": sum_of_pos_vars,
            "sum_of_head_vars": sum_of_head_vars,
            "predicted_var": predicted_var,
            "prediction_ratio": prediction_ratio,
            "var_K": var_K,
            "var_V": var_V,
            "elapsed_sec": elapsed,
        }

        print(f"  {mode:16s}: "
              f"pos_add={pos_additivity:.4f}  "
              f"head_add={head_additivity:.4f}  "
              f"pred={prediction_ratio:.4f}  "
              f"E[X]={mean_total:.6f}  "
              f"({elapsed:.1f}s)")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f"THEOREM B VALIDATION — {model_tag}")
    print(f"{'='*80}")

    print(f"\n(a) Variance Additivity  [expect ≈ 1.0]")
    print(f"{'Config':<20s} {'pos-level':>10s} {'head-level':>10s}  (pos = exact Thm B)")
    print("-" * 50)
    for key, res in sorted(all_results.items()):
        print(f"{key:<20s} {res['pos_additivity']:>10.4f} {res['head_additivity']:>10.4f}")

    print(f"\n(b) Analytical Prediction (Crypto Lemma):  Var(total) / Σ s²·σ²/12  [expect ≈ 1.0]")
    print(f"{'Config':<20s} {'measured':>12s} {'predicted':>12s} {'ratio':>8s}")
    print("-" * 60)
    for key, res in sorted(all_results.items()):
        print(f"{key:<20s} {res['var_total']:>12.4e} {res['predicted_var']:>12.4e} "
              f"{res['prediction_ratio']:>8.4f}")

    print(f"\n(c) Mean-zero check: E[X_total]  [expect ≈ 0]")
    for key, res in sorted(all_results.items()):
        std = np.sqrt(res["var_total"])
        z_score = abs(res["mean_X_total"]) / std if std > 0 else float("inf")
        print(f"  {key:<20s}: E[X]={res['mean_X_total']:.6f}, "
              f"|z|={z_score:.2f} {'✓' if z_score < 2 else '✗ (>2σ)'}")

    print(f"\n(d) K vs V variance breakdown:")
    for key, res in sorted(all_results.items()):
        total = res["var_K"] + res["var_V"]
        k_frac = res["var_K"] / total if total > 0 else 0
        print(f"  {key:<20s}: K={res['var_K']:.4e} ({k_frac:.1%}), V={res['var_V']:.4e}")

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "config": {
            "seq_len": args.seq_len,
            "n_chunks": args.n_chunks,
            "n_trials": args.n_trials,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "results": {k: v for k, v in all_results.items()},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()