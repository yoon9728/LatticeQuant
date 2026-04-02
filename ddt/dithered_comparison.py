"""
DDT — Dithered vs Deterministic Comparison (P2)
=================================================
Bridges the gap between theoretical setting (subtractive dither, D2 holds
conditional on block scale) and practical setting (deterministic, D2 violated).

Key outputs:
  1. Ranking consistency: Spearman rho between deterministic DL ranking,
     dithered E[DL] ranking, AND tr(M Sigma) ranking from P0 --
     completes the bridge: theory metric -> dithered -> deterministic
  2. Deterministic-dithered gap: quantifies excess loss from D2(a) violation.
     NOTE: this gap includes bias drift, variance structure differences,
     AND higher-order remainder differences -- it is NOT purely bias drift.
  3. Deterministic consistency check: det_dl_fresh vs P0 det_dl_p0

What this does NOT do:
  - Per-trial Q1 measurement under dithering (too expensive; would require
    gradient computation per trial). Mean-zero property of Q1 under dither
    is validated separately in Experiment B (variance_additivity.py).

Usage:
  python -m ddt.dithered_comparison \\
      --model meta-llama/Llama-3.1-8B \\
      --det-json results/ddt/caba_explain_v2_Llama-3.1-8B.json \\
      --bits 3 4 --n-configs 25 --n-dither-trials 50
"""

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Scipy for proper Spearman (tie handling + p-value)
try:
    from scipy.stats import spearmanr as scipy_spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Spearman with scipy fallback
# ============================================================

def spearman_corr(x, y):
    """Spearman rho with p-value. Uses scipy if available (tie-safe)."""
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(xa) < 3:
        return float("nan"), float("nan")
    if HAS_SCIPY:
        rho, p = scipy_spearmanr(xa, ya)
        return float(rho), float(p)
    # Manual fallback (no ties)
    n = len(xa)
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    d = rx - ry
    rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return float(rho), float("nan")


# ============================================================
# Permutation builders
# ============================================================

def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def make_identity_permutations(num_layers, num_kv_heads, head_dim):
    perms = {}
    for l in range(num_layers):
        perms[l] = {}
        for comp in ["K", "V"]:
            perms[l][comp] = {}
            for h in range(num_kv_heads):
                perms[l][comp][h] = torch.arange(head_dim)
    return perms


def make_random_permutations(num_layers, num_kv_heads, head_dim, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    perms = {}
    for l in range(num_layers):
        perms[l] = {}
        for comp in ["K", "V"]:
            perms[l][comp] = {}
            for h in range(num_kv_heads):
                perms[l][comp][h] = torch.randperm(head_dim, generator=g)
    return perms


# ============================================================
# Quantizers
# ============================================================

def quantize_deterministic(x: torch.Tensor, bits: int, alpha: float = 3.0) -> torch.Tensor:
    """Per-block deterministic uniform quantize-dequantize."""
    n_levels = 2 ** bits
    half = n_levels / 2
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    scale = alpha * rms / half
    x_scaled = x / scale
    x_quant = torch.round(x_scaled.clamp(-half, half - 1))
    return x_quant * scale


def quantize_dithered(
    x: torch.Tensor, bits: int, alpha: float = 3.0, generator: torch.Generator = None
) -> torch.Tensor:
    """Per-block subtractive-dither uniform quantize-dequantize.

    Implements the subtractive dither construction:
      1. Draw U ~ Uniform(-scale/2, scale/2) per element
      2. Quantize (x + U) with rounding
      3. Subtract U from the quantized output

    IMPORTANT: The block scale is data-dependent (RMS-based), so the
    Crypto Lemma holds conditional on the chosen block scale, not
    unconditionally. This is a standard high-rate surrogate used in
    practical lattice quantization experiments.

    Within each block (conditioned on scale, in the non-saturating regime):
      - E[e | scale] ~= 0  (approximate conditional unbiasedness)
      - e | scale ~ approximately Uniform on [-scale/2, scale/2]
    Clipping (clamping to [-half, half-1]) breaks exact uniformity at the
    boundary, but this affects < 0.3% of data at alpha=3."""
    n_levels = 2 ** bits
    half = n_levels / 2
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    scale = alpha * rms / half

    U = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    U = (U - 0.5) * scale

    x_dithered = x + U
    x_scaled = x_dithered / scale
    x_quant = torch.round(x_scaled.clamp(-half, half - 1))
    x_hat = x_quant * scale - U
    return x_hat


# ============================================================
# Delta-loss measurement
# ============================================================

@torch.no_grad()
def measure_delta_loss(
    model,
    input_ids: torch.Tensor,
    perms: Dict,
    bits: int,
    num_kv_heads: int,
    head_dim: int,
    clean_loss: float,
    dithered: bool = False,
    dither_seed: int = 0,
    block_size: int = 8,
) -> float:
    """Measure DL with either deterministic or dithered quantization."""
    assert head_dim % block_size == 0, (
        f"head_dim={head_dim} not divisible by block_size={block_size}"
    )

    hooks = []
    num_layers = len(model.model.layers)

    def make_quant_hook(layer_idx, comp):
        def hook(module, input, output):
            x = output.float()
            B, T, _ = x.shape
            x_heads = x.view(B, T, num_kv_heads, head_dim)
            x_out = torch.zeros_like(x_heads)

            for h in range(num_kv_heads):
                perm = perms[layer_idx][comp][h].to(x.device)
                inv_perm = torch.argsort(perm)
                v_h = x_heads[:, :, h, :]
                v_perm = v_h[:, :, perm]
                n_blocks = head_dim // block_size
                blocks = v_perm.reshape(B, T, n_blocks, block_size)

                if dithered:
                    g = torch.Generator(device=x.device)
                    g.manual_seed(dither_seed * 100000 + layer_idx * 1000 +
                                  (0 if comp == "K" else 500) + h)
                    blocks_qd = quantize_dithered(blocks, bits, generator=g)
                else:
                    blocks_qd = quantize_deterministic(blocks, bits)

                v_hat_perm = blocks_qd.reshape(B, T, head_dim)
                x_out[:, :, h, :] = v_hat_perm[:, :, inv_perm]

            return x_out.reshape(B, T, -1).to(output.dtype)
        return hook

    for idx in range(num_layers):
        layer = model.model.layers[idx]
        attn = layer.self_attn
        hooks.append(attn.k_proj.register_forward_hook(make_quant_hook(idx, "K")))
        hooks.append(attn.v_proj.register_forward_hook(make_quant_hook(idx, "V")))

    try:
        outputs = model(input_ids, labels=input_ids, use_cache=False)
        quant_loss = outputs.loss.item()
    finally:
        for h in hooks:
            h.remove()

    return quant_loss - clean_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT P2: Dithered vs Deterministic Comparison"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--det-json", type=str, required=True,
                        help="P0 JSON with deterministic results")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--n-configs", type=int, default=25)
    parser.add_argument("--n-dither-trials", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--det-consistency-tol", type=float, default=0.05,
                        help="Relative tolerance for det_fresh vs det_p0 (default 5%%)")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"dithered_comparison_{model_tag}.json")

    # ---- Load P0 deterministic results ----
    print("Loading P0 deterministic results...")
    with open(args.det_json) as f:
        det_data = json.load(f)
    det_configs = det_data["config_list"]
    print(f"  {len(det_configs)} configs loaded from {args.det_json}")

    # ---- Load model ----
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}
    if not args.no_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, **kwargs
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    block_size = 8
    assert head_dim % block_size == 0, f"head_dim={head_dim} % {block_size} != 0"
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Calibration data (same chunk as P0) ----
    device = get_model_device(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids

    n_chunks = det_data["config"]["n_chunks"]
    last_chunk_start = (n_chunks - 1) * args.seq_len
    clean_ids = all_input_ids[:, last_chunk_start:last_chunk_start + args.seq_len].to(device)

    with torch.no_grad():
        clean_loss = model(clean_ids, labels=clean_ids, use_cache=False).loss.item()
    print(f"  Clean loss: {clean_loss:.4f} (P0: {det_data['clean_loss']:.4f})")

    # ---- Build config subset (same P0 seeds) ----
    # baseline + random subset using identical seeds
    perm_configs = {}
    perm_configs["baseline"] = make_identity_permutations(
        num_layers, num_kv_heads, head_dim
    )

    n_random = args.n_configs - 1  # reserve 1 for baseline
    for i in range(n_random):
        seed = 42 + i
        perm_configs[f"random_s{seed}"] = make_random_permutations(
            num_layers, num_kv_heads, head_dim, seed=seed
        )
    print(f"  {len(perm_configs)} configs (baseline + {n_random} random)")

    # ---- Run experiments ----
    results = {}

    for bits in args.bits:
        print(f"\n{'='*60}")
        print(f"Bitwidth: {bits}")
        print(f"{'='*60}")

        bit_results = []

        for cfg_idx, (mode, perms) in enumerate(perm_configs.items()):
            t0 = time.time()

            # --- P0 lookup ---
            det_entry = None
            for c in det_configs:
                if c["mode"] == mode and c["bits"] == bits:
                    det_entry = c
                    break

            det_dl_p0 = det_entry["delta_loss"] if det_entry else None
            tr_m_sigma_p0 = det_entry.get("tr_M_Sigma") if det_entry else None

            # --- Deterministic DL fresh ---
            det_dl_fresh = measure_delta_loss(
                model, clean_ids, perms, bits,
                num_kv_heads, head_dim, clean_loss,
                dithered=False, block_size=block_size,
            )

            # --- Consistency check ---
            if det_dl_p0 is not None and abs(det_dl_p0) > 1e-6:
                rel_diff = abs(det_dl_fresh - det_dl_p0) / abs(det_dl_p0)
                if rel_diff > args.det_consistency_tol:
                    warnings.warn(
                        f"[{mode} {bits}b] det consistency: fresh={det_dl_fresh:.4f} "
                        f"vs P0={det_dl_p0:.4f} (rel_diff={rel_diff:.1%})"
                    )

            # --- Dithered DL (N trials) ---
            dith_dls = []
            for trial in range(args.n_dither_trials):
                dl = measure_delta_loss(
                    model, clean_ids, perms, bits,
                    num_kv_heads, head_dim, clean_loss,
                    dithered=True, dither_seed=trial,
                    block_size=block_size,
                )
                dith_dls.append(dl)

            dith_mean = float(np.mean(dith_dls))
            dith_std = float(np.std(dith_dls))
            dith_se = float(dith_std / np.sqrt(len(dith_dls)))

            # --- Det-dith gap (NOT purely bias drift — see docstring) ---
            det_dith_gap = det_dl_fresh - dith_mean

            elapsed = time.time() - t0

            entry = {
                "mode": mode,
                "bits": bits,
                "det_dl_p0": det_dl_p0,
                "det_dl_fresh": det_dl_fresh,
                "tr_M_Sigma_p0": tr_m_sigma_p0,
                "dith_mean": dith_mean,
                "dith_std": dith_std,
                "dith_se": dith_se,
                "det_dith_gap": float(det_dith_gap),
                "n_trials": len(dith_dls),
            }
            bit_results.append(entry)

            print(f"  [{cfg_idx+1:2d}/{len(perm_configs)}] {mode:16s}: "
                  f"det={det_dl_fresh:+.4f}  "
                  f"dith={dith_mean:+.4f}+/-{dith_se:.4f}  "
                  f"gap={det_dith_gap:+.4f}  "
                  f"({elapsed:.1f}s)")

        # --- Ranking comparisons (the bridge) ---
        # Core set: configs with dithered measurements
        valid = [r for r in bit_results
                 if r["det_dl_fresh"] is not None
                 and r["tr_M_Sigma_p0"] is not None]

        # Extended set: add P0-only configs (e.g. sorted) for tr(MΣ) vs det
        # These contribute to ρ(tr(MΣ), det) but not ρ(det, dith)
        # Use explicit det_for_rank / source fields to avoid semantic confusion
        p0_only_modes = set()
        for c in det_configs:
            if (c["bits"] == bits
                    and c["mode"] not in perm_configs
                    and c["delta_loss"] is not None
                    and c.get("tr_M_Sigma") is not None):
                p0_only_modes.add(c["mode"])

        extended = []
        for r in valid:
            extended.append({
                "det_for_rank": r["det_dl_fresh"],
                "tr_M_Sigma_p0": r["tr_M_Sigma_p0"],
                "mode": r["mode"],
                "source": "fresh",
            })
        for c in det_configs:
            if c["bits"] == bits and c["mode"] in p0_only_modes:
                extended.append({
                    "det_for_rank": c["delta_loss"],
                    "tr_M_Sigma_p0": c["tr_M_Sigma"],
                    "mode": c["mode"],
                    "source": "p0_only",
                })

        if len(valid) >= 5:
            det_dls = [r["det_dl_fresh"] for r in valid]
            dith_means = [r["dith_mean"] for r in valid]
            trms_vals = [r["tr_M_Sigma_p0"] for r in valid]

            rho_det_dith, p_det_dith = spearman_corr(det_dls, dith_means)
            rho_trms_dith, p_trms_dith = spearman_corr(trms_vals, dith_means)

            # Extended: tr(MΣ) vs det (includes sorted if available)
            ext_det = [r["det_for_rank"] for r in extended]
            ext_trms = [r["tr_M_Sigma_p0"] for r in extended]
            rho_trms_det, p_trms_det = spearman_corr(ext_trms, ext_det)

            n_ext = len(extended)
            n_p0_only = sum(1 for r in extended if r["source"] == "p0_only")
            ext_note = f" (+{n_p0_only} P0-only)" if n_p0_only > 0 else ""

            print(f"\n  --- Ranking Consistency ({bits}b, n={len(valid)}) ---")
            p_str = lambda p: f"p={p:.2e}" if p is not None and not math.isnan(p) else ""
            print(f"  rho(det, dith)    = {rho_det_dith:+.3f}  {p_str(p_det_dith)}")
            print(f"  rho(tr(MS), det)  = {rho_trms_det:+.3f}  {p_str(p_trms_det)}  [n={n_ext}{ext_note}]")
            print(f"  rho(tr(MS), dith) = {rho_trms_dith:+.3f}  {p_str(p_trms_dith)}")

            ranking = {
                "n_configs_dithered": len(valid),
                "n_configs_extended": n_ext,
                "rho_det_dith": float(rho_det_dith),
                "rho_trms_det": float(rho_trms_det),
                "rho_trms_dith": float(rho_trms_dith),
                "p_det_dith": float(p_det_dith) if not math.isnan(p_det_dith) else None,
                "p_trms_det": float(p_trms_det) if not math.isnan(p_trms_det) else None,
                "p_trms_dith": float(p_trms_dith) if not math.isnan(p_trms_dith) else None,
            }
        else:
            print(f"\n  Ranking: insufficient valid configs ({len(valid)})")
            ranking = {"n_configs": len(valid), "error": "insufficient"}

        # --- Gap statistics (single comprehension to avoid misalignment) ---
        gap_entries = [
            (r["det_dith_gap"], abs(r["det_dl_fresh"]))
            for r in bit_results
            if r["det_dl_fresh"] is not None
        ]
        gaps = [g for g, _ in gap_entries]
        gap_fracs = [abs(g) / d if d > 1e-6 else 0 for g, d in gap_entries]

        print(f"\n  --- Det-Dith Gap ({bits}b) ---")
        print(f"  Mean |gap| / |DL_det|: {np.mean(gap_fracs):.1%}")
        print(f"  Mean gap: {np.mean(gaps):+.4f}")

        results[f"{bits}b"] = {
            "configs": bit_results,
            "ranking": ranking,
            "gap_mean": float(np.mean(gaps)),
            "gap_frac_mean": float(np.mean(gap_fracs)),
        }

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "version": "P2_dithered_comparison_v2",
        "config": {
            "bits": args.bits,
            "n_configs": len(perm_configs),
            "n_dither_trials": args.n_dither_trials,
            "seq_len": args.seq_len,
            "det_consistency_tol": args.det_consistency_tol,
        },
        "clean_loss": clean_loss,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()