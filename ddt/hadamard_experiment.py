"""
DDT — Hadamard Rotation Experiment
====================================
Intervention test motivated by Proposition 7 (Isotropic Safety):
applying Walsh-Hadamard pre-rotation before block quantization
tends to homogenize blockwise energy, reducing permutation-induced
ΔL variation.

This experiment does NOT directly verify that Σ becomes isotropic.
It measures:
  1. Whether ΔL variation (std, range) across permutations decreases
  2. Whether worst-case ΔL decreases
  3. An anisotropy proxy (block RMS coefficient of variation) before/after

The Hadamard approach is conceptually related to rotation-based weight
quantization (QuIP, HIGGS, SpinQuant), which isotropize the sensitivity
side; here we apply rotation to the error side (KV cache).

Usage:
  python -m ddt.hadamard_experiment \\
      --model meta-llama/Llama-3.1-8B \\
      --bits 3 4 --n-configs 25
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from scipy.stats import spearmanr as scipy_spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Walsh-Hadamard Transform
# ============================================================

def fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """In-place Fast Walsh-Hadamard Transform on last dimension.

    O(n log n). Input (..., d) where d is power of 2.
    Normalized by 1/sqrt(d) so WHT is self-inverse: WHT(WHT(x)) = x.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be power of 2"
    h = 1
    result = x.clone()
    while h < d:
        for i in range(0, d, h * 2):
            a = result[..., i:i+h].clone()
            b = result[..., i+h:i+2*h].clone()
            result[..., i:i+h] = a + b
            result[..., i+h:i+2*h] = a - b
        h *= 2
    return result / math.sqrt(d)


# ============================================================
# Helpers
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


def quantize_uniform_blocks(x: torch.Tensor, bits: int, alpha: float = 3.0):
    """Per-block RMS-shared symmetric uniform quantize-dequantize."""
    n_levels = 2 ** bits
    half = n_levels / 2
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    scale = alpha * rms / half
    x_scaled = x / scale
    x_quant = torch.round(x_scaled.clamp(-half, half - 1))
    return x_quant * scale


def spearman_corr(x, y):
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(xa) < 3:
        return float("nan"), float("nan")
    if HAS_SCIPY:
        rho, p = scipy_spearmanr(xa, ya)
        return float(rho), float(p)
    n = len(xa)
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)), float("nan")


# ============================================================
# Anisotropy proxy: block RMS coefficient of variation
# ============================================================

def measure_block_rms_cv(
    model,
    input_ids: torch.Tensor,
    perms: Dict,
    num_kv_heads: int,
    head_dim: int,
    use_hadamard: bool = False,
    block_size: int = 8,
) -> float:
    """Measure block-quantization-aligned heterogeneity proxy.

    For each head, after optional Hadamard + permutation:
      - Reshape into blocks of block_size
      - Compute per-block RMS
      - CV = std(RMS) / mean(RMS)
    Returns mean CV across all layers x components x heads x tokens.

    This is NOT raw dimension anisotropy — it measures heterogeneity
    of blockwise energy AFTER the full quantization-aligned pipeline
    (Hadamard rotation + permutation + block partition).

    Low CV = homogeneous blocks = quantization error more uniform.
    High CV = heterogeneous blocks = some blocks get coarser quantization.
    """
    num_layers = len(model.model.layers)
    hooks = []
    cv_collector = []

    def make_cv_hook(layer_idx, comp):
        def hook(module, input, output):
            x = output.float()
            B, T, _ = x.shape
            x_heads = x.view(B, T, num_kv_heads, head_dim)

            for h in range(num_kv_heads):
                perm = perms[layer_idx][comp][h].to(x.device)
                v_h = x_heads[:, :, h, :]  # [B, T, d_h]

                if use_hadamard:
                    v_h = fast_walsh_hadamard(v_h)

                v_perm = v_h[:, :, perm]
                n_blocks = head_dim // block_size
                blocks = v_perm.reshape(B, T, n_blocks, block_size)

                # Per-block RMS: [B, T, n_blocks]
                block_rms = torch.sqrt((blocks ** 2).mean(dim=-1) + 1e-12)

                # CV per (batch, token): std/mean across blocks
                rms_mean = block_rms.mean(dim=-1, keepdim=True)
                rms_std = block_rms.std(dim=-1, keepdim=True, unbiased=False)
                cv = (rms_std / (rms_mean + 1e-12)).mean().item()
                cv_collector.append(cv)

            return output  # don't modify
        return hook

    for idx in range(num_layers):
        layer = model.model.layers[idx]
        attn = layer.self_attn
        hooks.append(attn.k_proj.register_forward_hook(make_cv_hook(idx, "K")))
        hooks.append(attn.v_proj.register_forward_hook(make_cv_hook(idx, "V")))

    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return float(np.mean(cv_collector)) if cv_collector else float("nan")


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
    use_hadamard: bool = False,
    block_size: int = 8,
) -> float:
    """Measure ΔL with optional Hadamard pre-rotation."""
    assert head_dim % block_size == 0

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

                if use_hadamard:
                    v_h = fast_walsh_hadamard(v_h)

                v_perm = v_h[:, :, perm]
                n_blocks = head_dim // block_size
                blocks = v_perm.reshape(B, T, n_blocks, block_size)
                blocks_qd = quantize_uniform_blocks(blocks, bits)
                v_hat_perm = blocks_qd.reshape(B, T, head_dim)
                v_hat = v_hat_perm[:, :, inv_perm]

                if use_hadamard:
                    v_hat = fast_walsh_hadamard(v_hat)  # self-inverse

                x_out[:, :, h, :] = v_hat

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
        description="DDT Hadamard Rotation Experiment"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--n-configs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--chunk-idx", type=int, default=3,
                        help="Calibration chunk index (default: 3, chosen to match current P0 setup)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"hadamard_experiment_{model_tag}.json")

    # ---- Load model ----
    print("Loading model...")
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
    assert head_dim % block_size == 0
    assert head_dim > 0 and (head_dim & (head_dim - 1)) == 0, (
        f"head_dim={head_dim} must be power of 2 for Walsh-Hadamard"
    )
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Calibration data ----
    device = get_model_device(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids

    chunk_start = args.chunk_idx * args.seq_len
    clean_ids = all_input_ids[:, chunk_start:chunk_start + args.seq_len].to(device)

    with torch.no_grad():
        clean_loss = model(clean_ids, labels=clean_ids, use_cache=False).loss.item()
    print(f"  Clean loss: {clean_loss:.4f}")

    # ---- Build permutation configs ----
    perm_configs = {}
    perm_configs["baseline"] = make_identity_permutations(
        num_layers, num_kv_heads, head_dim
    )
    for i in range(args.n_configs - 1):
        seed = 42 + i
        perm_configs[f"random_s{seed}"] = make_random_permutations(
            num_layers, num_kv_heads, head_dim, seed=seed
        )
    print(f"  {len(perm_configs)} permutation configs")

    # ---- WHT self-inverse check ----
    print(f"\n  WHT self-inverse check (d={head_dim})...", end=" ")
    test = torch.randn(1, head_dim)
    reconstructed = fast_walsh_hadamard(fast_walsh_hadamard(test))
    wht_error = (test - reconstructed).abs().max().item()
    print(f"max error = {wht_error:.2e} {'OK' if wht_error < 1e-5 else 'FAIL'}")

    # ---- Measure anisotropy proxy (baseline + 4 random perms) ----
    print("\n  Measuring block RMS CV (anisotropy proxy)...")
    cv_sample_modes = ["baseline"] + [f"random_s{42+i}" for i in range(4)]
    cvs_no_had = []
    cvs_had = []
    for mode in cv_sample_modes:
        if mode not in perm_configs:
            continue
        perms_cv = perm_configs[mode]
        cv_no = measure_block_rms_cv(
            model, clean_ids, perms_cv, num_kv_heads, head_dim,
            use_hadamard=False, block_size=block_size,
        )
        cv_yes = measure_block_rms_cv(
            model, clean_ids, perms_cv, num_kv_heads, head_dim,
            use_hadamard=True, block_size=block_size,
        )
        cvs_no_had.append(cv_no)
        cvs_had.append(cv_yes)
    cv_no_had = float(np.mean(cvs_no_had))
    cv_had = float(np.mean(cvs_had))
    print(f"  Block RMS CV (mean over {len(cvs_no_had)} configs): "
          f"no_had={cv_no_had:.4f}, had={cv_had:.4f} "
          f"(reduction: {(1 - cv_had/cv_no_had)*100:.1f}%)")

    # ---- Run experiments ----
    results = {}

    for bits in args.bits:
        print(f"\n{'='*60}")
        print(f"Bitwidth: {bits}")
        print(f"{'='*60}")

        bit_results = []

        for cfg_idx, (mode, perms) in enumerate(perm_configs.items()):
            t0 = time.time()

            dl_no_had = measure_delta_loss(
                model, clean_ids, perms, bits,
                num_kv_heads, head_dim, clean_loss,
                use_hadamard=False, block_size=block_size,
            )

            dl_had = measure_delta_loss(
                model, clean_ids, perms, bits,
                num_kv_heads, head_dim, clean_loss,
                use_hadamard=True, block_size=block_size,
            )

            elapsed = time.time() - t0

            entry = {
                "mode": mode,
                "bits": bits,
                "dl_no_hadamard": dl_no_had,
                "dl_hadamard": dl_had,
                "dl_reduction": dl_no_had - dl_had,
            }
            bit_results.append(entry)

            print(f"  [{cfg_idx+1:2d}/{len(perm_configs)}] {mode:16s}: "
                  f"no_had={dl_no_had:+.4f}  had={dl_had:+.4f}  "
                  f"delta={dl_no_had - dl_had:+.4f}  ({elapsed:.1f}s)")

        # ---- Summary statistics ----
        dls_no = np.array([r["dl_no_hadamard"] for r in bit_results])
        dls_had = np.array([r["dl_hadamard"] for r in bit_results])

        range_no = float(dls_no.max() - dls_no.min())
        range_had = float(dls_had.max() - dls_had.min())
        range_ratio = range_no / range_had if range_had > 1e-8 else float("inf")

        std_no = float(dls_no.std())
        std_had = float(dls_had.std())
        std_ratio = std_no / std_had if std_had > 1e-8 else float("inf")

        iqr_no = float(np.percentile(dls_no, 75) - np.percentile(dls_no, 25))
        iqr_had = float(np.percentile(dls_had, 75) - np.percentile(dls_had, 25))
        iqr_ratio = iqr_no / iqr_had if iqr_had > 1e-8 else float("inf")

        mean_no = float(dls_no.mean())
        mean_had = float(dls_had.mean())
        worst_no = float(dls_no.max())
        worst_had = float(dls_had.max())
        best_no = float(dls_no.min())
        best_had = float(dls_had.min())

        # Ranking consistency (support stat only)
        rho_ranking, p_ranking = spearman_corr(dls_no.tolist(), dls_had.tolist())

        print(f"\n  --- Summary ({bits}b) ---")
        print(f"  {'':20s} {'No Hadamard':>12s} {'Hadamard':>12s} {'Ratio':>8s}")
        print(f"  {'Mean ΔL':20s} {mean_no:12.4f} {mean_had:12.4f}")
        print(f"  {'Best ΔL':20s} {best_no:12.4f} {best_had:12.4f}")
        print(f"  {'Worst ΔL':20s} {worst_no:12.4f} {worst_had:12.4f}")
        print(f"  {'Std':20s} {std_no:12.4f} {std_had:12.4f} {std_ratio:7.1f}x")
        print(f"  {'IQR':20s} {iqr_no:12.4f} {iqr_had:12.4f} {iqr_ratio:7.1f}x")
        print(f"  {'Range':20s} {range_no:12.4f} {range_had:12.4f} {range_ratio:7.1f}x")
        if worst_no > 1e-6:
            print(f"  Worst-case reduction: {worst_no - worst_had:+.4f} "
                  f"({(worst_no - worst_had)/worst_no*100:+.1f}%)")
        p_str = f"p={p_ranking:.2e}" if not math.isnan(p_ranking) else ""
        print(f"  Ranking rho(no_had, had): {rho_ranking:+.3f} {p_str} [support stat]")

        results[f"{bits}b"] = {
            "configs": bit_results,
            "summary": {
                "mean_no_had": mean_no,
                "mean_had": mean_had,
                "best_no_had": best_no,
                "best_had": best_had,
                "worst_no_had": worst_no,
                "worst_had": worst_had,
                "std_no_had": std_no,
                "std_had": std_had,
                "std_ratio": std_ratio,
                "iqr_no_had": iqr_no,
                "iqr_had": iqr_had,
                "iqr_ratio": iqr_ratio,
                "range_no_had": range_no,
                "range_had": range_had,
                "range_ratio": range_ratio,
                "worst_reduction": float(worst_no - worst_had),
                "worst_reduction_pct": float((worst_no - worst_had) / worst_no * 100)
                    if worst_no > 1e-6 else 0,
                "ranking_rho": float(rho_ranking),
                "ranking_p": float(p_ranking) if not math.isnan(p_ranking) else None,
            },
        }

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "version": "hadamard_experiment_v2",
        "config": {
            "bits": args.bits,
            "n_configs": len(perm_configs),
            "seq_len": args.seq_len,
            "block_size": block_size,
            "chunk_idx": args.chunk_idx,
        },
        "clean_loss": clean_loss,
        "anisotropy_proxy": {
            "block_rms_cv_no_hadamard": cv_no_had,
            "block_rms_cv_hadamard": cv_had,
            "cv_reduction_pct": (1 - cv_had / cv_no_had) * 100 if cv_no_had > 1e-8 else 0,
        },
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()