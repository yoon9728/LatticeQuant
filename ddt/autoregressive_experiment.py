"""
DDT — Autoregressive KV Cache Reuse Experiment
=================================================
Tests whether prefill-based DDT findings transfer to a chunked
continuation setting with reused KV cache.

Setup:
  1. Prefill chunk 1 → generate clean KV cache
  2. Quantize KV cache (permute → block quantize → unpermute)
  3. Forward chunk 2 with past_key_values = quantized KV
  4. Measure loss on chunk 2

This is a closer approximation to deployment-time KV reuse than
prefill-only perturbation, but is NOT identical to online one-token-
at-a-time decode. It evaluates batched continuation with reused cache.

Chunk alignment with P0:
  - chunk 1 (index 2) = causal context for cache generation
  - chunk 2 (index 3) = evaluation text, same as P0 metric computation
  P0 metrics (tr(MΣ), Q1, MSE) were measured on chunk 2 data, so
  correlations between P0 metrics and ΔL_auto are on matched text.

Key outputs:
  ρ(ΔL_prefill, ΔL_auto) — if high, prefill is a valid proxy
  ρ(tr(MΣ), ΔL_auto) vs ρ(MSE, ΔL_auto) — DDT vs MSE in reuse setting
  ρ(Q1, ΔL_auto) — supplementary (Q1 is signed, interpret with caution)

Requires: transformers >= 4.36 (DynamicCache API)

Usage:
  python -m ddt.autoregressive_experiment \\
      --model meta-llama/Llama-3.1-8B \\
      --det-json results/ddt/caba_explain_v2_Llama-3.1-8B.json \\
      --bits 3 4 --n-configs 25
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Helpers
# ============================================================

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


# ============================================================
# KV cache utilities
# ============================================================

def kv_to_legacy(past_key_values) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Convert any cache format to legacy tuple of (key, value)."""
    # Already a tuple of tuples
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        if isinstance(past_key_values[0], tuple):
            return past_key_values

    # DynamicCache with to_legacy_cache
    if hasattr(past_key_values, 'to_legacy_cache'):
        try:
            return past_key_values.to_legacy_cache()
        except Exception:
            pass

    # DynamicCache with key_cache/value_cache lists
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        return tuple(
            (k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
        )

    raise TypeError(f"Cannot convert {type(past_key_values)} to legacy KV cache")


def legacy_to_cache(kv_tuple):
    """Wrap legacy tuple back to DynamicCache for model compatibility."""
    try:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            # Newer transformers: DynamicCache stores as lists directly
            if hasattr(cache, 'key_cache') and isinstance(cache.key_cache, list):
                cache.key_cache.append(k)
                cache.value_cache.append(v)
            else:
                cache.update(k, v, layer_idx)
        return cache
    except (ImportError, TypeError, AttributeError):
        # Fallback: return as-is (legacy tuple)
        return kv_tuple


def validate_kv_cache(kv_tuple, num_layers: int, num_kv_heads: int,
                      head_dim: int, label: str = "KV cache"):
    """Assert KV cache shape consistency."""
    assert len(kv_tuple) == num_layers, (
        f"{label}: expected {num_layers} layers, got {len(kv_tuple)}")
    k0, v0 = kv_tuple[0]
    assert k0.shape[-2] > 0, f"{label}: empty sequence dimension"
    assert k0.shape[-1] == head_dim, (
        f"{label}: head_dim mismatch: {k0.shape[-1]} vs {head_dim}")
    assert k0.shape[-3] == num_kv_heads, (
        f"{label}: num_kv_heads mismatch: {k0.shape[-3]} vs {num_kv_heads}")


def quantize_kv_cache(
    kv_tuple: Tuple,
    perms: Dict,
    bits: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 8,
) -> Tuple:
    """Quantize a legacy KV cache. Returns new tuple (does not modify input)."""
    assert head_dim % block_size == 0
    quantized = []

    for layer_idx, (key, value) in enumerate(kv_tuple):
        new_key = key.clone()
        new_value = value.clone()

        for h in range(num_kv_heads):
            for comp_name, tensor in [("K", new_key), ("V", new_value)]:
                perm = perms[layer_idx][comp_name][h].to(tensor.device)
                inv_perm = torch.argsort(perm)

                v_h = tensor[:, h, :, :].float()
                v_perm = v_h[:, :, perm]
                n_blocks = head_dim // block_size
                B, T = v_perm.shape[0], v_perm.shape[1]
                blocks = v_perm.reshape(B, T, n_blocks, block_size)
                blocks_qd = quantize_uniform_blocks(blocks, bits)
                v_hat_perm = blocks_qd.reshape(B, T, head_dim)
                v_hat = v_hat_perm[:, :, inv_perm]
                tensor[:, h, :, :] = v_hat.to(tensor.dtype)

        quantized.append((new_key, new_value))

    return tuple(quantized)


# ============================================================
# Sanity check: cache-reuse loss vs full-concatenated loss
# ============================================================

@torch.no_grad()
def sanity_check_cache_reuse(model, chunk1_ids, chunk2_ids, clean_kv_legacy):
    """Verify that cache-reuse loss matches full-concatenated loss.

    Compares:
      Path A: model(chunk2, past_key_values=clean_kv) → loss on chunk2
      Path B: model([chunk1+chunk2]) → loss on chunk2 portion only

    These should match within floating-point tolerance. If they don't,
    the cache-reuse path has a bug (position_ids, attention_mask, etc.).
    """
    device = chunk1_ids.device
    seq_len_c1 = chunk1_ids.shape[1]
    seq_len_c2 = chunk2_ids.shape[1]

    # Path A: cache reuse
    position_ids_c2 = torch.arange(seq_len_c1, seq_len_c1 + seq_len_c2,
                                    device=device).unsqueeze(0)
    attention_mask_c2 = torch.ones(1, seq_len_c1 + seq_len_c2,
                                    device=device, dtype=torch.long)

    out_a = model(
        chunk2_ids,
        past_key_values=legacy_to_cache(clean_kv_legacy),
        attention_mask=attention_mask_c2,
        position_ids=position_ids_c2,
        labels=chunk2_ids,
        use_cache=False,
    )
    loss_a = out_a.loss.item()

    # Path B: full concatenated, measure chunk2 loss only
    full_ids = torch.cat([chunk1_ids, chunk2_ids], dim=1)
    out_b = model(full_ids, use_cache=False)
    logits_b = out_b.logits  # [B, seq_len_c1+seq_len_c2, vocab]

    # Extract chunk2 logits and compute loss manually
    # HF causal LM: predict token[i+1] from logits[i]
    # For chunk2 portion: logits at positions [c1-1, c1, ..., c1+c2-2]
    # predict tokens at positions [c1, c1+1, ..., c1+c2-1]
    chunk2_logits = logits_b[:, seq_len_c1 - 1:seq_len_c1 + seq_len_c2 - 1, :]
    chunk2_labels = full_ids[:, seq_len_c1:seq_len_c1 + seq_len_c2]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_b = loss_fn(
        chunk2_logits.reshape(-1, chunk2_logits.shape[-1]),
        chunk2_labels.reshape(-1),
    ).item()

    gap = abs(loss_a - loss_b)
    ok = gap < 0.01  # tolerance for 8-bit base model

    return loss_a, loss_b, gap, ok


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT Autoregressive KV Cache Reuse Experiment"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--det-json", type=str, required=True,
                        help="P0 JSON for prefill ΔL and DDT metrics")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--n-configs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Abort if sanity check fails (recommended for paper results)")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"autoregressive_experiment_{model_tag}.json")

    # ---- Load P0 results ----
    print("Loading P0 results...")
    with open(args.det_json) as f:
        det_data = json.load(f)
    det_configs = det_data["config_list"]
    print(f"  {len(det_configs)} configs loaded")

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
    assert head_dim % block_size == 0
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Calibration data ----
    device = get_model_device(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids

    # Chunk 1 = context (index 2), Chunk 2 = evaluation (index 3, matches P0)
    chunk1_start = 2 * args.seq_len
    chunk2_start = 3 * args.seq_len
    chunk1_ids = all_input_ids[:, chunk1_start:chunk1_start + args.seq_len].to(device)
    chunk2_ids = all_input_ids[:, chunk2_start:chunk2_start + args.seq_len].to(device)

    print(f"  Chunk 1 (context): tokens [{chunk1_start}, {chunk1_start + args.seq_len})")
    print(f"  Chunk 2 (eval, = P0 chunk): tokens [{chunk2_start}, {chunk2_start + args.seq_len})")

    # ---- Generate clean KV cache from chunk 1 ----
    print("\n  Generating clean KV cache from chunk 1...")
    with torch.no_grad():
        out1 = model(chunk1_ids, use_cache=True)
    clean_kv_legacy = kv_to_legacy(out1.past_key_values)
    validate_kv_cache(clean_kv_legacy, num_layers, num_kv_heads, head_dim,
                      "Clean KV cache")
    print(f"  KV cache: {len(clean_kv_legacy)} layers, "
          f"key shape: {clean_kv_legacy[0][0].shape}")

    # ---- Sanity check: cache-reuse vs full-concatenated ----
    print("\n  Sanity check: cache-reuse loss vs full-concatenated loss...")
    loss_reuse, loss_concat, gap, ok = sanity_check_cache_reuse(
        model, chunk1_ids, chunk2_ids, clean_kv_legacy,
    )
    status = "PASS" if ok else "FAIL"
    print(f"  Cache-reuse loss:  {loss_reuse:.6f}")
    print(f"  Full-concat loss:  {loss_concat:.6f}")
    print(f"  Gap: {gap:.6f}  [{status}]")
    if not ok:
        print("  WARNING: cache-reuse path may have position/mask issues.")
        if args.strict:
            raise RuntimeError("Sanity check FAILED with --strict. Fix position/mask handling.")
        print("  Proceeding, but results should be interpreted with caution.")

    clean_loss_c2 = loss_reuse  # use cache-reuse path as baseline

    # ---- Precompute position_ids and attention_mask ----
    seq_len_c1 = chunk1_ids.shape[1]
    seq_len_c2 = chunk2_ids.shape[1]
    position_ids_c2 = torch.arange(
        seq_len_c1, seq_len_c1 + seq_len_c2, device=device
    ).unsqueeze(0)
    attention_mask_c2 = torch.ones(
        1, seq_len_c1 + seq_len_c2, device=device, dtype=torch.long
    )

    # ---- Build permutation configs (must match P0 exactly) ----
    perm_configs = {}
    perm_configs["baseline"] = make_identity_permutations(
        num_layers, num_kv_heads, head_dim
    )
    # P0 uses seeds 42, 43, ..., 42+N-2 for random configs
    for i in range(args.n_configs - 1):
        seed = 42 + i
        perm_configs[f"random_s{seed}"] = make_random_permutations(
            num_layers, num_kv_heads, head_dim, seed=seed
        )
    # NOTE: P0's "sorted" config requires actual activation data to reconstruct.
    # We exclude it here rather than using a fake placeholder.
    # Correlation is computed only on configs present in both P0 and this experiment.
    print(f"  {len(perm_configs)} permutation configs (baseline + {args.n_configs - 1} random)")

    # ---- Run experiments ----
    results = {}

    for bits in args.bits:
        print(f"\n{'='*60}")
        print(f"Bitwidth: {bits}")
        print(f"{'='*60}")

        bit_results = []

        # Build P0 lookup
        p0_by_mode = {}
        for c in det_configs:
            if c["bits"] == bits:
                p0_by_mode[c["mode"]] = c

        for cfg_idx, (mode, perms) in enumerate(perm_configs.items()):
            t0 = time.time()

            with torch.no_grad():
                quant_kv = quantize_kv_cache(
                    clean_kv_legacy, perms, bits,
                    num_kv_heads, head_dim, block_size,
                )

                out2_quant = model(
                    chunk2_ids,
                    past_key_values=legacy_to_cache(quant_kv),
                    attention_mask=attention_mask_c2,
                    position_ids=position_ids_c2,
                    labels=chunk2_ids,
                    use_cache=False,
                )
                quant_loss_c2 = out2_quant.loss.item()

            dl_auto = quant_loss_c2 - clean_loss_c2

            # P0 lookup
            p0 = p0_by_mode.get(mode, {})
            dl_prefill = p0.get("delta_loss")
            tr_m_sigma = p0.get("tr_M_Sigma")
            q1 = p0.get("linear_pred")
            tr_sigma = p0.get("tr_Sigma")

            elapsed = time.time() - t0

            entry = {
                "mode": mode,
                "bits": bits,
                "dl_auto": float(dl_auto),
                "dl_prefill": dl_prefill,
                "tr_M_Sigma": tr_m_sigma,
                "Q1": q1,
                "tr_Sigma": tr_sigma,
            }
            bit_results.append(entry)

            prefill_str = f"prefill={dl_prefill:+.4f}" if dl_prefill is not None else "prefill=N/A"
            print(f"  [{cfg_idx+1:2d}/{len(perm_configs)}] {mode:16s}: "
                  f"auto={dl_auto:+.4f}  {prefill_str}  ({elapsed:.1f}s)")

        # ---- Ranking correlations ----
        valid = [r for r in bit_results
                 if r["dl_prefill"] is not None
                 and r["tr_M_Sigma"] is not None]

        if len(valid) >= 5:
            dl_autos = [r["dl_auto"] for r in valid]
            dl_prefills = [r["dl_prefill"] for r in valid]
            trms = [r["tr_M_Sigma"] for r in valid]
            q1s = [r["Q1"] for r in valid]
            mses = [r["tr_Sigma"] for r in valid]

            rho_pf, p_pf = spearman_corr(dl_prefills, dl_autos)
            rho_trms, p_trms = spearman_corr(trms, dl_autos)
            rho_q1, p_q1 = spearman_corr(q1s, dl_autos)
            rho_mse, p_mse = spearman_corr(mses, dl_autos)

            p_str = lambda p: f"p={p:.2e}" if p is not None and not math.isnan(p) else ""

            print(f"\n  --- Ranking Correlations ({bits}b, n={len(valid)}) ---")
            print(f"  ρ(prefill, auto)   = {rho_pf:+.3f}  {p_str(p_pf)}  [primary]")
            print(f"  ρ(tr(MΣ), auto)    = {rho_trms:+.3f}  {p_str(p_trms)}  [primary]")
            print(f"  ρ(MSE, auto)       = {rho_mse:+.3f}  {p_str(p_mse)}  [primary]")
            print(f"  ρ(Q1, auto)        = {rho_q1:+.3f}  {p_str(p_q1)}  [supplementary]")

            ranking = {
                "n_configs": len(valid),
                "rho_prefill_auto": float(rho_pf),
                "rho_trms_auto": float(rho_trms),
                "rho_q1_auto": float(rho_q1),
                "rho_mse_auto": float(rho_mse),
                "p_prefill_auto": float(p_pf) if not math.isnan(p_pf) else None,
                "p_trms_auto": float(p_trms) if not math.isnan(p_trms) else None,
                "p_q1_auto": float(p_q1) if not math.isnan(p_q1) else None,
                "p_mse_auto": float(p_mse) if not math.isnan(p_mse) else None,
            }
        else:
            print(f"\n  Ranking: insufficient valid configs ({len(valid)})")
            ranking = {"n_configs": len(valid), "error": "insufficient"}

        # ---- Summary ----
        dl_autos_all = [r["dl_auto"] for r in bit_results]
        print(f"\n  --- Summary ({bits}b) ---")
        print(f"  Auto ΔL: mean={np.mean(dl_autos_all):.4f}, "
              f"std={np.std(dl_autos_all):.4f}, "
              f"range=[{min(dl_autos_all):.4f}, {max(dl_autos_all):.4f}]")

        results[f"{bits}b"] = {
            "configs": bit_results,
            "ranking": ranking,
            "summary": {
                "mean_dl_auto": float(np.mean(dl_autos_all)),
                "std_dl_auto": float(np.std(dl_autos_all)),
                "min_dl_auto": float(min(dl_autos_all)),
                "max_dl_auto": float(max(dl_autos_all)),
            },
        }

    # ---- Save ----
    import transformers
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "version": "autoregressive_experiment_v2",
        "transformers_version": transformers.__version__,
        "config": {
            "bits": args.bits,
            "n_configs": len(perm_configs),
            "seq_len": args.seq_len,
            "block_size": block_size,
            "chunk1_idx": 2,
            "chunk2_idx": 3,
        },
        "clean_loss_chunk2": clean_loss_c2,
        "sanity_check": {
            "cache_reuse_loss": loss_reuse,
            "full_concat_loss": loss_concat,
            "gap": gap,
            "passed": ok,
        },
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()