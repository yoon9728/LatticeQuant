"""
DDT — Quantizer Robustness Ablation
======================================
Tests whether DDT > MSE holds across quantizer hyperparameters.

Strategy:
  1. One clean backward pass → store sensitivity s = ∂L/∂v per (layer, position)
  2. For each ablation setting × permutation config:
     - Forward with quantization hooks → get error e AND ΔL
     - Q1 = Σ s·e  (cheap: dot product with pre-stored s)
     - MSE = Σ e²
  3. Compare ρ(Q1, ΔL) vs ρ(MSE, ΔL) per setting

Note: Q1 is a signed first-order predictor, not the full DDT risk metric
tr(MΣ). In moderate regime Q1 is the stronger predictor; in catastrophic
regime tr(MΣ) is preferred. This ablation uses Q1 as a first-order
DDT proxy because it is cheap to compute from pre-stored s.

Key insight: for a fixed model and corpus, s is independent of quantizer
hyperparameters (block size, α). We compute s once per corpus and reuse
it across all quantizer settings. For calibration-corpus ablations,
s is recomputed on the new corpus.

Ablation axes:
  block_size: {4, 8, 16}
  alpha:      {2, 3, 4}
  calibration: WikiText-2 vs C4 (s recomputed per corpus)

One model at a time. Recommended: Llama-3.1-8B + Qwen2.5-7B.

Usage:
  python -m ddt.ablation_experiment --model meta-llama/Llama-3.1-8B \\
      --ablation block_size --bits 3 4
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
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Helpers
# ============================================================

def get_model_device(model):
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


def spearman_rho(x, y):
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(xa) < 5:
        return float("nan"), float("nan")
    rho, p = spearmanr(xa, ya)
    return float(rho), float(p)


# ============================================================
# Sensitivity computation (one backward pass)
# ============================================================

def compute_sensitivities(model, input_ids, num_kv_heads, head_dim):
    """Compute s = ∂L/∂v for all layers via retain_grad().

    Uses output.retain_grad() on k_proj/v_proj outputs, then
    loss.backward() populates .grad attributes.

    Returns dict: sensitivities[layer_idx][comp] = tensor (B, T, H_kv, d_h)
    Stored on CPU in float16 to avoid GPU OOM. Caller should move to
    device per-layer when needed.
    """
    num_layers = len(model.model.layers)
    grad_targets = {}
    hooks = []

    def make_hook(layer_idx, comp):
        def hook(module, input, output):
            output.retain_grad()
            grad_targets[(layer_idx, comp)] = output
        return hook

    for idx in range(num_layers):
        layer = model.model.layers[idx]
        attn = layer.self_attn
        hooks.append(attn.k_proj.register_forward_hook(make_hook(idx, "K")))
        hooks.append(attn.v_proj.register_forward_hook(make_hook(idx, "V")))

    # Forward + backward
    model.zero_grad()
    outputs = model(input_ids, labels=input_ids, use_cache=False)
    outputs.loss.backward()

    # Remove hooks before processing
    for h in hooks:
        h.remove()

    # Extract and reshape gradients
    sensitivities = {}
    n_found = 0
    for (layer_idx, comp), tensor in grad_targets.items():
        g = tensor.grad
        if g is not None:
            B, T, _ = g.shape
            # Store on CPU in float16 to avoid GPU OOM
            s = g.detach().view(B, T, num_kv_heads, head_dim).half().cpu()
            n_found += 1
        else:
            s = None
        if layer_idx not in sensitivities:
            sensitivities[layer_idx] = {}
        sensitivities[layer_idx][comp] = s

    # Sanity check with debugging info
    expected = num_layers * 2  # K + V per layer
    if n_found != expected:
        missing = []
        for idx in range(num_layers):
            for comp in ["K", "V"]:
                s = sensitivities.get(idx, {}).get(comp)
                if s is None:
                    missing.append(f"layer {idx} {comp}")
        raise RuntimeError(
            f"Sensitivity sanity check failed: got {n_found}/{expected} gradients. "
            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    del grad_targets  # free references
    model.zero_grad()
    return sensitivities


# ============================================================
# Measurement: ΔL + Q1 + MSE per config
# ============================================================

@torch.no_grad()
def measure_config(
    model, input_ids, perms, bits, num_kv_heads, head_dim,
    clean_loss, sensitivities, block_size=8, alpha=3.0,
):
    """Measure ΔL, Q1, and MSE for one permutation config.

    Q1 = Σ s·e (using pre-computed sensitivities)
    MSE = Σ e² (total error power)
    """
    assert head_dim % block_size == 0
    hooks = []
    num_layers = len(model.model.layers)
    n_levels = 2 ** bits
    half = n_levels / 2
    device = input_ids.device

    q1_total = 0.0
    mse_total = 0.0

    def make_quant_hook(layer_idx, comp):
        def hook(module, input, output):
            nonlocal q1_total, mse_total
            x = output.float()
            B, T, _ = x.shape
            x_heads = x.view(B, T, num_kv_heads, head_dim)
            x_out = torch.zeros_like(x_heads)

            # Get pre-computed sensitivity for this layer/comp (already on device)
            s = sensitivities.get(layer_idx, {}).get(comp)

            for h in range(num_kv_heads):
                perm = perms[layer_idx][comp][h].to(device)
                inv_perm = torch.argsort(perm)
                v_h = x_heads[:, :, h, :]  # (B, T, head_dim)

                v_perm = v_h[:, :, perm]
                n_blocks = head_dim // block_size
                blocks = v_perm.reshape(B, T, n_blocks, block_size)

                rms = torch.sqrt((blocks ** 2).mean(dim=-1, keepdim=True) + 1e-12)
                scale = alpha * rms / half
                scaled = blocks / scale
                quant = torch.round(scaled.clamp(-half, half - 1))
                blocks_qd = quant * scale

                v_hat_perm = blocks_qd.reshape(B, T, head_dim)
                v_hat = v_hat_perm[:, :, inv_perm]

                # Error
                e_h = v_hat - v_h  # (B, T, head_dim)
                mse_total += e_h.pow(2).sum().item()

                # Q1 contribution: s · e
                if s is not None:
                    s_h = s[:, :, h, :]  # (B, T, head_dim)
                    q1_total += (s_h * e_h).sum().item()

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

    return {
        "delta_loss": quant_loss - clean_loss,
        "Q1": q1_total,
        "MSE": mse_total,
    }


# ============================================================
# Calibration data loaders
# ============================================================

def load_wikitext(tokenizer, seq_len, chunk_idx=0):
    """Load WikiText-2 test split, chunk_idx-th slice."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    start = chunk_idx * seq_len
    return ids[:, start:start + seq_len]


def load_c4(tokenizer, seq_len, chunk_idx=0, n_samples=8):
    """Load C4 validation data, chunk_idx-th slice. First n_samples concatenated."""
    dataset = load_dataset("allenai/c4", "en", split="validation",
                           streaming=True)
    texts = []
    for i, sample in enumerate(dataset):
        texts.append(sample["text"])
        if i >= n_samples - 1:
            break
    text = "\n\n".join(texts)
    ids = tokenizer(text, return_tensors="pt").input_ids
    start = chunk_idx * seq_len
    return ids[:, start:start + seq_len]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT Quantizer Robustness Ablation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["calibration", "block_size", "alpha"])
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--n-configs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--test-sensitivity", action="store_true",
                        help="Only test sensitivity computation and exit (sanity check)")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"ablation_{args.ablation}_{model_tag}.json")

    print(f"Model: {args.model}")
    print(f"Ablation: {args.ablation}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    device = get_model_device(model)
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Ablation grid ----
    if args.ablation == "calibration":
        grid = [
            {"name": "wikitext2", "corpus": "wikitext", "block_size": 8, "alpha": 3.0},
            {"name": "c4", "corpus": "c4", "block_size": 8, "alpha": 3.0},
        ]
    elif args.ablation == "block_size":
        grid = [
            {"name": "bs4", "corpus": "wikitext", "block_size": 4, "alpha": 3.0},
            {"name": "bs8", "corpus": "wikitext", "block_size": 8, "alpha": 3.0},
            {"name": "bs16", "corpus": "wikitext", "block_size": 16, "alpha": 3.0},
        ]
    elif args.ablation == "alpha":
        grid = [
            {"name": "a2", "corpus": "wikitext", "block_size": 8, "alpha": 2.0},
            {"name": "a3", "corpus": "wikitext", "block_size": 8, "alpha": 3.0},
            {"name": "a4", "corpus": "wikitext", "block_size": 8, "alpha": 4.0},
        ]

    # ---- Permutation configs ----
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

    # ---- Run per corpus group ----
    # Group grid by corpus: same corpus shares sensitivity computation
    corpus_groups = {}
    for setting in grid:
        corpus = setting["corpus"]
        if corpus not in corpus_groups:
            corpus_groups[corpus] = []
        corpus_groups[corpus].append(setting)

    all_results = {}

    for corpus, settings in corpus_groups.items():
        # Load calibration data
        if corpus == "c4":
            cal_ids = load_c4(tokenizer, args.seq_len).to(device)
            print(f"\n  Corpus: C4 validation (chunk 0, {args.seq_len} tokens)")
        else:
            cal_ids = load_wikitext(tokenizer, args.seq_len).to(device)
            print(f"\n  Corpus: WikiText-2 test (chunk 0, {args.seq_len} tokens)")

        # Clean loss
        with torch.no_grad():
            clean_loss = model(cal_ids, labels=cal_ids, use_cache=False).loss.item()
        print(f"  Clean loss: {clean_loss:.4f}")

        # Compute sensitivities once per corpus
        print(f"  Computing sensitivities (one backward pass)...")
        t0 = time.time()
        sensitivities = compute_sensitivities(
            model, cal_ids, num_kv_heads, head_dim
        )
        n_layers_with_s = sum(
            1 for l in sensitivities.values()
            for c in l.values() if c is not None
        )
        print(f"  Done in {time.time()-t0:.1f}s "
              f"({n_layers_with_s} sensitivity tensors)")

        if args.test_sensitivity:
            # Show sample stats and exit
            for comp in ["K", "V"]:
                s0 = sensitivities[0][comp]
                if s0 is not None:
                    print(f"  Layer 0 {comp}: shape={list(s0.shape)}, "
                          f"dtype={s0.dtype}, device={s0.device}, "
                          f"norm={s0.float().norm():.4f}")
            print("\n  Sensitivity test PASSED. Exiting.")
            return

        # Run each setting in this corpus group
        for setting in settings:
            name = setting["name"]
            block_size = setting["block_size"]
            alpha = setting["alpha"]

            if head_dim % block_size != 0:
                print(f"\n  SKIP {name}: head_dim={head_dim} % block_size={block_size} != 0")
                continue

            print(f"\n{'='*60}")
            print(f"Setting: {name} (block_size={block_size}, α={alpha})")
            print(f"{'='*60}")

            # Move sensitivities to GPU for this setting's config loop
            sens_gpu = {}
            for l_idx, comps in sensitivities.items():
                sens_gpu[l_idx] = {}
                for comp, s_cpu in comps.items():
                    sens_gpu[l_idx][comp] = s_cpu.to(device) if s_cpu is not None else None

            setting_results = {}

            for bits in args.bits:
                print(f"\n  --- {bits}b ---")
                bit_results = []

                for cfg_idx, (mode, perms) in enumerate(perm_configs.items()):
                    result = measure_config(
                        model, cal_ids, perms, bits, num_kv_heads, head_dim,
                        clean_loss, sens_gpu,
                        block_size=block_size, alpha=alpha,
                    )
                    result["mode"] = mode
                    bit_results.append(result)

                    if (cfg_idx + 1) % 10 == 0:
                        print(f"    [{cfg_idx+1}/{len(perm_configs)}]")

                dls = [r["delta_loss"] for r in bit_results]
                q1s = [r["Q1"] for r in bit_results]
                mses = [r["MSE"] for r in bit_results]

                rho_q1, p_q1 = spearman_rho(q1s, dls)
                rho_mse, p_mse = spearman_rho(mses, dls)

                dl_std = np.std(dls)
                dl_cv = dl_std / abs(np.mean(dls)) if abs(np.mean(dls)) > 1e-6 else 0

                p_str = lambda p: f"p={p:.2e}" if not math.isnan(p) else ""

                print(f"  ΔL: mean={np.mean(dls):.4f}, std={dl_std:.4f}, CV={dl_cv:.2f}")
                print(f"  ρ(Q1, ΔL)  = {rho_q1:+.3f}  {p_str(p_q1)}  [first-order DDT]")
                print(f"  ρ(MSE, ΔL) = {rho_mse:+.3f}  {p_str(p_mse)}  [baseline]")

                diff = abs(rho_q1) - abs(rho_mse)
                if abs(diff) < 0.1:
                    winner = "TIE"
                elif diff > 0:
                    winner = "Q1"
                else:
                    winner = "MSE"
                print(f"  → {winner} "
                      f"(|ρ_Q1|={abs(rho_q1):.3f} vs |ρ_MSE|={abs(rho_mse):.3f}, "
                      f"Δ={diff:+.3f})")

                setting_results[f"{bits}b"] = {
                    "configs": bit_results,
                    "rho_q1_dl": rho_q1,
                    "p_q1_dl": p_q1 if not math.isnan(p_q1) else None,
                    "rho_mse_dl": rho_mse,
                    "p_mse_dl": p_mse if not math.isnan(p_mse) else None,
                    "dl_mean": float(np.mean(dls)),
                    "dl_std": float(dl_std),
                    "dl_cv": float(dl_cv),
                }

            all_results[name] = {
                "setting": setting,
                "clean_loss": clean_loss,
                "bits_results": setting_results,
            }

            # Free GPU cache for this setting
            del sens_gpu
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("SUMMARY: Q1 (first-order DDT) vs MSE across settings")
    print(f"{'='*60}")
    print(f"  {'Setting':10s} {'Bit':4s} {'ρ(Q1,ΔL)':10s} {'ρ(MSE,ΔL)':10s} {'Result':8s}")
    for name, data in all_results.items():
        for bit_key, br in data["bits_results"].items():
            rq = br["rho_q1_dl"]
            rm = br["rho_mse_dl"]
            diff = abs(rq) - abs(rm)
            if abs(diff) < 0.1:
                w = "TIE"
            elif diff > 0:
                w = "Q1"
            else:
                w = "MSE"
            print(f"  {name:10s} {bit_key:4s} {rq:+10.3f} {rm:+10.3f} {w:8s}")

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "ablation_type": args.ablation,
        "n_configs": len(perm_configs),
        "results": all_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()