"""
K-path Softmax Flip Diagnostic
================================
Measure WHY key quantization destroys Qwen but not Llama.

For each layer:
  1. Capture clean K, Q, attention logits
  2. Quantize K → Δk = K_hat - K
  3. Per-dimension: σ²_k,i and E[Δk_i²]
  4. Logit perturbation: q·Δk/√d for each (query, key) pair
  5. Compare perturbation magnitude vs softmax logit gap
     (gap = max_logit - 2nd_max_logit)

If |q·Δk/√d| > logit gap → softmax flips → catastrophic

Usage:
  python -m ddt.kpath_diagnostic --model meta-llama/Llama-3.1-8B
  python -m ddt.kpath_diagnostic --model Qwen/Qwen2.5-7B
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_uniform(x, bits=4):
    """Per-token symmetric uniform quantization."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    max_val = 2 ** (bits - 1) - 1
    x_q = (x / scale * max_val).round().clamp(-max_val, max_val) * scale / max_val
    return x_q


class KPathCapture:
    """Capture Q, K, and attention logits at each layer."""

    def __init__(self, model):
        self.model = model
        self.data = {}  # layer -> {q, k, logits}
        self.hooks = []

    def register(self):
        for idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn

            def make_hook(li):
                def hook(module, args, kwargs, output):
                    # Capture attention weights (pre-softmax logits not directly available)
                    # Instead, we'll capture Q and K from projections
                    pass
                return hook

            # Hook q_proj and k_proj to capture Q and K
            def make_q_hook(li):
                def hook(module, input, output):
                    if li not in self.data:
                        self.data[li] = {}
                    self.data[li]['q'] = output.detach()
                return hook

            def make_k_hook(li):
                def hook(module, input, output):
                    if li not in self.data:
                        self.data[li] = {}
                    self.data[li]['k'] = output.detach()
                return hook

            self.hooks.append(attn.q_proj.register_forward_hook(make_q_hook(idx)))
            self.hooks.append(attn.k_proj.register_forward_hook(make_k_hook(idx)))

    def clear(self):
        self.data.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def analyze_layer(q_raw, k_raw, n_heads, n_kv_heads, head_dim, bits=4):
    """
    Analyze K-path vulnerability for one layer.

    q_raw: (1, T, n_heads * head_dim) from q_proj
    k_raw: (1, T, n_kv_heads * head_dim) from k_proj
    """
    T = q_raw.shape[1]
    device = q_raw.device

    # Reshape to heads
    q = q_raw.squeeze(0).float()  # (T, n_heads * head_dim)
    k = k_raw.squeeze(0).float()  # (T, n_kv_heads * head_dim)

    # --- Per-dimension K statistics ---
    k_per_dim_var = k.var(dim=0)  # (n_kv_heads * head_dim,)
    am = k_per_dim_var.mean().item()
    gm = k_per_dim_var.clamp(min=1e-30).log().mean().exp().item()
    am_gm = am / max(gm, 1e-30)

    k_per_dim_std = k.std(dim=0)  # (n_kv_heads * head_dim,)
    max_min_ratio = (k_per_dim_std.max() / k_per_dim_std.clamp(min=1e-10).min()).item()

    # --- Quantize K ---
    k_q = quantize_uniform(k_raw, bits).squeeze(0).float()
    dk = k_q - k  # (T, n_kv_heads * head_dim)

    # Per-dimension error variance
    dk_per_dim_var = (dk ** 2).mean(dim=0)  # (n_kv_heads * head_dim,)
    dk_per_dim_std = dk_per_dim_var.sqrt()

    # Error concentration: top-10% dimensions' error / total error
    dk_sorted = dk_per_dim_var.sort(descending=True).values
    n_dims = len(dk_sorted)
    top10_pct = dk_sorted[:max(n_dims // 10, 1)].sum() / dk_sorted.sum()

    # --- Logit perturbation ---
    # For GQA: each KV head serves multiple Q heads
    # Reshape for dot product
    heads_per_kv = n_heads // n_kv_heads

    # Compute q·Δk/√d for a sample of (query, key) pairs
    # Use first KV head for simplicity
    q_head0 = q[:, :head_dim]  # (T, head_dim) — first query head
    k_head0 = k[:, :head_dim]  # (T, head_dim) — first KV head
    dk_head0 = dk[:, :head_dim]  # (T, head_dim)

    # Clean logits: q·k/√d
    clean_logits = (q_head0 @ k_head0.T) / (head_dim ** 0.5)  # (T, T)

    # Logit perturbation: q·Δk/√d
    logit_perturb = (q_head0 @ dk_head0.T) / (head_dim ** 0.5)  # (T, T)

    # Softmax gap: for each query, gap between top-1 and top-2 logit
    sorted_logits = clean_logits.sort(dim=-1, descending=True).values
    # Handle causal mask: only look at valid positions
    # Use simple approximation: gap for each row
    gaps = (sorted_logits[:, 0] - sorted_logits[:, 1])  # (T,)

    # Flip probability: |perturbation| > gap
    perturb_abs = logit_perturb.abs()
    # For each query, max perturbation across all keys
    max_perturb_per_query = perturb_abs.max(dim=-1).values  # (T,)

    # Fraction of queries where max perturbation exceeds gap
    flip_fraction = (max_perturb_per_query > gaps.abs()).float().mean().item()

    # Statistics
    perturb_mean = perturb_abs.mean().item()
    perturb_max = perturb_abs.max().item()
    perturb_std = logit_perturb.std().item()
    gap_mean = gaps.abs().mean().item()
    gap_median = gaps.abs().median().item()

    # Perturbation / gap ratio (how close to flipping)
    ratio_mean = (max_perturb_per_query / gaps.abs().clamp(min=1e-10)).mean().item()

    return {
        'am_gm': am_gm,
        'max_min_ratio': max_min_ratio,
        'error_top10_pct': top10_pct.item(),
        'perturb_mean': perturb_mean,
        'perturb_max': perturb_max,
        'perturb_std': perturb_std,
        'gap_mean': gap_mean,
        'gap_median': gap_median,
        'perturb_gap_ratio': ratio_mean,
        'flip_fraction': flip_fraction,
        'k_dims': n_kv_heads * head_dim,
    }


def run_model(model_name, bits=4, seq_len=512):
    """Run K-path diagnostic for one model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  K-path Diagnostic: {model_name} @ {bits}bit")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Get attention config
    attn0 = model.model.layers[0].self_attn
    n_heads = attn0.config.num_attention_heads
    n_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    n_layers = len(model.model.layers)

    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    # Calibration tokens
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)

    # Capture Q and K
    capturer = KPathCapture(model)
    capturer.register()

    with torch.no_grad():
        model(input_ids, use_cache=False)

    capturer.remove()

    # Analyze each layer
    print(f"\n  {'L':>3} {'AM/GM':>8} {'max/min':>8} {'err_top10%':>10} "
          f"{'|pert|':>8} {'gap':>8} {'pert/gap':>8} {'flip%':>6}")
    print(f"  {'-'*72}")

    results = []
    for li in range(n_layers):
        if li not in capturer.data:
            continue
        if 'q' not in capturer.data[li] or 'k' not in capturer.data[li]:
            continue

        r = analyze_layer(
            capturer.data[li]['q'],
            capturer.data[li]['k'],
            n_heads, n_kv_heads, head_dim, bits
        )
        r['layer'] = li
        results.append(r)

        flag = " ***" if r['flip_fraction'] > 0.1 else ""
        print(f"  {li:3d} {r['am_gm']:8.1f} {r['max_min_ratio']:8.1f} "
              f"{r['error_top10_pct']:10.3f} "
              f"{r['perturb_mean']:8.4f} {r['gap_mean']:8.4f} "
              f"{r['perturb_gap_ratio']:8.2f} {r['flip_fraction']:6.1%}{flag}")

    # Summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {model_name}")
    am_gms = [r['am_gm'] for r in results]
    ratios = [r['perturb_gap_ratio'] for r in results]
    flips = [r['flip_fraction'] for r in results]

    print(f"  AM/GM: mean={np.mean(am_gms):.1f}  max={np.max(am_gms):.1f}")
    print(f"  Perturbation/Gap ratio: mean={np.mean(ratios):.2f}  "
          f"max={np.max(ratios):.2f}")
    print(f"  Flip fraction: mean={np.mean(flips):.1%}  "
          f"max={np.max(flips):.1%}")
    print(f"  Layers with >10% flips: "
          f"{sum(1 for f in flips if f > 0.1)}/{len(flips)}")
    print(f"  {'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="K-path softmax flip diagnostic")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    results = run_model(args.model, args.bits, args.seq_len)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir,
                        f"kpath_diagnostic_{tag}_{args.bits}bit.json")

    with open(path, 'w') as f:
        json.dump({'model': args.model, 'bits': args.bits, 'layers': results},
                  f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()