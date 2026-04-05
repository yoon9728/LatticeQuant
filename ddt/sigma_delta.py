"""
σ²_δ Diagnostic Validation
============================
Directly measure σ²_δ = Var(q·Δk/√d) for each layer,
compute predicted χ² = exp(σ²_δ) - 1,
and compare with actual K-path PPL degradation.

If exp(σ²_δ)-1 correlates with PPL degradation across 3 models,
the diagnostic is validated.

Usage:
  python -m ddt.sigma_delta --model meta-llama/Llama-3.1-8B
  python -m ddt.sigma_delta --model Qwen/Qwen2.5-7B
  python -m ddt.sigma_delta --model mistralai/Mistral-7B-v0.3
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_uniform(x, bits=4):
    """Per-token symmetric uniform quantization."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    max_val = 2 ** (bits - 1) - 1
    x_q = (x / scale * max_val).round().clamp(-max_val, max_val) * scale / max_val
    return x_q


def run_model(model_name, bits=4, seq_len=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  σ²_δ Diagnostic: {model_name} @ {bits}bit")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Config
    attn0 = model.model.layers[0].self_attn
    n_heads = attn0.config.num_attention_heads
    n_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    heads_per_kv = n_heads // n_kv_heads
    n_layers = len(model.model.layers)

    # Calibration tokens
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)
    T = input_ids.shape[1]
    print(f"  T={T}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    # Capture Q and K at every layer
    captures = {}
    hooks = []

    for idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def make_q_hook(li):
            def hook(module, input, output):
                if li not in captures:
                    captures[li] = {}
                captures[li]['q'] = output.detach().float()
            return hook

        def make_k_hook(li):
            def hook(module, input, output):
                if li not in captures:
                    captures[li] = {}
                captures[li]['k'] = output.detach().float()
            return hook

        hooks.append(attn.q_proj.register_forward_hook(make_q_hook(idx)))
        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(idx)))

    with torch.no_grad():
        model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    # Measure σ²_δ per layer
    print(f"\n  {'L':>3} {'σ²_δ':>10} {'exp(σ²)-1':>10} {'regime':>12}")
    print(f"  {'-'*40}")

    results = []
    for li in range(n_layers):
        q_raw = captures[li]['q'].squeeze(0)  # (T, n_heads * head_dim)
        k_raw = captures[li]['k'].squeeze(0)  # (T, n_kv_heads * head_dim)

        # Quantize K
        k_q = quantize_uniform(k_raw.unsqueeze(0), bits).squeeze(0)
        dk = k_q - k_raw  # (T, n_kv_heads * head_dim)

        # Reshape to heads
        q_heads = q_raw.view(T, n_heads, head_dim)      # (T, n_heads, d)
        k_heads = k_raw.view(T, n_kv_heads, head_dim)   # (T, n_kv, d)
        dk_heads = dk.view(T, n_kv_heads, head_dim)      # (T, n_kv, d)

        # Compute σ²_δ averaged over all head pairs
        sigma2_list = []

        for kv_h in range(n_kv_heads):
            # All Q heads that use this KV head
            q_start = kv_h * heads_per_kv
            q_end = q_start + heads_per_kv

            dk_h = dk_heads[:, kv_h, :]  # (T, d) — key error for this KV head

            for q_h in range(q_start, min(q_end, n_heads)):
                q_h_vec = q_heads[:, q_h, :]  # (T, d) — queries for this Q head

                # δ_{ij} = q_i · Δk_j / √d
                # Shape: (T_query, T_key)
                delta = (q_h_vec @ dk_h.T) / (head_dim ** 0.5)

                # Per-query variance of δ: Var_j(δ_{ij}) for each query i
                # Then average over query positions
                per_query_var = delta.var(dim=-1)  # (T,)
                sigma2 = per_query_var.mean().item()
                sigma2_list.append(sigma2)

        sigma2_avg = np.mean(sigma2_list)
        predicted_chi2 = np.exp(sigma2_avg) - 1 if sigma2_avg < 50 else float('inf')

        if sigma2_avg < 0.1:
            regime = "safe"
        elif sigma2_avg < 1.0:
            regime = "borderline"
        else:
            regime = "CATASTROPHIC"

        results.append({
            'layer': li,
            'sigma2_delta': sigma2_avg,
            'predicted_chi2': predicted_chi2,
            'regime': regime,
            'sigma2_per_head': sigma2_list,
        })

        flag = " ***" if sigma2_avg > 1.0 else ""
        print(f"  {li:3d} {sigma2_avg:10.4f} {predicted_chi2:10.4f} {regime:>12}{flag}")

    # Summary
    all_sigma2 = [r['sigma2_delta'] for r in results]
    max_sigma2 = max(all_sigma2)
    mean_sigma2 = np.mean(all_sigma2)
    catastrophic_layers = sum(1 for s in all_sigma2 if s > 1.0)
    pred = np.exp(max_sigma2)-1 if max_sigma2 < 50 else float('inf')

    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {model_name}")
    print(f"  {'='*50}")
    print(f"  σ²_δ: mean={mean_sigma2:.4f}  max={max_sigma2:.4f}")
    print(f"  Catastrophic layers (σ²>1): {catastrophic_layers}/{n_layers}")
    print(f"  Overall regime: {'CATASTROPHIC' if max_sigma2 > 1 else 'borderline' if max_sigma2 > 0.1 else 'safe'}")
    print(f"  Predicted max χ²: {pred}")
    return results, {
        'model': model_name,
        'bits': bits,
        'mean_sigma2': mean_sigma2,
        'max_sigma2': max_sigma2,
        'catastrophic_layers': catastrophic_layers,
    }


def main():
    parser = argparse.ArgumentParser(description="σ²_δ diagnostic")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    results, summary = run_model(args.model, args.bits, args.seq_len)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir,
                        f"sigma_delta_{tag}_{args.bits}bit.json")

    save = {
        'model': args.model,
        'bits': args.bits,
        'summary': summary,
        'layers': [{
            'layer': r['layer'],
            'sigma2_delta': r['sigma2_delta'],
            'predicted_chi2': r['predicted_chi2'],
            'regime': r['regime'],
        } for r in results],
    }
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()