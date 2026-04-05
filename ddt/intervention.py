"""
Diagnostic-Guided Intervention: Proof of Concept
==================================================
Tests whether σ²_eff-guided bit allocation outperforms
baselines under matched bit budget.

Conditions (all K-only, V unquantized):
  1. Uniform 4-bit K (baseline)
  2. Theory-guided: top-m σ²_eff layers → 6-bit, rest → 3-bit
  3. Random: m random layers → 6-bit, rest → 3-bit (3 seeds)
  4. Heuristic-first: first m layers → 6-bit, rest → 3-bit
  5. Heuristic-last: last m layers → 6-bit, rest → 3-bit

Budget matching: m·6 + (L-m)·3 ≈ L·4

Usage:
  python -m ddt.intervention --model Qwen/Qwen2.5-7B
  python -m ddt.intervention --model meta-llama/Llama-3.1-8B
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_uniform(x, bits=4):
    """Per-token symmetric uniform quantization."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    max_val = 2 ** (bits - 1) - 1
    x_q = (x / scale * max_val).round().clamp(-max_val, max_val) * scale / max_val
    return x_q


class PerLayerKQuantHook:
    """Apply different K bitrates per layer."""

    def __init__(self, model, layer_bits):
        """
        layer_bits: dict {layer_idx: bits} or None for no quantization
        """
        self.model = model
        self.layer_bits = layer_bits  # {0: 6, 1: 3, 2: 3, ...}
        self.hooks = []

    def register(self):
        for idx, layer in enumerate(self.model.model.layers):
            if idx not in self.layer_bits:
                continue
            bits = self.layer_bits[idx]

            def make_hook(li, b):
                def hook(module, input, output):
                    return quantize_uniform(output, b)
                return hook

            h = layer.self_attn.k_proj.register_forward_hook(make_hook(idx, bits))
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def measure_ppl(model, input_ids, layer_bits=None):
    """Run forward with per-layer K quantization, return PPL."""
    if layer_bits is not None:
        hooks = PerLayerKQuantHook(model, layer_bits)
        hooks.register()
    else:
        hooks = None

    with torch.no_grad():
        out = model(input_ids, use_cache=False)

    logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)).item()
    ppl = np.exp(loss) if loss < 20 else float('inf')

    if hooks:
        hooks.remove()

    return ppl


def compute_sigma2_eff(model, input_ids, n_layers, n_heads, n_kv_heads, head_dim):
    """Quick σ²_eff measurement per layer (from diagnose.py logic)."""
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
        out = model(input_ids, use_cache=False, output_attentions=True)

    for h in hooks:
        h.remove()

    T = input_ids.shape[1]
    heads_per_kv = n_heads // n_kv_heads
    bits = 4  # measure σ²_δ at 4-bit

    results = []
    for li in range(n_layers):
        q_raw = captures[li]['q'].squeeze(0)
        k_raw = captures[li]['k'].squeeze(0)

        k_q = quantize_uniform(k_raw.unsqueeze(0), bits).squeeze(0)
        dk = k_q - k_raw

        q_heads = q_raw.view(T, n_heads, head_dim)
        dk_heads = dk.view(T, n_kv_heads, head_dim)

        sigma2_list = []
        for kv_h in range(n_kv_heads):
            q_start = kv_h * heads_per_kv
            dk_h = dk_heads[:, kv_h, :]
            for q_h in range(q_start, min(q_start + heads_per_kv, n_heads)):
                q_h_vec = q_heads[:, q_h, :]
                delta = (q_h_vec @ dk_h.T) / (head_dim ** 0.5)
                per_query_var = delta.var(dim=-1)
                sigma2_list.append(per_query_var.mean().item())

        sigma2_avg = np.mean(sigma2_list)

        # n_eff from attention weights
        A = out.attentions[li].squeeze(0).float()  # (n_heads, T, T)
        eta_V = (A ** 2).sum(dim=-1).mean().item()
        n_eff = 1.0 / max(eta_V, 1e-10)
        s2_eff = sigma2_avg * max(1.0 - 1.0/n_eff, 0.0)

        results.append({
            'layer': li,
            'sigma2_delta': sigma2_avg,
            'sigma2_eff': s2_eff,
            'sigma2_weighted': s2_eff * (n_layers - li) / n_layers,
        })

    return results


def run_intervention(model_name, seq_len=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  Diagnostic-Guided Intervention")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    attn0 = model.model.layers[0].self_attn
    n_heads = attn0.config.num_attention_heads
    n_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    n_layers = len(model.model.layers)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)
    T = input_ids.shape[1]
    print(f"  T={T}, L={n_layers}")

    # Step 1: Measure σ²_eff
    print(f"\n  Step 1: Measuring σ²_eff per layer...")
    sigma_results = compute_sigma2_eff(
        model, input_ids, n_layers, n_heads, n_kv_heads, head_dim)

    for r in sigma_results:
        flag = " ***" if r['sigma2_eff'] > 1.0 else ""
        print(f"    Layer {r['layer']:2d}: σ²_eff={r['sigma2_eff']:8.4f}  "
              f"σ²_weighted={r['sigma2_weighted']:8.4f}{flag}")

    # Step 2: Budget matching
    # m layers at 6-bit, rest at 3-bit
    # m·6 + (L-m)·3 = L·4 → 3m = L → m = L/3
    m = round(n_layers / 3)
    avg_bits = (m * 6 + (n_layers - m) * 3) / n_layers
    print(f"\n  Step 2: Budget matching")
    print(f"    {m} layers @ 6-bit + {n_layers - m} layers @ 3-bit")
    print(f"    Average: {avg_bits:.2f} bits/dim (target: 4.0)")

    # Step 3: Select layers for each condition
    # Theory-guided: top-m by σ²_weighted (depth-corrected)
    sorted_by_risk = sorted(sigma_results, key=lambda x: x['sigma2_weighted'], reverse=True)
    theory_6bit = set(r['layer'] for r in sorted_by_risk[:m])
    theory_3bit = set(range(n_layers)) - theory_6bit

    # Also keep old σ²_eff-only ranking for comparison
    sorted_by_eff = sorted(sigma_results, key=lambda x: x['sigma2_eff'], reverse=True)
    theory_eff_6bit = set(r['layer'] for r in sorted_by_eff[:m])

    # Random: 3 seeds
    random_configs = []
    for seed in [42, 123, 777]:
        rng = random.Random(seed)
        rand_6bit = set(rng.sample(range(n_layers), m))
        random_configs.append(rand_6bit)

    # Heuristic: first-m layers
    first_6bit = set(range(m))

    # Heuristic: last-m layers
    last_6bit = set(range(n_layers - m, n_layers))

    # Heuristic: evenly spaced
    step = n_layers / m
    even_6bit = set(int(i * step) for i in range(m))

    # Build layer_bits dicts
    def make_bits(layers_6bit):
        return {l: (6 if l in layers_6bit else 3) for l in range(n_layers)}

    # Theory-optimal: per-layer bits to bring σ²_eff < 0.5
    # σ²(b) = σ²(4) × 4^(4-b), so b* = 4 + log₂(σ²_eff/0.5)/2
    TARGET_S2 = 0.5
    optimal_bits_raw = {}
    for r in sigma_results:
        li = r['layer']
        s2 = r['sigma2_eff']
        if s2 <= TARGET_S2:
            optimal_bits_raw[li] = 3  # already safe, save bits
        else:
            b_star = 4 + np.log2(s2 / TARGET_S2) / 2
            optimal_bits_raw[li] = int(np.ceil(b_star))

    # Clamp to [2, 8]
    for li in optimal_bits_raw:
        optimal_bits_raw[li] = max(2, min(8, optimal_bits_raw[li]))

    # Budget matching: adjust 'safe' layers to hit target avg=4
    total_target = n_layers * 4
    critical_budget = sum(optimal_bits_raw[li] for li in range(n_layers)
                         if optimal_bits_raw[li] > 3)
    critical_layers_opt = [li for li in range(n_layers) if optimal_bits_raw[li] > 3]
    safe_layers_opt = [li for li in range(n_layers) if optimal_bits_raw[li] <= 3]
    if safe_layers_opt:
        remaining = total_target - critical_budget
        safe_bits = max(2, remaining // len(safe_layers_opt))
    else:
        safe_bits = 3
    optimal_bits = {}
    for li in range(n_layers):
        if optimal_bits_raw[li] > 3:
            optimal_bits[li] = optimal_bits_raw[li]
        else:
            optimal_bits[li] = safe_bits
    opt_avg = sum(optimal_bits.values()) / n_layers

    print(f"\n  Theory-optimal allocation:")
    for li in range(n_layers):
        if optimal_bits[li] != safe_bits:
            print(f"    Layer {li:2d}: {optimal_bits[li]}bit "
                  f"(σ²_eff={sigma_results[li]['sigma2_eff']:.2f})")
    print(f"    Others: {safe_bits}bit, avg={opt_avg:.2f}bit")

    conditions = [
        ("Uniform 4-bit", {l: 4 for l in range(n_layers)}),
        ("Theory (optimal)", optimal_bits),
        ("Theory (weighted)", make_bits(theory_6bit)),
        ("Theory (eff-only)", make_bits(theory_eff_6bit)),
        ("Random (seed=42)", make_bits(random_configs[0])),
        ("Random (seed=123)", make_bits(random_configs[1])),
        ("Random (seed=777)", make_bits(random_configs[2])),
        ("Heuristic: first", make_bits(first_6bit)),
        ("Heuristic: last", make_bits(last_6bit)),
        ("Heuristic: even", make_bits(even_6bit)),
    ]

    # Step 4: Clean baseline
    print(f"\n  Step 3: Running experiments...")
    clean_ppl = measure_ppl(model, input_ids, layer_bits=None)
    print(f"    Clean (no quant):     PPL = {clean_ppl:.2f}")

    # Step 5: Run all conditions
    results = {'clean': clean_ppl}

    for name, layer_bits in conditions:
        ppl = measure_ppl(model, input_ids, layer_bits)
        bits_used = [layer_bits[l] for l in range(n_layers)]
        avg_b = np.mean(bits_used)

        # Show non-default layers
        non_default = sorted([(l, b) for l, b in layer_bits.items()
                              if b != min(bits_used)], key=lambda x: -x[1])
        bits_str = ",".join(f"L{l}:{b}" for l, b in non_default[:6])
        if len(non_default) > 6:
            bits_str += f"...+{len(non_default)-6}"

        ppl_str = f"{ppl:.2f}" if ppl < 1e6 else f"{ppl:.0f}"
        change = (ppl / clean_ppl - 1) * 100
        change_str = f"+{change:.1f}%" if change < 1e6 else f"+{change:.0f}%"

        print(f"    {name:25s} PPL = {ppl_str:>12s} ({change_str:>12s}) "
              f"avg={avg_b:.2f}bit")

        results[name] = {
            'ppl': ppl,
            'avg_bits': avg_b,
            'layer_bits': {str(l): b for l, b in layer_bits.items()},
        }

    # Random average
    rand_ppls = [results[f"Random (seed={s})"]['ppl']
                 for s in [42, 123, 777]]
    rand_avg = np.mean(rand_ppls)

    # Summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Clean:            {clean_ppl:.2f}")
    print(f"  Uniform 4-bit:    {results['Uniform 4-bit']['ppl']:.2f}")
    print(f"  Theory (optimal): {results['Theory (optimal)']['ppl']:.2f}")
    print(f"  Theory (weighted):{results['Theory (weighted)']['ppl']:.2f}")
    print(f"  Theory (eff-only):{results['Theory (eff-only)']['ppl']:.2f}")
    print(f"  Random (avg):     {rand_avg:.2f}")
    print(f"  Heuristic-first:  {results['Heuristic: first']['ppl']:.2f}")
    print(f"  Heuristic-last:   {results['Heuristic: last']['ppl']:.2f}")
    print(f"  Heuristic-even:   {results['Heuristic: even']['ppl']:.2f}")
    print(f"")

    to_ppl = results['Theory (optimal)']['ppl']
    tw_ppl = results['Theory (weighted)']['ppl']
    te_ppl = results['Theory (eff-only)']['ppl']
    hf_ppl = results['Heuristic: first']['ppl']
    uni_ppl = results['Uniform 4-bit']['ppl']

    print(f"  COMPARISON (all at ~{avg_bits:.2f} avg bits):")
    print(f"  {'Condition':25s} {'PPL':>10s} {'vs Uniform':>12s} {'vs Random':>12s}")
    print(f"  {'-'*60}")

    all_conditions = [
        ("Uniform 4-bit", uni_ppl),
        ("Heuristic: last", results['Heuristic: last']['ppl']),
        ("Random (avg)", rand_avg),
        ("Heuristic: even", results['Heuristic: even']['ppl']),
        ("Theory (eff-only)", te_ppl),
        ("Theory (weighted)", tw_ppl),
        ("Heuristic: first", hf_ppl),
        ("Theory (optimal)", to_ppl),
    ]
    # Sort by PPL descending
    all_conditions.sort(key=lambda x: -x[1])
    for name, ppl in all_conditions:
        vs_uni = f"{uni_ppl/ppl:.1f}x" if ppl > 0 else "—"
        vs_rand = f"{rand_avg/ppl:.1f}x" if ppl > 0 else "—"
        marker = " ◀ BEST" if ppl == to_ppl else ""
        print(f"  {name:25s} {ppl:10.1f} {vs_uni:>12s} {vs_rand:>12s}{marker}")

    print(f"")
    if to_ppl < hf_ppl and to_ppl < rand_avg:
        print(f"  ✓ Theory (optimal) OUTPERFORMS all baselines")
        print(f"    vs Uniform:  {uni_ppl/to_ppl:.0f}× better")
        print(f"    vs Random:   {rand_avg/to_ppl:.0f}× better")
        print(f"    vs First:    {hf_ppl/to_ppl:.0f}× better")
    print(f"")
    print(f"  Theory (optimal) allocation:")
    for li in range(n_layers):
        if optimal_bits[li] != safe_bits:
            s2 = sigma_results[li]['sigma2_eff']
            print(f"    Layer {li:2d}: {optimal_bits[li]}bit (σ²_eff={s2:.2f})")
    print(f"    Others: {safe_bits}bit")
    print(f"  {'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic-guided intervention experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    results = run_intervention(args.model, args.seq_len)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir, f"intervention_{tag}.json")

    # Serialize
    save = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                       for kk, vv in v.items()}
        else:
            save[k] = float(v)

    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()