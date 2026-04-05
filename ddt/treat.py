"""
Cause-Specific Treatment Pipeline
==================================
Optimal treatment = cheapest treatments first, then bit allocation on residual.

Pipeline:
  1. Diagnose: measure σ²_eff per layer under per-token 4-bit
  2. Apply free treatment: per-channel K quantization
  3. Re-diagnose: measure σ²_eff under per-channel
  4. Allocate K bits on residual σ²_eff only
  5. V at 3-bit everywhere (contractive, proven safe by Thm 2)
  6. Evaluate all combinations

Usage:
  python -m ddt.treat --model Qwen/Qwen2.5-7B
  python -m ddt.treat --model meta-llama/Llama-3.1-8B
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_uniform(x, bits=4, mode='per_token'):
    """Symmetric uniform quantization.
    mode='per_token':   one scale per token (dim=-1)
    mode='per_channel': one scale per dimension (dim=-2)
    """
    if mode == 'per_channel':
        scale = x.abs().amax(dim=-2, keepdim=True).clamp(min=1e-10)
    else:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    max_val = 2 ** (bits - 1) - 1
    return (x / scale * max_val).round().clamp(-max_val, max_val) * scale / max_val


def measure_sigma2(model, input_ids, bits=4, quant_mode='per_token'):
    """Measure σ²_eff per layer under given quantization mode."""
    n_layers = len(model.model.layers)
    attn0 = model.model.layers[0].self_attn
    n_heads = attn0.config.num_attention_heads
    n_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim
    heads_per_kv = n_heads // n_kv_heads
    T = input_ids.shape[1]

    captures = {}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        def make_hook(li, name):
            def hook(module, input, output):
                if li not in captures:
                    captures[li] = {}
                captures[li][name] = output.detach().float()
            return hook
        hooks.append(attn.q_proj.register_forward_hook(make_hook(idx, 'q')))
        hooks.append(attn.k_proj.register_forward_hook(make_hook(idx, 'k')))

    with torch.no_grad():
        out = model(input_ids, use_cache=False, output_attentions=True)
    for h in hooks:
        h.remove()

    results = []
    for li in range(n_layers):
        q_raw = captures[li]['q'].squeeze(0)
        k_raw = captures[li]['k'].squeeze(0)
        k_q = quantize_uniform(k_raw.unsqueeze(0), bits, mode=quant_mode).squeeze(0)
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
                sigma2_list.append(delta.var(dim=-1).mean().item())

        sigma2_avg = np.mean(sigma2_list)
        A = out.attentions[li].squeeze(0).float()
        eta_V = (A ** 2).sum(dim=-1).mean().item()
        n_eff = 1.0 / max(eta_V, 1e-10)
        s2_eff = sigma2_avg * max(1.0 - 1.0 / n_eff, 0.0)

        q_norm = (q_raw.view(T, n_heads, head_dim) ** 2).sum(dim=-1).mean().item() / head_dim

        results.append({
            'layer': li, 'sigma2_eff': s2_eff, 'q_norm': q_norm,
            'noise_factor': sigma2_avg / max(q_norm, 1e-10),
        })
    return results


def compute_optimal_bits(sigma2_eff, target=0.5, b_min=2, b_max=8):
    if sigma2_eff <= target:
        return 4
    b = int(np.ceil(4 + np.log2(sigma2_eff / target) / 2))
    return max(b_min, min(b_max, b))


def evaluate(model, input_ids, k_config, v_bits=None):
    """k_config: {layer: (bits, mode)} for K. v_bits: int or None."""
    n_layers = len(model.model.layers)
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if idx in k_config:
            kb, km = k_config[idx]
            def make_k_hook(bits, mode):
                def hook(mod, inp, out):
                    return quantize_uniform(out, bits, mode=mode)
                return hook
            hooks.append(attn.k_proj.register_forward_hook(make_k_hook(kb, km)))
        if v_bits is not None:
            def make_v_hook(bits):
                def hook(mod, inp, out):
                    return quantize_uniform(out, bits, mode='per_token')
                return hook
            hooks.append(attn.v_proj.register_forward_hook(make_v_hook(v_bits)))

    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    logits = out.logits
    sl = logits[:, :-1, :].contiguous()
    lab = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1)).item()
    return np.exp(loss) if loss < 20 else float('inf')


def run_treatment(model_name, seq_len=512, target_sigma2=0.5,
                  output_dir="results/ddt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  Cause-Specific Treatment Pipeline")
    print(f"  Model: {model_name}")
    print(f"  Target σ²_eff: {target_sigma2}")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = len(model.model.layers)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)

    # ====================================
    # Phase 1-2: Diagnose under both modes
    # ====================================
    print(f"\n  Phase 1: Diagnosing...")
    diag_pt = measure_sigma2(model, input_ids, bits=4, quant_mode='per_token')
    diag_pc = measure_sigma2(model, input_ids, bits=4, quant_mode='per_channel')

    print(f"\n  {'L':>3} {'σ²_pt':>8} {'σ²_pc':>8} {'reduc':>7} {'status':>6}")
    print(f"  {'-'*40}")
    for li in range(n_layers):
        s_pt = diag_pt[li]['sigma2_eff']
        s_pc = diag_pc[li]['sigma2_eff']
        red = s_pt / max(s_pc, 1e-10) if s_pc > 0.001 else float('inf')
        st = "CRIT" if s_pc > 1.0 else ("MOD" if s_pc > 0.1 else "safe")
        if s_pt > 0.3 or s_pc > 0.3:
            print(f"  {li:3d} {s_pt:8.2f} {s_pc:8.2f} {red:6.1f}× {st:>6}")

    crit_pt = sum(1 for r in diag_pt if r['sigma2_eff'] > 1.0)
    crit_pc = sum(1 for r in diag_pc if r['sigma2_eff'] > 1.0)
    print(f"\n  CRITICAL: {crit_pt} (per-token) → {crit_pc} (per-channel)")

    # ====================================
    # Phase 3: Prescribe per-layer treatment
    # ====================================
    print(f"\n  Phase 2: Prescribing...")

    # Treatment A: per-token + optimal bits
    alloc_A = {li: (compute_optimal_bits(diag_pt[li]['sigma2_eff'], target_sigma2),
                    'per_token') for li in range(n_layers)}
    avg_A = np.mean([b for b, m in alloc_A.values()])

    # Treatment B: per-channel + optimal bits (stacked)
    alloc_B = {li: (compute_optimal_bits(diag_pc[li]['sigma2_eff'], target_sigma2),
                    'per_channel') for li in range(n_layers)}
    avg_B = np.mean([b for b, m in alloc_B.values()])

    print(f"    Treatment A (per-token + opt):   avg K = {avg_A:.2f}bit")
    print(f"    Treatment B (per-channel + opt):  avg K = {avg_B:.2f}bit")
    print(f"    Per-channel saves: {avg_A - avg_B:.2f} K bits/layer")

    print(f"\n  Per-layer prescription:")
    for li in range(n_layers):
        b_a = alloc_A[li][0]
        b_b = alloc_B[li][0]
        s_pt = diag_pt[li]['sigma2_eff']
        s_pc = diag_pc[li]['sigma2_eff']
        if b_a != 4 or b_b != 4:
            save = b_a - b_b
            save_s = f" [{save}bit saved]" if save > 0 else ""
            print(f"    L{li:2d}: per-token→{b_a}bit, per-channel→{b_b}bit  "
                  f"(σ²: {s_pt:.1f}→{s_pc:.1f}){save_s}")

    # ====================================
    # Phase 4: Evaluate progressively
    # ====================================
    print(f"\n  Phase 3: Evaluating (progressive improvement)...\n")

    clean = evaluate(model, input_ids, {}, v_bits=None)

    # Build conditions progressively
    uni_pt4 = {l: (4, 'per_token') for l in range(n_layers)}
    uni_pc4 = {l: (4, 'per_channel') for l in range(n_layers)}

    results = {}

    conditions = [
        ("1. Baseline: pt uniform K4, V4",    uni_pt4, 4, 4.0, 4.0),
        ("2. +per-channel K (free)",          uni_pc4, 4, 4.0, 4.0),
        ("3. +optimal K bits",               alloc_B, None, avg_B, None),
        ("4. +V at 3-bit (contractive)",     alloc_B, 3, avg_B, 3.0),
    ]

    print(f"    {'Step':40s} {'PPL':>8s} {'K avg':>6s} {'V':>4s} {'K+V':>6s}")
    print(f"    {'-'*68}")
    print(f"    {'Clean':40s} {clean:8.2f} {'—':>6s} {'—':>4s} {'—':>6s}")

    prev_ppl = None
    for name, k_cfg, v_b, k_avg, v_avg in conditions:
        ppl = evaluate(model, input_ids, k_cfg, v_bits=v_b)
        total = (k_avg + (v_avg if v_avg else 0))
        v_str = f"{v_b}" if v_b else "—"
        total_str = f"{total:.1f}" if v_avg else "—"
        delta = ""
        if prev_ppl and ppl < prev_ppl:
            delta = f"  ↓{(1-ppl/prev_ppl)*100:.0f}%"
        elif prev_ppl and ppl >= prev_ppl:
            delta = f"  ={ppl/prev_ppl:.2f}×"
        print(f"    {name:40s} {ppl:8.2f} {k_avg:6.2f} {v_str:>4s} {total_str:>6s}{delta}")
        results[name] = ppl
        prev_ppl = ppl

    # ====================================
    # Summary
    # ====================================
    baseline_ppl = results["1. Baseline: pt uniform K4, V4"]
    full_ppl = results["4. +V at 3-bit (contractive)"]

    print(f"\n  {'='*65}")
    print(f"  TREATMENT SUMMARY")
    print(f"  {'='*65}")
    print(f"  Clean:              {clean:.2f}")
    print(f"  Before treatment:   {baseline_ppl:.2f}  (K4+V4 = 8 bits/dim)")
    print(f"  After treatment:    {full_ppl:.2f}  (K{avg_B:.1f}+V3 = {avg_B+3:.1f} bits/dim)")
    print(f"")

    if baseline_ppl > 100:
        print(f"  PPL improvement:    {baseline_ppl:.0f} → {full_ppl:.2f} "
              f"({baseline_ppl/full_ppl:.0f}× better)")
    else:
        print(f"  PPL improvement:    {baseline_ppl:.2f} → {full_ppl:.2f} "
              f"({(1-full_ppl/baseline_ppl)*100:.1f}% better)")
    print(f"  Memory savings:     {(1-(avg_B+3)/8)*100:.1f}% vs uniform K4+V4")
    print(f"  Quality overhead:   +{(full_ppl/clean-1)*100:.1f}% vs clean")
    print(f"")
    print(f"  Treatment stack (each step's contribution):")
    r = list(results.values())
    print(f"    Per-channel (free):     {r[0]:.1f} → {r[1]:.1f}")
    print(f"    Optimal K bits:         {r[1]:.1f} → {r[2]:.1f}")
    print(f"    V 3-bit (safe):         {r[2]:.1f} → {r[3]:.1f}")
    print(f"  {'='*65}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tag = model_name.replace("/", "_")
    path = os.path.join(output_dir, f"treatment_{tag}.json")
    save = {
        'model': model_name, 'clean_ppl': clean,
        'baseline_ppl': baseline_ppl, 'full_ppl': full_ppl,
        'avg_k_bits': avg_B, 'v_bits': 3,
        'total_bits': avg_B + 3,
        'results': {k: v for k, v in results.items()},
        'allocation': {str(l): {'bits': alloc_B[l][0], 'mode': alloc_B[l][1]}
                      for l in range(n_layers)},
    }
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--target-sigma2", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()
    run_treatment(args.model, args.seq_len, args.target_sigma2, args.output_dir)


if __name__ == "__main__":
    main()