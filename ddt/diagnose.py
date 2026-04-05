"""
Complete KV Cache Quantization Diagnostic
==========================================
Unified diagnostic measuring both K-path and V-path vulnerability.

K-path: σ²_δ = Var(q·Δk/√d)
  → D_K ≤ (exp(σ²_δ) - 1) · η_V
  → σ²_δ < 1: safe (linear regime)
  → σ²_δ > 1: catastrophic (exponential blowup)

V-path: D_V = (1/T) Σⱼ cⱼ · dⱼ
  → Always bounded (linear propagation)
  → cⱼ = column attention concentration
  → dⱼ = per-token quantization MSE

Output: Per-layer diagnosis + bottleneck identification + overall verdict.

Usage:
  python -m ddt.diagnose --model meta-llama/Llama-3.1-8B
  python -m ddt.diagnose --model Qwen/Qwen2.5-7B
  python -m ddt.diagnose --model mistralai/Mistral-7B-v0.3
"""

import argparse
import json
import os

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


def run_diagnostic(model_name, bits=4, seq_len=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  KV Cache Quantization Diagnostic")
    print(f"  Model: {model_name} @ {bits}bit")
    print(f"{'='*70}")

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
    print(f"  T={T}, heads={n_heads}Q/{n_kv_heads}KV, d={head_dim}, L={n_layers}")

    # =========================================================
    # Phase 1: Capture Q, K, V, and attention weights
    # =========================================================
    print(f"\n  Phase 1: Capturing Q, K, V, attention weights...")

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

        def make_v_hook(li):
            def hook(module, input, output):
                if li not in captures:
                    captures[li] = {}
                captures[li]['v'] = output.detach().float()
            return hook

        hooks.append(attn.q_proj.register_forward_hook(make_q_hook(idx)))
        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(idx)))
        hooks.append(attn.v_proj.register_forward_hook(make_v_hook(idx)))

    # Run with output_attentions AND use_cache to get both attention weights and post-RoPE keys
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=True)

    attn_weights = out.attentions  # tuple of (1, n_heads, T, T) per layer
    past_kv = out.past_key_values  # post-RoPE KV cache

    for h in hooks:
        h.remove()

    # Also compute clean PPL
    clean_logits = out.logits
    shift_logits = clean_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    clean_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)).item()
    clean_ppl = np.exp(clean_loss)
    print(f"  Clean PPL: {clean_ppl:.2f}")

    # =========================================================
    # Phase 1b: Post-RoPE key outlier analysis
    # =========================================================
    print(f"  Phase 1b: Post-RoPE key analysis... (cache type: {type(past_kv).__name__})")

    post_rope_results = []
    for li in range(n_layers):
        # Access post-RoPE keys from DynamicCache
        k_post = None
        try:
            # transformers 5.x: DynamicCache with .layers[li].keys
            k_post = past_kv.layers[li].keys
        except (AttributeError, IndexError):
            pass

        if k_post is None:
            try:
                # Older: past_kv.key_cache[li]
                k_post = past_kv.key_cache[li]
            except (AttributeError, IndexError):
                pass

        if k_post is None:
            try:
                # Oldest: tuple of (key, value)
                k_post = past_kv[li][0]
            except (IndexError, TypeError):
                pass

        if k_post is None:
            post_rope_results.append({
                'layer': li,
                'post_rope_outlier_ratio': 0.0,
                'post_rope_outlier_max': 0.0,
            })
            continue

        k_post = k_post.squeeze(0).float()  # (n_kv_heads, T, head_dim)

        # Per-head post-RoPE outlier ratio
        post_ratios = []
        for kv_h in range(min(k_post.shape[0], n_kv_heads)):
            k_h = k_post[kv_h]  # (T, head_dim)
            per_dim_var = k_h.var(dim=0)  # (head_dim,)
            ratio = (per_dim_var.max() / per_dim_var.mean().clamp(min=1e-10)).item()
            post_ratios.append(ratio)

        post_rope_results.append({
            'layer': li,
            'post_rope_outlier_ratio': np.mean(post_ratios),
            'post_rope_outlier_max': np.max(post_ratios),
        })

    # =========================================================
    # Phase 2: K-path diagnostic (σ²_δ)
    # =========================================================
    print(f"\n  Phase 2: K-path (σ²_δ)...")

    k_results = []
    for li in range(n_layers):
        q_raw = captures[li]['q'].squeeze(0)  # (T, n_heads * d)
        k_raw = captures[li]['k'].squeeze(0)  # (T, n_kv * d)

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
        sigma2_max = np.max(sigma2_list)
        pred_chi2 = np.exp(min(sigma2_avg, 50)) - 1

        # Outlier ratio: max/mean of per-dimension K variance
        # High → outlier dims dominate per-token scale → rotation helps
        k_heads = k_raw.view(T, n_kv_heads, head_dim)
        dim_var_list = []
        for kv_h in range(n_kv_heads):
            k_h = k_heads[:, kv_h, :]  # (T, d)
            per_dim_var = k_h.var(dim=0)  # (d,)
            ratio = (per_dim_var.max() / per_dim_var.mean()).item()
            dim_var_list.append(ratio)
        k_outlier_ratio = np.mean(dim_var_list)

        # Query norm: average ‖q‖²/d per head
        q_heads = q_raw.view(T, n_heads, head_dim)
        q_norm_list = []
        for q_h in range(n_heads):
            qn = (q_heads[:, q_h, :] ** 2).sum(dim=-1).mean().item() / head_dim
            q_norm_list.append(qn)
        q_norm_avg = np.mean(q_norm_list)
        q_norm_max = np.max(q_norm_list)

        k_results.append({
            'layer': li,
            'sigma2_mean': sigma2_avg,
            'sigma2_max': sigma2_max,
            'pred_chi2': pred_chi2,
            'k_outlier_ratio': k_outlier_ratio,
            'q_norm_avg': q_norm_avg,
            'q_norm_max': q_norm_max,
        })

    # =========================================================
    # Phase 3: V-path diagnostic (D_V = (1/T) Σ cⱼdⱼ)
    # =========================================================
    print(f"  Phase 3: V-path (D_V)...")

    v_results = []
    for li in range(n_layers):
        v_raw = captures[li]['v'].squeeze(0)  # (T, n_kv * d)

        # Quantize V
        v_q = quantize_uniform(v_raw.unsqueeze(0), bits).squeeze(0)
        dv = v_q - v_raw

        # Per-token MSE: dⱼ = ‖Δv_j‖²/d
        d_j = (dv ** 2).mean(dim=-1)  # (T,) — avg across all dims

        # Attention weights for this layer
        A = attn_weights[li].squeeze(0).float()  # (n_heads, T, T)

        # Column concentration: cⱼ = Σᵢ aᵢⱼ²
        # Average over heads
        c_j_per_head = (A ** 2).sum(dim=1)  # (n_heads, T)
        c_j = c_j_per_head.mean(dim=0)  # (T,) — avg over heads

        # D_V = (1/T) Σⱼ cⱼ · dⱼ
        D_V = (c_j * d_j).sum().item() / T

        # η_V = (1/T) Σᵢ ‖aᵢ‖²₂ = (1/T) Σⱼ cⱼ (by definition)
        eta_V = c_j.sum().item() / T

        # Homoscedastic surrogate: D_homo = η_V · mean(dⱼ)
        D_homo = eta_V * d_j.mean().item()

        # n_eff: effective number of attended tokens (per query avg)
        # n_eff_i = 1/‖aᵢ‖₂² → σ²_eff = σ²_δ · (1 - 1/n_eff)
        n_eff = 1.0 / max(eta_V, 1e-10)

        # λ_max(C_V): attention-weighted value covariance top eigenvalue
        # C_V = Σⱼ ā_j vⱼ vⱼᵀ  (d × d)
        # ā_j = (1/T) Σᵢ aᵢⱼ = column average attention
        v_heads = v_raw.view(T, n_kv_heads, head_dim)

        col_avg = A.mean(dim=0).mean(dim=0)  # (T,) — avg over heads & queries

        lambda_max_list = []
        trace_list = []
        for kv_h in range(n_kv_heads):
            v_h = v_heads[:, kv_h, :]  # (T, d)
            # C_V = Vᵀ diag(w) V where w = col_avg
            v_weighted = v_h * col_avg.unsqueeze(1).sqrt()  # (T, d)
            C_V_h = v_weighted.T @ v_weighted  # (d, d)
            eigs = torch.linalg.eigvalsh(C_V_h)
            lambda_max_list.append(eigs[-1].item())
            trace_list.append(eigs.sum().item())

        lambda_max = np.mean(lambda_max_list)
        trace_cv = np.mean(trace_list)
        d_eff = trace_cv / max(lambda_max, 1e-10)

        v_results.append({
            'layer': li,
            'D_V': D_V,
            'eta_V': eta_V,
            'D_homo': D_homo,
            'd_j_mean': d_j.mean().item(),
            'd_j_max': d_j.max().item(),
            'n_eff': n_eff,
            'lambda_max': lambda_max,
            'trace_cv': trace_cv,
            'd_eff': d_eff,
        })

    # =========================================================
    # Phase 4: Diagnosis (proper 4-level path)
    # =========================================================

    # Baselines (derived from safe models: Mistral/Llama)
    QUERY_BASELINE = 1.5
    NOISE_BASELINE = 0.3

    print(f"\n  {'='*70}")
    print(f"  LAYER-BY-LAYER DIAGNOSIS")
    print(f"  {'='*70}")
    print(f"  {'L':>3} {'σ²_eff':>8} {'σ²_wt':>8} {'risk':>8} {'cause':>10} "
          f"{'q_fac':>6} {'n_fac':>6} "
          f"{'D_K':>10} {'D_V':>10}")
    print(f"  {'-'*80}")

    layer_results = []
    for li in range(n_layers):
        kr = k_results[li]
        vr = v_results[li]

        s2 = kr['sigma2_mean']
        pred = kr['pred_chi2']
        dv = vr['D_V']
        eta = vr['eta_V']
        n_eff = vr['n_eff']
        lam_max = vr['lambda_max']
        trace_cv = vr['trace_cv']
        d_eff = vr['d_eff']

        outlier_r = kr['k_outlier_ratio']
        q_norm = kr['q_norm_avg']
        post_r = post_rope_results[li]['post_rope_outlier_ratio']

        # --- Level 1: Risk (σ²_eff) ---
        s2_eff = s2 * max(1.0 - 1.0/n_eff, 0.0)

        # Depth-corrected: early layers affect more downstream layers
        s2_weighted = s2_eff * (n_layers - li) / n_layers

        if s2_eff < 0.1:
            risk = "SAFE"
        elif s2_eff < 1.0:
            risk = "MODERATE"
        else:
            risk = "CRITICAL"

        # --- Level 2: Factor decomposition ---
        query_factor = q_norm  # ‖q‖²/d
        noise_factor = s2 / max(q_norm, 1e-10)  # E[Δk²] implied

        q_high = query_factor > 2 * QUERY_BASELINE  # > 3.0
        n_high = noise_factor > 2 * NOISE_BASELINE   # > 0.6

        # --- Level 2+3: Cause classification ---
        if risk == "SAFE":
            cause = "—"
        elif q_high and n_high:
            # COMPOUND: both factors abnormal → sub-classify noise
            if post_r > 20:
                cause = "COMP+ROPE"
            elif outlier_r > 10:
                cause = "COMP+OUTL"
            else:
                cause = "COMP+BIT"
        elif q_high:
            cause = "QUERY"
        elif n_high:
            # NOISE dominant → sub-classify
            if post_r > 20:
                cause = "ROPE"
            elif outlier_r > 10:
                cause = "OUTLIER"
            else:
                cause = "NOISE-BIT"
        else:
            cause = "BIT"

        # --- D_K bounds ---
        D_K_loose = pred * trace_cv
        D_K_tight = pred * lam_max

        layer_results.append({
            'layer': li,
            'sigma2_delta': s2,
            'sigma2_eff': s2_eff,
            'sigma2_weighted': s2_weighted,
            'sigma2_max_head': kr['sigma2_max'],
            'risk': risk,
            'query_factor': query_factor,
            'noise_factor': noise_factor,
            'cause': cause,
            'k_outlier_ratio': outlier_r,
            'post_rope_outlier_ratio': post_r,
            'q_norm': q_norm,
            'D_V': dv,
            'D_K_loose': D_K_loose,
            'D_K_tight': D_K_tight,
            'eta_V': eta,
            'n_eff': n_eff,
            'lambda_max': lam_max,
            'trace_cv': trace_cv,
            'd_eff': d_eff,
        })

        flag = " ***" if risk == "CRITICAL" else ""
        print(f"  {li:3d} {s2_eff:8.4f} {s2_weighted:8.4f} {risk:>8} {cause:>10} "
              f"{query_factor:6.2f} {noise_factor:6.2f} "
              f"{D_K_tight:10.2e} {dv:10.2e}{flag}")

    # =========================================================
    # Phase 5: Overall summary
    # =========================================================
    all_s2 = [r['sigma2_delta'] for r in layer_results]
    all_s2_eff = [r['sigma2_eff'] for r in layer_results]
    all_s2_wt = [r['sigma2_weighted'] for r in layer_results]
    all_dv = [r['D_V'] for r in layer_results]
    all_d_eff = [r['d_eff'] for r in layer_results]
    all_n_eff = [r['n_eff'] for r in layer_results]
    all_qf = [r['query_factor'] for r in layer_results]
    all_nf = [r['noise_factor'] for r in layer_results]

    crit_layers = sum(1 for r in layer_results if r['risk'] == 'CRITICAL')
    mod_layers = sum(1 for r in layer_results if r['risk'] == 'MODERATE')

    max_s2_eff = max(all_s2_eff)
    mean_s2_eff = np.mean(all_s2_eff)
    max_s2 = max(all_s2)
    mean_s2 = np.mean(all_s2)

    if max_s2_eff < 0.1:
        overall = "SAFE"
    elif max_s2_eff < 1.0:
        overall = "MODERATE"
    elif crit_layers <= 2:
        overall = "RISKY"
    else:
        overall = "CRITICAL"

    print(f"\n  {'='*70}")
    print(f"  OVERALL DIAGNOSIS: {model_name}")
    print(f"  {'='*70}")
    print(f"  Clean PPL:           {clean_ppl:.2f}")
    print(f"  Overall risk:        {overall}")
    print(f"")
    print(f"  σ²_δ mean/max:       {mean_s2:.4f} / {max_s2:.4f}")
    print(f"  σ²_eff mean/max:     {mean_s2_eff:.4f} / {max_s2_eff:.4f}")
    print(f"  σ²_weighted mean/max:{np.mean(all_s2_wt):.4f} / {np.max(all_s2_wt):.4f}  (depth-corrected)")
    print(f"  CRITICAL layers:     {crit_layers}/{n_layers}")
    print(f"  MODERATE layers:     {mod_layers}/{n_layers}")
    print(f"")
    print(f"  Factor analysis:")
    print(f"    Query factor:      mean={np.mean(all_qf):.2f}  max={np.max(all_qf):.2f}  "
          f"(baseline={QUERY_BASELINE})")
    print(f"    Noise factor:      mean={np.mean(all_nf):.2f}  max={np.max(all_nf):.2f}  "
          f"(baseline={NOISE_BASELINE})")
    print(f"")
    all_post_r = [r['post_rope_outlier_ratio'] for r in layer_results]
    all_pre_r = [r['k_outlier_ratio'] for r in layer_results]
    print(f"  Outlier analysis:")
    print(f"    Pre-RoPE ratio:    mean={np.mean(all_pre_r):.1f}  max={np.max(all_pre_r):.1f}")
    print(f"    Post-RoPE ratio:   mean={np.mean(all_post_r):.1f}  max={np.max(all_post_r):.1f}")
    print(f"    RoPE amplification:{np.mean(all_post_r)/max(np.mean(all_pre_r),0.01):.1f}x")
    print(f"")

    # Cause breakdown
    cause_counts = {}
    for r in layer_results:
        c = r['cause']
        if c != '—':
            cause_counts[c] = cause_counts.get(c, 0) + 1
    print(f"  Cause breakdown:")
    if cause_counts:
        for c, n in sorted(cause_counts.items(), key=lambda x: -x[1]):
            print(f"    {c:>12}: {n} layers")
    else:
        print(f"    All layers SAFE.")
    print(f"")

    # --- Level 4: Treatment prescription ---
    print(f"  PRESCRIPTION:")
    if overall == "SAFE":
        print(f"    Uniform {bits}bit K/V allocation is safe.")
    elif overall == "MODERATE":
        print(f"    K:{bits+1}bit / V:{bits-1}bit asymmetric allocation recommended.")
        if any('QUERY' in r['cause'] for r in layer_results):
            print(f"    Query-aware layers: give K extra bit where q_norm > {2*QUERY_BASELINE:.1f}.")
    elif overall == "RISKY":
        crit_list = [r for r in layer_results if r['risk'] == 'CRITICAL']
        for r in crit_list:
            print(f"    Layer {r['layer']} ({r['cause']}): ", end="")
            if 'QUERY' in r['cause']:
                print(f"K needs {bits+2}bit (q_norm={r['q_norm']:.1f})")
            elif 'ROPE' in r['cause']:
                print(f"per-channel K quant or PolarQuant")
            elif 'OUTL' in r['cause']:
                print(f"Hadamard rotation on K")
            else:
                print(f"K needs {bits+1}-{bits+2}bit")
    else:  # CRITICAL
        print(f"    {bits}bit per-token K quantization will FAIL.")
        print(f"")
        # Group treatments by cause
        has_query = any('QUERY' in r['cause'] or 'COMP' in r['cause']
                       for r in layer_results if r['risk'] == 'CRITICAL')
        has_rope = any('ROPE' in r['cause']
                      for r in layer_results if r['risk'] != 'SAFE')
        has_outlier = any('OUTL' in r['cause']
                         for r in layer_results if r['risk'] != 'SAFE')
        has_bit = any(r['cause'] in ('BIT', 'NOISE-BIT', 'COMP+BIT')
                     for r in layer_results if r['risk'] != 'SAFE')

        if has_query:
            q_crit = [r for r in layer_results
                     if r['risk'] == 'CRITICAL' and 'QUERY' in r['cause'] or 'COMP' in r['cause']]
            print(f"    [QUERY] Layers with abnormal query norm (>{2*QUERY_BASELINE:.1f}):")
            print(f"      → Query-aware K bit allocation: "
                  f"give {bits+2}-{bits+3}bit to these layers.")
        if has_rope:
            print(f"    [ROPE] Post-RoPE outlier detected:")
            print(f"      → Per-channel K quantization or PolarQuant.")
        if has_outlier:
            print(f"    [OUTLIER] Pre-RoPE outlier detected:")
            print(f"      → Hadamard rotation (QuaRot/RHT) on K.")
        if has_bit and not has_query and not has_rope and not has_outlier:
            print(f"    [BIT] Insufficient precision:")
            print(f"      → Increase K to ≥{bits+2}bit.")

    print(f"  {'='*70}")

    return layer_results, {
        'model': model_name,
        'bits': bits,
        'clean_ppl': clean_ppl,
        'overall': overall,
        'sigma2_mean': mean_s2,
        'sigma2_max': max_s2,
        'sigma2_eff_mean': mean_s2_eff,
        'sigma2_eff_max': max_s2_eff,
        'query_factor_mean': float(np.mean(all_qf)),
        'noise_factor_mean': float(np.mean(all_nf)),
        'n_eff_mean': float(np.mean(all_n_eff)),
        'd_eff_mean': float(np.mean(all_d_eff)),
        'critical_layers': crit_layers,
        'cause_counts': cause_counts,
        'D_V_mean': float(np.mean(all_dv)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Complete KV cache quantization diagnostic")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    results, summary = run_diagnostic(args.model, args.bits, args.seq_len)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir,
                        f"diagnosis_{tag}_{args.bits}bit.json")

    save = {
        'summary': summary,
        'layers': [{
            'layer': r['layer'],
            'sigma2_delta': r['sigma2_delta'],
            'sigma2_eff': r['sigma2_eff'],
            'sigma2_weighted': r['sigma2_weighted'],
            'risk': r['risk'],
            'query_factor': r['query_factor'],
            'noise_factor': r['noise_factor'],
            'cause': r['cause'],
            'k_outlier_ratio': r['k_outlier_ratio'],
            'post_rope_outlier_ratio': r['post_rope_outlier_ratio'],
            'q_norm': r['q_norm'],
            'D_V': r['D_V'],
            'D_K_loose': r['D_K_loose'],
            'D_K_tight': r['D_K_tight'],
            'n_eff': r['n_eff'],
            'd_eff': r['d_eff'],
        } for r in results],
    }
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()