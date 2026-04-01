"""
Theorem 1 Validation
=====================
Validates the attention-output distortion decomposition:

    D_attn ≈ η_K · D_K + η_V · D_V

where D_K, D_V are per-dim quantization MSE of K, V respectively.
D_pred uses empirical MSE (not theoretical c·σ²·4^{-b}), isolating
the Thm 1 decomposition accuracy from E₈ high-rate model accuracy.

Three validation modes:

  V-path (default, offline):
    Captures A and V from a single forward pass, quantizes V offline,
    recomputes A @ V_hat, and checks:
        η_V_empirical  =  MSE(A@V, A@V_hat) / MSE(V, V_hat)  ≈  η_V_predicted

    Exact for the fixed-attention V-branch (A held constant).
    Cost: 1 forward pass total.

  K+V total (--measure-actual, hook-based):
    For each layer, hooks k_proj and v_proj to inject E₈-quantized
    values, then captures the attention output.  Compares:
        D_actual  vs  D_pred = η_K · D_K_emp + η_V · D_V_emp

    Cost: 2 forward passes per layer (clean + quantized).

  Decomposed (--measure-actual --decompose):
    Additionally runs K-only and V-only quantized passes, enabling:
      - Independent K-path / V-path validation
      - Additivity check: D_K_only + D_V_only ≈ D_KV (cross-terms small?)

    Cost: 4 forward passes per layer.

Usage:
  python allocation/thm1_validate.py \\
      --model meta-llama/Llama-3.1-8B --load-in-8bit \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json --bits 4

  python allocation/thm1_validate.py \\
      --model meta-llama/Llama-3.1-8B --load-in-8bit \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json --bits 4 \\
      --measure-actual --decompose
"""

import torch
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, '..'))

from sensitivity import ModelSpec
from core.e8_quantizer import encode_e8


# ============================================================
# Offline V-path validation
# ============================================================

def quantize_tensor_e8(tensor: torch.Tensor, bits: int) -> tuple:
    """
    Quantize a (B, heads, T, d) tensor with E₈ at `bits` per dim.
    Returns (quantized_tensor, per_dim_mse).
    All computation on GPU — does not call scalar compute_scale.
    """
    B, H, T, d = tensor.shape
    assert d % 8 == 0

    t = tensor.float()
    # Per-head second moment → scale (same formula as compressed_kv_cache)
    t_flat = t.reshape(B * H, T * (d // 8), 8)
    sigma2 = (t_flat ** 2).mean(dim=(1, 2))             # (B*H,)
    scale_sq = 2 * torch.pi * torch.e * sigma2 * (4.0 ** (-bits))
    scale = scale_sq.sqrt()                              # (B*H,)
    scale = torch.where(sigma2 < 1e-12, torch.ones_like(scale), scale)

    # Normalize, quantize, denormalize
    scale_exp = scale.unsqueeze(-1).unsqueeze(-1)         # (B*H, 1, 1)
    normed = t_flat / scale_exp
    q = encode_e8(normed.reshape(-1, 8)).reshape(t_flat.shape)
    recon = q * scale_exp

    # Reshape back
    recon_full = recon.reshape(B, H, T, d)
    mse = ((t - recon_full) ** 2).mean().item()

    return recon_full, mse


def validate_v_path(
    A: torch.Tensor,        # (B, n_heads, T, T)
    V: torch.Tensor,        # (B, T, n_kv_heads * head_dim)  — raw v_proj output
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    bits: int,
    eta_V_predicted: float,
) -> dict:
    """
    Offline V-path validation for one layer.

    Three predictions compared to D_V_attn (actual attention-output MSE):
      1. homo:  η_V · D_V_raw  (homoscedastic, original Thm 1)
      2. hetero: Σᵢ Σⱼ aᵢⱼ² · e²ⱼ  (uses ACTUAL per-token MSE, assumes uncorrelated)
      3. actual: ‖A(V−V̂)‖²  (includes both heteroscedasticity and cross-correlation)

    Comparing these tells us the source of any gap:
      homo ≈ hetero ≈ actual  → Thm 1 is accurate
      homo ≠ hetero ≈ actual  → gap from per-token heteroscedasticity
      homo ≈ hetero ≠ actual  → gap from cross-token error correlation
    """
    B = A.shape[0]
    T = A.shape[2]
    group_size = n_heads // n_kv_heads
    BH = B * n_heads

    # Reshape V to per-head: (B, n_kv, T, d)
    V_h = V.float().view(B, T, n_kv_heads, head_dim).permute(0, 2, 1, 3)

    # Quantize V
    V_hat, D_V_raw = quantize_tensor_e8(V_h, bits)

    # ---- Per-token MSE diagnostic ----
    delta_V = V_h - V_hat                                    # (B, n_kv, T, d)
    per_token_mse = (delta_V ** 2).mean(dim=-1)              # (B, n_kv, T)
    token_mse_cv = (per_token_mse.std() / per_token_mse.mean()).item()

    # ---- Expand for GQA ----
    V_exp = V_h.repeat_interleave(group_size, dim=1)          # (B, H, T, d)
    V_hat_exp = V_hat.repeat_interleave(group_size, dim=1)
    per_token_mse_exp = per_token_mse.repeat_interleave(group_size, dim=1)  # (B, H, T)

    del V_h, V_hat, delta_V

    # ---- Actual attention-output distortion ----
    A_f = A.float()
    O_clean = torch.bmm(
        A_f.reshape(BH, T, T), V_exp.reshape(BH, T, head_dim),
    ).reshape(B, n_heads, T, head_dim)
    O_hat = torch.bmm(
        A_f.reshape(BH, T, T), V_hat_exp.reshape(BH, T, head_dim),
    ).reshape(B, n_heads, T, head_dim)
    D_V_attn = ((O_clean - O_hat) ** 2).mean().item()

    del O_clean, O_hat, V_exp, V_hat_exp

    # ---- Prediction 1: homoscedastic (η_V · D_V_raw) ----
    D_homo = eta_V_predicted * D_V_raw

    # ---- Prediction 2: heteroscedastic (uses actual per-token MSE, assumes uncorrelated) ----
    # D_hetero = (1/(T·H)) Σ_h Σ_i Σ_j a²_ij · e²_j
    A_sq = A_f ** 2
    D_hetero = torch.bmm(
        A_sq.reshape(BH, T, T),
        per_token_mse_exp.reshape(BH, T, 1),
    ).reshape(B, n_heads, T).mean().item()

    del A_sq, A_f, per_token_mse_exp

    # ---- Decompose the gap ----
    # If D_homo >> D_hetero ≈ D_actual → heteroscedasticity is the issue
    # If D_homo ≈ D_hetero >> D_actual → cross-correlation is the issue
    eta_V_empirical = D_V_attn / D_V_raw if D_V_raw > 0 else float('inf')
    ratio = eta_V_empirical / eta_V_predicted if eta_V_predicted > 0 else float('inf')

    return {
        'D_V_raw': D_V_raw,
        'D_V_attn': D_V_attn,
        'D_homo': D_homo,
        'D_hetero': D_hetero,
        'eta_V_predicted': eta_V_predicted,
        'eta_V_empirical': eta_V_empirical,
        'ratio': ratio,
        'token_mse_cv': token_mse_cv,         # CV of per-token MSE (0 = perfectly homoscedastic)
        'hetero_vs_actual': D_hetero / D_V_attn if D_V_attn > 0 else float('inf'),
        'homo_vs_hetero': D_homo / D_hetero if D_hetero > 0 else float('inf'),
    }


# ============================================================
# Hook-based K+V validation (empirical MSE, decomposed)
# ============================================================

def _run_quantized_pass(
    model, spec, input_ids, layer_idx, bits,
    quantize_K: bool, quantize_V: bool,
    mse_store: dict,
) -> torch.Tensor:
    """
    Single forward pass with optional K/V quantization at `layer_idx`.
    Returns attention module output.  Stores empirical per-dim MSE in
    mse_store['D_K'] / mse_store['D_V'].
    """
    attn = getattr(spec.layers[layer_idx], spec.attn_attr)
    k_proj = getattr(attn, spec.k_proj_attr)
    v_proj = getattr(attn, spec.v_proj_attr)
    hooks = []
    captured = {}

    def make_quantize_hook(key, n_kv_heads, head_dim):
        def hook(module, input, output):
            B, T, _ = output.shape
            t = output.float().view(B, T, n_kv_heads, head_dim).permute(0, 2, 1, 3)
            t_hat, mse = quantize_tensor_e8(t, bits)
            mse_store[key] = mse  # empirical per-dim MSE
            return t_hat.permute(0, 2, 1, 3).reshape(B, T, -1).to(output.dtype)
        return hook

    def capture_hook(module, input, output):
        captured['out'] = output[0].detach() if isinstance(output, tuple) else output.detach()

    if quantize_K:
        hooks.append(k_proj.register_forward_hook(
            make_quantize_hook('D_K', spec.n_kv_heads, spec.head_dim)))
    if quantize_V:
        hooks.append(v_proj.register_forward_hook(
            make_quantize_hook('D_V', spec.n_kv_heads, spec.head_dim)))
    hooks.append(attn.register_forward_hook(capture_hook))

    try:
        with torch.no_grad():
            model(input_ids, use_cache=False, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    return captured['out']


def validate_kv_total(
    model,
    spec: ModelSpec,
    input_ids: torch.Tensor,
    layer_idx: int,
    bits: int,
    sensitivity_layer: dict,
    decompose: bool = False,
) -> dict:
    """
    Hook-based validation with empirical D_K, D_V.

    Always runs: clean pass + K+V quantized pass.
    If decompose=True: also runs K-only and V-only passes.

    D_pred uses empirical MSE (not theoretical c·σ²·4^{-b}), so the
    ratio D_actual/D_pred isolates Thm 1 decomposition accuracy from
    E₈ high-rate model accuracy.
    """
    attn = getattr(spec.layers[layer_idx], spec.attn_attr)

    # ---- Clean pass ----
    captured_clean = {}
    def clean_hook(module, input, output):
        captured_clean['out'] = output[0].detach() if isinstance(output, tuple) else output.detach()
    h = attn.register_forward_hook(clean_hook)
    with torch.no_grad():
        model(input_ids, use_cache=False, output_attentions=True)
    h.remove()
    O_clean = captured_clean['out']

    # Use GQA-corrected effective sensitivities (fall back to legacy if absent)
    eta_K = sensitivity_layer.get('eta_K_eff', sensitivity_layer['eta_K'])
    eta_V = sensitivity_layer.get('eta_V_eff', sensitivity_layer['eta_V'])

    # ---- K+V quantized pass ----
    mse_kv = {}
    O_kv = _run_quantized_pass(
        model, spec, input_ids, layer_idx, bits,
        quantize_K=True, quantize_V=True, mse_store=mse_kv)

    D_kv_actual = ((O_clean.float() - O_kv.float()) ** 2).mean().item()
    D_K_emp = mse_kv.get('D_K', 0.0)
    D_V_emp = mse_kv.get('D_V', 0.0)
    D_kv_pred = eta_K * D_K_emp + eta_V * D_V_emp
    ratio_kv = D_kv_actual / D_kv_pred if D_kv_pred > 0 else float('inf')

    result = {
        'D_kv_actual': D_kv_actual,
        'D_kv_pred': D_kv_pred,
        'D_K_emp': D_K_emp,
        'D_V_emp': D_V_emp,
        'eta_K_D_K': eta_K * D_K_emp,
        'eta_V_D_V': eta_V * D_V_emp,
        'ratio_kv': ratio_kv,
    }

    del O_kv

    # ---- Decomposed: K-only and V-only ----
    if decompose:
        # V-only
        mse_v = {}
        O_v = _run_quantized_pass(
            model, spec, input_ids, layer_idx, bits,
            quantize_K=False, quantize_V=True, mse_store=mse_v)
        D_v_actual = ((O_clean.float() - O_v.float()) ** 2).mean().item()
        D_v_pred = eta_V * mse_v.get('D_V', 0.0)
        result['D_v_only_actual'] = D_v_actual
        result['D_v_only_pred'] = D_v_pred
        result['ratio_v_only'] = D_v_actual / D_v_pred if D_v_pred > 0 else float('inf')
        del O_v

        # K-only
        mse_k = {}
        O_k = _run_quantized_pass(
            model, spec, input_ids, layer_idx, bits,
            quantize_K=True, quantize_V=False, mse_store=mse_k)
        D_k_actual = ((O_clean.float() - O_k.float()) ** 2).mean().item()
        D_k_pred = eta_K * mse_k.get('D_K', 0.0)
        result['D_k_only_actual'] = D_k_actual
        result['D_k_only_pred'] = D_k_pred
        result['ratio_k_only'] = D_k_actual / D_k_pred if D_k_pred > 0 else float('inf')
        del O_k

        # Additivity check: D_k + D_v ≈ D_kv?
        result['additivity'] = (D_k_actual + D_v_actual) / D_kv_actual if D_kv_actual > 0 else float('inf')

    del O_clean
    return result


# ============================================================
# Main extractor: captures A, K, V per layer with hooks
# ============================================================

class Thm1Validator:
    def __init__(self, model, sensitivity: dict):
        self.model = model
        self.spec = ModelSpec.from_model(model)
        self.sensitivity = sensitivity
        self._hooks = []
        self._a_buf = {}
        self._v_buf = {}

    def _register_capture_hooks(self):
        spec = self.spec
        for idx in range(spec.n_layers):
            attn = getattr(spec.layers[idx], spec.attn_attr)
            v_proj = getattr(attn, spec.v_proj_attr)

            # Capture V from v_proj
            def make_v_hook(layer_idx):
                def hook(module, input, output):
                    self._v_buf[layer_idx] = output.detach()
                return hook
            self._hooks.append(v_proj.register_forward_hook(make_v_hook(idx)))

            # Capture A from attention output (needs output_attentions=True)
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            self._a_buf[layer_idx] = attn_weights.detach()
                    # Null out weights to save memory
                    return output[:1] + (None,) + output[2:]
                return hook
            self._hooks.append(attn.register_forward_hook(make_attn_hook(idx)))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def run(
        self,
        tokenizer,
        bits: int = 4,
        seq_len: int = 2048,
        measure_actual: bool = False,
        decompose: bool = False,
    ) -> dict:
        spec = self.spec
        device = next(self.model.parameters()).device
        sens_layers = self.sensitivity['layers']

        # ---- Calibration tokens ----
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = "\n\n".join(ds['text'])
        input_ids = tokenizer(
            text, return_tensors='pt', truncation=True, max_length=seq_len,
        ).input_ids.to(device)
        print(f"  Calibration: {input_ids.shape[1]} tokens, {bits}b quantization")

        # ---- Phase A: V-path (offline) ----
        print("\n  [Phase A] V-path validation (offline)...")
        self._register_capture_hooks()
        t0 = time.time()

        with torch.no_grad():
            self.model(input_ids, use_cache=False, output_attentions=True)

        self._remove_hooks()
        print(f"    Forward pass: {time.time() - t0:.1f}s")

        v_results = []
        for l in range(spec.n_layers):
            if l not in self._a_buf:
                print(f"    Layer {l}: no attention weights captured (skipped)")
                continue

            A = self._a_buf.pop(l)
            V = self._v_buf.pop(l)

            vr = validate_v_path(
                A, V,
                n_heads=spec.n_heads,
                n_kv_heads=spec.n_kv_heads,
                head_dim=spec.head_dim,
                bits=bits,
                eta_V_predicted=sens_layers[l].get('eta_V_eff', sens_layers[l]['eta_V']),
            )
            vr['layer'] = l
            v_results.append(vr)

            del A, V

        self._a_buf.clear()
        self._v_buf.clear()

        # ---- Phase B: K+V total (hook-based, optional) ----
        kv_results = []
        if measure_actual:
            passes_per_layer = 2 + (2 if decompose else 0)  # clean + KV + (K-only + V-only)
            total_passes = spec.n_layers * passes_per_layer
            print(f"\n  [Phase B] K+V validation (hook-based, "
                  f"{passes_per_layer} passes/layer × {spec.n_layers} = {total_passes})...")
            for l in range(spec.n_layers):
                kvr = validate_kv_total(
                    self.model, spec, input_ids,
                    layer_idx=l, bits=bits,
                    sensitivity_layer=sens_layers[l],
                    decompose=decompose,
                )
                kvr['layer'] = l
                kv_results.append(kvr)

                if (l + 1) % 8 == 0 or l == spec.n_layers - 1:
                    print(f"    {l + 1}/{spec.n_layers} layers done")

        return {
            'bits': bits,
            'seq_len': input_ids.shape[1],
            'v_path': v_results,
            'kv_total': kv_results,
        }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Theorem 1 validation')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--sensitivity', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--measure-actual', action='store_true',
                        help='Run hook-based K+V total measurement')
    parser.add_argument('--decompose', action='store_true',
                        help='Also run K-only and V-only passes (with --measure-actual)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map='cuda:0', attn_implementation='eager')
    if args.load_in_8bit:
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs['torch_dtype'] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"  Loaded: {torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM")

    with open(args.sensitivity) as f:
        sensitivity = json.load(f)

    validator = Thm1Validator(model, sensitivity)
    results = validator.run(
        tokenizer, bits=args.bits, seq_len=args.seq_len,
        measure_actual=args.measure_actual,
        decompose=args.decompose,
    )
    results['model'] = args.model

    # ---- Print V-path results ----
    print(f"\n{'':=<100}")
    print(f"  Theorem 1 V-path diagnostic ({args.bits}b)")
    print(f"  D_homo = η_V·D_raw | D_hetero = Σ aᵢⱼ²·eⱼ² (uncorr) | D_actual = ‖A·δV‖²")
    print(f"{'':=<100}")
    print(f"{'Layer':>6} | {'D_homo':>10} | {'D_hetero':>10} | {'D_actual':>10} | "
          f"{'ho/he':>6} | {'he/ac':>6} | {'CV(e)':>6} | {'ratio':>6}")
    print("-" * 82)

    ratios_v = []
    for vr in results['v_path']:
        print(f"{vr['layer']:>6} | {vr['D_homo']:>10.4e} | "
              f"{vr['D_hetero']:>10.4e} | {vr['D_V_attn']:>10.4e} | "
              f"{vr['homo_vs_hetero']:>6.2f} | {vr['hetero_vs_actual']:>6.2f} | "
              f"{vr['token_mse_cv']:>6.2f} | {vr['ratio']:>6.3f}")
        ratios_v.append(vr['ratio'])

    ratios_v = np.array(ratios_v)
    ho_he = np.array([vr['homo_vs_hetero'] for vr in results['v_path']])
    he_ac = np.array([vr['hetero_vs_actual'] for vr in results['v_path']])
    cvs = np.array([vr['token_mse_cv'] for vr in results['v_path']])

    print(f"\n  Overall ratio (η_emp/η_pred): mean={ratios_v.mean():.4f}")
    print(f"  ho/he (homo vs hetero):  mean={ho_he.mean():.3f}  — if >>1: token MSE heterogeneity matters")
    print(f"  he/ac (hetero vs actual): mean={he_ac.mean():.3f}  — if ≠1: cross-token correlation matters")
    print(f"  CV(e) (token MSE spread): mean={cvs.mean():.3f}")

    # ---- Print K+V total results ----
    if results['kv_total']:
        print(f"\n{'':=<72}")
        print(f"  Theorem 1 K+V: D_pred(empirical) vs D_actual ({args.bits}b)")
        print(f"{'':=<72}")
        header = (f"{'Layer':>6} | {'D_pred':>10} | {'D_actual':>10} | "
                  f"{'ratio':>8} | {'K%':>6}")
        if args.decompose:
            header += f" | {'K-only':>8} | {'V-only':>8} | {'addit':>6}"
        print(header)
        print("-" * (50 + (30 if args.decompose else 0)))

        ratios_kv = []
        for kvr in results['kv_total']:
            k_frac = kvr['eta_K_D_K'] / kvr['D_kv_pred'] if kvr['D_kv_pred'] > 0 else 0
            line = (f"{kvr['layer']:>6} | {kvr['D_kv_pred']:>10.4e} | "
                    f"{kvr['D_kv_actual']:>10.4e} | "
                    f"{kvr['ratio_kv']:>8.4f} | {k_frac:>5.1%}")
            if args.decompose and 'ratio_k_only' in kvr:
                line += (f" | {kvr['ratio_k_only']:>8.4f} | "
                         f"{kvr['ratio_v_only']:>8.4f} | "
                         f"{kvr.get('additivity', 0):>6.3f}")
            print(line)
            ratios_kv.append(kvr['ratio_kv'])

        ratios_kv = np.array(ratios_kv)
        print(f"\n  K+V ratio: mean={ratios_kv.mean():.4f}, "
              f"std={ratios_kv.std():.4f}, "
              f"range=[{ratios_kv.min():.4f}, {ratios_kv.max():.4f}]")

    # ---- Save ----
    if args.output_dir is None:
        args.output_dir = str(Path(args.sensitivity).parent)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model_short = args.model.split('/')[-1]
    save_path = Path(args.output_dir) / f'thm1_{model_short}_{args.bits}b.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {save_path}")


if __name__ == '__main__':
    main()