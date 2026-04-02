"""
LatticeQuant v3 — Theorem 2 Validation (P4)
=============================================
Validates the cross-layer error propagation model:

    D_final ≈ Σ_l  Γ_l · D_attn_l

where Γ_l is the amplification factor from propagation.py and
D_attn_l = η_K·D_K + η_V·D_V is the local attention-output distortion
from Theorem 1.

Three validation checks:

  (A) Per-layer Γ accuracy:
      Quantize only layer l's KV → measure final hidden state error →
      Γ_l_actual = D_final_l / D_attn_l
      Compare Γ_l_actual (from real quantization) vs Γ_l_iso (from
      isotropic noise injection in propagation.py).
      This tests whether isotropic Gaussian is a good surrogate for
      actual E₈ quantization error structure.

  (B) Additivity:
      Quantize ALL layers simultaneously → measure total D_final →
      compare to Σ_l Γ_l · D_attn_l.
      Tests whether layer errors propagate independently (no significant
      cross-layer interaction).

  (C) Γ curve analysis:
      Monotonicity check, decay rate, correlation with remaining depth.

Usage:
  python allocation/thm2_validate.py \\
      --model meta-llama/Llama-3.1-8B --load-in-8bit \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json \\
      --propagation results/propagation_Llama-3.1-8B.json \\
      --bits 4

  # Skip per-layer (just additivity + analysis):
  python allocation/thm2_validate.py \\
      --model meta-llama/Llama-3.1-8B --load-in-8bit \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json \\
      --propagation results/propagation_Llama-3.1-8B.json \\
      --bits 4 --skip-per-layer
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

from sensitivity import ModelSpec, _resolve_attr
from thm1_validate import quantize_tensor_e8


# ============================================================
# Final-hidden-state capture (reuse pattern from propagation.py)
# ============================================================

def _find_final_norm(model) -> Optional[Any]:
    """Locate the final LayerNorm before the LM head."""
    for path in ('model.norm', 'model.model.norm', 'transformer.ln_f',
                 'transformer.norm', 'gpt_neox.final_layer_norm'):
        mod = _resolve_attr(model, path)
        if mod is not None:
            return mod
    return None


class _HiddenCapture:
    def __init__(self):
        self.value: Optional[torch.Tensor] = None

    def hook(self, module, input, output):
        if isinstance(output, tuple):
            self.value = output[0].detach()
        else:
            self.value = output.detach()


# ============================================================
# Per-layer quantization with final hidden state measurement
# ============================================================

def _run_with_kv_quant(
    model, spec, input_ids, final_norm,
    quant_layers: List[int],
    bits: int,
) -> tuple:
    """
    Forward pass with E₈ quantization applied to specified layers' KV.

    Returns:
        final_hidden: (B, T, D) final hidden state after final LN
        per_layer_mse: dict[layer_idx] -> {'D_K': float, 'D_V': float}
    """
    hooks = []
    per_layer_mse = {}

    def make_quantize_hook(layer_idx, key, n_kv_heads, head_dim):
        def hook(module, input, output):
            B, T, _ = output.shape
            t = output.float().view(B, T, n_kv_heads, head_dim).permute(0, 2, 1, 3)
            t_hat, mse = quantize_tensor_e8(t, bits)
            if layer_idx not in per_layer_mse:
                per_layer_mse[layer_idx] = {}
            per_layer_mse[layer_idx][key] = mse
            return t_hat.permute(0, 2, 1, 3).reshape(B, T, -1).to(output.dtype)
        return hook

    # Register quantization hooks for specified layers
    for l in quant_layers:
        attn = getattr(spec.layers[l], spec.attn_attr)
        k_proj = getattr(attn, spec.k_proj_attr)
        v_proj = getattr(attn, spec.v_proj_attr)
        hooks.append(k_proj.register_forward_hook(
            make_quantize_hook(l, 'D_K', spec.n_kv_heads, spec.head_dim)))
        hooks.append(v_proj.register_forward_hook(
            make_quantize_hook(l, 'D_V', spec.n_kv_heads, spec.head_dim)))

    # Capture final hidden state
    cap = _HiddenCapture()
    hooks.append(final_norm.register_forward_hook(cap.hook))

    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return cap.value, per_layer_mse


# ============================================================
# Thm2 Validator
# ============================================================

class Thm2Validator:
    def __init__(self, model, sensitivity: dict, propagation: dict):
        self.model = model
        self.spec = ModelSpec.from_model(model)
        self.sensitivity = sensitivity
        self.propagation = propagation
        self.final_norm = _find_final_norm(model)
        if self.final_norm is None:
            raise ValueError("Cannot locate final LayerNorm.")

    def _get_clean_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Clean forward pass → final hidden state."""
        cap = _HiddenCapture()
        handle = self.final_norm.register_forward_hook(cap.hook)
        try:
            with torch.no_grad():
                self.model(input_ids, use_cache=False)
        finally:
            handle.remove()
        return cap.value

    def _predicted_D_attn(self, layer_idx: int, D_K_emp: float, D_V_emp: float) -> dict:
        """Thm1-predicted local attention output distortion using empirical MSE."""
        sl = self.sensitivity['layers'][layer_idx]
        eta_K = sl.get('eta_K_eff', sl['eta_K'])
        eta_V = sl.get('eta_V_eff', sl['eta_V'])
        D_attn = eta_K * D_K_emp + eta_V * D_V_emp
        return {
            'eta_K': eta_K, 'eta_V': eta_V,
            'D_K_emp': D_K_emp, 'D_V_emp': D_V_emp,
            'D_attn': D_attn,
            'K_frac': eta_K * D_K_emp / D_attn if D_attn > 0 else 0,
        }

    def run(
        self,
        tokenizer,
        bits: int = 4,
        seq_len: int = 2048,
        skip_per_layer: bool = False,
    ) -> dict:
        spec = self.spec
        device = next(self.model.parameters()).device
        n_layers = spec.n_layers
        gamma_iso = [lr['gamma'] for lr in self.propagation['layers']]

        # ---- Calibration tokens ----
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = "\n\n".join(ds['text'])
        input_ids = tokenizer(
            text, return_tensors='pt', truncation=True, max_length=seq_len,
        ).input_ids.to(device)
        print(f"  Calibration: {input_ids.shape[1]} tokens, {bits}b quantization")

        # ---- Clean pass ----
        print("  Clean forward pass ...")
        t0 = time.time()
        clean_hidden = self._get_clean_hidden(input_ids)  # (B, T, D)
        print(f"    Done: {time.time() - t0:.1f}s")

        # ==================================================
        # (A) Per-layer Γ validation
        # ==================================================
        per_layer_results = []
        if not skip_per_layer:
            print(f"\n  [Phase A] Per-layer Γ validation ({n_layers} layers) ...")
            t_a = time.time()
            for l in range(n_layers):
                quant_hidden, mse_dict = _run_with_kv_quant(
                    self.model, spec, input_ids, self.final_norm,
                    quant_layers=[l], bits=bits,
                )

                # Final hidden state distortion from quantizing layer l alone
                D_final_l = ((clean_hidden.float() - quant_hidden.float()) ** 2).mean().item()

                # Local attention distortion (Thm1)
                lm = mse_dict.get(l, {})
                pred = self._predicted_D_attn(l, lm.get('D_K', 0), lm.get('D_V', 0))
                D_attn_l = pred['D_attn']

                # Actual Γ from quantization
                gamma_actual = D_final_l / D_attn_l if D_attn_l > 0 else float('inf')

                # Compare to isotropic Γ
                gamma_ratio = gamma_actual / gamma_iso[l] if gamma_iso[l] > 0 else float('inf')

                per_layer_results.append({
                    'layer': l,
                    'D_final_l': D_final_l,
                    'D_attn_l': D_attn_l,
                    'gamma_actual': gamma_actual,
                    'gamma_iso': gamma_iso[l],
                    'gamma_ratio': gamma_ratio,  # should be ≈ 1.0
                    **pred,
                })

                del quant_hidden

                if (l + 1) % 8 == 0 or l == n_layers - 1:
                    print(f"    {l + 1}/{n_layers} layers done")

            print(f"    Phase A: {time.time() - t_a:.1f}s")

        # ==================================================
        # (B) Additivity: all layers quantized
        # ==================================================
        print(f"\n  [Phase B] Additivity (all {n_layers} layers quantized) ...")
        t_b = time.time()
        all_hidden, all_mse = _run_with_kv_quant(
            self.model, spec, input_ids, self.final_norm,
            quant_layers=list(range(n_layers)), bits=bits,
        )

        D_final_total = ((clean_hidden.float() - all_hidden.float()) ** 2).mean().item()
        del all_hidden

        # Predicted total: Σ_l Γ_l × D_attn_l
        D_pred_total = 0.0
        per_layer_contributions = []
        for l in range(n_layers):
            lm = all_mse.get(l, {})
            pred = self._predicted_D_attn(l, lm.get('D_K', 0), lm.get('D_V', 0))
            contrib = gamma_iso[l] * pred['D_attn']
            D_pred_total += contrib
            per_layer_contributions.append({
                'layer': l,
                'gamma_iso': gamma_iso[l],
                'D_attn_l': pred['D_attn'],
                'contrib': contrib,
            })

        additivity_ratio = D_final_total / D_pred_total if D_pred_total > 0 else float('inf')

        # If per-layer was run, also compute sum-of-singles
        D_sum_singles = 0.0
        if per_layer_results:
            D_sum_singles = sum(r['D_final_l'] for r in per_layer_results)

        print(f"    Phase B: {time.time() - t_b:.1f}s")

        # ==================================================
        # (C) Γ curve analysis
        # ==================================================
        gamma_arr = np.array(gamma_iso)
        diffs = np.diff(gamma_arr)
        n_decreasing = np.sum(diffs < 0)
        monotonic_frac = n_decreasing / len(diffs) if len(diffs) > 0 else 0

        # Correlation with remaining depth
        remaining_depth = np.arange(n_layers, 0, -1)
        corr_depth = np.corrcoef(gamma_arr, remaining_depth)[0, 1]

        # Log-linear fit: log(Γ) ≈ a + b·l
        log_gamma = np.log(gamma_arr + 1e-12)
        layer_idx = np.arange(n_layers)
        if n_layers > 1:
            slope, intercept = np.polyfit(layer_idx, log_gamma, 1)
            # Per-layer decay factor
            decay_per_layer = np.exp(slope)
        else:
            slope, intercept, decay_per_layer = 0, 0, 1

        curve_analysis = {
            'monotonic_frac': monotonic_frac,
            'n_decreasing_steps': int(n_decreasing),
            'n_total_steps': n_layers - 1,
            'corr_remaining_depth': corr_depth,
            'log_linear_slope': slope,
            'log_linear_intercept': intercept,
            'decay_per_layer': decay_per_layer,
            'gamma_range': [float(gamma_arr.min()), float(gamma_arr.max())],
            'gamma_mean': float(gamma_arr.mean()),
        }

        elapsed = time.time() - t0
        return {
            'bits': bits,
            'seq_len': input_ids.shape[1],
            'per_layer': per_layer_results,
            'additivity': {
                'D_final_total': D_final_total,
                'D_pred_total': D_pred_total,
                'ratio': additivity_ratio,
                'D_sum_singles': D_sum_singles,
                'singles_vs_all': D_sum_singles / D_final_total if D_final_total > 0 and D_sum_singles > 0 else 0,
                'per_layer_contributions': per_layer_contributions,
            },
            'curve_analysis': curve_analysis,
            'elapsed_sec': elapsed,
        }


# ============================================================
# CLI + pretty printing
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LatticeQuant v3: Theorem 2 validation (Γ accuracy)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--sensitivity', type=str, required=True)
    parser.add_argument('--propagation', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--skip-per-layer', action='store_true',
                        help='Skip per-layer Γ measurement (just additivity + curve)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map='cuda:0')
    if args.load_in_8bit:
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs['torch_dtype'] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"  Loaded: {torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM")

    with open(args.sensitivity) as f:
        sensitivity = json.load(f)
    with open(args.propagation) as f:
        propagation = json.load(f)

    validator = Thm2Validator(model, sensitivity, propagation)
    results = validator.run(
        tokenizer, bits=args.bits, seq_len=args.seq_len,
        skip_per_layer=args.skip_per_layer,
    )
    results['model'] = args.model

    # ---- Print (A): Per-layer Γ ----
    if results['per_layer']:
        print(f"\n{'':=<90}")
        print(f"  (A) Per-layer Γ: isotropic vs actual ({args.bits}b)")
        print(f"  ratio = Γ_actual / Γ_iso  (should be ≈ 1.0)")
        print(f"{'':=<90}")
        print(f"{'Layer':>6} | {'Γ_iso':>10} | {'Γ_actual':>10} | "
              f"{'ratio':>8} | {'D_final_l':>10} | {'D_attn_l':>10} | {'K%':>6}")
        print("-" * 78)

        ratios = []
        for r in results['per_layer']:
            print(f"{r['layer']:>6} | {r['gamma_iso']:>10.2f} | "
                  f"{r['gamma_actual']:>10.2f} | {r['gamma_ratio']:>8.4f} | "
                  f"{r['D_final_l']:>10.4e} | {r['D_attn_l']:>10.4e} | "
                  f"{r['K_frac']:>5.1%}")
            ratios.append(r['gamma_ratio'])

        ratios = np.array(ratios)
        print(f"\n  Γ_actual/Γ_iso: mean={ratios.mean():.4f}, "
              f"std={ratios.std():.4f}, "
              f"range=[{ratios.min():.4f}, {ratios.max():.4f}]")
        if abs(ratios.mean() - 1.0) < 0.2:
            print("  → Isotropic noise is a good surrogate for actual quant error.")
        else:
            print(f"  → Systematic bias: isotropic Γ {'over' if ratios.mean() < 1 else 'under'}estimates actual.")

    # ---- Print (B): Additivity ----
    add = results['additivity']
    print(f"\n{'':=<70}")
    print(f"  (B) Additivity: Σ Γ_l·D_attn_l vs D_final_total ({args.bits}b)")
    print(f"{'':=<70}")
    print(f"  D_final_total (all layers quantized): {add['D_final_total']:.6e}")
    print(f"  D_pred_total  (Σ Γ_l · D_attn_l):    {add['D_pred_total']:.6e}")
    print(f"  ratio (actual/pred):                  {add['ratio']:.4f}")
    if add['D_sum_singles'] > 0:
        print(f"  D_sum_singles (Σ D_final_l):          {add['D_sum_singles']:.6e}")
        print(f"  singles/all:                          {add['singles_vs_all']:.4f}")
        print(f"    (>1 means cross-layer errors partially cancel; "
              f"<1 means they compound)")

    # Top-5 contributors
    contribs = sorted(add['per_layer_contributions'], key=lambda x: x['contrib'], reverse=True)
    print(f"\n  Top-5 contributors to predicted total:")
    for c in contribs[:5]:
        frac = c['contrib'] / add['D_pred_total'] if add['D_pred_total'] > 0 else 0
        print(f"    Layer {c['layer']:>2}: Γ={c['gamma_iso']:.1f} × D_attn={c['D_attn_l']:.4e} "
              f"= {c['contrib']:.4e} ({frac:.1%})")

    # ---- Print (C): Curve analysis ----
    ca = results['curve_analysis']
    print(f"\n{'':=<50}")
    print(f"  (C) Γ curve analysis")
    print(f"{'':=<50}")
    print(f"  Monotonic decrease: {ca['n_decreasing_steps']}/{ca['n_total_steps']} "
          f"({ca['monotonic_frac']:.1%})")
    print(f"  Corr(Γ, remaining depth): {ca['corr_remaining_depth']:.4f}")
    print(f"  Log-linear decay: Γ_l ≈ exp({ca['log_linear_intercept']:.2f} "
          f"+ {ca['log_linear_slope']:.4f}·l)")
    print(f"  Per-layer decay factor: {ca['decay_per_layer']:.4f}")
    print(f"  Γ range: [{ca['gamma_range'][0]:.2f}, {ca['gamma_range'][1]:.2f}]")

    # ---- Save ----
    if args.output_dir is None:
        args.output_dir = str(Path(args.sensitivity).parent)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_short = args.model.split('/')[-1]
    save_path = Path(args.output_dir) / f'thm2_{model_short}_{args.bits}b.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {save_path}")


if __name__ == '__main__':
    main()