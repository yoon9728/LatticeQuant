"""
LatticeQuant v3 — Attention Sensitivity Extraction (P1)
========================================================
Computes per-layer, per-head sensitivity metrics for attention-aware
bit allocation (Theorem 1 of the v3 paper).

Metrics extracted:
  η_V,l  = (1/T) Σᵢ ‖aᵢ‖²₂          (value sensitivity, Q-head mean)
  η_K,l  = (1/Td²) Σᵢ ‖qᵢ‖² · Σₖ aᵢₖ² ‖vₖ - μᵢ‖²   (key sensitivity, Q-head mean)
  σ²_K,l = E[K²] per dim              (key second moment)
  σ²_V,l = E[V²] per dim              (value second moment)
  w_X    = mean_g[η_X,g · σ²_X,g]     (GQA-grouped effective weight for allocator)
  η_X_eff = w_X / σ²_X                (scalarised grouped sensitivity for Thm 1 validation)

Note on σ²: We use the second moment E[x²], NOT centered variance Var(x).
This matches the E₈ quantizer pipeline which does not center before
quantizing: scale = f(E[x²]), so distortion ∝ E[x²].  Using Var(x)
would underestimate distortion for heads with nonzero mean.

GQA-aware weighting:
  For grouped-query attention, the effective V/K weights used by downstream
  allocation are computed in grouped form:
      w_X = mean_g [ η_X,g · σ²_X,g ]
  where g indexes KV heads (groups), η_X,g is the average sensitivity of the
  Q-heads that share KV head g, and σ²_X,g is the second moment of that KV head.
  This avoids the product-of-means ≠ mean-of-products mismatch
  (negative covariance causes mean(η)·mean(σ²) to overpredict).

Design:
  - Hook-based per-layer processing: A is computed in the hook, used, then
    nulled out. Only one layer's attention matrix (B, H, T, T) is in GPU
    memory at a time. For T=2048, H=32: 512 MB per layer, not 16 GB total.
  - All arithmetic on GPU; only final scalars move to CPU via .item().
  - GQA-aware: V is expanded to match Q-head count before vectorised ops,
    while grouped weights are computed at the KV-head level.
  - Architecture auto-detection for Llama / Mistral / Qwen / Phi families.
    Separate-QKV (q_proj+k_proj+v_proj) is required; fused-QKV not yet
    supported. Add a branch in _detect_projections() to extend.

Usage:
  python allocation/sensitivity.py --model meta-llama/Llama-3.1-8B --load-in-8bit
  python allocation/sensitivity.py --model meta-llama/Llama-3.1-8B --load-in-8bit --seq-len 1024
"""

import torch
import json
import time
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ============================================================
# Architecture detection
# ============================================================

def _resolve_attr(obj, dotted_path):
    for attr in dotted_path.split('.'):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return None
    return obj


@dataclass
class ModelSpec:
    layers: Any
    attn_attr: str
    q_proj_attr: str
    k_proj_attr: str
    v_proj_attr: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    group_size: int

    @staticmethod
    def from_model(model):
        config = model.config
        layers = None
        for path in ('model.layers', 'model.model.layers', 'transformer.h',
                     'transformer.layers', 'gpt_neox.layers'):
            candidate = _resolve_attr(model, path)
            if candidate is not None and hasattr(candidate, '__len__'):
                layers = candidate
                break
        if layers is None:
            raise ValueError("Cannot locate decoder layers.")

        attn_attr = None
        for name in ('self_attn', 'attn', 'attention', 'self_attention'):
            if hasattr(layers[0], name):
                attn_attr = name
                break
        if attn_attr is None:
            raise ValueError(f"Cannot find attention module on {type(layers[0]).__name__}.")

        first_attn = getattr(layers[0], attn_attr)
        q_attr = k_attr = v_attr = None
        for q, k, v in [('q_proj', 'k_proj', 'v_proj'),
                        ('query', 'key', 'value'),
                        ('wq', 'wk', 'wv')]:
            if all(hasattr(first_attn, a) for a in (q, k, v)):
                q_attr, k_attr, v_attr = q, k, v
                break
        if q_attr is None:
            if hasattr(first_attn, 'c_attn') or hasattr(first_attn, 'qkv_proj'):
                raise ValueError("Fused QKV projection detected. Not yet supported.")
            raise ValueError(f"Cannot find Q/K/V projections on {type(first_attn).__name__}.")

        n_heads = config.num_attention_heads
        n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        head_dim = getattr(config, 'head_dim', None)
        if head_dim is None:
            head_dim = config.hidden_size // n_heads
        n_layers = config.num_hidden_layers
        group_size = n_heads // n_kv_heads

        return ModelSpec(layers=layers, attn_attr=attn_attr,
                         q_proj_attr=q_attr, k_proj_attr=k_attr, v_proj_attr=v_attr,
                         n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                         head_dim=head_dim, group_size=group_size)


# ============================================================
# Per-layer sensitivity computation (pure torch, GPU-only)
# ============================================================

def compute_layer_sensitivity(A, Q, K, V, n_heads, n_kv_heads, head_dim):
    B, _, T, _ = A.shape
    assert n_heads % n_kv_heads == 0, f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
    group_size = n_heads // n_kv_heads

    Q_h = Q.float().view(B, T, n_heads, head_dim).permute(0, 2, 1, 3)
    K_h = K.float().view(B, T, n_kv_heads, head_dim).permute(0, 2, 1, 3)
    V_h = V.float().view(B, T, n_kv_heads, head_dim).permute(0, 2, 1, 3)

    sigma2_K = (K_h ** 2).mean().item()
    sigma2_V = (V_h ** 2).mean().item()
    sigma2_K_groups = (K_h ** 2).mean(dim=(0, 2, 3))
    sigma2_V_groups = (V_h ** 2).mean(dim=(0, 2, 3))
    del K_h

    A_f = A.float()
    eta_V_heads = (A_f ** 2).sum(dim=-1).mean(dim=(0, 2))

    V_exp = V_h.repeat_interleave(group_size, dim=1)
    mu = torch.bmm(
        A_f.reshape(B * n_heads, T, T),
        V_exp.reshape(B * n_heads, T, head_dim),
    ).reshape(B, n_heads, T, head_dim)

    A_sq = A_f ** 2
    del A_f

    v_norms_sq = (V_exp ** 2).sum(dim=-1)
    term_vsq = torch.bmm(
        A_sq.reshape(B * n_heads, T, T),
        v_norms_sq.reshape(B * n_heads, T, 1),
    ).reshape(B, n_heads, T)

    a2v = torch.bmm(
        A_sq.reshape(B * n_heads, T, T),
        V_exp.reshape(B * n_heads, T, head_dim),
    ).reshape(B, n_heads, T, head_dim)
    term_cross = (mu * a2v).sum(dim=-1)
    del a2v

    mu_norm_sq = (mu ** 2).sum(dim=-1)
    s2 = A_sq.sum(dim=-1)

    # ---- Heteroscedastic η_V: weight by per-token relative magnitude ----
    # Per-head scaling: tokens with ‖vⱼ‖² << σ²_head get quantized to near-zero
    # → much less error. This corrects η_V by weighting token j's attention
    # contribution by σ²_V,j / σ̄²_V,g (relative magnitude within KV head).
    # Validated: he/ac ≈ 1.00 (Thm 1 diagnostic), so this model is exact
    # up to the proportional-error approximation.
    sigma2_V_token = (V_h ** 2).mean(dim=-1)                         # (B, n_kv, T)
    sigma2_V_head_mean = sigma2_V_token.mean(dim=2, keepdim=True)    # (B, n_kv, 1)
    v_token_ratio = sigma2_V_token / sigma2_V_head_mean.clamp(min=1e-12)
    v_token_ratio_exp = v_token_ratio.repeat_interleave(group_size, dim=1)  # (B, H, T)

    eta_V_het_heads = torch.bmm(
        A_sq.reshape(B * n_heads, T, T),
        v_token_ratio_exp.reshape(B * n_heads, T, 1),
    ).reshape(B, n_heads, T).mean(dim=(0, 2))                       # (H,)

    del sigma2_V_token, sigma2_V_head_mean, v_token_ratio, v_token_ratio_exp

    del A_sq

    jacobian_norm_sq = term_vsq - 2.0 * term_cross + mu_norm_sq * s2
    jacobian_norm_sq.clamp_(min=0.0)
    del term_vsq, term_cross, mu_norm_sq, s2, mu, V_exp

    q_norm_sq = (Q_h ** 2).sum(dim=-1)
    eta_K_heads = (
        (q_norm_sq * jacobian_norm_sq).mean(dim=(0, 2)) / (head_dim ** 2)
    )
    del jacobian_norm_sq, q_norm_sq, Q_h

    # Group Q-head sensitivities by shared KV head.
    # ASSUMPTION: contiguous grouping — Q-heads [0..gs-1] share KV-head 0,
    # [gs..2gs-1] share KV-head 1, etc.  True for Llama/Mistral/Qwen.
    # If a model uses non-contiguous mapping, this view() is wrong.
    eta_V_groups = eta_V_heads.view(n_kv_heads, group_size).mean(dim=1)
    eta_K_groups = eta_K_heads.view(n_kv_heads, group_size).mean(dim=1)
    eta_V_het_groups = eta_V_het_heads.view(n_kv_heads, group_size).mean(dim=1)
    del V_h

    eta_K = eta_K_heads.mean().item()
    eta_V = eta_V_heads.mean().item()
    eta_V_het = eta_V_het_heads.mean().item()

    w_K_grouped = (eta_K_groups * sigma2_K_groups).mean().item()
    w_V_grouped = (eta_V_groups * sigma2_V_groups).mean().item()
    w_V_het = (eta_V_het_groups * sigma2_V_groups).mean().item()
    w_K_legacy = eta_K * sigma2_K
    w_V_legacy = eta_V * sigma2_V

    eta_K_eff = w_K_grouped / sigma2_K if sigma2_K > 0 else 0.0
    eta_V_eff = w_V_grouped / sigma2_V if sigma2_V > 0 else 0.0
    eta_V_het_eff = w_V_het / sigma2_V if sigma2_V > 0 else 0.0

    return {
        'eta_K': eta_K, 'eta_V': eta_V,
        'eta_K_eff': eta_K_eff, 'eta_V_eff': eta_V_eff,
        'eta_V_het': eta_V_het, 'eta_V_het_eff': eta_V_het_eff,
        'sigma2_K': sigma2_K, 'sigma2_V': sigma2_V,
        'w_K': w_K_grouped, 'w_V': w_V_grouped,
        'w_V_het': w_V_het,
        'w_K_legacy': w_K_legacy, 'w_V_legacy': w_V_legacy,
        'eta_K_per_head': eta_K_heads.cpu().tolist(),
        'eta_V_per_head': eta_V_heads.cpu().tolist(),
        'eta_V_het_per_head': eta_V_het_heads.cpu().tolist(),
        'eta_K_groups': eta_K_groups.cpu().tolist(),
        'eta_V_groups': eta_V_groups.cpu().tolist(),
        'sigma2_K_groups': sigma2_K_groups.cpu().tolist(),
        'sigma2_V_groups': sigma2_V_groups.cpu().tolist(),
    }


# ============================================================
# Hook-based extractor
# ============================================================

class SensitivityExtractor:
    def __init__(self, model):
        self.model = model
        self.spec = ModelSpec.from_model(model)
        attn_impl = getattr(model.config, '_attn_implementation', None)
        if attn_impl is not None and attn_impl != 'eager':
            raise RuntimeError(
                f"Model uses attn_implementation='{attn_impl}', but "
                f"sensitivity extraction requires 'eager'."
            )
        self._hooks = []
        self._q_buf = {}
        self._k_buf = {}
        self._v_buf = {}
        self._results = {}

    def _proj_hook(self, buf, layer_idx):
        def hook(module, input, output):
            buf[layer_idx] = output.detach()
        return hook

    def _attn_hook(self, layer_idx):
        spec = self.spec
        def hook(module, input, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(f"Layer {layer_idx}: unexpected attention output format.")
            attn_weights = output[1]
            if attn_weights is None:
                raise RuntimeError(f"Layer {layer_idx}: attention weights are None.")
            for name, buf in [('Q', self._q_buf), ('K', self._k_buf), ('V', self._v_buf)]:
                if layer_idx not in buf:
                    raise RuntimeError(f"Layer {layer_idx}: {name} buffer missing.")
            Q = self._q_buf.pop(layer_idx)
            K = self._k_buf.pop(layer_idx)
            V = self._v_buf.pop(layer_idx)
            self._results[layer_idx] = compute_layer_sensitivity(
                A=attn_weights, Q=Q, K=K, V=V,
                n_heads=spec.n_heads, n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim)
            del Q, K, V
            return output[:1] + (None,) + output[2:]
        return hook

    def _register_hooks(self):
        spec = self.spec
        for idx in range(spec.n_layers):
            attn = getattr(spec.layers[idx], spec.attn_attr)
            self._hooks.append(getattr(attn, spec.q_proj_attr).register_forward_hook(
                self._proj_hook(self._q_buf, idx)))
            self._hooks.append(getattr(attn, spec.k_proj_attr).register_forward_hook(
                self._proj_hook(self._k_buf, idx)))
            self._hooks.append(getattr(attn, spec.v_proj_attr).register_forward_hook(
                self._proj_hook(self._v_buf, idx)))
            self._hooks.append(attn.register_forward_hook(self._attn_hook(idx)))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._q_buf.clear()
        self._k_buf.clear()
        self._v_buf.clear()

    def run(self, tokenizer, seq_len=2048, text=None):
        spec = self.spec
        device = next(self.model.parameters()).device
        if text is None:
            from datasets import load_dataset
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            text = "\n\n".join(ds['text'])
        input_ids = tokenizer(
            text, return_tensors='pt', truncation=True, max_length=seq_len,
        ).input_ids.to(device)
        actual_len = input_ids.shape[1]
        print(f"  Calibration: {actual_len} tokens on {device}")

        self._register_hooks()
        t0 = time.time()
        try:
            with torch.no_grad():
                self.model(input_ids, output_attentions=True, use_cache=False)
        finally:
            self._remove_hooks()
        elapsed = time.time() - t0
        print(f"  Forward pass: {elapsed:.1f}s")

        layers = [self._results[i] for i in range(spec.n_layers)]
        self._results.clear()
        return {
            'model_spec': {
                'n_layers': spec.n_layers, 'n_heads': spec.n_heads,
                'n_kv_heads': spec.n_kv_heads, 'head_dim': spec.head_dim,
                'group_size': spec.group_size,
            },
            'seq_len': actual_len, 'layers': layers, 'elapsed_sec': elapsed,
        }


# ============================================================
# Convenience + CLI
# ============================================================

def extract_sensitivity(model_name, seq_len=2048, load_in_8bit=False, output_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = dict(device_map='cuda:0', attn_implementation='eager')
    if load_in_8bit:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw['torch_dtype'] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    print(f"  Loaded: {torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM")

    extractor = SensitivityExtractor(model)
    results = extractor.run(tokenizer, seq_len=seq_len)
    results['model'] = model_name

    print(f"\n{'Layer':>6} | {'η_V':>8} | {'η_V_eff':>8} | "
          f"{'w_V':>10} | {'wV_old':>10} | "
          f"{'η_K':>8} | {'η_K_eff':>8} | "
          f"{'w_K':>10} | {'wK_old':>10}")
    print("-" * 100)
    for i, lr in enumerate(results['layers']):
        print(f"{i:>6} | {lr['eta_V']:>8.4f} | {lr['eta_V_eff']:>8.4f} | "
              f"{lr['w_V']:>10.4e} | {lr['w_V_legacy']:>10.4e} | "
              f"{lr['eta_K']:>8.4f} | {lr['eta_K_eff']:>8.4f} | "
              f"{lr['w_K']:>10.4e} | {lr['w_K_legacy']:>10.4e}")

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / 'results')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_short = model_name.split('/')[-1]
    save_path = Path(output_dir) / f'sensitivity_{model_short}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {save_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='LatticeQuant v3: sensitivity extraction')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    extract_sensitivity(args.model, args.seq_len, args.load_in_8bit, args.output_dir)

if __name__ == '__main__':
    main()