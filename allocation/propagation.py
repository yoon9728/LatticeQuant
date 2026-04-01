"""
LatticeQuant v3 — Cross-Layer Error Propagation (P3)
=====================================================
Measures per-layer amplification factors Γ_l (Theorem 2).

Method: perturbation injection.
  1. Clean forward pass → capture final hidden state (after final LN, before LM head).
  2. For each layer l, inject isotropic Gaussian noise ε, run the rest of
     the network, measure:
         Γ_l = E[‖Δh_final‖²] / E[‖ε‖²]
  3. Average over n_samples noise draws for stability.

Injection point (--inject-point):
  'residual'  — inject at decoder layer output (after attention + MLP + residual).
                Measures aggregate downstream amplification.  Coarser: skips
                within-layer MLP amplification of the attention error.
  'attention' — inject at self_attn module output, i.e. the attention branch
                output before it is merged back into the residual stream in
                standard pre-norm decoder implementations.  Closer to
                Theorem 2's local error definition.

Design choices:
  - Measurement at final-LN output (not logits): avoids the LM head's
    weight magnitudes contaminating the amplification factor.
  - Γ_l is a layer-level aggregate scalar; head-level structure is averaged.
  - Hook-based: one full forward per (layer, sample).  For L=32, n=3
    this is 96+1=97 forward passes.  At ~2s each ≈ 3 min on 8B model.
  - Noise scale: σ_noise ≈ σ_hidden · 4^{-b/2}, roughly matching the
    high-rate scaling order of b-bit quantization noise.
  - All computation on GPU; only final scalars move to CPU.

Usage:
  python allocation/propagation.py --model meta-llama/Llama-3.1-8B --load-in-8bit
  python allocation/propagation.py --model meta-llama/Llama-3.1-8B --load-in-8bit --inject-point attention
"""

import torch
import json
import time
import argparse
from typing import List, Optional, Any
from pathlib import Path

import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, '..'))

from sensitivity import ModelSpec, _resolve_attr


# ============================================================
# Final-hidden-state capture
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
    """Hook helper that stores the output of a module."""
    def __init__(self):
        self.value: Optional[torch.Tensor] = None

    def hook(self, module, input, output):
        # final norm output is a single tensor (not a tuple)
        if isinstance(output, tuple):
            self.value = output[0].detach()
        else:
            self.value = output.detach()


# ============================================================
# Propagation measurer
# ============================================================

class PropagationMeasurer:
    """
    Measure cross-layer error amplification Γ_l.

    Usage::

        measurer = PropagationMeasurer(model)
        results = measurer.run(tokenizer, seq_len=2048, n_samples=3)
        # results['layers'][l] = {'gamma': float, 'gamma_samples': [...]}
    """

    INJECT_POINTS = ('residual', 'attention')

    def __init__(self, model):
        self.model = model
        self.spec = ModelSpec.from_model(model)
        self.final_norm = _find_final_norm(model)
        if self.final_norm is None:
            raise ValueError(
                "Cannot locate final LayerNorm.  Tried: model.norm, "
                "model.model.norm, transformer.ln_f, transformer.norm, "
                "gpt_neox.final_layer_norm"
            )

    def _clean_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run clean forward, return final hidden state (after final LN)."""
        cap = _HiddenCapture()
        handle = self.final_norm.register_forward_hook(cap.hook)
        try:
            with torch.no_grad():
                self.model(input_ids, use_cache=False)
        finally:
            handle.remove()
        return cap.value  # (B, T, hidden_size)

    def _get_inject_target(self, layer_idx: int, inject_point: str):
        """Return the module to hook for noise injection."""
        spec = self.spec
        layer = spec.layers[layer_idx]
        if inject_point == 'residual':
            return layer
        elif inject_point == 'attention':
            return getattr(layer, spec.attn_attr)
        else:
            raise ValueError(f"Unknown inject_point: {inject_point}")

    def _perturbed_pass(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        noise: torch.Tensor,
        inject_point: str = 'residual',
    ) -> torch.Tensor:
        """
        Run forward with additive noise at the specified injection point.

        inject_point='residual':  hook decoder layer output (after full block)
        inject_point='attention': hook self_attn module output (attention branch,
                                  before merge into residual stream)

        Returns final hidden state (after final LN).
        """
        target = self._get_inject_target(layer_idx, inject_point)

        def inject_hook(module, input, output):
            if isinstance(output, tuple):
                if len(output) == 0:
                    raise RuntimeError(
                        f"Layer {layer_idx} ({inject_point}): "
                        f"module returned empty tuple"
                    )
                return (output[0] + noise,) + output[1:]
            elif torch.is_tensor(output):
                return output + noise
            else:
                raise RuntimeError(
                    f"Layer {layer_idx} ({inject_point}): "
                    f"unexpected output type {type(output).__name__}. "
                    f"Expected tuple or Tensor."
                )

        cap = _HiddenCapture()
        h_inject = target.register_forward_hook(inject_hook)
        h_capture = self.final_norm.register_forward_hook(cap.hook)

        try:
            with torch.no_grad():
                self.model(input_ids, use_cache=False)
        finally:
            h_inject.remove()
            h_capture.remove()

        return cap.value  # (B, T, hidden_size)

    def run(
        self,
        tokenizer,
        seq_len: int = 2048,
        n_samples: int = 3,
        noise_bits: float = 4.0,
        inject_point: str = 'residual',
        text: Optional[str] = None,
    ) -> dict:
        """
        Measure Γ_l for every layer.

        Parameters
        ----------
        tokenizer    : PreTrainedTokenizer
        seq_len      : calibration sequence length
        n_samples    : number of noise draws per layer (averaged)
        noise_bits   : noise magnitude roughly matches this bitrate's
                       quantization error (high-rate scaling order).
                       σ_noise = σ_hidden · 4^{-b/2}.
        inject_point : 'residual' (decoder layer output) or
                       'attention' (self_attn output, closer to Thm 2).
        text         : calibration text (defaults to wikitext-2 test split)

        Returns
        -------
        dict with:
            layers: [{gamma, gamma_samples, noise_scale}, ...]
            elapsed_sec: wall time
        """
        if inject_point not in self.INJECT_POINTS:
            raise ValueError(
                f"inject_point must be one of {self.INJECT_POINTS}, "
                f"got '{inject_point}'"
            )

        spec = self.spec
        device = next(self.model.parameters()).device

        # ---- Calibration tokens ----
        if text is None:
            from datasets import load_dataset
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            text = "\n\n".join(ds['text'])

        input_ids = tokenizer(
            text, return_tensors='pt', truncation=True, max_length=seq_len,
        ).input_ids.to(device)
        actual_len = input_ids.shape[1]
        print(f"  Calibration: {actual_len} tokens on {device}")
        print(f"  Injection point: {inject_point}")

        # ---- Clean pass ----
        print("  Clean forward pass …")
        t0 = time.time()
        clean_hidden = self._clean_pass(input_ids)  # (B, T, D)

        # ---- Noise scale ----
        # σ_noise ≈ σ_hidden · 4^{-b/2}: roughly matches the high-rate
        # scaling order of b-bit quantization noise (not an exact match
        # to E₈ or any specific quantizer geometry).
        hidden_sigma = clean_hidden.float().std()
        noise_sigma = hidden_sigma * (4.0 ** (-noise_bits / 2.0))
        print(f"  Hidden σ={hidden_sigma.item():.4f}, "
              f"noise σ={noise_sigma.item():.4e} "
              f"(~{noise_bits:.0f}-bit high-rate order)")

        # ---- Per-layer measurement ----
        results_layers: List[dict] = []
        total_passes = spec.n_layers * n_samples

        for l in range(spec.n_layers):
            gammas = []
            for s in range(n_samples):
                noise = torch.randn_like(clean_hidden) * noise_sigma

                perturbed_hidden = self._perturbed_pass(
                    input_ids, l, noise, inject_point)

                # Γ_l = E[‖Δh‖²] / E[‖ε‖²]
                delta_sq = ((perturbed_hidden.float() - clean_hidden.float()) ** 2).mean()
                noise_sq = (noise.float() ** 2).mean()
                gamma = (delta_sq / noise_sq).item()
                gammas.append(gamma)

                del noise, perturbed_hidden

            gamma_mean = sum(gammas) / len(gammas)
            results_layers.append({
                'gamma': gamma_mean,
                'gamma_samples': gammas,
                'noise_scale': noise_sigma.item(),
            })

            done = (l + 1) * n_samples
            print(f"    Layer {l:>2}: Γ = {gamma_mean:.4f}  "
                  f"[{done}/{total_passes} passes]")

        elapsed = time.time() - t0
        print(f"  Total: {elapsed:.1f}s ({elapsed / total_passes:.1f}s per pass)")

        return {
            'model_spec': {
                'n_layers': spec.n_layers,
                'n_heads': spec.n_heads,
                'n_kv_heads': spec.n_kv_heads,
                'head_dim': spec.head_dim,
            },
            'seq_len': actual_len,
            'noise_bits': noise_bits,
            'n_samples': n_samples,
            'inject_point': inject_point,
            'hidden_sigma': hidden_sigma.item(),
            'layers': results_layers,
            'elapsed_sec': elapsed,
        }


# ============================================================
# Convenience: load + measure + save
# ============================================================

def measure_propagation(
    model_name: str,
    seq_len: int = 2048,
    n_samples: int = 3,
    noise_bits: float = 4.0,
    inject_point: str = 'residual',
    load_in_8bit: bool = False,
    output_dir: Optional[str] = None,
) -> dict:
    """End-to-end: load model → measure Γ → save JSON."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map='cuda:0')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='cuda:0')
    model.eval()

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {mem:.1f} GB VRAM")

    measurer = PropagationMeasurer(model)
    results = measurer.run(tokenizer, seq_len=seq_len,
                           n_samples=n_samples, noise_bits=noise_bits,
                           inject_point=inject_point)
    results['model'] = model_name

    # ---- Summary ----
    gammas = [lr['gamma'] for lr in results['layers']]
    print(f"\n{'Layer':>6} | {'Γ':>10}")
    print("-" * 20)
    for i, g in enumerate(gammas):
        bar = '█' * min(int(g * 20), 60)
        print(f"{i:>6} | {g:>10.4f}  {bar}")

    print(f"\n  Γ range: [{min(gammas):.4f}, {max(gammas):.4f}]")
    print(f"  Γ mean:  {sum(gammas)/len(gammas):.4f}")

    # ---- Save ----
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / 'results')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_short = model_name.split('/')[-1]
    save_path = Path(output_dir) / f'propagation_{model_short}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {save_path}")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LatticeQuant v3: Measure cross-layer error propagation')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--n-samples', type=int, default=3)
    parser.add_argument('--noise-bits', type=float, default=4.0)
    parser.add_argument('--inject-point', type=str, default='residual',
                        choices=['residual', 'attention'],
                        help="'residual': decoder layer output. "
                             "'attention': self_attn output (closer to Thm 2).")
    parser.add_argument('--load-in-8bit', action='store_true',
                        help='Load model weights in 8-bit (BitsAndBytes)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    measure_propagation(args.model, args.seq_len, args.n_samples,
                        args.noise_bits, args.inject_point,
                        args.load_in_8bit, args.output_dir)


if __name__ == '__main__':
    main()