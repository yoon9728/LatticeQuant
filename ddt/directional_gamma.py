"""
Directional Gamma: Isotropic vs Anisotropic Noise Amplification
================================================================
Key experiment: does the DIRECTION of quantization error matter
for cross-layer amplification?

Method:
  For each layer l:
    1. Inject isotropic noise -> measure Gamma_iso
    2. Inject anisotropic noise (scaled by per-dim hidden state variance)
       -> measure Gamma_aniso
    3. Compare: if Gamma_aniso >> Gamma_iso, then the network amplifies
       structured errors more than random errors.

Why anisotropic noise matches quantization error:
  For per-channel/per-dim quantization, error variance per dimension
  is proportional to signal variance in that dimension.
  High-variance dimensions get larger absolute error.
  This is the dominant source of error anisotropy in practice
  (KIVI, KVQuant, scalar quantizers all have this property).

Usage:
  python -m ddt.directional_gamma --model meta-llama/Llama-3.1-8B
  python -m ddt.directional_gamma --model Qwen/Qwen2.5-7B
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LayerOutputCapture:
    """Capture hidden states at each decoder layer output."""

    def __init__(self, model):
        self.model = model
        self.hidden_states = {}  # layer_idx -> (1, T, d)
        self.hooks = []
        self.layers = self._find_layers()

    def _find_layers(self):
        layers = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, layer in enumerate(self.model.model.layers):
                layers.append((i, layer))
        return layers

    def register(self):
        for idx, layer in self.layers:
            def make_hook(i):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.hidden_states[i] = output[0].detach()
                    else:
                        self.hidden_states[i] = output.detach()
                return hook
            h = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(h)

    def clear(self):
        self.hidden_states.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def find_final_norm(model):
    """Locate final LayerNorm before LM head."""
    for path in ['model.norm', 'model.model.norm']:
        parts = path.split('.')
        mod = model
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                mod = None
                break
        if mod is not None:
            return mod
    return None


def get_final_hidden(model, input_ids, final_norm):
    """Forward pass, return final hidden state after final LN."""
    container = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            container['h'] = output[0].detach()
        else:
            container['h'] = output.detach()

    handle = final_norm.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids, use_cache=False)
    handle.remove()
    return container['h']


def inject_and_measure(model, input_ids, final_norm, layer_idx, noise, clean_final):
    """Inject noise at decoder layer output, measure final hidden state change."""
    layers = list(model.model.layers)
    target = layers[layer_idx]

    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + noise,) + output[1:]
        return output + noise

    container = {}
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            container['h'] = output[0].detach()
        else:
            container['h'] = output.detach()

    h1 = target.register_forward_hook(inject_hook)
    h2 = final_norm.register_forward_hook(capture_hook)

    with torch.no_grad():
        model(input_ids, use_cache=False)

    h1.remove()
    h2.remove()

    perturbed = container['h']
    delta_sq = ((perturbed.float() - clean_final.float()) ** 2).mean()
    noise_sq = (noise.float() ** 2).mean()
    gamma = (delta_sq / noise_sq).item()
    return gamma


def run_experiment(model_name, seq_len=512, n_samples=3):
    """Main experiment: isotropic vs anisotropic Gamma."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    final_norm = find_final_norm(model)
    if final_norm is None:
        raise RuntimeError("Cannot find final norm")

    # Calibration tokens
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)
    print(f"  Tokens: {input_ids.shape[1]}")

    # Clean forward + capture per-layer hidden states
    print("  Capturing per-layer hidden states...")
    capturer = LayerOutputCapture(model)
    capturer.register()
    clean_final = get_final_hidden(model, input_ids, final_norm)

    # Also need hidden states for anisotropic noise
    # Run again with capture hooks
    capturer.clear()
    with torch.no_grad():
        model(input_ids, use_cache=False)
    layer_hidden = dict(capturer.hidden_states)
    capturer.remove()

    n_layers = len(layer_hidden)
    hidden_sigma = clean_final.float().std().item()
    noise_scale = hidden_sigma * (4.0 ** (-4.0 / 2.0))  # ~4bit equivalent
    print(f"  Layers: {n_layers}, hidden sigma: {hidden_sigma:.4f}, "
          f"noise scale: {noise_scale:.4e}")

    # Per-layer experiment
    results = []
    for l in range(n_layers):
        h_l = layer_hidden[l].float()  # (1, T, d)

        # Per-dimension variance profile of hidden states at this layer
        per_dim_std = h_l.std(dim=1, keepdim=True)  # (1, 1, d)

        # Normalize so total noise power is same for both conditions
        per_dim_std_normalized = per_dim_std / (per_dim_std.square().mean().sqrt() + 1e-10)

        gammas_iso = []
        gammas_aniso = []

        for s in range(n_samples):
            torch.manual_seed(42 + l * 100 + s)
            base_noise = torch.randn_like(clean_final)

            # Isotropic: same noise power in all dimensions
            noise_iso = base_noise * noise_scale
            g_iso = inject_and_measure(
                model, input_ids, final_norm, l, noise_iso, clean_final)
            gammas_iso.append(g_iso)

            # Anisotropic: noise power proportional to hidden state variance
            noise_aniso = base_noise * per_dim_std_normalized.to(base_noise.dtype) * noise_scale
            g_aniso = inject_and_measure(
                model, input_ids, final_norm, l, noise_aniso, clean_final)
            gammas_aniso.append(g_aniso)

        g_iso_mean = np.mean(gammas_iso)
        g_aniso_mean = np.mean(gammas_aniso)
        ratio = g_aniso_mean / max(g_iso_mean, 1e-10)

        # Per-dim variance stats (how anisotropic is this layer?)
        dim_var = per_dim_std.squeeze().cpu().numpy() ** 2
        am_gm = np.mean(dim_var) / max(np.exp(np.mean(np.log(dim_var + 1e-30))), 1e-30)

        results.append({
            'layer': l,
            'gamma_iso': g_iso_mean,
            'gamma_aniso': g_aniso_mean,
            'ratio': ratio,
            'am_gm_hidden': float(am_gm),
            'per_dim_std_max_min': float(per_dim_std.max() / max(per_dim_std.min(), 1e-10)),
        })

        flag = "***" if ratio > 2.0 else ""
        print(f"  Layer {l:2d}: Γ_iso={g_iso_mean:8.2f}  "
              f"Γ_aniso={g_aniso_mean:8.2f}  "
              f"ratio={ratio:6.2f}  "
              f"AM/GM={am_gm:8.1f}  {flag}")

    # Summary
    ratios = [r['ratio'] for r in results]
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Γ_aniso/Γ_iso: mean={np.mean(ratios):.2f}  "
          f"max={np.max(ratios):.2f}  min={np.min(ratios):.2f}")
    print(f"  Layers with ratio > 2: "
          f"{sum(1 for r in ratios if r > 2)}/{len(ratios)}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Directional Gamma: iso vs aniso amplification")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    results = run_experiment(args.model, args.seq_len, args.n_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir, f"directional_gamma_{tag}.json")

    save = {
        'model': args.model,
        'seq_len': args.seq_len,
        'n_samples': args.n_samples,
        'layers': results,
        'summary': {
            'ratio_mean': float(np.mean([r['ratio'] for r in results])),
            'ratio_max': float(np.max([r['ratio'] for r in results])),
        }
    }
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()