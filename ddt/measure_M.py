"""
DDT — Attention Sensitivity Matrix Mj Measurement (v2)
========================================================
Week 1: Verify that Mj is computable and kappa(Mj) differs across models.

Mathematical definition (V-path):
  When value at position j is perturbed by ej, loss changes by:
    dL ~ sum_j s_j^T ej
  where s_j = sum_i sum_h a_{ij}^h * (W_O^h)^T g_i

  Layer-aggregated: M_l = (1/T) sum_j s_j s_j^T
  True cost: E[(dL)^2] ~ sum_j tr(M_j Sigma_{e,j})

Hook placement:
  - Forward hook on self_attn: capture attention weights
  - Backward hook on self_attn: capture dL/d(self_attn output)

GQA: Each KV head is shared by (n_q_heads / n_kv_heads) query heads.

Usage:
  python -m ddt.measure_M --model meta-llama/Llama-3.1-8B --n-chunks 5
  python -m ddt.measure_M --model Qwen/Qwen2.5-7B --n-chunks 5
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_calibration_data(tokenizer, dataset_name="wikitext", n_chunks=5,
                         seq_len=512, seed=42):
    """Load calibration chunks. Returns list of (1, seq_len) tensors."""
    from datasets import load_dataset

    rng = np.random.RandomState(seed)

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 100])
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []
        for ex in ds:
            texts.append(ex["text"])
            if len(texts) >= 500:
                break
        text = "\n\n".join(texts)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    max_start = len(tokens) - seq_len
    if max_start <= 0:
        raise ValueError(f"Text too short: {len(tokens)} tokens < {seq_len}")

    starts = rng.choice(max_start, size=n_chunks, replace=False)
    return [tokens[s : s + seq_len].unsqueeze(0) for s in sorted(starts)]


class SelfAttnHookManager:
    """Hooks on self_attn modules to capture attention weights and gradients."""

    def __init__(self, model):
        self.model = model
        self.attn_weights = {}
        self.attn_out_grads = {}
        self.hooks = []
        self.layer_configs = {}
        self._register(model)

    def _find_layers(self):
        layers = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer, "self_attn"):
                    layers.append((i, layer))
        if not layers:
            raise RuntimeError("Cannot find attention layers.")
        return layers

    def _register(self, model):
        layers = self._find_layers()

        for layer_idx, decoder_layer in layers:
            attn = decoder_layer.self_attn

            # Extract config — transformers 5.0+ uses attn.config
            config = {}
            config["n_heads"] = attn.config.num_attention_heads
            config["n_kv_heads"] = attn.config.num_key_value_heads
            config["head_dim"] = attn.head_dim
            config["W_O"] = attn.o_proj.weight.detach().float()

            self.layer_configs[layer_idx] = config

            # Forward hook: capture attention weights
            def make_fwd_hook(idx):
                def hook(module, args, kwargs, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        weights = output[1]
                        if weights is not None:
                            self.attn_weights[idx] = weights.detach().float()
                    return output
                return hook

            h1 = attn.register_forward_hook(make_fwd_hook(layer_idx),
                                            with_kwargs=True)
            self.hooks.append(h1)

            # Backward hook: capture dL/d(self_attn output)
            def make_bwd_hook(idx):
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        self.attn_out_grads[idx] = grad_output[0].detach().float()
                return hook

            h2 = attn.register_full_backward_hook(make_bwd_hook(layer_idx))
            self.hooks.append(h2)

        print(f"  Registered hooks on {len(layers)} self_attn modules")
        sample = self.layer_configs[layers[0][0]]
        print(f"    Config: n_heads={sample['n_heads']}, "
              f"n_kv_heads={sample['n_kv_heads']}, "
              f"head_dim={sample['head_dim']}, "
              f"W_O shape={tuple(sample['W_O'].shape)}")

    @property
    def n_layers(self):
        return len(self.layer_configs)

    @property
    def layer_indices(self):
        return sorted(self.layer_configs.keys())

    def clear(self):
        self.attn_weights.clear()
        self.attn_out_grads.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_value_sensitivity(A, g, W_O, n_heads, n_kv_heads, head_dim):
    """
    Compute dL/dv_j for each position j.

    Args:
        A:     (n_heads, T, T) attention weights
        g:     (T, d_model) dL/d(self_attn output)
        W_O:   (d_model, n_heads * head_dim) output projection weight

    Returns:
        grad_v: (T, n_kv_heads * head_dim)
    """
    T = g.shape[0]
    device = g.device
    heads_per_kv = n_heads // n_kv_heads

    # Project gradient into per-head output space
    # g @ W_O: (T, d_model) @ (d_model, n_heads*head_dim) = (T, n_heads*head_dim)
    g_proj = (g @ W_O).view(T, n_heads, head_dim)

    # For each KV head, aggregate over query heads in the group
    grad_v_parts = []
    for kv_h in range(n_kv_heads):
        qh_start = kv_h * heads_per_kv
        qh_end = (kv_h + 1) * heads_per_kv

        grad_kv_h = torch.zeros(T, head_dim, device=device)
        for qh in range(qh_start, qh_end):
            grad_kv_h = grad_kv_h + A[qh].T @ g_proj[:, qh, :]

        grad_v_parts.append(grad_kv_h)

    return torch.cat(grad_v_parts, dim=1)


def compute_M_from_grad_v(grad_v, T):
    """
    Compute M_l = (1/T) sum_j grad_v[j] @ grad_v[j]^T and spectral properties.
    Uses SVD for efficiency.
    """
    d_v = grad_v.shape[1]
    grad_scaled = grad_v / (T ** 0.5)

    try:
        S = torch.linalg.svdvals(grad_scaled)
        eigenvalues = (S ** 2).cpu().numpy()
    except Exception as e:
        print(f"    [WARN] SVD failed ({e}), using diagonal fallback")
        M_diag = (grad_scaled ** 2).sum(dim=0).cpu().numpy()
        eigenvalues = np.sort(M_diag)[::-1]

    eigenvalues = np.maximum(eigenvalues, 0)
    tr = float(eigenvalues.sum())

    if tr < 1e-20:
        return {'eigenvalues': eigenvalues.tolist(), 'trace': tr,
                'lambda_max': 0, 'lambda_min': 0, 'kappa': 1.0,
                'kappa_eff': 1.0, 'top10_ratio': 1.0,
                'n_effective_dims': 0, 'd_v': d_v}

    lmax = float(eigenvalues[0])
    noise_floor = lmax * 1e-10
    pos_eigs = eigenvalues[eigenvalues > noise_floor]
    lmin = float(pos_eigs[-1]) if len(pos_eigs) > 0 else noise_floor

    return {
        'eigenvalues': eigenvalues.tolist(),
        'trace': tr,
        'lambda_max': lmax,
        'lambda_min': float(lmin),
        'kappa': lmax / max(lmin, 1e-20),
        'kappa_eff': lmax / max(tr / d_v, 1e-20),
        'top10_ratio': float(eigenvalues[:10].sum() / max(tr, 1e-20)),
        'n_effective_dims': int((eigenvalues > noise_floor).sum()),
        'd_v': d_v,
    }


def measure_M(model, tokenizer, chunks, device):
    """Run forward+backward on chunks, compute M per layer."""

    hook_mgr = SelfAttnHookManager(model)
    chunk_metrics = {idx: [] for idx in hook_mgr.layer_indices}

    for ci, chunk in enumerate(chunks):
        print(f"\n  Chunk {ci+1}/{len(chunks)}: {chunk.shape[1]} tokens")
        hook_mgr.clear()
        model.zero_grad()

        ids = chunk.to(device)
        outputs = model(ids, output_attentions=True, use_cache=False)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss.backward()

        n_attn = len(hook_mgr.attn_weights)
        n_grad = len(hook_mgr.attn_out_grads)
        print(f"    Loss={loss.item():.4f}  "
              f"hooks: attn={n_attn}/{hook_mgr.n_layers}  "
              f"grad={n_grad}/{hook_mgr.n_layers}")

        if n_grad == 0:
            print(f"    [ERROR] No gradients captured!")
            continue

        for layer_idx in hook_mgr.layer_indices:
            if layer_idx not in hook_mgr.attn_weights:
                continue
            if layer_idx not in hook_mgr.attn_out_grads:
                continue

            cfg = hook_mgr.layer_configs[layer_idx]
            A = hook_mgr.attn_weights[layer_idx].squeeze(0)
            g = hook_mgr.attn_out_grads[layer_idx].squeeze(0)
            W_O = cfg["W_O"].to(device)

            grad_v = compute_value_sensitivity(
                A, g, W_O, cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"])

            if grad_v.norm().item() < 1e-12:
                print(f"    [WARN] Layer {layer_idx}: grad_v ~ 0")
                continue

            # Per-KV-head M (d_v=head_dim=128 for all models → fair comparison)
            T = g.shape[0]
            n_kv = cfg["n_kv_heads"]
            d_h = cfg["head_dim"]
            grad_v_heads = grad_v.view(T, n_kv, d_h)  # (T, n_kv, d_h)

            head_results = []
            for kv_h in range(n_kv):
                hr = compute_M_from_grad_v(grad_v_heads[:, kv_h, :], T)
                head_results.append(hr)

            # Aggregate across KV heads: mean of scalar metrics
            result = {
                'kappa': float(np.mean([h['kappa'] for h in head_results])),
                'kappa_eff': float(np.mean([h['kappa_eff'] for h in head_results])),
                'kappa_max_head': float(np.max([h['kappa'] for h in head_results])),
                'kappa_eff_max_head': float(np.max([h['kappa_eff'] for h in head_results])),
                'trace': float(np.mean([h['trace'] for h in head_results])),
                'top10_ratio': float(np.mean([h['top10_ratio'] for h in head_results])),
                'n_effective_dims': float(np.mean([h['n_effective_dims'] for h in head_results])),
                'd_v': d_h,  # now 128 for all models
                'per_head_kappa_eff': [float(h['kappa_eff']) for h in head_results],
            }
            result['grad_v_norm'] = float(grad_v.norm().item())
            result['T'] = T
            chunk_metrics[layer_idx].append(result)

        del outputs, loss
        torch.cuda.empty_cache()

    hook_mgr.remove_hooks()

    # Aggregate: average scalar metrics, keep median-kappa eigenvalues
    averaged = {}
    for layer_idx in hook_mgr.layer_indices:
        results_list = chunk_metrics[layer_idx]
        if not results_list:
            continue

        def agg(key):
            vals = [r[key] for r in results_list]
            return float(np.mean(vals)), float(np.std(vals))

        kappa_m, kappa_s = agg('kappa')
        kappa_eff_m, kappa_eff_s = agg('kappa_eff')
        kappa_eff_max_m, _ = agg('kappa_eff_max_head')
        tr_m, tr_s = agg('trace')
        top10_m, _ = agg('top10_ratio')
        gv_m, _ = agg('grad_v_norm')

        kappas = [r['kappa'] for r in results_list]
        med_idx = int(np.argsort(kappas)[len(kappas) // 2])

        cfg = hook_mgr.layer_configs[layer_idx]
        averaged[layer_idx] = {
            'kappa_mean': kappa_m, 'kappa_std': kappa_s,
            'kappa_eff_mean': kappa_eff_m, 'kappa_eff_std': kappa_eff_s,
            'kappa_eff_max_head': kappa_eff_max_m,
            'trace_mean': tr_m, 'trace_std': tr_s,
            'top10_ratio': top10_m,
            'grad_v_norm': gv_m,
            'n_chunks': len(results_list),
            'd_v': results_list[0]['d_v'],
            'n_heads': cfg['n_heads'],
            'n_kv_heads': cfg['n_kv_heads'],
            'head_dim': cfg['head_dim'],
            'per_head_kappa_eff': results_list[med_idx].get('per_head_kappa_eff', []),        }

    return averaged


def print_summary(results, model_name):
    print(f"\n{'='*70}")
    print(f"  Value Sensitivity Mj: {model_name}")
    print(f"{'='*70}")
    print(f"  {'L':>3} {'k(M)':>10} {'+/-std':>8} {'k_eff':>10} "
          f"{'tr(M)':>12} {'top10':>6} {'|gv|':>10}")
    print(f"  {'-'*63}")

    kappas, traces = [], []
    for li in sorted(results.keys()):
        r = results[li]
        kappas.append(r['kappa_mean'])
        traces.append(r['trace_mean'])
        print(f"  {li:3d} {r['kappa_mean']:10.1f} {r['kappa_std']:8.1f} "
              f"{r['kappa_eff_mean']:10.1f} {r['trace_mean']:12.4e} "
              f"{r['top10_ratio']:6.3f} {r['grad_v_norm']:10.4f}")

    print(f"\n  kappa: mean={np.mean(kappas):.1f}  median={np.median(kappas):.1f}  "
          f"max={np.max(kappas):.1f}  min={np.min(kappas):.1f}")
    print(f"  tr(M): mean={np.mean(traces):.4e}  max={np.max(traces):.4e}")

    stds = [results[k]['kappa_std'] for k in sorted(results)]
    means = [results[k]['kappa_mean'] for k in sorted(results)]
    avg_cv = np.mean([s / max(m, 1) for s, m in zip(stds, means)])
    print(f"  Stability: CV(kappa)={avg_cv:.3f} "
          f"({'OK' if avg_cv < 0.2 else 'UNSTABLE'})")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="DDT: Measure Mj")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-chunks", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "c4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB)")

    print(f"\nLoading {args.model} (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    chunks = get_calibration_data(
        tokenizer, args.dataset, args.n_chunks, args.seq_len, args.seed)
    print(f"Calibration: {len(chunks)} chunks x {args.seq_len} tokens")

    t0 = time.time()
    results = measure_M(model, tokenizer, chunks, device)
    print(f"Total time: {time.time()-t0:.1f}s")

    if not results:
        print("[FATAL] No results.")
        return

    print_summary(results, args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir,
                        f"M_{tag}_{args.dataset}_n{args.n_chunks}.json")

    kappas = [r['kappa_mean'] for r in results.values()]
    save = {
        'model': args.model, 'dataset': args.dataset,
        'n_chunks': args.n_chunks, 'seq_len': args.seq_len, 'seed': args.seed,
        'aggregate': {
            'kappa_mean': float(np.mean(kappas)),
            'kappa_median': float(np.median(kappas)),
            'kappa_max': float(np.max(kappas)),
        },
        'layers': {str(k): v for k, v in results.items()},
    }
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()