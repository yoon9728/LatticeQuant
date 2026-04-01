"""
DDT — CABA Anomaly Directional Explanation
============================================
Paper-defining experiment for Directional Distortion Theory.

Core question:
  Why does permuting dimensions within quantization blocks change PPL
  dramatically, even when MSE stays similar or decreases?

DDT answer:
  Loss degradation is governed by tr(M·Σ), not tr(Σ) alone.
  - M = sensitivity matrix (gradient outer product, captures direction)
  - Σ = error covariance (depends on permutation + quantizer)
  Permutation changes which dimensions share a block, altering Σ's
  alignment with M's high-eigenvalue directions.

Experiment design:
  For each (model, permutation_mode={baseline, sorted, random×N_seeds}):
    1. Measure M via backward pass, averaged over multiple calibration
       chunks for stability (addresses reviewer concern on M noise).
    2. Quantize K/V with permutation → compute error → Σ.
    3. Compute tr(MΣ) [directional predictor] and tr(Σ) [error power].
    4. Measure actual Δloss on the same calibration batch (direct
       predictor comparison, not just PPL reference from caba_eval).
    5. Compare rankings: does tr(MΣ) predict degradation better than tr(Σ)?

Quantizer note:
  Uses per-block RMS-shared symmetric uniform quantization (matches
  caba_eval.py exactly).  This is a block-quantizer proxy to isolate
  block-assignment / permutation effects, independent of E₈ lattice
  geometry.  The uniform quantizer is sufficient because the directional
  phenomenon arises from block structure, not lattice shape.

Architecture note:
  Sensitivity is measured w.r.t. k_proj / v_proj layer outputs.  In the
  evaluated implementations (Llama, Qwen, Mistral), these outputs
  coincide with the cached K/V representations — no further linear
  transform is applied between projection and caching.  This is checked
  at runtime by verifying that the hooked output shape matches
  (batch, seq_len, num_kv_heads * head_dim).

Usage:
  python -m ddt.caba_explain \\
      --model Qwen/Qwen2.5-7B \\
      --caba results/caba_qwen2.5_7b.json \\
      --bits 4

  python -m ddt.caba_explain \\
      --model meta-llama/Llama-3.1-8B \\
      --caba results/caba_llama_3.1_8b.json \\
      --bits 4
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# Permutation utilities (self-contained, matches caba_eval.py)
# ============================================================

def load_sorted_permutations(caba_path: str) -> Dict:
    """Load per-layer/head variance-sorted permutations from caba_analysis output."""
    with open(caba_path) as f:
        data = json.load(f)
    perms = {}
    for layer_key, layer_data in data.get("permutations", {}).items():
        layer_idx = int(layer_key.split("_")[1])
        perms[layer_idx] = {}
        for comp in ["K", "V"]:
            perms[layer_idx][comp] = {}
            for head_key, head_data in layer_data.get(comp, {}).items():
                head_idx = int(head_key.split("_")[1])
                perms[layer_idx][comp][head_idx] = torch.tensor(
                    head_data["perm"], dtype=torch.long
                )
    return perms


def make_identity_permutations(num_layers: int, num_kv_heads: int, head_dim: int) -> Dict:
    """Identity permutation (baseline)."""
    perms = {}
    for l in range(num_layers):
        perms[l] = {}
        for comp in ["K", "V"]:
            perms[l][comp] = {}
            for h in range(num_kv_heads):
                perms[l][comp][h] = torch.arange(head_dim)
    return perms


def make_random_permutations(
    num_layers: int, num_kv_heads: int, head_dim: int, seed: int = 42
) -> Dict:
    """Random permutation with explicit seed for reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    perms = {}
    for l in range(num_layers):
        perms[l] = {}
        for comp in ["K", "V"]:
            perms[l][comp] = {}
            for h in range(num_kv_heads):
                perms[l][comp][h] = torch.randperm(head_dim, generator=g)
    return perms


# ============================================================
# Block quantization (matches caba_eval.py _qd_uniform exactly)
# ============================================================

def quantize_uniform_blocks(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-block RMS-shared symmetric uniform quantize-dequantize.

    This is a block-quantizer proxy — not E₈.  Used here to isolate
    block-assignment effects independent of lattice geometry.

    Args:
        x: (..., 8) — last dim is block_size=8.
        bits: quantization bitwidth.

    Returns:
        Quantized-dequantized tensor, same shape.
    """
    n_levels = 2 ** bits
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    x_scaled = x / rms
    half = n_levels / 2
    x_quant = torch.round(x_scaled.clamp(-half, half - 1))
    return x_quant * rms


# ============================================================
# Device utilities
# ============================================================

def get_model_device(model) -> torch.device:
    """Get the device of the first model parameter.

    Safer than model.device for device_map='auto' (sharded) models.
    """
    return next(model.parameters()).device


# ============================================================
# Sensitivity Matrix Measurement
# ============================================================

class SensitivityMeasurer:
    """
    Measures per-head sensitivity matrices M^{l,h} for K and V.

    DDT Definition 1:
      s^l_j = dL/dv^l_j   (multi-layer sensitivity)
      M^l_h = (1/T) sum_j  s^{l,h}_j (s^{l,h}_j)^T  in R^{d_h x d_h}

    Implementation:
      Forward hook on k_proj/v_proj saves output tensors.
      Gradient hook captures dL/d(projection output) during backward.
      M is accumulated over multiple calibration chunks (--n-chunks)
      for stability.

    Architecture note:
      In Llama/Qwen/Mistral, k_proj/v_proj outputs are the tensors
      that enter the KV cache.  No further linear transform is applied
      between projection output and caching.  This is verified at hook
      installation by checking output shape.
    """

    def __init__(self, model, num_kv_heads: int, head_dim: int):
        self.model = model
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hooks = []
        self.kv_tensors = {}
        self.kv_grads = {}
        self._shape_verified = False

    def _make_grad_hook(self, layer_idx: int, comp: str):
        """Gradient hook: saves gradient to CPU."""
        def hook(grad):
            self.kv_grads[(layer_idx, comp)] = grad.detach().cpu()
        return hook

    def _make_fwd_hook(self, layer_idx: int, comp: str):
        """Forward hook: saves output, registers gradient hook.

        The requires_grad_(True) call ensures gradient tracking for
        intermediate activations in BitsAndBytes 8-bit models.  This
        does NOT alter the forward computation — it only retains
        gradient flow through this node.  Validated by sanity check:
        if gradient is captured (non-None, non-zero), the hook worked.
        """
        expected_kv_dim = self.num_kv_heads * self.head_dim

        def hook(module, input, output):
            # Architecture verification (once)
            if not self._shape_verified:
                if output.shape[-1] != expected_kv_dim:
                    print(f"  WARNING: {comp}_proj output dim {output.shape[-1]} "
                          f"!= expected {expected_kv_dim}.  M may not correspond "
                          f"to cached KV tensors.")
                else:
                    self._shape_verified = True

            self.kv_tensors[(layer_idx, comp)] = output.detach().cpu()
            if not output.requires_grad:
                output.requires_grad_(True)
            output.register_hook(self._make_grad_hook(layer_idx, comp))
        return hook

    def _install_hooks(self):
        """Install forward hooks on k_proj and v_proj of all layers."""
        for idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self.hooks.append(
                attn.k_proj.register_forward_hook(self._make_fwd_hook(idx, "K"))
            )
            self.hooks.append(
                attn.v_proj.register_forward_hook(self._make_fwd_hook(idx, "V"))
            )

    def _cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.enable_grad()
    def _measure_single_chunk(self, input_ids: torch.Tensor) -> Tuple[Dict, float, int]:
        """Run forward+backward on one chunk, return per-head data + loss."""
        self.kv_tensors.clear()
        self.kv_grads.clear()
        self._install_hooks()

        outputs = self.model(input_ids, labels=input_ids, use_cache=False)
        loss = outputs.loss
        loss_val = loss.item()

        loss.backward()
        self._cleanup()
        self.model.zero_grad()
        torch.cuda.empty_cache()

        num_layers = len(self.model.model.layers)
        T = input_ids.shape[1]
        chunk_data = {}
        grad_count = 0

        for l in range(num_layers):
            for comp in ["K", "V"]:
                key = (l, comp)
                if key not in self.kv_grads:
                    continue

                raw = self.kv_tensors[key][0].view(T, self.num_kv_heads, self.head_dim)
                grad = self.kv_grads[key][0].view(T, self.num_kv_heads, self.head_dim)
                grad_count += 1

                # Sanity: gradient should be non-zero
                grad_norm = grad.norm().item()
                if grad_norm < 1e-12:
                    print(f"  WARNING: near-zero gradient at layer {l} {comp} "
                          f"(norm={grad_norm:.2e}). Hook may not be working.")

                for h in range(self.num_kv_heads):
                    g_h = grad[:, h, :].float()
                    v_h = raw[:, h, :].float()
                    chunk_data[(l, comp, h)] = {"grad": g_h, "tensor": v_h}

        self.kv_tensors.clear()
        self.kv_grads.clear()

        return chunk_data, loss_val, grad_count

    def measure(
        self,
        all_input_ids: torch.Tensor,
        seq_len: int = 2048,
        n_chunks: int = 4,
    ) -> Tuple[Dict, List[float]]:
        """
        Measure M averaged over n_chunks calibration sequences.

        Args:
            all_input_ids: Full tokenized calibration text [1, total_tokens].
                           Chunks are sliced from this.

        Multi-chunk averaging reduces sensitivity to specific text
        content, addressing the concern that single-backward M is too
        noisy for paper-quality results.

        Returns:
            (results_dict, losses_list)
            results_dict: (layer, comp, head) -> {M, M_eigenvalues, M_trace, tensor}
                'M' is averaged over all chunks (population estimate).
                'tensor' is from the LAST chunk only (for Sigma computation).
                This is intentional: M estimates the expected sensitivity
                landscape (benefits from averaging), while Sigma measures
                error covariance on a specific input (the same input used
                for delta-loss measurement in Phase 3).
            losses_list: per-chunk loss values.
        """
        total_tokens = all_input_ids.shape[1]
        device = get_model_device(self.model)
        losses = []
        M_accum = {}
        last_tensors = {}
        last_grads = {}    # raw gradients from last chunk (for linear predictor)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * seq_len
            if start + seq_len > total_tokens:
                print(f"  Chunk {chunk_idx}: not enough tokens, stopping")
                break

            input_ids = all_input_ids[:, start:start + seq_len].to(device)
            print(f"  Chunk {chunk_idx}/{n_chunks}: tokens [{start}, {start + seq_len})")

            chunk_data, loss_val, grad_count = self._measure_single_chunk(input_ids)
            losses.append(loss_val)
            print(f"    Loss: {loss_val:.4f}, gradients: {grad_count} pairs")

            if chunk_idx == 0:
                expected = len(self.model.model.layers) * 2
                if grad_count < expected:
                    print(f"    WARNING: expected {expected} gradient pairs, got {grad_count}")
                    print(f"    Try --no-8bit if gradients are missing.")

            T = seq_len
            for key, data in chunk_data.items():
                g_h = data["grad"]
                M_chunk = (g_h.T @ g_h) / T

                if key not in M_accum:
                    M_accum[key] = M_chunk
                else:
                    M_accum[key] = M_accum[key] + M_chunk

                last_tensors[key] = data["tensor"]
                last_grads[key] = data["grad"]

            del chunk_data
            torch.cuda.empty_cache()

        actual_chunks = len(losses)
        results = {}
        for key, M_sum in M_accum.items():
            M = M_sum / actual_chunks
            eigvals = torch.linalg.eigvalsh(M).flip(0)

            results[key] = {
                "M": M,
                "M_eigenvalues": eigvals,
                "M_trace": M.trace().item(),
                "tensor": last_tensors[key],
                "grad": last_grads[key],
            }

        print(f"\n  M averaged over {actual_chunks} chunks")
        m_traces = [d["M_trace"] for d in results.values()]
        if m_traces:
            print(f"  M trace range: [{min(m_traces):.6e}, {max(m_traces):.6e}]")

        return results, losses


# ============================================================
# Directional Risk Analysis
# ============================================================

def compute_directional_metrics(
    M_data: Dict,
    perms: Dict,
    bits: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 8,
) -> Dict:
    """
    Compute tr(M Sigma) and error power for a given permutation set.

    For each (layer, component, head):
      1. Apply permutation to raw KV tensor
      2. Reshape into 8-dim blocks -> quantize -> dequantize
      3. Compute error e = x_hat - x (in original dimension order)
      4. Compute Sigma_u = (1/T) e^T e  [uncentered second moment]
      5. Compute Sigma_c = Sigma_u - mu mu^T  [centered covariance]
      6. Compute tr(M Sigma) for both

    Returns K/V/total separated aggregates.
    """
    per_head = {}
    agg = {
        scope: {
            "tr_M_Sigma": 0.0, "tr_Sigma": 0.0,
            "tr_M_Sigma_c": 0.0, "tr_Sigma_c": 0.0,
            "linear_pred": 0.0,
        }
        for scope in ["K", "V", "total"]
    }

    for (l, comp, h), data in M_data.items():
        M = data["M"]
        v_h = data["tensor"]
        g_h = data["grad"]     # [T, d_h] — raw gradient from last chunk
        T = v_h.shape[0]

        perm = perms[l][comp][h]
        inv_perm = torch.argsort(perm)

        v_perm = v_h[:, perm]
        n_blocks = head_dim // block_size
        blocks = v_perm.reshape(T, n_blocks, block_size)
        blocks_qd = quantize_uniform_blocks(blocks, bits)
        v_hat_perm = blocks_qd.reshape(T, head_dim)
        v_hat = v_hat_perm[:, inv_perm]

        e = v_hat - v_h

        # === First-order predictor (Theorem A) ===
        # ΔL ≈ Σ_j s_j^T e_j  (position-wise gradient·error inner product)
        linear_pred_val = (g_h * e).sum().item()

        # Uncentered second moment
        Sigma_u = (e.T @ e) / T

        # Centered covariance
        e_mean = e.mean(dim=0, keepdim=True)
        Sigma_c = Sigma_u - (e_mean.T @ e_mean)

        tr_M_Sigma_u = torch.trace(M @ Sigma_u).item()
        tr_M_Sigma_c = torch.trace(M @ Sigma_c).item()
        tr_Sigma_u = Sigma_u.trace().item()
        tr_Sigma_c = Sigma_c.trace().item()
        mean_bias_norm = e_mean.norm().item()

        # Spectral analysis + HLP bounds (Theorem C)
        M_eigvals = data["M_eigenvalues"]
        Sigma_eigvals = torch.linalg.eigvalsh(Sigma_u).flip(0)

        hlp_upper = (M_eigvals * Sigma_eigvals).sum().item()
        hlp_lower = (M_eigvals * Sigma_eigvals.flip(0)).sum().item()
        hlp_span = hlp_upper - hlp_lower
        hlp_position = ((tr_M_Sigma_u - hlp_lower) / hlp_span
                        if hlp_span > 1e-20 else 0.5)

        per_head[(l, comp, h)] = {
            "linear_pred": linear_pred_val,
            "tr_M_Sigma": tr_M_Sigma_u,
            "tr_M_Sigma_centered": tr_M_Sigma_c,
            "error_power": tr_Sigma_u,
            "error_power_centered": tr_Sigma_c,
            "mean_bias_norm": mean_bias_norm,
            "hlp_upper": hlp_upper,
            "hlp_lower": hlp_lower,
            "hlp_position": hlp_position,
            "M_trace": data["M_trace"],
            "M_top1_frac": (M_eigvals[0] / M_eigvals.sum()).item()
                           if M_eigvals.sum() > 0 else 0,
        }

        for scope in [comp, "total"]:
            agg[scope]["linear_pred"] += linear_pred_val
            agg[scope]["tr_M_Sigma"] += tr_M_Sigma_u
            agg[scope]["tr_Sigma"] += tr_Sigma_u
            agg[scope]["tr_M_Sigma_c"] += tr_M_Sigma_c
            agg[scope]["tr_Sigma_c"] += tr_Sigma_c

    return {"per_head": per_head, **{f"agg_{k}": v for k, v in agg.items()}}


def aggregate_per_layer(
    metrics: Dict, num_layers: int, num_kv_heads: int
) -> List[Dict]:
    """Aggregate per-head metrics to per-layer with K/V separation."""
    per_head = metrics["per_head"]
    layers = []
    for l in range(num_layers):
        layer = {"layer": l}
        for scope in ["K", "V", "total"]:
            tr_MS = 0.0
            tr_S = 0.0
            comps = [scope] if scope != "total" else ["K", "V"]
            for comp in comps:
                for h in range(num_kv_heads):
                    key = (l, comp, h)
                    if key in per_head:
                        d = per_head[key]
                        tr_MS += d["tr_M_Sigma"]
                        tr_S += d["error_power"]
            layer[f"tr_M_Sigma_{scope}"] = tr_MS
            layer[f"error_power_{scope}"] = tr_S
        layers.append(layer)
    return layers


def find_top_offending_heads(metrics: Dict, n: int = 10) -> List[Dict]:
    """Find heads with highest directional risk tr(M Sigma)."""
    per_head = metrics["per_head"]
    ranked = sorted(per_head.items(), key=lambda kv: kv[1]["tr_M_Sigma"], reverse=True)
    return [
        {
            "layer": k[0], "comp": k[1], "head": k[2],
            "tr_M_Sigma": v["tr_M_Sigma"],
            "error_power": v["error_power"],
            "hlp_position": v["hlp_position"],
            "M_top1_frac": v["M_top1_frac"],
        }
        for (k, v) in ranked[:n]
    ]


# ============================================================
# Same-batch actual delta-loss measurement
# ============================================================

@torch.no_grad()
def measure_actual_delta_loss(
    model,
    input_ids: torch.Tensor,
    perms: Dict,
    bits: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 8,
) -> float:
    """
    Measure actual loss increase from quantizing KV with given permutation.

    This is a prefill-style perturbation experiment: all positions are
    processed in a single forward pass with quantized projections.  It
    is NOT identical to autoregressive decode-time cache reuse, but
    serves as a direct local predictor for tr(M Sigma) validation.

    Injects quantization error via forward hooks on k_proj/v_proj:
    intercept output, apply permute -> quantize -> unpermute, replace.
    """
    hooks = []
    num_layers = len(model.model.layers)

    def make_quant_hook(layer_idx, comp):
        def hook(module, input, output):
            x = output.float()
            B, T, _ = x.shape
            x_heads = x.view(B, T, num_kv_heads, head_dim)
            x_out = torch.zeros_like(x_heads)

            for h in range(num_kv_heads):
                perm = perms[layer_idx][comp][h].to(x.device)
                inv_perm = torch.argsort(perm)
                v_h = x_heads[:, :, h, :]
                v_perm = v_h[:, :, perm]
                n_blocks = head_dim // block_size
                blocks = v_perm.reshape(B, T, n_blocks, block_size)
                blocks_qd = quantize_uniform_blocks(blocks, bits)
                v_hat_perm = blocks_qd.reshape(B, T, head_dim)
                x_out[:, :, h, :] = v_hat_perm[:, :, inv_perm]

            return x_out.reshape(B, T, -1).to(output.dtype)
        return hook

    for idx in range(num_layers):
        layer = model.model.layers[idx]
        attn = layer.self_attn
        hooks.append(attn.k_proj.register_forward_hook(make_quant_hook(idx, "K")))
        hooks.append(attn.v_proj.register_forward_hook(make_quant_hook(idx, "V")))

    outputs = model(input_ids, labels=input_ids, use_cache=False)
    quant_loss = outputs.loss.item()

    for h in hooks:
        h.remove()

    return quant_loss


# ============================================================
# PPL result loading
# ============================================================

def load_ppl_results(results_dir: str, model_tag: str) -> Dict:
    """Load actual PPL from existing caba_eval JSON files."""
    ppl_data = {}
    results_path = Path(results_dir)
    tag_lower = model_tag.lower().replace("-", "_")

    for f in sorted(results_path.glob("caba_ppl_*.json")):
        fname_lower = f.stem.lower()
        if tag_lower not in fname_lower:
            continue
        try:
            d = json.load(open(f))
            suffix = f.stem
            for sep_idx in range(len(suffix)):
                remaining = suffix[sep_idx:]
                if remaining.startswith(("baseline", "sorted", "random")):
                    suffix = remaining
                    break

            ppl_data[suffix] = {
                "ppl": d.get("ppl_quant", d.get("ppl", 0)),
                "ppl_fp16": d.get("ppl_fp16", 0),
                "delta_pct": d.get("delta_pct", 0),
                "mode": d.get("mode", "unknown"),
                "bits": d.get("bits", 0),
                "file": f.name,
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: could not parse {f.name}: {e}")
    return ppl_data


# ============================================================
# Model loading
# ============================================================

def load_model(model_name: str, load_in_8bit: bool = True):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"dtype": torch.float16, "device_map": "auto"}
    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, **kwargs
    )
    model.eval()
    return model, tokenizer


# ============================================================
# Statistical utilities
# ============================================================

def spearman_rank_corr(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between two lists."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT CABA Anomaly Directional Explanation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--caba", type=str, default=None,
                        help="Path to caba_analysis JSON (sorted permutation)")
    parser.add_argument("--bits", type=int, nargs="+", default=[4])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-chunks", type=int, default=4,
                        help="Calibration chunks for M averaging (default: 4)")
    parser.add_argument("--n-random-seeds", type=int, default=5,
                        help="Random permutation seeds (default: 5)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true",
                        help="Load model in fp16 instead of 8-bit")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory with existing caba_eval PPL results")
    parser.add_argument("--skip-delta-loss", action="store_true",
                        help="Skip same-batch delta-loss measurement")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"caba_explain_{model_tag}.json")

    print(f"Model:         {args.model}")
    print(f"Bits:          {args.bits}")
    print(f"Seq len:       {args.seq_len}")
    print(f"M chunks:      {args.n_chunks}")
    print(f"Random seeds:  {args.n_random_seeds}")
    print(f"Output:        {args.output}")

    # ---- Load model ----
    print("\nLoading model...")
    model, tokenizer = load_model(args.model, load_in_8bit=not args.no_8bit)

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Load calibration data (once) ----
    device = get_model_device(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"  Calibration tokens: {all_input_ids.shape[1]}")

    # ---- Phase 1: Measure M ----
    print(f"\n{'='*60}")
    print(f"Phase 1: Measuring sensitivity matrices M ({args.n_chunks} chunks)")
    print(f"{'='*60}")
    measurer = SensitivityMeasurer(model, num_kv_heads, head_dim)
    M_data, chunk_losses = measurer.measure(
        all_input_ids, seq_len=args.seq_len, n_chunks=args.n_chunks
    )
    print(f"  Measured {len(M_data)} (layer, comp, head) entries")
    print(f"  Chunk losses: {[f'{l:.4f}' for l in chunk_losses]}")

    # ---- Clean loss for delta-loss (uses LAST chunk = same data as Sigma) ----
    last_chunk_start = (len(chunk_losses) - 1) * args.seq_len
    clean_ids = all_input_ids[:, last_chunk_start:last_chunk_start + args.seq_len].to(device)

    with torch.no_grad():
        clean_loss = model(clean_ids, labels=clean_ids, use_cache=False).loss.item()
    print(f"  Clean loss: {clean_loss:.4f}")

    # ---- Phase 2: Permutations ----
    print(f"\n{'='*60}")
    print("Phase 2: Building permutation sets")
    print(f"{'='*60}")
    perm_sets = {"baseline": make_identity_permutations(num_layers, num_kv_heads, head_dim)}

    if args.caba is not None:
        perm_sets["sorted"] = load_sorted_permutations(args.caba)
        print(f"  Loaded sorted permutations from {args.caba}")

    for seed_i in range(args.n_random_seeds):
        seed = 42 + seed_i
        perm_sets[f"random_s{seed}"] = make_random_permutations(
            num_layers, num_kv_heads, head_dim, seed=seed
        )
    print(f"  Modes: {list(perm_sets.keys())}")

    # ---- Phase 3: Directional metrics + delta-loss ----
    print(f"\n{'='*60}")
    print("Phase 3: Computing directional metrics + delta-loss")
    print(f"{'='*60}")
    all_results = {}

    for bits in args.bits:
        print(f"\n--- Bitwidth: {bits} ---")
        for mode, perms in perm_sets.items():
            t0 = time.time()

            metrics = compute_directional_metrics(
                M_data, perms, bits, num_layers, num_kv_heads, head_dim
            )
            per_layer = aggregate_per_layer(metrics, num_layers, num_kv_heads)
            top_heads = find_top_offending_heads(metrics, n=10)

            delta_loss = None
            if not args.skip_delta_loss:
                quant_loss = measure_actual_delta_loss(
                    model, clean_ids, perms, bits, num_kv_heads, head_dim
                )
                delta_loss = quant_loss - clean_loss

            elapsed = time.time() - t0
            key = f"{mode}_{bits}b"
            all_results[key] = {
                "mode": mode, "bits": bits,
                "agg_K": metrics["agg_K"],
                "agg_V": metrics["agg_V"],
                "agg_total": metrics["agg_total"],
                "delta_loss": delta_loss,
                "per_layer": per_layer,
                "top_heads": top_heads,
                "elapsed_sec": elapsed,
            }

            dl_str = f"dl={delta_loss:.4f}" if delta_loss is not None else "dl=skip"
            lp = metrics['agg_total']['linear_pred']
            print(
                f"  {mode:16s}: "
                f"linpred={lp:.4f}  "
                f"tr(MS)={metrics['agg_total']['tr_M_Sigma']:.4e}  "
                f"{dl_str}  ({elapsed:.1f}s)"
            )

    # ---- Phase 4: PPL reference ----
    print(f"\n{'='*60}")
    print("Phase 4: PPL reference")
    print(f"{'='*60}")
    ppl_data = load_ppl_results(args.results_dir, model_tag)
    for k, v in sorted(ppl_data.items()):
        print(f"  {k}: PPL={v['ppl']:.1f}")

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print(f"DIRECTIONAL ANALYSIS — {model_tag}")
    print(f"{'='*80}")
    print(f"{'Config':<24s} {'linear_pred':>12s} {'delta_loss':>10s} "
          f"{'tr(MS)':>12s} {'err_pwr':>12s} {'PPL':>10s}")
    print("-" * 90)

    for key, res in sorted(all_results.items()):
        ppl_str = ""
        ppl_candidates = [key]
        if res["mode"].startswith("random"):
            ppl_candidates.append(f"random_{res['bits']}b")
        for pk in ppl_candidates:
            if pk in ppl_data:
                ppl_str = f"{ppl_data[pk]['ppl']:.1f}"
                break

        dl_str = f"{res['delta_loss']:.4f}" if res['delta_loss'] is not None else "-"
        lp = res['agg_total']['linear_pred']
        print(f"{key:<24s} {lp:>12.4f} {dl_str:>10s} "
              f"{res['agg_total']['tr_M_Sigma']:>12.4e} "
              f"{res['agg_total']['tr_Sigma']:>12.4e} {ppl_str:>10s}")

    # ---- K vs V breakdown ----
    print(f"\n--- K vs V breakdown ---")
    print(f"{'Config':<24s} {'tr(MS)_K':>12s} {'tr(MS)_V':>12s} {'K_frac':>8s}")
    print("-" * 60)
    for key, res in sorted(all_results.items()):
        k_val = res["agg_K"]["tr_M_Sigma"]
        v_val = res["agg_V"]["tr_M_Sigma"]
        total = k_val + v_val
        k_frac = k_val / total if total > 0 else 0
        print(f"{key:<24s} {k_val:>12.4e} {v_val:>12.4e} {k_frac:>8.1%}")

    # ---- Ranking + Spearman ----
    print(f"\n--- Ranking Diagnostic ---")
    for bits in args.bits:
        items = {k: v for k, v in all_results.items() if v["bits"] == bits}
        if len(items) < 3:
            continue

        keys_list = sorted(items.keys())
        lin_preds = [items[k]["agg_total"]["linear_pred"] for k in keys_list]
        err_pwrs = [items[k]["agg_total"]["tr_Sigma"] for k in keys_list]
        dir_risks = [items[k]["agg_total"]["tr_M_Sigma"] for k in keys_list]
        delta_losses = [items[k]["delta_loss"] for k in keys_list]

        has_dl = all(dl is not None for dl in delta_losses)

        print(f"\n  {bits}b ({len(keys_list)} configs):")
        if has_dl:
            rho_lin = spearman_rank_corr(lin_preds, delta_losses)
            rho_dir = spearman_rank_corr(dir_risks, delta_losses)
            rho_mse = spearman_rank_corr(err_pwrs, delta_losses)
            print(f"    Spearman(linear_pred, dl) = {rho_lin:.4f}  ← Theorem A")
            print(f"    Spearman(tr(MS), dl)      = {rho_dir:.4f}  ← Theorem B (variance)")
            print(f"    Spearman(err_pwr, dl)     = {rho_mse:.4f}  ← naive MSE")
            # Pearson for linear_pred vs dl (should be ~1.0 in linear regime)
            if len(lin_preds) >= 3:
                corr = np.corrcoef(lin_preds, delta_losses)[0, 1]
                print(f"    Pearson(linear_pred, dl)  = {corr:.4f}")
            winner = max(
                [("linear_pred", abs(rho_lin)), ("tr(MS)", abs(rho_dir)), ("err_pwr", abs(rho_mse))],
                key=lambda x: x[1]
            )
            print(f"    >>> Best predictor: {winner[0]}")

    # ---- Random seed stats ----
    for bits in args.bits:
        r_keys = [k for k in all_results
                  if k.startswith("random_") and all_results[k]["bits"] == bits]
        if len(r_keys) < 2:
            continue

        r_lin = [all_results[k]["agg_total"]["linear_pred"] for k in r_keys]
        r_dir = [all_results[k]["agg_total"]["tr_M_Sigma"] for k in r_keys]
        r_err = [all_results[k]["agg_total"]["tr_Sigma"] for k in r_keys]
        r_dl = [all_results[k]["delta_loss"] for k in r_keys
                if all_results[k]["delta_loss"] is not None]

        print(f"\n  Random seed stats ({bits}b, {len(r_keys)} seeds):")
        print(f"    linear_pred: {np.mean(r_lin):.4f} +/- {np.std(r_lin):.4f}")
        print(f"    tr(MS): {np.mean(r_dir):.4e} +/- {np.std(r_dir):.4e}")
        print(f"    err_pwr: {np.mean(r_err):.4e} +/- {np.std(r_err):.4e}")
        if r_dl:
            print(f"    dl: {np.mean(r_dl):.4f} +/- {np.std(r_dl):.4f}")

        sorted_key = f"sorted_{bits}b"
        if sorted_key in all_results:
            s_dir = all_results[sorted_key]["agg_total"]["tr_M_Sigma"]
            outside = s_dir < min(r_dir) or s_dir > max(r_dir)
            print(f"    sorted tr(MS)={s_dir:.4e} — "
                  f"{'outside' if outside else 'inside'} random cloud")

    # ---- Centered vs uncentered ----
    print(f"\n--- Centered vs Uncentered ---")
    any_large = False
    for key, res in sorted(all_results.items()):
        u = res["agg_total"]["tr_M_Sigma"]
        c = res["agg_total"]["tr_M_Sigma_c"]
        diff_pct = abs(u - c) / abs(u) * 100 if abs(u) > 1e-20 else 0
        if diff_pct > 1:
            print(f"  {key}: uncentered={u:.4e}, centered={c:.4e}, diff={diff_pct:.1f}%")
            any_large = True
    if not any_large:
        print(f"  All differences < 1% (mean-zero quantizer assumption confirmed)")

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "config": {
            "seq_len": args.seq_len,
            "n_chunks": args.n_chunks,
            "n_random_seeds": args.n_random_seeds,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "clean_loss": clean_loss,
        "chunk_losses": chunk_losses,
        "bits": args.bits,
        "metrics": {},
        "ppl_reference": ppl_data,
    }

    for key, res in all_results.items():
        output_data["metrics"][key] = {
            "mode": res["mode"],
            "bits": res["bits"],
            "agg_K": res["agg_K"],
            "agg_V": res["agg_V"],
            "agg_total": res["agg_total"],
            "delta_loss": res["delta_loss"],
            "per_layer": res["per_layer"],
            "top_heads": res["top_heads"],
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()