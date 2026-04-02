"""
DDT — CABA Anomaly Directional Explanation (v2)
=================================================
Paper-defining experiment for Directional Distortion Theory.

v2 changes (addressing reviewer critique):
  - Permutation count: 7 -> 50+ (statistical significance)
  - Bit-width sweep: [2, 3, 4, 5] (regime coverage)
  - Three metrics compared on identical config set:
      Q1 (linear predictor, Theorem A),
      tr(M Sigma) (directional risk, Theorem B proxy),
      tr(Sigma) (MSE / error power, naive baseline)
  - Bootstrap CI for all Spearman/Pearson correlations
  - End-to-end evidence: MSE-best vs DDT-best config comparison
  - Flat output structure for direct figure generation

Core question:
  Why does permuting dimensions within quantization blocks change PPL
  dramatically, even when MSE stays similar or decreases?

DDT answer:
  Loss degradation is governed by tr(M·Sigma), not tr(Sigma) alone.
  - M = sensitivity matrix (gradient outer product, captures direction)
  - Sigma = error covariance (depends on permutation + quantizer)
  Permutation changes which dimensions share a block, altering Sigma's
  alignment with M's high-eigenvalue directions.

Theory-code correspondence:
  Q1 = sum_{l,j} (s^l_j)^T e^l_j          [Theorem A, Eq. (3)]
     Code: (g_h * e).sum() per head, aggregated over (l, comp, h).
     g_h = dL/d(v^l_j)|_clean from backward pass (last chunk).
     e = v_hat - v from quantize-dequantize on same last chunk.

  tr(M Sigma) = directional risk proxy     [Theorem B(ii) analog]
     M = (1/T) sum_j s_j s_j^T  (multi-chunk averaged).
     Sigma_u = (1/T) e^T e  (uncentered, deterministic error).
     In dithered setting this equals the theorem's exact variance;
     in deterministic setting it serves as a ranking proxy.

  tr(Sigma) = MSE / error power            [naive baseline]
     Isotropic assumption: treats all error directions equally.

  delta_loss = L(v_hat) - L(v)             [Theorem A, measured]
     Forward pass with quantization hooks on all layers simultaneously.
     Includes cross-layer effects (Q2 + higher-order).
     Computed on same data batch as Q1 and Sigma for fair comparison.

  HLP bounds: sum a_i b_{d+1-i} <= tr(M Sigma) <= sum a_i b_i
     [Theorem C(ii)]
     a_i, b_i = eigenvalues of M, Sigma in descending order.

  Centered/uncentered gap:
     Sigma_u - Sigma_c = mu mu^T where mu = E[e|D]  [Remark 5]
     Quantifies bias drift in deterministic quantizers.

Quantizer note:
  Uses per-block RMS-shared symmetric uniform quantization (matches
  caba_eval.py exactly).  This is a block-quantizer proxy to isolate
  block-assignment / permutation effects, independent of E_8 lattice
  geometry.  The uniform quantizer is sufficient because the directional
  phenomenon arises from block structure, not lattice shape.

Architecture note:
  Sensitivity is measured w.r.t. k_proj / v_proj layer outputs.  In the
  evaluated implementations (Llama, Qwen, Mistral), these outputs
  coincide with the cached K/V representations -- no further linear
  transform is applied between projection and caching.  This is checked
  at runtime by verifying that the hooked output shape matches
  (batch, seq_len, num_kv_heads * head_dim).

Usage:
  # Full P0 experiment (50 permutations x 4 bitwidths = 200 configs)
  python -m ddt.caba_explain_v2 \\
      --model meta-llama/Llama-3.1-8B \\
      --caba results/caba_llama_3.1_8b.json \\
      --bits 2 3 4 5 --n-random-seeds 48

  python -m ddt.caba_explain_v2 \\
      --model Qwen/Qwen2.5-7B \\
      --caba results/caba_qwen2.5_7b.json \\
      --bits 2 3 4 5 --n-random-seeds 48
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

# scipy for proper Spearman/Pearson with tie-handling and p-values
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    from scipy.stats import pearsonr as _scipy_pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Using manual Spearman (no tie handling). "
          "Install scipy for publication-quality statistics.")


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

def quantize_uniform_blocks(x: torch.Tensor, bits: int, alpha: float = 3.0) -> torch.Tensor:
    """Per-block symmetric uniform quantize-dequantize with bit-dependent step size.

    This is a block-quantizer proxy -- not E_8.  Used here to isolate
    block-assignment effects independent of lattice geometry.

    Quantization grid: 2^bits levels spanning [-alpha*rms, alpha*rms].
    Step size (scale): alpha * rms / 2^(b-1).

    Each additional bit halves the step size, yielding ~6 dB/bit MSE
    reduction (standard uniform quantization property).

    Bug fix note (v2):
      The v1 quantizer used scale=rms regardless of bits, so step size
      was constant and only the clipping range changed with bits.
      For b>=3 with RMS-normalized data in [-3,3], clipping never occurred,
      making 3b/4b/5b produce identical results.
      The fix: scale = alpha*rms / half, making step size ∝ 2^{-b}.

    Args:
        x: (..., 8) -- last dim is block_size=8.
        bits: quantization bitwidth (integer).
        alpha: number of RMS units covered by the full grid range.
               alpha=3.0 covers ±3*rms (99.7% of Gaussian data).

    Returns:
        Quantized-dequantized tensor, same shape.
    """
    n_levels = 2 ** bits
    half = n_levels / 2   # float: 2b->2.0, 3b->4.0, 4b->8.0, 5b->16.0
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    scale = alpha * rms / half   # step size in original coordinates
    x_scaled = x / scale
    x_quant = torch.round(x_scaled.clamp(-half, half - 1))
    return x_quant * scale


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
        does NOT alter the forward computation -- it only retains
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
            results_dict: (layer, comp, head) -> {M, M_eigenvalues, M_trace, tensor, grad}
                'M' is averaged over all chunks (population estimate).
                'tensor' is from the LAST chunk only (for Sigma computation).
                'grad' is from the LAST chunk only (for Q1 computation).
                This is intentional: M estimates the expected sensitivity
                landscape (benefits from averaging), while Sigma and Q1
                measure quantities on a specific input (the same input used
                for delta-loss measurement).
            losses_list: per-chunk loss values.
        """
        total_tokens = all_input_ids.shape[1]
        device = get_model_device(self.model)
        losses = []
        M_accum = {}
        last_tensors = {}
        last_grads = {}

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
                # M_chunk = (1/T) sum_j s_j s_j^T  [paper Definition 1]
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

    Theory-code correspondence for each (layer, component, head):
      1. Apply permutation to raw KV tensor
      2. Reshape into 8-dim blocks -> quantize -> dequantize
      3. Compute error e = x_hat - x (in original dimension order)
      4. Q1 contribution: sum_j s_j^T e_j  [Theorem A, Eq. (3)]
         Uses last-chunk gradient (same data as delta-loss measurement)
      5. Sigma_u = (1/T) e^T e  [uncentered second moment]
      6. Sigma_c = Sigma_u - mu mu^T  [centered covariance, Remark 5]
      7. tr(M Sigma_u) = directional risk  [Theorem B proxy]
      8. tr(Sigma_u) = MSE / error power  [naive baseline]
      9. HLP bounds from eigenvalues  [Theorem C(ii)]

    Returns K/V/total separated aggregates + per-head details.
    """
    assert head_dim % block_size == 0, (
        f"head_dim={head_dim} not divisible by block_size={block_size}. "
        f"Block quantization requires exact division."
    )
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
        v_h = data["tensor"]      # [T, d_h] -- last chunk KV tensor
        g_h = data["grad"]        # [T, d_h] -- last chunk gradient = s^l_j
        T = v_h.shape[0]

        perm = perms[l][comp][h]
        inv_perm = torch.argsort(perm)

        # Permute -> block-quantize -> unpermute
        v_perm = v_h[:, perm]
        n_blocks = head_dim // block_size
        blocks = v_perm.reshape(T, n_blocks, block_size)
        blocks_qd = quantize_uniform_blocks(blocks, bits)
        v_hat_perm = blocks_qd.reshape(T, head_dim)
        v_hat = v_hat_perm[:, inv_perm]

        # Quantization error in original coordinate system
        e = v_hat - v_h

        # === Q1: First-order predictor [Theorem A, Eq. (3)] ===
        # Q1 = sum_j (s^l_j)^T e^l_j  for this head
        # g_h[j, :] = s^{l,h}_j, e[j, :] = e^{l,h}_j
        # Element-wise multiply and sum = sum of inner products over positions
        linear_pred_val = (g_h * e).sum().item()

        # === Uncentered second moment: Sigma_u = (1/T) e^T e ===
        Sigma_u = (e.T @ e) / T

        # === Centered covariance: Sigma_c = Sigma_u - mu mu^T [Remark 5] ===
        e_mean = e.mean(dim=0, keepdim=True)    # mu = E[e|D], position-averaged
        Sigma_c = Sigma_u - (e_mean.T @ e_mean)

        # === tr(M Sigma) [Theorem B proxy] ===
        tr_M_Sigma_u = torch.trace(M @ Sigma_u).item()
        tr_M_Sigma_c = torch.trace(M @ Sigma_c).item()

        # === tr(Sigma) = MSE [naive baseline] ===
        tr_Sigma_u = Sigma_u.trace().item()
        tr_Sigma_c = Sigma_c.trace().item()

        # === Bias norm: ||mu|| quantifies D2(a) violation [Remark 5] ===
        mean_bias_norm = e_mean.norm().item()

        # === HLP bounds [Theorem C(ii)] ===
        # a_i = M eigenvalues (descending), b_i = Sigma eigenvalues (descending)
        # Upper: sum a_i b_i (co-aligned), Lower: sum a_i b_{d+1-i} (counter-aligned)
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
    serves as a direct measurement target for Q1 validation.

    Theory correspondence:
      Returns delta_loss = L(v_hat) - L(v) = Delta L
      This is the EXACT quantity that Q1 approximates:
        Q1 = sum_{l,j} s_j^T e_j  ≈  Delta L  [Theorem A]
      The gap (Delta L - Q1) = Q2 + R3 (higher-order terms).

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

    try:
        outputs = model(input_ids, labels=input_ids, use_cache=False)
        quant_loss = outputs.loss.item()
    finally:
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
# Statistical utilities (publication-quality)
# ============================================================

def spearman_corr(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Spearman rank correlation with p-value.

    Uses scipy if available (handles ties correctly via midrank).
    Falls back to manual O(n) implementation otherwise.

    Returns:
        (rho, p_value).  p_value is NaN if scipy unavailable.
    """
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    if HAS_SCIPY:
        res = _scipy_spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)
    else:
        # Manual Spearman (no tie handling -- use scipy for paper)
        xa, ya = np.array(x), np.array(y)
        rx = np.argsort(np.argsort(xa)).astype(float)
        ry = np.argsort(np.argsort(ya)).astype(float)
        d = rx - ry
        rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
        return float(rho), float("nan")


def pearson_corr(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Pearson correlation with p-value.

    Returns:
        (r, p_value).  p_value is NaN if scipy unavailable.
    """
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    if HAS_SCIPY:
        res = _scipy_pearsonr(x, y)
        return float(res.statistic), float(res.pvalue)
    else:
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan")


def bootstrap_ci(
    x: List[float],
    y: List[float],
    corr_fn,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a correlation statistic.

    Resamples (x_i, y_i) pairs with replacement, computes the
    correlation on each resample, returns percentile CI.

    Args:
        x, y: paired observations.
        corr_fn: function(x, y) -> (statistic, p_value).
        n_boot: number of bootstrap resamples.
        alpha: significance level (0.05 for 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    x_arr, y_arr = np.array(x), np.array(y)
    n = len(x_arr)

    point, _ = corr_fn(x, y)

    rng = np.random.RandomState(seed)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        stat, _ = corr_fn(x_arr[idx].tolist(), y_arr[idx].tolist())
        boot_stats[i] = stat

    # Filter NaN (can happen if resampled data is degenerate)
    boot_valid = boot_stats[~np.isnan(boot_stats)]
    if len(boot_valid) < n_boot * 0.9:
        print(f"  WARNING: {n_boot - len(boot_valid)}/{n_boot} bootstrap "
              f"resamples produced NaN")

    ci_lo = float(np.percentile(boot_valid, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_valid, 100 * (1 - alpha / 2)))

    return point, ci_lo, ci_hi


# ============================================================
# End-to-end evidence: MSE-best vs DDT-best
# ============================================================

def compute_end_to_end_evidence(config_list: List[Dict], bits: int) -> Dict:
    """For a given bitwidth, compare the config chosen by each metric.

    For each of the three metrics (Q1/linear_pred, tr(MΣ), MSE/tr(Σ)):
      - Find the config that the metric ranks as "best" (lowest value)
      - Report that config's actual delta_loss

    If DDT-best has lower actual delta_loss than MSE-best, DDT is a
    better design tool.

    Args:
        config_list: list of dicts with keys
            {mode, bits, linear_pred, tr_M_Sigma, tr_Sigma, delta_loss}
        bits: filter to this bitwidth

    Returns:
        Dict with per-metric best config and actual delta_loss.
    """
    items = [c for c in config_list if c["bits"] == bits and c["delta_loss"] is not None]
    if len(items) < 3:
        return {"error": f"too few configs with delta_loss at {bits}b"}

    result = {}
    for metric_name, metric_key in [
        ("Q1_linear_pred", "linear_pred"),
        ("tr_M_Sigma", "tr_M_Sigma"),
        ("MSE_tr_Sigma", "tr_Sigma"),
    ]:
        # Best = lowest metric value (least predicted degradation)
        best = min(items, key=lambda c: c[metric_key])
        worst = max(items, key=lambda c: c[metric_key])

        result[metric_name] = {
            "best_mode": best["mode"],
            "best_metric_val": best[metric_key],
            "best_actual_delta_loss": best["delta_loss"],
            "worst_mode": worst["mode"],
            "worst_metric_val": worst[metric_key],
            "worst_actual_delta_loss": worst["delta_loss"],
        }

    # Oracle: config with actual lowest delta_loss
    oracle_best = min(items, key=lambda c: c["delta_loss"])
    result["oracle"] = {
        "best_mode": oracle_best["mode"],
        "best_actual_delta_loss": oracle_best["delta_loss"],
    }

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT CABA Anomaly Directional Explanation (v2)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--caba", type=str, default=None,
                        help="Path to caba_analysis JSON (sorted permutation)")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Bitwidths to sweep (default: 2 3 4 5)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-chunks", type=int, default=4,
                        help="Calibration chunks for M averaging (default: 4)")
    parser.add_argument("--n-random-seeds", type=int, default=48,
                        help="Random permutation seeds (default: 48). "
                             "Total configs = (n_random_seeds + 1 or 2) x len(bits)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Bootstrap resamples for CI (default: 10000)")
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
        args.output = str(out_dir / f"caba_explain_v2_{model_tag}.json")

    n_special = 1  # baseline
    if args.caba is not None:
        n_special += 1  # + sorted
    n_total_per_bit = n_special + args.n_random_seeds
    n_total = n_total_per_bit * len(args.bits)

    print(f"{'='*60}")
    print(f"DDT CABA Explain v2 — P0 Experiment")
    print(f"{'='*60}")
    print(f"Model:           {args.model}")
    print(f"Bits:            {args.bits}")
    print(f"Seq len:         {args.seq_len}")
    print(f"M chunks:        {args.n_chunks}")
    print(f"Random seeds:    {args.n_random_seeds}")
    print(f"Configs/bit:     {n_total_per_bit}")
    print(f"Total configs:   {n_total}")
    print(f"Bootstrap:       {args.n_bootstrap}")
    print(f"Output:          {args.output}")
    print(f"Skip delta-loss: {args.skip_delta_loss}")

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

    # ---- Clean loss (uses LAST chunk = same data as Sigma and Q1) ----
    last_chunk_start = (len(chunk_losses) - 1) * args.seq_len
    clean_ids = all_input_ids[:, last_chunk_start:last_chunk_start + args.seq_len].to(device)

    with torch.no_grad():
        clean_loss = model(clean_ids, labels=clean_ids, use_cache=False).loss.item()
    print(f"  Clean loss (last chunk): {clean_loss:.4f}")

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
    print(f"  Total permutation modes: {len(perm_sets)}")
    print(f"  Modes: baseline" +
          (", sorted" if args.caba else "") +
          f", random_s42..random_s{42 + args.n_random_seeds - 1}")

    # ---- Phase 3: Directional metrics + delta-loss ----
    print(f"\n{'='*60}")
    print(f"Phase 3: Computing directional metrics + delta-loss ({n_total} configs)")
    print(f"{'='*60}")

    # Flat config list for figures and statistical analysis
    config_list = []
    # Structured results (backward compat)
    all_results = {}

    total_elapsed = 0.0
    config_count = 0

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
            total_elapsed += elapsed
            config_count += 1

            key = f"{mode}_{bits}b"

            # Flat config entry (for figures)
            config_entry = {
                "mode": mode,
                "bits": bits,
                "key": key,
                "linear_pred": metrics["agg_total"]["linear_pred"],
                "tr_M_Sigma": metrics["agg_total"]["tr_M_Sigma"],
                "tr_Sigma": metrics["agg_total"]["tr_Sigma"],
                "tr_M_Sigma_c": metrics["agg_total"]["tr_M_Sigma_c"],
                "tr_Sigma_c": metrics["agg_total"]["tr_Sigma_c"],
                "linear_pred_K": metrics["agg_K"]["linear_pred"],
                "linear_pred_V": metrics["agg_V"]["linear_pred"],
                "tr_M_Sigma_K": metrics["agg_K"]["tr_M_Sigma"],
                "tr_M_Sigma_V": metrics["agg_V"]["tr_M_Sigma"],
                "tr_Sigma_K": metrics["agg_K"]["tr_Sigma"],
                "tr_Sigma_V": metrics["agg_V"]["tr_Sigma"],
                "delta_loss": delta_loss,
            }
            config_list.append(config_entry)

            # Structured result (backward compat)
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
            if config_count <= 10 or mode in ("baseline", "sorted") or config_count % 10 == 0:
                print(
                    f"  [{config_count:3d}/{n_total}] {mode:16s} {bits}b: "
                    f"Q1={lp:+.4f}  "
                    f"tr(MS)={metrics['agg_total']['tr_M_Sigma']:.4e}  "
                    f"MSE={metrics['agg_total']['tr_Sigma']:.4e}  "
                    f"{dl_str}  ({elapsed:.1f}s)"
                )

    print(f"\n  Total: {config_count} configs in {total_elapsed:.0f}s "
          f"({total_elapsed/config_count:.1f}s/config)")

    # ---- Phase 4: Statistical Analysis ----
    print(f"\n{'='*60}")
    print("Phase 4: Statistical Analysis (per-bitwidth)")
    print(f"{'='*60}")

    correlation_results = {}

    for bits in args.bits:
        items = [c for c in config_list if c["bits"] == bits]
        has_dl = all(c["delta_loss"] is not None for c in items)
        n_cfg = len(items)

        print(f"\n--- {bits}b ({n_cfg} configs) ---")

        if not has_dl or n_cfg < 5:
            print("  Skipping: insufficient data or no delta_loss")
            continue

        delta_losses = [c["delta_loss"] for c in items]
        bit_corr = {}

        for metric_name, metric_key in [
            ("Q1_linear_pred", "linear_pred"),
            ("tr_M_Sigma", "tr_M_Sigma"),
            ("MSE_tr_Sigma", "tr_Sigma"),
        ]:
            vals = [c[metric_key] for c in items]

            # Spearman with bootstrap CI
            rho_point, rho_lo, rho_hi = bootstrap_ci(
                vals, delta_losses, spearman_corr,
                n_boot=args.n_bootstrap, seed=bits * 1000,
            )
            _, rho_p = spearman_corr(vals, delta_losses)

            # Pearson with bootstrap CI
            r_point, r_lo, r_hi = bootstrap_ci(
                vals, delta_losses, pearson_corr,
                n_boot=args.n_bootstrap, seed=bits * 1000 + 1,
            )
            _, r_p = pearson_corr(vals, delta_losses)

            bit_corr[metric_name] = {
                "spearman_rho": rho_point,
                "spearman_ci_95": [rho_lo, rho_hi],
                "spearman_p": rho_p,
                "pearson_r": r_point,
                "pearson_ci_95": [r_lo, r_hi],
                "pearson_p": r_p,
            }

            p_str = f"p={rho_p:.2e}" if not math.isnan(rho_p) else "p=N/A"
            print(f"  {metric_name:20s}: "
                  f"ρ={rho_point:+.3f} [{rho_lo:+.3f}, {rho_hi:+.3f}]  {p_str}  |  "
                  f"r={r_point:+.3f} [{r_lo:+.3f}, {r_hi:+.3f}]")

        # Identify best predictor
        best_metric = max(bit_corr.items(), key=lambda kv: abs(kv[1]["spearman_rho"]))
        print(f"  >>> Best predictor (|ρ|): {best_metric[0]}")

        correlation_results[f"{bits}b"] = {
            "n_configs": n_cfg,
            "correlations": bit_corr,
            "best_predictor": best_metric[0],
        }

    # ---- All-bits pooled analysis (supplementary: cross-bit scale dominates) ----
    # NOTE: Pooled correlation is inflated by bitwidth differences (2b MSE >> 5b MSE).
    # Main claims use per-bitwidth correlations above. Pooled is for reference only.
    all_items_with_dl = [c for c in config_list if c["delta_loss"] is not None]
    if len(all_items_with_dl) >= 10:
        print(f"\n--- ALL BITS POOLED ({len(all_items_with_dl)} configs) [supplementary] ---")

        delta_losses_all = [c["delta_loss"] for c in all_items_with_dl]
        pooled_corr = {}

        for metric_name, metric_key in [
            ("Q1_linear_pred", "linear_pred"),
            ("tr_M_Sigma", "tr_M_Sigma"),
            ("MSE_tr_Sigma", "tr_Sigma"),
        ]:
            vals = [c[metric_key] for c in all_items_with_dl]

            rho_point, rho_lo, rho_hi = bootstrap_ci(
                vals, delta_losses_all, spearman_corr,
                n_boot=args.n_bootstrap, seed=99999,
            )
            _, rho_p = spearman_corr(vals, delta_losses_all)

            r_point, r_lo, r_hi = bootstrap_ci(
                vals, delta_losses_all, pearson_corr,
                n_boot=args.n_bootstrap, seed=99998,
            )

            pooled_corr[metric_name] = {
                "spearman_rho": rho_point,
                "spearman_ci_95": [rho_lo, rho_hi],
                "spearman_p": rho_p,
                "pearson_r": r_point,
                "pearson_ci_95": [r_lo, r_hi],
            }

            p_str = f"p={rho_p:.2e}" if not math.isnan(rho_p) else "p=N/A"
            print(f"  {metric_name:20s}: "
                  f"ρ={rho_point:+.3f} [{rho_lo:+.3f}, {rho_hi:+.3f}]  {p_str}  |  "
                  f"r={r_point:+.3f} [{r_lo:+.3f}, {r_hi:+.3f}]")

        correlation_results["all_bits_pooled"] = {
            "n_configs": len(all_items_with_dl),
            "correlations": pooled_corr,
        }

    # ---- Phase 5: End-to-end evidence ----
    print(f"\n{'='*60}")
    print("Phase 5: End-to-End Evidence (MSE-best vs DDT-best)")
    print(f"{'='*60}")

    e2e_results = {}

    for bits in args.bits:
        e2e = compute_end_to_end_evidence(config_list, bits)
        if "error" in e2e:
            print(f"  {bits}b: {e2e['error']}")
            continue

        e2e_results[f"{bits}b"] = e2e

        print(f"\n  --- {bits}b ---")
        print(f"  {'Metric':<20s} {'Best config':<20s} {'Δloss':>10s}  "
              f"{'Worst config':<20s} {'Δloss':>10s}")
        print(f"  {'-'*84}")

        for metric_name in ["Q1_linear_pred", "tr_M_Sigma", "MSE_tr_Sigma"]:
            d = e2e[metric_name]
            print(f"  {metric_name:<20s} {d['best_mode']:<20s} {d['best_actual_delta_loss']:>10.4f}  "
                  f"{d['worst_mode']:<20s} {d['worst_actual_delta_loss']:>10.4f}")

        oracle = e2e["oracle"]
        print(f"  {'ORACLE':<20s} {oracle['best_mode']:<20s} {oracle['best_actual_delta_loss']:>10.4f}")

        # Compare: did DDT-best (tr(MΣ)) beat MSE-best?
        # Role separation: Q1 = prediction/validation, tr(MΣ) = selection/design
        ddt_dl = e2e["tr_M_Sigma"]["best_actual_delta_loss"]
        mse_dl = e2e["MSE_tr_Sigma"]["best_actual_delta_loss"]
        if ddt_dl < mse_dl:
            improvement = (mse_dl - ddt_dl) / mse_dl * 100
            print(f"  >>> DDT-best Δloss {ddt_dl:.4f} < MSE-best {mse_dl:.4f}  "
                  f"({improvement:.1f}% better)")
        elif ddt_dl > mse_dl:
            print(f"  >>> MSE-best Δloss {mse_dl:.4f} < DDT-best {ddt_dl:.4f}")
        else:
            print(f"  >>> Tied: same config selected by both metrics")

    # ---- Phase 6: K vs V breakdown ----
    print(f"\n{'='*60}")
    print("Phase 6: K vs V Sensitivity Breakdown")
    print(f"{'='*60}")

    kv_breakdown = {}
    for bits in args.bits:
        items = [c for c in config_list if c["bits"] == bits]
        k_fracs = []
        for c in items:
            total = c["tr_M_Sigma_K"] + c["tr_M_Sigma_V"]
            k_frac = c["tr_M_Sigma_K"] / total if total > 1e-20 else 0.5
            k_fracs.append(k_frac)
        mean_k = np.mean(k_fracs)
        std_k = np.std(k_fracs)
        kv_breakdown[f"{bits}b"] = {
            "K_frac_mean": float(mean_k),
            "K_frac_std": float(std_k),
        }
        print(f"  {bits}b: K-path fraction = {mean_k:.1%} ± {std_k:.1%}")

    # ---- Phase 7: Centered vs uncentered (bias drift diagnostic) ----
    print(f"\n{'='*60}")
    print("Phase 7: Bias Drift Diagnostic (Centered vs Uncentered)")
    print(f"{'='*60}")

    bias_drift_results = {}
    for bits in args.bits:
        items = [c for c in config_list if c["bits"] == bits]
        gaps = []
        for c in items:
            u = c["tr_M_Sigma"]
            cent = c["tr_M_Sigma_c"]
            gap_pct = abs(u - cent) / abs(u) * 100 if abs(u) > 1e-20 else 0
            gaps.append(gap_pct)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        bias_drift_results[f"{bits}b"] = {
            "gap_pct_mean": float(mean_gap),
            "gap_pct_std": float(std_gap),
        }
        print(f"  {bits}b: centered/uncentered gap = {mean_gap:.1f}% ± {std_gap:.1f}%")

    # ---- Phase 8: Mean-zero condition (z-scores) ----
    print(f"\n{'='*60}")
    print("Phase 8: Mean-Zero Condition Check")
    print(f"{'='*60}")
    # Under D2(a), E[Q1|D] = 0.  For deterministic quantizer, this is approximate.
    # z-score = linear_pred / (std of linear_pred across random seeds)
    for bits in args.bits:
        random_items = [c for c in config_list
                        if c["bits"] == bits and c["mode"].startswith("random")]
        if len(random_items) < 5:
            continue
        lp_values = [c["linear_pred"] for c in random_items]
        lp_mean = np.mean(lp_values)
        lp_std = np.std(lp_values)
        z = lp_mean / (lp_std / np.sqrt(len(lp_values))) if lp_std > 1e-20 else 0
        print(f"  {bits}b: Q1 mean = {lp_mean:.4f}, std = {lp_std:.4f}, "
              f"z = {z:.2f} (|z|<2 → consistent with mean-zero)")

    # ---- Phase 9: PPL reference ----
    print(f"\n{'='*60}")
    print("Phase 9: PPL Reference (from existing caba_eval results)")
    print(f"{'='*60}")
    ppl_data = load_ppl_results(args.results_dir, model_tag)
    if ppl_data:
        for k, v in sorted(ppl_data.items()):
            print(f"  {k}: PPL={v['ppl']:.1f}")
    else:
        print("  No PPL reference files found.")

    # ---- Save ----
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "version": "v2_P0",
        "config": {
            "seq_len": args.seq_len,
            "n_chunks": args.n_chunks,
            "n_random_seeds": args.n_random_seeds,
            "n_bootstrap": args.n_bootstrap,
            "bits": args.bits,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "n_total_configs": n_total,
        },
        "clean_loss": clean_loss,
        "chunk_losses": chunk_losses,

        # === Flat config list (primary data for figures) ===
        # Each entry: {mode, bits, key, linear_pred, tr_M_Sigma, tr_Sigma,
        #              tr_M_Sigma_c, tr_Sigma_c, delta_loss, K/V splits}
        "config_list": config_list,

        # === Statistical analysis ===
        "correlations": correlation_results,
        "end_to_end_evidence": e2e_results,
        "kv_breakdown": kv_breakdown,
        "bias_drift": bias_drift_results,

        # === Structured results (backward compat, includes per-layer/top-heads) ===
        "metrics": {
            key: {
                "mode": res["mode"],
                "bits": res["bits"],
                "agg_K": res["agg_K"],
                "agg_V": res["agg_V"],
                "agg_total": res["agg_total"],
                "delta_loss": res["delta_loss"],
                "per_layer": res["per_layer"],
                "top_heads": res["top_heads"],
            }
            for key, res in all_results.items()
        },

        "ppl_reference": ppl_data,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")
    print(f"\nTotal runtime: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()