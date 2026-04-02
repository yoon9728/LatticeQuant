"""
DDT — Autoregressive KV Cache Reuse Experiment
=================================================
Tests whether prefill-based DDT findings transfer to a chunked
continuation setting with reused KV cache.

Uses the same DynamicCache subclass pattern as caba_eval.py:
quantization happens inside update(), avoiding all cache internal access.

Flow:
  1. Forward chunk 1 with DynamicCache() → clean cache
  2. Forward chunk 2 with clean cache → clean_loss
  3. For each permutation config:
     a. Forward chunk 1 with QuantCache(perms, bits) → quantized cache
     b. Forward chunk 2 with quantized cache → quant_loss
     c. ΔL_auto = quant_loss - clean_loss

Chunk alignment with P0:
  chunk 1 (index 2) = causal context for cache generation
  chunk 2 (index 3) = evaluation text, same as P0 metric computation

Requires: transformers >= 4.36

Usage:
  python -m ddt.autoregressive_experiment \\
      --model meta-llama/Llama-3.1-8B \\
      --det-json results/ddt/caba_explain_v2_Llama-3.1-8B.json \\
      --bits 3 4 --n-configs 25 --strict
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

try:
    from scipy.stats import spearmanr as scipy_spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# QuantCache — DynamicCache with permuted block quantization
# ============================================================

class QuantCache(DynamicCache):
    """DynamicCache that quantizes K/V on update().

    Same pattern as CABACache in caba_eval.py.
    Quantization: permute dims → per-block uniform QDQ → unpermute.
    """

    def __init__(self, perms: Dict, bits: int = 4, block_size: int = 8):
        super().__init__()
        self.perms = perms
        self.bits = bits
        self.block_size = block_size
        self.n_levels = 2 ** bits
        self._shape_checked = False

    def _quantize_head(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """Quantize a single head: permute → block QDQ → unpermute.

        x: (batch, seq_len, head_dim)
        perm: (head_dim,)
        """
        inv_perm = torch.argsort(perm)
        x_f = x.float()
        x_perm = x_f[:, :, perm]

        B, T, hd = x_perm.shape
        n_blocks = hd // self.block_size
        blocks = x_perm.reshape(B, T, n_blocks, self.block_size)

        # Per-block RMS-shared symmetric uniform QDQ
        half = self.n_levels / 2
        rms = torch.sqrt((blocks ** 2).mean(dim=-1, keepdim=True) + 1e-12)
        scale = 3.0 * rms / half  # alpha=3
        scaled = blocks / scale
        quant = torch.round(scaled.clamp(-half, half - 1))
        blocks_qd = quant * scale

        x_hat_perm = blocks_qd.reshape(B, T, hd)
        x_hat = x_hat_perm[:, :, inv_perm]
        return x_hat.to(x.dtype)

    def _quantize_tensor(self, tensor: torch.Tensor, layer_idx: int,
                          component: str) -> torch.Tensor:
        """Quantize all heads in a K or V tensor.

        tensor: (batch, num_heads, seq_len, head_dim)
        """
        assert tensor.ndim == 4, f"Expected 4D tensor, got {tensor.ndim}D"
        if not self._shape_checked:
            assert tensor.shape[-1] % self.block_size == 0, (
                f"head_dim={tensor.shape[-1]} not divisible by block_size={self.block_size}")
            self._shape_checked = True
        B, H, T, D = tensor.shape
        out = torch.empty_like(tensor)
        device = tensor.device

        for h in range(H):
            perm = self.perms[layer_idx][component][h].to(device)
            head_data = tensor[:, h, :, :]  # (B, T, D)
            out[:, h, :, :] = self._quantize_head(head_data, perm)

        return out

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override: quantize K/V before storing in cache."""
        k_quant = self._quantize_tensor(key_states, layer_idx, "K")
        v_quant = self._quantize_tensor(value_states, layer_idx, "V")
        return super().update(k_quant, v_quant, layer_idx, cache_kwargs)


# ============================================================
# Helpers
# ============================================================

def spearman_corr(x, y):
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(xa) < 3:
        return float("nan"), float("nan")
    if HAS_SCIPY:
        rho, p = scipy_spearmanr(xa, ya)
        return float(rho), float(p)
    n = len(xa)
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)), float("nan")


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def make_identity_permutations(num_layers, num_kv_heads, head_dim):
    perms = {}
    for l in range(num_layers):
        perms[l] = {}
        for comp in ["K", "V"]:
            perms[l][comp] = {}
            for h in range(num_kv_heads):
                perms[l][comp][h] = torch.arange(head_dim)
    return perms


def make_random_permutations(num_layers, num_kv_heads, head_dim, seed=42):
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
# Sanity check
# ============================================================

@torch.no_grad()
def sanity_check_cache_reuse(model, chunk1_ids, chunk2_ids):
    """Verify cache-reuse loss matches full-concatenated loss.

    Path A: chunk1 → cache → chunk2 with cache → loss
    Path B: [chunk1+chunk2] one pass → chunk2 loss
    """
    device = chunk1_ids.device
    c1_len = chunk1_ids.shape[1]
    c2_len = chunk2_ids.shape[1]

    # Path A: cache reuse
    cache_a = DynamicCache()
    model(chunk1_ids, past_key_values=cache_a, use_cache=True)

    pos_ids = torch.arange(c1_len, c1_len + c2_len, device=device).unsqueeze(0)
    attn_mask = torch.ones(1, c1_len + c2_len, device=device, dtype=torch.long)

    out_a = model(
        chunk2_ids,
        past_key_values=cache_a,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        labels=chunk2_ids,
        use_cache=False,
    )
    loss_a = out_a.loss.item()

    # Path B: full concatenated
    full_ids = torch.cat([chunk1_ids, chunk2_ids], dim=1)
    out_b = model(full_ids, use_cache=False)
    logits_b = out_b.logits

    # Manual chunk2 CE loss (shifted by 1)
    c2_logits = logits_b[:, c1_len - 1:c1_len + c2_len - 1, :]
    c2_labels = full_ids[:, c1_len:c1_len + c2_len]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_b = loss_fn(
        c2_logits.reshape(-1, c2_logits.shape[-1]),
        c2_labels.reshape(-1),
    ).item()

    gap = abs(loss_a - loss_b)
    ok = gap < 0.01
    return loss_a, loss_b, gap, ok


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDT Autoregressive KV Cache Reuse Experiment"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--det-json", type=str, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--n-configs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Abort if sanity check fails")
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]
    if args.output is None:
        out_dir = Path("results/ddt")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"autoregressive_experiment_{model_tag}.json")

    # ---- Load P0 results ----
    print("Loading P0 results...")
    with open(args.det_json) as f:
        det_data = json.load(f)
    det_configs = det_data["config_list"]
    print(f"  {len(det_configs)} configs loaded")

    # ---- Load model ----
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}
    if not args.no_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, **kwargs
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    block_size = 8
    assert head_dim % block_size == 0
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

    # ---- Calibration data ----
    device = get_model_device(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    all_input_ids = tokenizer(text, return_tensors="pt").input_ids

    chunk1_start = 2 * args.seq_len
    chunk2_start = 3 * args.seq_len
    chunk1_ids = all_input_ids[:, chunk1_start:chunk1_start + args.seq_len].to(device)
    chunk2_ids = all_input_ids[:, chunk2_start:chunk2_start + args.seq_len].to(device)

    print(f"  Chunk 1 (context): tokens [{chunk1_start}, {chunk1_start + args.seq_len})")
    print(f"  Chunk 2 (eval, = P0 chunk): tokens [{chunk2_start}, {chunk2_start + args.seq_len})")

    # ---- Sanity check ----
    print("\n  Sanity check: cache-reuse loss vs full-concatenated loss...")
    with torch.no_grad():
        loss_reuse, loss_concat, gap, ok = sanity_check_cache_reuse(
            model, chunk1_ids, chunk2_ids,
        )
    status = "PASS" if ok else "FAIL"
    print(f"  Cache-reuse loss:  {loss_reuse:.6f}")
    print(f"  Full-concat loss:  {loss_concat:.6f}")
    print(f"  Gap: {gap:.6f}  [{status}]")
    if not ok:
        print("  WARNING: cache-reuse path may have position/mask issues.")
        if args.strict:
            raise RuntimeError("Sanity check FAILED with --strict.")
        print("  Proceeding with caution.")

    # ---- Clean baseline: chunk1 cache → chunk2 loss ----
    print("\n  Measuring clean chunk 2 loss...")
    c1_len = chunk1_ids.shape[1]
    c2_len = chunk2_ids.shape[1]
    pos_ids_c2 = torch.arange(c1_len, c1_len + c2_len, device=device).unsqueeze(0)
    attn_mask_c2 = torch.ones(1, c1_len + c2_len, device=device, dtype=torch.long)

    with torch.no_grad():
        clean_cache = DynamicCache()
        model(chunk1_ids, past_key_values=clean_cache, use_cache=True)

        out_clean = model(
            chunk2_ids,
            past_key_values=clean_cache,
            attention_mask=attn_mask_c2,
            position_ids=pos_ids_c2,
            labels=chunk2_ids,
            use_cache=False,
        )
        clean_loss_c2 = out_clean.loss.item()
    print(f"  Clean chunk 2 loss: {clean_loss_c2:.4f}")

    # ---- Build permutation configs (must match P0 exactly) ----
    perm_configs = {}
    perm_configs["baseline"] = make_identity_permutations(
        num_layers, num_kv_heads, head_dim
    )
    for i in range(args.n_configs - 1):
        seed = 42 + i
        perm_configs[f"random_s{seed}"] = make_random_permutations(
            num_layers, num_kv_heads, head_dim, seed=seed
        )
    print(f"  {len(perm_configs)} permutation configs")

    # ---- Run experiments ----
    results = {}

    for bits in args.bits:
        print(f"\n{'='*60}")
        print(f"Bitwidth: {bits}")
        print(f"{'='*60}")

        bit_results = []

        p0_by_mode = {}
        for c in det_configs:
            if c["bits"] == bits:
                p0_by_mode[c["mode"]] = c

        for cfg_idx, (mode, perms) in enumerate(perm_configs.items()):
            t0 = time.time()

            with torch.no_grad():
                # Forward chunk 1 with quantizing cache
                quant_cache = QuantCache(perms, bits=bits, block_size=block_size)
                model(chunk1_ids, past_key_values=quant_cache, use_cache=True)

                # Forward chunk 2 with quantized cache
                out_quant = model(
                    chunk2_ids,
                    past_key_values=quant_cache,
                    attention_mask=attn_mask_c2,
                    position_ids=pos_ids_c2,
                    labels=chunk2_ids,
                    use_cache=False,
                )
                quant_loss = out_quant.loss.item()

            dl_auto = quant_loss - clean_loss_c2

            p0 = p0_by_mode.get(mode, {})
            dl_prefill = p0.get("delta_loss")
            tr_m_sigma = p0.get("tr_M_Sigma")
            q1 = p0.get("linear_pred")
            tr_sigma = p0.get("tr_Sigma")

            elapsed = time.time() - t0

            entry = {
                "mode": mode,
                "bits": bits,
                "dl_auto": float(dl_auto),
                "dl_prefill": dl_prefill,
                "tr_M_Sigma": tr_m_sigma,
                "Q1": q1,
                "tr_Sigma": tr_sigma,
            }
            bit_results.append(entry)

            pf_str = f"prefill={dl_prefill:+.4f}" if dl_prefill is not None else "prefill=N/A"
            print(f"  [{cfg_idx+1:2d}/{len(perm_configs)}] {mode:16s}: "
                  f"auto={dl_auto:+.4f}  {pf_str}  ({elapsed:.1f}s)")

        # ---- Ranking correlations ----
        valid = [r for r in bit_results
                 if r["dl_prefill"] is not None
                 and r["tr_M_Sigma"] is not None]

        if len(valid) >= 5:
            dl_autos = [r["dl_auto"] for r in valid]
            dl_prefills = [r["dl_prefill"] for r in valid]
            trms = [r["tr_M_Sigma"] for r in valid]
            q1s = [r["Q1"] for r in valid]
            mses = [r["tr_Sigma"] for r in valid]

            rho_pf, p_pf = spearman_corr(dl_prefills, dl_autos)
            rho_trms, p_trms = spearman_corr(trms, dl_autos)
            rho_q1, p_q1 = spearman_corr(q1s, dl_autos)
            rho_mse, p_mse = spearman_corr(mses, dl_autos)

            p_str = lambda p: f"p={p:.2e}" if p is not None and not math.isnan(p) else ""

            print(f"\n  --- Ranking Correlations ({bits}b, n={len(valid)}) ---")
            print(f"  ρ(prefill, auto)   = {rho_pf:+.3f}  {p_str(p_pf)}  [primary]")
            print(f"  ρ(tr(MΣ), auto)    = {rho_trms:+.3f}  {p_str(p_trms)}  [primary]")
            print(f"  ρ(MSE, auto)       = {rho_mse:+.3f}  {p_str(p_mse)}  [primary]")
            print(f"  ρ(Q1, auto)        = {rho_q1:+.3f}  {p_str(p_q1)}  [supplementary]")

            ranking = {
                "n_configs": len(valid),
                "rho_prefill_auto": float(rho_pf),
                "rho_trms_auto": float(rho_trms),
                "rho_q1_auto": float(rho_q1),
                "rho_mse_auto": float(rho_mse),
                "p_prefill_auto": float(p_pf) if not math.isnan(p_pf) else None,
                "p_trms_auto": float(p_trms) if not math.isnan(p_trms) else None,
                "p_q1_auto": float(p_q1) if not math.isnan(p_q1) else None,
                "p_mse_auto": float(p_mse) if not math.isnan(p_mse) else None,
            }
        else:
            print(f"\n  Ranking: insufficient valid configs ({len(valid)})")
            ranking = {"n_configs": len(valid), "error": "insufficient"}

        dl_all = [r["dl_auto"] for r in bit_results]
        print(f"\n  --- Summary ({bits}b) ---")
        print(f"  Auto ΔL: mean={np.mean(dl_all):.4f}, "
              f"std={np.std(dl_all):.4f}, "
              f"range=[{min(dl_all):.4f}, {max(dl_all):.4f}]")

        results[f"{bits}b"] = {
            "configs": bit_results,
            "ranking": ranking,
            "summary": {
                "mean_dl_auto": float(np.mean(dl_all)),
                "std_dl_auto": float(np.std(dl_all)),
                "min_dl_auto": float(min(dl_all)),
                "max_dl_auto": float(max(dl_all)),
            },
        }

    # ---- Save ----
    import transformers
    output_data = {
        "model": args.model,
        "model_tag": model_tag,
        "version": "autoregressive_experiment_v3",
        "transformers_version": transformers.__version__,
        "config": {
            "bits": args.bits,
            "n_configs": len(perm_configs),
            "seq_len": args.seq_len,
            "block_size": block_size,
            "chunk1_idx": 2,
            "chunk2_idx": 3,
        },
        "clean_loss_chunk2": clean_loss_c2,
        "sanity_check": {
            "cache_reuse_loss": loss_reuse,
            "full_concat_loss": loss_concat,
            "gap": gap,
            "passed": ok,
        },
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()