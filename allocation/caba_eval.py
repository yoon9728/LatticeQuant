"""
CABA Phase 2 — Permuted Block Assignment PPL Evaluation

CompressedKVCache 패턴을 따라 DynamicCache 상속 + update() 오버라이드.
Block assignment permutation을 적용한 후 per-block uniform quantization.

Three modes:
  baseline  — identity permutation (기존 consecutive blocks)
  sorted    — variance-sorted permutation from caba_analysis.py
  random    — random permutation (ablation control)

Quantizer: per-block uniform (E₈ proxy).
  AM/GM 효과 검증에 충분. E₈ 연결은 결과 확인 후.

Usage:
    python -m allocation.caba_eval \\
        --model Qwen/Qwen2.5-7B --bits 4 --mode baseline

    python -m allocation.caba_eval \\
        --model Qwen/Qwen2.5-7B --bits 4 --mode sorted \\
        --caba results/caba_qwen2.5_7b.json

    python -m allocation.caba_eval \\
        --model Qwen/Qwen2.5-7B --bits 4 --mode random
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

# E₈ quantizer import
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
try:
    from e8_quantizer import encode_e8, compute_scale
    E8_AVAILABLE = True
except ImportError:
    E8_AVAILABLE = False


# ============================================================
# Permutation loading
# ============================================================

def load_sorted_permutations(caba_path: str) -> Dict:
    """caba_analysis.py 결과에서 per-layer/head permutation 로드."""
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


def make_random_permutations(
    num_layers: int, num_kv_heads: int, head_dim: int, seed: int = 42
) -> Dict:
    """Random permutation (ablation control). Seeded for reproducibility."""
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


def make_identity_permutations(
    num_layers: int, num_kv_heads: int, head_dim: int
) -> Dict:
    """Identity permutation (baseline)."""
    perms = {}
    for l in range(num_layers):
        perms[l] = {}
        for comp in ["K", "V"]:
            perms[l][comp] = {}
            for h in range(num_kv_heads):
                perms[l][comp][h] = torch.arange(head_dim)
    return perms


def precompute_inverse_permutations(perms: Dict) -> Dict:
    """Inverse permutation 미리 계산."""
    inv = {}
    for l, ldata in perms.items():
        inv[l] = {}
        for comp, cdata in ldata.items():
            inv[l][comp] = {}
            for h, perm in cdata.items():
                inv_p = torch.empty_like(perm)
                inv_p[perm] = torch.arange(len(perm))
                inv[l][comp][h] = inv_p
    return inv


def validate_permutations(
    perms: Dict, num_layers: int, num_kv_heads: int, head_dim: int
):
    """Permutation이 모델 구조와 일치하는지 검증.

    잘못된 JSON을 로드하거나 모델을 바꿔치기한 경우를 방어.
    """
    for l in range(num_layers):
        assert l in perms, (
            f"Layer {l} missing in permutations "
            f"(have {sorted(perms.keys())})"
        )
        for comp in ["K", "V"]:
            assert comp in perms[l], (
                f"Component {comp} missing in layer {l}"
            )
            assert len(perms[l][comp]) == num_kv_heads, (
                f"Layer {l} {comp}: expected {num_kv_heads} heads, "
                f"got {len(perms[l][comp])}"
            )
            for h in range(num_kv_heads):
                assert h in perms[l][comp], (
                    f"Head {h} missing in layer {l} {comp}"
                )
                perm = perms[l][comp][h]
                assert len(perm) == head_dim, (
                    f"Layer {l} {comp} head {h}: "
                    f"perm length {len(perm)} != head_dim {head_dim}"
                )
    print(f"  Permutation validation passed: "
          f"{num_layers} layers × {num_kv_heads} heads × {head_dim} dims")


# ============================================================
# CABACache — DynamicCache with permuted block quantization
# ============================================================

class CABACache(DynamicCache):
    """Permuted block assignment + per-block quantization.

    CompressedKVCache와 동일한 인터페이스:
      model(input_ids, past_key_values=cache, use_cache=True)
    로 사용.

    update()에서:
      1. per-head dimension permutation 적용
      2. per-block quantize-dequantize (uniform 또는 E₈)
      3. inverse permutation으로 원래 순서 복원
      4. DynamicCache에 저장

    quantizer:
      "uniform" — per-block RMS-shared symmetric uniform
      "e8"     — per-block E₈ lattice (compute_scale + encode_e8)
    """

    def __init__(
        self,
        perms: Dict,
        inv_perms: Dict,
        bits: int = 4,
        quantizer: str = "uniform",
        scale_mode: str = "per_head",
    ):
        super().__init__()
        self.perms = perms
        self.inv_perms = inv_perms
        self.bits = bits
        self.quantizer = quantizer
        self.scale_mode = scale_mode
        self.n_levels = 2 ** bits

        if quantizer == "e8" and not E8_AVAILABLE:
            raise ImportError(
                "E₈ quantizer not found. Check core/e8_quantizer.py path."
            )

    def _quantize_dequantize_block(self, x: torch.Tensor) -> torch.Tensor:
        """Per-block quantize-dequantize.

        Args:
            x: (..., 8) — last dim is block_size.

        Returns:
            quantized-dequantized tensor, same shape.
        """
        if self.quantizer == "e8":
            return self._qd_e8(x)
        else:
            return self._qd_uniform(x)

    def _qd_uniform(self, x: torch.Tensor) -> torch.Tensor:
        """Per-block RMS-shared symmetric uniform quantization."""
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
        x_scaled = x / rms
        half = self.n_levels / 2
        x_clamp = x_scaled.clamp(-half, half - 1)
        x_quant = torch.round(x_clamp)
        return x_quant * rms

    def _qd_e8(self, x: torch.Tensor, head_sigma2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """E₈ lattice quantization.

        Scale modes:
          head_sigma2=None  → per-block scale (각 8-dim block 개별 σ²)
          head_sigma2=값    → per-head scale (CompressedKVCache와 동일)

        Per-head scale이 수치적으로 안정적:
          near-zero block에서 per-block scale이 극단적으로 작아지는 문제 방지.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, 8)  # (N, 8)

        import math
        if head_sigma2 is not None:
            # Per-head: 모든 block이 같은 scale 공유
            target = 2 * math.pi * math.e * head_sigma2 * (4.0 ** (-self.bits))
            scale = torch.sqrt(target + 1e-30)
            x_scaled = x_flat / scale
        else:
            # Per-block: 각 block 개별 scale
            sigma2 = (x_flat ** 2).mean(dim=-1)  # (N,)
            target = 2 * math.pi * math.e * sigma2 * (4.0 ** (-self.bits))
            scales = torch.sqrt(target + 1e-30)  # (N,)
            x_scaled = x_flat / scales.unsqueeze(-1)

        q = encode_e8(x_scaled)

        if head_sigma2 is not None:
            x_hat = q * scale
        else:
            x_hat = q * scales.unsqueeze(-1)

        return x_hat.reshape(orig_shape)

    def _permute_quantize_unpermute(
        self,
        tensor: torch.Tensor,
        layer_idx: int,
        component: str,
    ) -> torch.Tensor:
        """Single tensor (K or V) 에 permutation + quantization 적용.

        Args:
            tensor: (batch, num_kv_heads, seq_len, head_dim)
            layer_idx: layer index
            component: "K" or "V"

        Returns:
            quantized tensor, same shape and dtype.
        """
        orig_dtype = tensor.dtype
        device = tensor.device
        batch, heads, seq, hd = tensor.shape
        block_size = 8

        if hd % block_size != 0:
            raise ValueError(
                f"head_dim={hd} is not divisible by block_size={block_size}. "
                f"CABA requires head_dim % 8 == 0."
            )

        t = tensor.float()
        out = torch.empty_like(t)

        for h in range(heads):
            perm = self.perms[layer_idx][component][h].to(device)
            inv_perm = self.inv_perms[layer_idx][component][h].to(device)

            # (batch, seq, head_dim)
            head_data = t[:, h, :, :]

            # Permute: gather along dim=-1
            permuted = head_data[:, :, perm]

            # Reshape to blocks: (batch, seq, n_blocks, 8)
            n_blocks = hd // block_size
            blocks = permuted.reshape(batch, seq, n_blocks, block_size)

            # Quantize-dequantize
            if self.quantizer == "e8" and self.scale_mode == "per_head":
                # Per-head σ²: one scale for all blocks in this head
                # Matches CompressedKVCache._compress_tensor
                head_sigma2 = (blocks ** 2).mean()
                blocks_qd = self._qd_e8(blocks, head_sigma2=head_sigma2)
            elif self.quantizer == "e8":
                blocks_qd = self._qd_e8(blocks)
            else:
                blocks_qd = self._qd_uniform(blocks)

            # Reshape back
            permuted_qd = blocks_qd.reshape(batch, seq, hd)

            # Inverse permute
            out[:, h, :, :] = permuted_qd[:, :, inv_perm]

        return out.to(orig_dtype)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override DynamicCache.update with CABA quantization."""
        # Quantize K and V with permuted block assignment
        k_quant = self._permute_quantize_unpermute(key_states, layer_idx, "K")
        v_quant = self._permute_quantize_unpermute(value_states, layer_idx, "V")

        # Store in DynamicCache (parent)
        return super().update(k_quant, v_quant, layer_idx, cache_kwargs)


# ============================================================
# PPL evaluation — same sliding window as e2e_eval.py
# ============================================================

@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer,
    cache_factory,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Sliding-window PPL on wikitext-2, e2e_eval.py와 동일한 방식."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    print(f"    Dataset: {seq_len:,} tokens, stride={stride}, max_len={max_length}")

    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        chunk_ids = input_ids[:, begin:end].to("cuda:0")
        cache = cache_factory()

        outputs = model(
            chunk_ids,
            past_key_values=cache,
            use_cache=True,
        )

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk_ids[:, 1:].contiguous()

        eval_start = max(0, chunk_ids.size(1) - trg_len - 1)
        eval_logits = shift_logits[:, eval_start:, :]
        eval_labels = shift_labels[:, eval_start:]

        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(
            eval_logits.reshape(-1, eval_logits.size(-1)),
            eval_labels.reshape(-1),
        )

        nlls.append(loss.item())
        n_tokens += eval_labels.numel()

        prev_end = end
        if end >= seq_len - 1:
            break

        if len(nlls) % 20 == 0:
            ppl_running = math.exp(sum(nlls) / n_tokens)
            print(f"      {n_tokens:,} tokens, running PPL={ppl_running:.2f}")

    ppl = math.exp(sum(nlls) / n_tokens)
    print(f"    Final: {n_tokens:,} tokens, PPL={ppl:.4f}")
    return ppl


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CABA Phase 2: PPL evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5])
    parser.add_argument(
        "--quantizer", type=str, default="uniform",
        choices=["uniform", "e8"],
        help="Block quantizer: uniform (fast proxy) or e8 (actual E₈ lattice)",
    )
    parser.add_argument(
        "--scale-mode", type=str, default="per_head",
        choices=["per_head", "per_block"],
        help="E₈ scale computation: per_head (stable, matches CompressedKVCache) or per_block",
    )
    parser.add_argument(
        "--mode", type=str, default="sorted",
        choices=["baseline", "sorted", "random"],
    )
    parser.add_argument(
        "--caba", type=str, default=None,
        help="Path to caba_analysis.py output JSON (required for sorted mode)",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for random mode (reproducibility)")
    parser.add_argument("--fp16-ppl", type=float, default=None,
                       help="Pre-computed FP16 baseline PPL (skip re-evaluation)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "sorted" and args.caba is None:
        parser.error("--caba is required for sorted mode")

    if args.output is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        args.output = f"results/caba_ppl_{model_short}_{args.mode}_{args.quantizer}_{args.bits}b.json"

    print(f"Model: {args.model}")
    print(f"Mode:  {args.mode}")
    print(f"Bits:  {args.bits}")
    print(f"Quant: {args.quantizer}")

    # Load model
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Model config
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"  {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")

    # Permutations
    print(f"\nPreparing permutations ({args.mode})...")
    if args.mode == "sorted":
        perms = load_sorted_permutations(args.caba)
        print(f"  Loaded from {args.caba}")
    elif args.mode == "random":
        perms = make_random_permutations(num_layers, num_kv_heads, head_dim, seed=args.seed)
        print(f"  Generated random permutations (seed={args.seed})")
    else:
        perms = make_identity_permutations(num_layers, num_kv_heads, head_dim)
        print(f"  Identity (baseline)")

    validate_permutations(perms, num_layers, num_kv_heads, head_dim)
    inv_perms = precompute_inverse_permutations(perms)

    # Cache factory (called per sliding window)
    def cache_factory():
        return CABACache(perms, inv_perms, bits=args.bits,
                        quantizer=args.quantizer, scale_mode=args.scale_mode)

    # FP16 baseline
    if args.fp16_ppl is not None:
        ppl_fp16 = args.fp16_ppl
        print(f"\n[FP16 baseline] Using provided value: {ppl_fp16:.4f}")
    else:
        print("\n[FP16 baseline]")
        ppl_fp16 = evaluate_ppl(
            model, tokenizer,
            cache_factory=DynamicCache,
            max_length=args.max_length,
            stride=args.stride,
        )

    # Evaluate with CABA
    print(f"\n[{args.mode} {args.bits}b]")
    t0 = time.time()
    ppl_quant = evaluate_ppl(
        model, tokenizer,
        cache_factory=cache_factory,
        max_length=args.max_length,
        stride=args.stride,
    )
    elapsed = time.time() - t0

    # Report
    delta = (ppl_quant / ppl_fp16 - 1) * 100
    print(f"\n{'=' * 50}")
    print(f"Results: {args.model.split('/')[-1]} | {args.mode} | {args.bits}b")
    print(f"{'=' * 50}")
    print(f"  FP16 PPL:  {ppl_fp16:.4f}")
    print(f"  Quant PPL: {ppl_quant:.4f} ({delta:+.2f}%)")
    print(f"  Time:      {elapsed:.1f}s")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "model": args.model,
        "mode": args.mode,
        "bits": args.bits,
        "quantizer": args.quantizer,
        "scope": "within-head permutation",
        "ppl_fp16": ppl_fp16,
        "ppl_quant": ppl_quant,
        "delta_pct": delta,
        "max_length": args.max_length,
        "stride": args.stride,
        "elapsed_seconds": elapsed,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()