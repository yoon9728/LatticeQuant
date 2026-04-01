"""
Perplexity Evaluation
======================
E₈ lattice KV cache quantization with perplexity evaluation on WikiText-2.

Features:
  1. Per-head σ² estimation
  2. Optional RHT within each head (Gaussianizes coordinates)
  3. Outlier clipping before quantization

Usage:
  python llm/perplexity_eval.py --model meta-llama/Llama-3.1-8B --all
  python llm/perplexity_eval.py --model meta-llama/Llama-3.1-8B --bits 4.0 --no-rht
"""

import torch
import numpy as np
import time
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from e8_quantizer import encode_e8, compute_scale, G_E8
from pipeline import fast_hadamard_transform, inverse_fast_hadamard_transform

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset


# ============================================================
# RHT for head dimensions
# ============================================================

class HeadRHT:
    """
    Randomized Hadamard Transform for a single head dimension.
    Shared random signs between encoder/decoder (per layer+head).
    """
    def __init__(self, head_dim: int, layer_idx: int, head_idx: int, kv_type: int):
        seed = layer_idx * 10000 + head_idx * 10 + kv_type
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.signs = torch.where(
            torch.rand(head_dim, generator=gen) < 0.5,
            torch.ones(head_dim), -torch.ones(head_dim)
        )
        self._device_signs = {}
    
    def get_signs(self, device):
        if device not in self._device_signs:
            self._device_signs[device] = self.signs.to(device)
        return self._device_signs[device]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signs = self.get_signs(x.device)
        return fast_hadamard_transform(x * signs)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        signs = self.get_signs(x.device)
        return inverse_fast_hadamard_transform(x) * signs


# ============================================================
# Improved Quantized KV Cache
# ============================================================

class QuantizedKVCacheV2(DynamicCache):
    """
    Improved quantized KV cache with:
      - Per-head σ² estimation
      - Optional RHT per head
      - Outlier clipping
    """
    
    def __init__(self, bits_per_dim: float, use_rht: bool = True, 
                 clip_sigma: float = 0.0):
        super().__init__()
        self.bits = bits_per_dim
        self.use_rht = use_rht
        self.clip_sigma = clip_sigma  # 0 = no clipping, >0 = clip at N*sigma
        self.rht_cache = {}  # (layer, head, kv_type) → HeadRHT
    
    def get_rht(self, layer_idx: int, head_idx: int, kv_type: int, 
                head_dim: int) -> HeadRHT:
        key = (layer_idx, head_idx, kv_type)
        if key not in self.rht_cache:
            self.rht_cache[key] = HeadRHT(head_dim, layer_idx, head_idx, kv_type)
        return self.rht_cache[key]
    
    def quantize_tensor(self, tensor: torch.Tensor, layer_idx: int,
                        kv_type: int) -> torch.Tensor:
        """
        Quantize KV tensor with per-head processing.
        tensor: (batch, num_heads, seq_len, head_dim)
        """
        orig_dtype = tensor.dtype
        t = tensor.float()
        batch, num_heads, seq_len, head_dim = t.shape
        
        if head_dim % 8 != 0:
            return tensor
        
        result = torch.zeros_like(t)
        
        for h in range(num_heads):
            head_data = t[:, h, :, :]  # (batch, seq_len, head_dim)
            vectors = head_data.reshape(-1, head_dim)  # (N, head_dim)
            
            # Step 1: Optional RHT
            if self.use_rht and (head_dim & (head_dim - 1)) == 0:  # power of 2
                rht = self.get_rht(layer_idx, h, kv_type, head_dim)
                vectors = rht.forward(vectors)
            
            # Step 2: Per-head σ² estimation
            sigma2 = (vectors ** 2).mean().item()
            if sigma2 < 1e-12:
                result[:, h, :, :] = head_data
                continue
            
            # Step 3: Outlier clipping
            if self.clip_sigma > 0:
                sigma = np.sqrt(sigma2)
                clip_val = self.clip_sigma * sigma
                vectors = vectors.clamp(-clip_val, clip_val)
                # Recompute σ² after clipping
                sigma2 = (vectors ** 2).mean().item()
                if sigma2 < 1e-12:
                    result[:, h, :, :] = head_data
                    continue
            
            # Step 4: Scale selection
            scale = compute_scale(sigma2, self.bits)
            
            # Step 5: E₈ block quantization
            blocks = vectors.reshape(-1, 8)
            blocks_norm = blocks / scale
            q = encode_e8(blocks_norm)
            blocks_hat = q * scale
            vectors_hat = blocks_hat.reshape(-1, head_dim)
            
            # Step 6: Inverse RHT
            if self.use_rht and (head_dim & (head_dim - 1)) == 0:
                vectors_hat = rht.inverse(vectors_hat)
            
            result[:, h, :, :] = vectors_hat.reshape(batch, seq_len, head_dim)
        
        return result.to(orig_dtype)
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override: quantize K/V before storing."""
        key_q = self.quantize_tensor(key_states, layer_idx, kv_type=0)
        value_q = self.quantize_tensor(value_states, layer_idx, kv_type=1)
        return super().update(key_q, value_q, layer_idx, cache_kwargs)


# ============================================================
# Perplexity Evaluation
# ============================================================

def evaluate_perplexity(model, tokenizer, device: str = 'cuda',
                        bits_per_dim: float = 0,
                        use_rht: bool = True,
                        clip_sigma: float = 0.0,
                        max_length: int = 2048,
                        stride: int = 512) -> float:
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    
    rht_str = "+RHT" if (use_rht and bits_per_dim > 0) else ""
    clip_str = f"+clip{clip_sigma}σ" if (clip_sigma > 0 and bits_per_dim > 0) else ""
    label = f"{bits_per_dim}b E₈{rht_str}{clip_str}" if bits_per_dim > 0 else "FP16 baseline"
    print(f"Evaluating: {label}")
    print(f"  Tokens: {seq_len:,}, Max length: {max_length}, Stride: {stride}")
    
    nlls = []
    n_chunks = 0
    t_start = time.time()
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - max(begin_loc, 0)
        if begin_loc > 0:
            trg_len = min(stride, end_loc - begin_loc)
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            if bits_per_dim > 0:
                past_kv = QuantizedKVCacheV2(bits_per_dim, use_rht=use_rht,
                                              clip_sigma=clip_sigma)
                outputs = model(input_chunk, labels=target_ids,
                              past_key_values=past_kv, use_cache=True)
            else:
                outputs = model(input_chunk, labels=target_ids)
            
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood.item())
        n_chunks += 1
        
        if n_chunks % 20 == 0:
            elapsed = time.time() - t_start
            current_ppl = np.exp(np.mean(nlls))
            tokens_done = min(end_loc, seq_len)
            print(f"  [{tokens_done:,}/{seq_len:,}] "
                  f"ppl={current_ppl:.2f}, time={elapsed:.0f}s")
        
        if end_loc == seq_len:
            break
    
    ppl = np.exp(np.mean(nlls))
    elapsed = time.time() - t_start
    print(f"  Final: {ppl:.4f} ({elapsed:.1f}s)")
    
    return ppl


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--bits', type=float, default=4.0)
    parser.add_argument('--all', action='store_true',
                        help='Run full comparison: baseline + variants')
    parser.add_argument('--no-rht', action='store_true')
    parser.add_argument('--clip', type=float, default=0.0,
                        help='Outlier clipping at N*sigma (0=disabled)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=512)
    args = parser.parse_args()
    
    print("LatticeQuant Perplexity Evaluation (v2)")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if '8B' in args.model or '8b' in args.model:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=args.device)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model: {args.model} ({params:.2f}B)")
    print()
    
    if args.all:
        results = {}
        
        # Baseline
        ppl_base = evaluate_perplexity(model, tokenizer, args.device, 0,
                                        max_length=args.max_length,
                                        stride=args.stride)
        results['FP16'] = ppl_base
        print()
        
        for bits in [3.0, 3.5, 4.0, 5.0]:
            ppl_q = evaluate_perplexity(model, tokenizer, args.device,
                                         bits, use_rht=False,
                                         max_length=args.max_length,
                                         stride=args.stride)
            results[f'{bits}b'] = ppl_q
            print()
        
        # Summary
        print("=" * 70)
        print(f"Summary: Perplexity on wikitext2 ({args.model})")        
        print("-" * 70)
        print(f"{'Config':>18} | {'PPL':>10} | {'Δ vs FP16':>10} | {'Δ%':>8}")
        print("-" * 70)
        
        for config, ppl in results.items():
            delta = ppl - ppl_base
            delta_pct = (ppl / ppl_base - 1) * 100
            print(f"{config:>18} | {ppl:>10.4f} | {delta:>+10.4f} | {delta_pct:>+7.2f}%")
        
        print("=" * 70)
        
    else:
        # Single run
        ppl = evaluate_perplexity(model, tokenizer, args.device,
                                   bits_per_dim=args.bits,
                                   use_rht=not args.no_rht,
                                   clip_sigma=args.clip,
                                   max_length=args.max_length,
                                   stride=args.stride)


if __name__ == '__main__':
    main()