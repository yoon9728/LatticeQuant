"""
Triton KV Cache Decompression
===============================
Integrates Triton dequant kernel with CompressedKVCache.

Optimized decompress path:
  - No GPU memory allocation in hot path (head_ids via arithmetic)
  - No GPU→CPU sync (.item() removed)
  - torch.empty instead of torch.zeros
  - Single scatter write (fallback first, then overwrite repr)

Dependencies:
  - triton_dequant.py (triton_unpack_e8_4bit)
  - compressed_kv_cache.py (CompressedKVCache, CompressedChunk)
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton_dequant import triton_unpack_e8_4bit
from compressed_kv_cache import CompressedKVCache, CompressedChunk, CompressedKVLayer


# ============================================================
# Triton-accelerated chunk decompression (optimized)
# ============================================================

def triton_decompress_chunk(chunk: CompressedChunk) -> torch.Tensor:
    """
    Decompress a CompressedChunk using Triton kernel + fallback merge.
    
    Optimized:
      - head_indices via integer division (no arange/repeat_interleave)
      - torch.empty (no memset)
      - single write pattern: fallback first, then overwrite repr
      - no .item() GPU sync
    """
    device = chunk.repr_mask.device
    batch, heads, seq, hd = chunk.orig_shape
    bph = chunk.blocks_per_head
    n_heads = batch * heads
    total_blocks = n_heads * bph
    
    # Validate layout
    assert total_blocks * 8 == batch * heads * seq * hd, \
        f"Block count mismatch: {total_blocks}*8 != {batch}*{heads}*{seq}*{hd}"
    
    repr_mask = chunk.repr_mask
    n_repr = chunk.packed.shape[0]
    n_oor = chunk.fallback.shape[0]
    
    # Allocate output (no memset needed — every position will be written)
    result = torch.empty(total_blocks, 8, dtype=torch.float16, device=device)
    
    # Step 1: Write fallback vectors first (OOR positions)
    if n_oor > 0:
        assert chunk.fallback.shape[0] == n_oor, \
            f"Fallback shape mismatch: {chunk.fallback.shape[0]} != {n_oor}"
        result[~repr_mask] = chunk.fallback
    
    # Step 2: Overwrite representable positions via Triton
    if n_repr > 0:
        # Compute head index for each packed vector via integer arithmetic
        # repr_mask layout: [head0 blocks..., head1 blocks..., ...]
        # block_global_idx // bph = head_index
        head_ids_repr = chunk.repr_head_ids
        
        dequanted = triton_unpack_e8_4bit(
            chunk.packed,
            chunk.scales,
            head_ids_repr,
        )
        result[repr_mask] = dequanted
    
    return result.reshape(batch, heads, seq, hd)


# ============================================================
# Patched CompressedKVCache with Triton decompression
# ============================================================

class TritonCompressedKVCache(CompressedKVCache):
    """
    CompressedKVCache with Triton-accelerated decompression.
    """
    
    def _decompress_chunk(self, chunk: CompressedChunk) -> torch.Tensor:
        return triton_decompress_chunk(chunk)


# ============================================================
# Tests
# ============================================================

def test_triton_vs_python_correctness():
    from compressed_kv_cache import CompressedKVCache as PythonCache
    
    print("Test: Triton vs Python correctness")
    print("=" * 60)
    
    device = 'cuda'
    batch, heads, seq, hd = 1, 8, 128, 128
    
    py_cache = PythonCache(bits_per_dim=4)
    tr_cache = TritonCompressedKVCache(bits_per_dim=4)
    
    torch.manual_seed(42)
    k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
    k_py, v_py = py_cache.update(k.clone(), v.clone(), layer_idx=0)
    
    torch.manual_seed(42)
    k2 = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
    v2 = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
    k_tr, v_tr = tr_cache.update(k2.clone(), v2.clone(), layer_idx=0)
    
    diff_k = (k_py.float() - k_tr.float()).abs().max().item()
    diff_v = (v_py.float() - v_tr.float()).abs().max().item()
    
    match = diff_k < 1e-3 and diff_v < 1e-3
    print(f"  K_diff={diff_k:.2e} V_diff={diff_v:.2e} {'PASS' if match else 'FAIL'}")
    print()
    return match


def test_fallback_heavy():
    """
    Force high OOR rate by using very small scale (large lattice coords).
    Verifies merge correctness when fallback dominates.
    """
    from e8_quantizer import encode_e8, compute_scale
    from compact_storage import pack_e8, check_representable
    
    print("Test: Fallback-heavy case (forced high OOR)")
    print("=" * 60)
    
    device = 'cuda'
    batch, heads, seq, hd = 1, 8, 64, 128
    bits = 4
    
    # Use very small bits_per_dim equivalent to get many OOR
    # Trick: use bits=4 cache but feed data with variance >> 1
    py_cache = CompressedKVCache(bits_per_dim=bits)
    tr_cache = TritonCompressedKVCache(bits_per_dim=bits)
    
    # Large variance → many coords exceed range
    torch.manual_seed(42)
    k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16) * 10.0
    v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16) * 10.0
    
    k_py, _ = py_cache.update(k.clone(), v.clone(), layer_idx=0)
    k_tr, _ = tr_cache.update(k.clone(), v.clone(), layer_idx=0)
    
    diff = (k_py.float() - k_tr.float()).abs().max().item()
    match = diff < 1e-2
    
    # Report OOR rate
    oor_pct = tr_cache.total_oor / max(tr_cache.total_blocks, 1) * 100
    
    print(f"  OOR rate: {oor_pct:.1f}%")
    print(f"  Max diff: {diff:.2e}")
    print(f"  {'PASS' if match else 'FAIL'}")
    print()
    return match


def test_incremental_decode():
    print("Test: Triton incremental decode")
    print("=" * 60)
    
    device = 'cuda'
    cache = TritonCompressedKVCache(bits_per_dim=4)
    batch, heads, hd = 1, 8, 128
    
    torch.manual_seed(42)
    k = torch.randn(batch, heads, 64, hd, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, 64, hd, device=device, dtype=torch.float16)
    cache.update(k, v, layer_idx=0)
    print(f"  Prefill: seq={cache.get_seq_length(0)}")
    
    for step in range(4):
        k_new = torch.randn(batch, heads, 1, hd, device=device, dtype=torch.float16)
        v_new = torch.randn(batch, heads, 1, hd, device=device, dtype=torch.float16)
        k_out, v_out = cache.update(k_new, v_new, layer_idx=0)
    
    assert cache.get_seq_length(0) == 68
    assert not torch.isnan(k_out).any()
    print(f"  Final: seq={cache.get_seq_length(0)}, no NaN → PASS")
    print()


def test_decompress_latency():
    """
    Pure decompression latency: create chunks first, then benchmark
    only the decompress call. No compression cost included.
    """
    from compressed_kv_cache import CompressedKVCache as PythonCache
    
    print("Test: Pure decompression latency (Triton vs Python)")
    print("=" * 60)
    
    device = 'cuda'
    bits = 4
    
    configs = [
        ("Prefill 512",  1, 8, 512, 128),
        ("Prefill 2048", 1, 8, 2048, 128),
        ("Decode 1tok",  1, 8, 1, 128),
        ("Decode 8tok",  1, 8, 8, 128),
    ]
    
    print(f"  {'Config':<16} | {'Python':>10} | {'Triton':>10} | {'Speedup':>8}")
    print(f"  {'-'*52}")
    
    for label, batch, heads, seq, hd in configs:
        torch.manual_seed(42)
        k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        
        # Build chunks (cost not measured)
        py_cache = PythonCache(bits_per_dim=bits)
        py_cache.update(k.clone(), v.clone(), layer_idx=0)
        chunk = py_cache._comp_keys[0].chunks[0]
        
        # Benchmark Python decompress only
        n_iter = 200 if seq <= 8 else 50
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = py_cache._decompress_chunk(chunk)
        torch.cuda.synchronize()
        t_py = (time.time() - t0) / n_iter
        
        # Benchmark Triton decompress only
        for _ in range(10):
            _ = triton_decompress_chunk(chunk)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(n_iter):
            _ = triton_decompress_chunk(chunk)
        torch.cuda.synchronize()
        t_tr = (time.time() - t0) / n_iter
        
        speedup = t_py / t_tr if t_tr > 0 else float('inf')
        
        def fmt(t):
            return f"{t*1e6:.0f}us" if t < 0.001 else f"{t*1e3:.2f}ms"
        
        print(f"  {label:<16} | {fmt(t_py):>10} | {fmt(t_tr):>10} | {speedup:>7.1f}x")
    
    print()


def test_full_cache_latency():
    """
    Full KV cache decompress at Llama-3.1-8B scale (4 layers).
    """
    print("Test: Full cache decompress (4L × 8H × 2048T × 128D)")
    print("=" * 60)
    
    device = 'cuda'
    batch, heads, seq, hd = 1, 8, 2048, 128
    n_layers = 4
    
    cache = TritonCompressedKVCache(bits_per_dim=4)
    torch.manual_seed(42)
    for layer in range(n_layers):
        k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        cache.update(k, v, layer_idx=layer)
    
    # Collect all chunks
    all_chunks = []
    for layer_idx in range(n_layers):
        for chunk in cache._comp_keys[layer_idx].chunks:
            all_chunks.append(chunk)
        for chunk in cache._comp_values[layer_idx].chunks:
            all_chunks.append(chunk)
    
    def decompress_all():
        for chunk in all_chunks:
            triton_decompress_chunk(chunk)
    
    # Warmup
    for _ in range(5):
        decompress_all()
    torch.cuda.synchronize()
    
    n_iter = 50
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        decompress_all()
    torch.cuda.synchronize()
    t_total = (time.time() - t0) / n_iter
    
    total_vecs = n_layers * 2 * heads * seq * (hd // 8)
    
    print(f"  Decompress time:  {t_total*1e3:.2f} ms")
    print(f"  Vectors (K+V):    {total_vecs:,}")
    print(f"  Throughput:       {total_vecs/t_total/1e6:.1f}M vec/s")
    print()
    print(cache.memory_report())
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Triton KV Cache Integration")
    print("=" * 60)
    print()
    
    r1 = test_triton_vs_python_correctness()
    r2 = test_fallback_heavy()
    test_incremental_decode()
    test_decompress_latency()
    test_full_cache_latency()
    
    if r1 and r2:
        print("PASSED.")
    else:
        print("FAILED.")