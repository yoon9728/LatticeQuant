"""
Triton E₈ Dequant Kernel
==========================
GPU kernel to decompress packed E₈ uint8 → float16 on-the-fly.

Replaces the Python-loop unpack_e8() with a fused Triton kernel.
This enables:
  - On-the-fly decompression during attention (no double-storage)
  - Measured decode latency for the paper
  - Throughput benchmarking (vectors/sec)

Kernel logic (per vector):
  1. Load b bytes → assemble into packed integer
  2. Extract coset bit
  3. Extract 7 unsigned coordinates
  4. Extract coord7 upper bits
  5. Recover coord7 LSB from parity constraint
  6. Convert to signed, apply coset offset
  7. Multiply by scale → float16 output

Supports bits_per_dim = 3, 4, 5.

Dependencies:
  - triton
  - torch
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Triton kernel: unpack + dequant for 4 bits/dim
# ============================================================
# Specialized for bits=4 (most common case, uint32-aligned).
# 4 bytes per vector → load as single uint32 → pure bit ops.

@triton.jit
def _unpack_e8_4bit_kernel(
    packed_ptr,      # (N, 4) uint8 → treat as (N,) uint32
    scale_ptr,       # (num_heads,) float32 — per-head scale
    head_idx_ptr,    # (N,) int32 — which head each vector belongs to
    out_ptr,         # (N, 8) float16 output
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Unpack 4-bit packed E₈ vectors to float16.
    
    Bit layout (32 bits total):
      [coset(1) | c0(4) | c1(4) | c2(4) | c3(4) | c4(4) | c5(4) | c6(4) | c7_upper(3)]
      Total: 1 + 7*4 + 3 = 32 bits = 4 bytes ✓
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    
    # Load 4 bytes as uint32 (big-endian assembly)
    b0 = tl.load(packed_ptr + offs * 4 + 0, mask=mask, other=0).to(tl.int64)
    b1 = tl.load(packed_ptr + offs * 4 + 1, mask=mask, other=0).to(tl.int64)
    b2 = tl.load(packed_ptr + offs * 4 + 2, mask=mask, other=0).to(tl.int64)
    b3 = tl.load(packed_ptr + offs * 4 + 3, mask=mask, other=0).to(tl.int64)
    packed = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    
    # Extract fields (reverse of packing order)
    R: tl.constexpr = 8  # 2^(4-1)
    COORD_MASK: tl.constexpr = 15  # (1 << 4) - 1
    UPPER_MASK: tl.constexpr = 7   # (1 << 3) - 1
    
    # coord7 upper (3 bits)
    c7_upper = packed & UPPER_MASK
    packed = packed >> 3
    
    # coords 6..0 (4 bits each, extract in reverse)
    c6 = (packed & COORD_MASK) - R
    packed = packed >> 4
    c5 = (packed & COORD_MASK) - R
    packed = packed >> 4
    c4 = (packed & COORD_MASK) - R
    packed = packed >> 4
    c3 = (packed & COORD_MASK) - R
    packed = packed >> 4
    c2 = (packed & COORD_MASK) - R
    packed = packed >> 4
    c1 = (packed & COORD_MASK) - R
    packed = packed >> 4
    c0 = (packed & COORD_MASK) - R
    packed = packed >> 4
    
    # coset bit
    coset = packed & 1
    
    # Recover coord7 LSB from parity: sum(c0..c6) + lsb ≡ 0 (mod 2)
    partial_sum = c0 + c1 + c2 + c3 + c4 + c5 + c6
    # Need: lsb such that partial_sum + c7_upper*2 + lsb - R ≡ 0 mod 2
    # Since c7_upper*2 and R=8 are both even: lsb ≡ partial_sum mod 2
    lsb = tl.abs(partial_sum) % 2
    c7 = c7_upper * 2 + lsb - R
    
    # Load per-head scale
    h_idx = tl.load(head_idx_ptr + offs, mask=mask, other=0)
    scale = tl.load(scale_ptr + h_idx, mask=mask, other=1.0)
    
    # Apply coset offset (0 for integer, 0.5 for half-integer)
    offset = coset.to(tl.float32) * 0.5
    
    # Compute output: (signed_coord + offset) * scale
    o0 = (c0.to(tl.float32) + offset) * scale
    o1 = (c1.to(tl.float32) + offset) * scale
    o2 = (c2.to(tl.float32) + offset) * scale
    o3 = (c3.to(tl.float32) + offset) * scale
    o4 = (c4.to(tl.float32) + offset) * scale
    o5 = (c5.to(tl.float32) + offset) * scale
    o6 = (c6.to(tl.float32) + offset) * scale
    o7 = (c7.to(tl.float32) + offset) * scale
    
    # Store as float16
    tl.store(out_ptr + offs * 8 + 0, o0.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 1, o1.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 2, o2.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 3, o3.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 4, o4.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 5, o5.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 6, o6.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs * 8 + 7, o7.to(tl.float16), mask=mask)


# ============================================================
# Python wrapper
# ============================================================

def triton_unpack_e8_4bit(
    packed: torch.Tensor,
    scales: torch.Tensor,
    head_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Triton-accelerated E₈ unpack + dequant for 4 bits/dim.
    
    Args:
        packed: (N, 4) uint8 tensor — packed E₈ vectors
        scales: (num_heads,) float32 — per-head lattice scale
        head_indices: (N,) int32 — head index for each vector
    
    Returns:
        (N, 8) float16 tensor — decompressed vectors
    """
    assert packed.dtype == torch.uint8
    assert packed.shape[1] == 4
    assert packed.is_cuda
    
    N = packed.shape[0]
    out = torch.empty(N, 8, dtype=torch.float16, device=packed.device)
    
    # Flatten packed to contiguous uint8 for byte-level access
    packed_flat = packed.contiguous().view(-1)
    
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    
    _unpack_e8_4bit_kernel[grid](
        packed_flat,
        scales,
        head_indices,
        out.view(-1),  # flat float16
        N=N,
        BLOCK=BLOCK,
    )
    
    return out


# ============================================================
# Tests
# ============================================================

def test_correctness():
    """
    Verify Triton kernel produces same output as Python unpack_e8.
    """
    from compact_storage import pack_e8, unpack_e8, check_representable
    from e8_quantizer import encode_e8, compute_scale
    
    print("Test: Triton vs Python unpack correctness")
    print("=" * 60)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    bits = 4
    sigma2 = 1.0
    scale_val = compute_scale(sigma2, bits)
    N = 100_000
    
    x = torch.randn(N, 8, device=device)
    q = encode_e8(x / scale_val)
    
    # Filter representable
    in_range = check_representable(q, bits)
    q_repr = q[in_range]
    M = q_repr.shape[0]
    
    # Pack
    packed = pack_e8(q_repr, bits)
    
    # Python unpack
    q_python = unpack_e8(packed, bits)
    
    # Triton unpack: need scales and head_indices
    # For this test, all vectors use same scale
    scales = torch.tensor([scale_val], dtype=torch.float32, device=device)
    head_indices = torch.zeros(M, dtype=torch.int32, device=device)
    
    out_triton = triton_unpack_e8_4bit(packed, scales, head_indices)
    
    # Python dequant
    dequant_python = (q_python * scale_val).half()
    
    # Compare
    diff = (dequant_python.float() - out_triton.float()).abs().max().item()
    match = diff < 1e-2  # float16 tolerance
    
    print(f"  Vectors: {M:,} (from {N:,}, {M/N*100:.1f}% representable)")
    print(f"  Max diff: {diff:.4e}")
    print(f"  {'PASS' if match else 'FAIL'}")
    print()
    
    return match


def test_throughput():
    """
    Benchmark Triton kernel throughput.
    """
    from compact_storage import pack_e8, check_representable
    from e8_quantizer import encode_e8, compute_scale
    
    print("Test: Triton dequant throughput")
    print("=" * 60)
    
    device = 'cuda'
    
    for N in [10_000, 100_000, 1_000_000]:
        torch.manual_seed(42)
        bits = 4
        scale_val = compute_scale(1.0, bits)
        
        x = torch.randn(N, 8, device=device)
        q = encode_e8(x / scale_val)
        in_range = check_representable(q, bits)
        packed = pack_e8(q[in_range], bits)
        M = packed.shape[0]
        
        scales = torch.tensor([scale_val], dtype=torch.float32, device=device)
        head_indices = torch.zeros(M, dtype=torch.int32, device=device)
        
        # Warmup
        for _ in range(5):
            _ = triton_unpack_e8_4bit(packed, scales, head_indices)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iter = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = triton_unpack_e8_4bit(packed, scales, head_indices)
        torch.cuda.synchronize()
        t1 = time.time()
        
        elapsed = (t1 - t0) / n_iter
        vecs_per_sec = M / elapsed
        elements_per_sec = M * 8 / elapsed
        
        print(f"  N={M:>10,}: {elapsed*1e6:>8.1f} us | "
              f"{vecs_per_sec/1e6:>7.2f}M vec/s | "
              f"{elements_per_sec/1e9:>6.2f}G elem/s")
    
    print()


def test_vs_python_speed():
    """
    Compare Triton vs Python unpack speed.
    """
    from compact_storage import pack_e8, unpack_e8, check_representable
    from e8_quantizer import encode_e8, compute_scale
    
    print("Test: Triton vs Python speed comparison")
    print("=" * 60)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    bits = 4
    scale_val = compute_scale(1.0, bits)
    N = 500_000
    
    x = torch.randn(N, 8, device=device)
    q = encode_e8(x / scale_val)
    in_range = check_representable(q, bits)
    packed = pack_e8(q[in_range], bits)
    M = packed.shape[0]
    
    scales = torch.tensor([scale_val], dtype=torch.float32, device=device)
    head_indices = torch.zeros(M, dtype=torch.int32, device=device)
    
    # Python timing
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        q_py = unpack_e8(packed, bits)
        _ = (q_py * scale_val).half()
    torch.cuda.synchronize()
    t_python = (time.time() - t0) / 10
    
    # Triton timing
    for _ in range(5):
        _ = triton_unpack_e8_4bit(packed, scales, head_indices)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = triton_unpack_e8_4bit(packed, scales, head_indices)
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 100
    
    speedup = t_python / t_triton
    
    print(f"  Vectors: {M:,}")
    print(f"  Python:  {t_python*1e3:.2f} ms")
    print(f"  Triton:  {t_triton*1e3:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Triton E₈ Dequant Kernel")
    print("=" * 60)
    print()
    
    r1 = test_correctness()
    test_throughput()
    test_vs_python_speed()
    
    if r1:
        print("PASSED: Triton kernel matches Python unpack.")
    else:
        print("FAILED: correctness mismatch.")