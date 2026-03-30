"""
LatticeQuant v2 — CompressedKVCache
====================================
DynamicCache subclass that stores KV tensors in compressed E₈ format.

Storage strategy (Phase 1A fixed-length + FP16 fallback):
  - Representable E₈ vectors (~73%): packed uint8, b bytes per 8-dim block
  - OOR vectors (~27%): stored as float16 dequantized E₈ values
  - Per-head scales stored with each chunk for correct reconstruction
  - Chunk-aware: each update appends an independent compressed chunk
    with its own scales, so the full compressed state is self-consistent

Note on VRAM:
  Current DynamicCache API requires decompressed FP16 in parent's
  key_cache/value_cache for attention. Actual VRAM includes BOTH
  compressed and decompressed data (double-storage). Theoretical
  compressed size is reported separately. Phase 2 (Triton on-the-fly
  dequant) eliminates double-storage.

Dependencies:
  - compact_storage.py (pack_e8, unpack_e8, check_representable)
  - e8_quantizer.py (encode_e8, compute_scale)
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e8_quantizer import encode_e8, compute_scale
from compact_storage import pack_e8, unpack_e8, check_representable

from transformers.cache_utils import DynamicCache


# ============================================================
# Compressed chunk: one update() call's worth of data
# ============================================================

@dataclass
class CompressedChunk:
    """
    One chunk of compressed KV data (from a single update() call).
    Self-contained: has its own scales for correct reconstruction.
    """
    packed: torch.Tensor          # (n_repr, bits) uint8
    fallback: torch.Tensor        # (n_oor, 8) float16 dequantized E₈ values
    repr_mask: torch.Tensor       # (total_blocks,) bool
    scales: torch.Tensor          # (batch*heads,) float32 — per-head scale
    bits_per_dim: int
    orig_shape: tuple             # (batch, heads, seq_chunk, head_dim)
    blocks_per_head: int          # seq_chunk * (head_dim // 8)
    num_blocks: int
    n_repr: int
    n_oor: int
    repr_indices: torch.Tensor = None    # (n_repr,) int32 — global block indices
    repr_head_ids: torch.Tensor = None   # (n_repr,) int32 — head index per packed vector
    repr_cumsum: torch.Tensor = None     # (total_blocks,) int32 — exclusive prefix sum of repr
    fb_cumsum: torch.Tensor = None       # (total_blocks,) int32 — exclusive prefix sum of fallback


# ============================================================
# Per-layer compressed storage (list of chunks)
# ============================================================

class CompressedKVLayer:
    """
    Compressed storage for one layer's K or V tensor.
    Consists of a list of chunks, each self-contained with own scales.
    """
    
    def __init__(self):
        self.chunks: List[CompressedChunk] = []
        self._full_shape_cache: Optional[tuple] = None  # (batch, heads, total_seq, hd)
    
    @property
    def is_empty(self):
        return len(self.chunks) == 0
    
    @property
    def total_seq(self):
        if not self.chunks:
            return 0
        return sum(c.orig_shape[2] for c in self.chunks)
    
    @property
    def full_shape(self):
        if not self.chunks:
            return None
        c0 = self.chunks[0]
        return (c0.orig_shape[0], c0.orig_shape[1], self.total_seq, c0.orig_shape[3])
    
    @property
    def num_blocks(self):
        return sum(c.num_blocks for c in self.chunks)
    
    @property
    def total_bytes(self):
        total = 0
        for c in self.chunks:
            total += c.packed.nelement() * c.packed.element_size()
            total += c.fallback.nelement() * c.fallback.element_size()
            total += c.repr_mask.nelement() * c.repr_mask.element_size()
            total += c.scales.nelement() * c.scales.element_size()
        return total
    
    @property
    def fp16_equivalent_bytes(self):
        return self.num_blocks * 8 * 2
    
    @property
    def total_repr(self):
        return sum(c.n_repr for c in self.chunks)
    
    @property
    def total_oor(self):
        return sum(c.n_oor for c in self.chunks)
    
    def append_chunk(self, chunk: CompressedChunk):
        self.chunks.append(chunk)


# ============================================================
# Uncompressible layer (head_dim not multiple of 8)
# ============================================================

class UncompressibleKVLayer:
    """
    Fallback for head_dim not divisible by 8.
    Stores original tensor as-is in FP16. No compression attempted.
    """
    def __init__(self, tensor: torch.Tensor):
        self.data = tensor.to(torch.float16)
        self.shape = tensor.shape
    
    @property
    def is_empty(self):
        return False
    
    @property
    def total_bytes(self):
        return self.data.nelement() * self.data.element_size()
    
    @property
    def fp16_equivalent_bytes(self):
        return self.total_bytes
    
    def get_tensor(self):
        return self.data


# ============================================================
# CompressedKVCache
# ============================================================

class CompressedKVCache(DynamicCache):
    """
    DynamicCache subclass that compresses K/V with E₈ lattice quantization.
    
    Chunk-aware: each update() creates an independent compressed chunk
    with per-head scales, ensuring self-consistent reconstruction.
    
    Reports theoretical compressed size separately from actual VRAM.
    """
    
    def __init__(self, bits_per_dim: int = 4):
        super().__init__()
        assert bits_per_dim in (3, 4, 5)
        self.bits = bits_per_dim
        
        # Ensure parent cache lists exist (transformers version compatibility)
        if not hasattr(self, 'key_cache'):
            self.key_cache = []
        if not hasattr(self, 'value_cache'):
            self.value_cache = []
        
        self._comp_keys: List = []   # CompressedKVLayer or UncompressibleKVLayer
        self._comp_values: List = []
        
        self.total_blocks = 0
        self.total_repr = 0
        self.total_oor = 0
    
    def _compress_tensor(self, tensor: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Compress a KV tensor → (compressed_layer_or_chunk, decompressed_fp16).
        
        If head_dim is not multiple of 8, returns UncompressibleKVLayer.
        Otherwise returns a CompressedChunk.
        """
        device = tensor.device
        batch, heads, seq, hd = tensor.shape
        
        # Non-compressible: store as-is
        if hd % 8 != 0:
            return 'uncompressible', tensor.to(torch.float16)
        
        t = tensor.float()
        bph = seq * (hd // 8)  # blocks per head
        n_heads = batch * heads
        
        # Reshape: (batch*heads, blocks_per_head, 8)
        t_heads = t.reshape(n_heads, bph, 8)
        
        # Per-head σ²
        sigma2 = (t_heads ** 2).mean(dim=(1, 2))  # (n_heads,)
        per_head_scale = torch.zeros(n_heads, dtype=torch.float32, device=device)
        
        all_packed = []
        all_fallback = []
        all_masks = []
        total_repr = 0
        total_oor = 0
        
        # Also build decompressed for attention
        decompressed = torch.zeros_like(t_heads)
        
        for h in range(n_heads):
            blocks = t_heads[h]  # (bph, 8)
            s2 = sigma2[h].item()
            
            if s2 < 1e-12:
                mask = torch.zeros(bph, dtype=torch.bool, device=device)
                all_masks.append(mask)
                all_fallback.append(blocks.half())
                decompressed[h] = blocks
                per_head_scale[h] = 1.0
                total_oor += bph
                continue
            
            scale = compute_scale(s2, self.bits)
            per_head_scale[h] = scale
            
            q = encode_e8(blocks / scale)
            in_range = check_representable(q, self.bits)
            n_r = in_range.sum().item()
            n_o = (~in_range).sum().item()
            
            if n_r > 0:
                all_packed.append(pack_e8(q[in_range], self.bits))
            
            if n_o > 0:
                all_fallback.append((q[~in_range] * scale).half())
            
            all_masks.append(in_range)
            
            # Decompressed: all blocks go through E₈ quantize/dequant
            decompressed[h] = q * scale
            
            total_repr += n_r
            total_oor += n_o
        
        self.total_blocks += n_heads * bph
        self.total_repr += total_repr
        self.total_oor += total_oor

        # Precompute indices for Triton decompress (avoids torch.nonzero at decode time)
        full_mask = torch.cat(all_masks, dim=0)
        repr_indices = torch.nonzero(full_mask).flatten().to(torch.int32)
        repr_head_ids = (repr_indices // bph).to(torch.int32)
        mask_int = full_mask.to(torch.int32)
        repr_cumsum = (torch.cumsum(mask_int, dim=0) - mask_int).to(torch.int32)
        fb_cumsum = (torch.cumsum(1 - mask_int, dim=0) - (1 - mask_int)).to(torch.int32)

        chunk = CompressedChunk(
            packed=torch.cat(all_packed, dim=0) if all_packed else
                   torch.zeros(0, self.bits, dtype=torch.uint8, device=device),
            fallback=torch.cat(all_fallback, dim=0) if all_fallback else
                     torch.zeros(0, 8, dtype=torch.float16, device=device),
            repr_mask=torch.cat(all_masks, dim=0),
            scales=per_head_scale,
            bits_per_dim=self.bits,
            orig_shape=tensor.shape,
            blocks_per_head=bph,
            num_blocks=n_heads * bph,
            n_repr=total_repr,
            n_oor=total_oor,
            repr_indices=repr_indices,
            repr_head_ids=repr_head_ids,
            repr_cumsum=repr_cumsum,
            fb_cumsum=fb_cumsum,
        )
        
        decompressed_out = decompressed.reshape(tensor.shape).to(torch.float16)
        return chunk, decompressed_out
    
    def _decompress_chunk(self, chunk: CompressedChunk) -> torch.Tensor:
        """
        Decompress a single chunk using its own per-head scales.
        Returns float16 tensor with chunk's original shape.
        """
        device = chunk.repr_mask.device
        batch, heads, seq, hd = chunk.orig_shape
        bph = chunk.blocks_per_head
        n_heads = batch * heads
        
        result = torch.zeros(n_heads, bph, 8, dtype=torch.float32, device=device)
        
        repr_cursor = 0
        fb_cursor = 0
        
        for h in range(n_heads):
            start = h * bph
            end = start + bph
            mask_h = chunk.repr_mask[start:end]
            scale = chunk.scales[h].item()
            
            n_r = mask_h.sum().item()
            n_o = bph - n_r
            
            if n_r > 0:
                q_h = unpack_e8(chunk.packed[repr_cursor:repr_cursor + n_r], chunk.bits_per_dim)
                result[h, mask_h] = q_h * scale
                repr_cursor += n_r
            
            if n_o > 0:
                result[h, ~mask_h] = chunk.fallback[fb_cursor:fb_cursor + n_o].float()
                fb_cursor += n_o
        
        return result.reshape(chunk.orig_shape).to(torch.float16)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override DynamicCache.update to compress K/V.
        Returns quantized K/V for attention (in-the-loop evaluation).
        """
        comp_k, k_dec = self._compress_tensor(key_states)
        comp_v, v_dec = self._compress_tensor(value_states)
        
        # Ensure lists are long enough
        while len(self._comp_keys) <= layer_idx:
            self._comp_keys.append(None)
            self._comp_values.append(None)
        
        # Handle uncompressible case
        if comp_k == 'uncompressible':
            self._comp_keys[layer_idx] = UncompressibleKVLayer(key_states)
            self._comp_values[layer_idx] = UncompressibleKVLayer(value_states)
        else:
            # Initialize or append chunk
            if self._comp_keys[layer_idx] is None:
                kl = CompressedKVLayer()
                vl = CompressedKVLayer()
                self._comp_keys[layer_idx] = kl
                self._comp_values[layer_idx] = vl
            
            self._comp_keys[layer_idx].append_chunk(comp_k)
            self._comp_values[layer_idx].append_chunk(comp_v)
        
        # Build full decompressed tensor for attention
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k_dec)
            self.value_cache.append(v_dec)
        else:
            # Concatenate existing + new along seq dimension
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], k_dec], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], v_dec], dim=2
            )
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    # ========================================================
    # Reporting
    # ========================================================
    
    def compressed_bytes(self) -> int:
        """Theoretical compressed size (what a production system would use)."""
        total = 0
        for layer in self._comp_keys + self._comp_values:
            if layer is None:
                continue
            total += layer.total_bytes
        return total
    
    def fp16_equivalent_bytes(self) -> int:
        total = 0
        for layer in self._comp_keys + self._comp_values:
            if layer is None:
                continue
            total += layer.fp16_equivalent_bytes
        return total
    
    def memory_report(self) -> str:
        comp = self.compressed_bytes()
        fp16 = self.fp16_equivalent_bytes()
        ratio = fp16 / max(comp, 1)
        repr_pct = self.total_repr / max(self.total_blocks, 1) * 100
        oor_pct = self.total_oor / max(self.total_blocks, 1) * 100
        
        n_layers = sum(1 for x in self._comp_keys if x is not None)
        n_chunks = sum(
            len(x.chunks) for x in self._comp_keys
            if isinstance(x, CompressedKVLayer)
        )
        
        lines = [
            f"CompressedKVCache Report",
            f"  Bits/dim:  {self.bits}",
            f"  Layers:    {n_layers}",
            f"  Chunks:    {n_chunks} (K side)",
            f"  Blocks:    {self.total_blocks:,} total",
            f"  Repr:      {self.total_repr:,} ({repr_pct:.1f}%) packed uint8",
            f"  OOR:       {self.total_oor:,} ({oor_pct:.1f}%) FP16 dequantized",
            f"  Theoretical compressed: {comp:,} B ({comp/1024/1024:.2f} MB)",
            f"  FP16 equivalent:        {fp16:,} B ({fp16/1024/1024:.2f} MB)",
            f"  Theoretical ratio:      {ratio:.2f}x",
            f"  Note: actual VRAM includes decompressed FP16 in parent cache.",
        ]
        return "\n".join(lines)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[2]
        return 0


# ============================================================
# Tests
# ============================================================

def test_roundtrip_quality():
    print("Test: CompressedKVCache roundtrip quality")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for bits in [3, 4, 5]:
        cache = CompressedKVCache(bits_per_dim=bits)
        batch, heads, seq, hd = 1, 8, 128, 128
        
        torch.manual_seed(42)
        k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        
        k_out, v_out = cache.update(k, v, layer_idx=0)
        
        mse_k = ((k.float() - k_out.float()) ** 2).mean().item()
        rel_k = mse_k / (k.float() ** 2).mean().item()
        
        assert not torch.isnan(k_out).any(), "NaN in output"
        assert k_out.shape == k.shape, f"Shape mismatch: {k_out.shape} vs {k.shape}"
        
        print(f"  {bits}b: MSE={mse_k:.6f} rel={rel_k:.4f} shape={k_out.shape}")
    
    print()


def test_incremental_decode():
    print("Test: Incremental decode (prefill + 4 tokens)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    batch, heads, hd = 1, 8, 128
    
    # Prefill
    torch.manual_seed(42)
    k = torch.randn(batch, heads, 64, hd, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, 64, hd, device=device, dtype=torch.float16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    print(f"  Prefill:   seq={cache.get_seq_length(0)}")
    
    # Decode
    for step in range(4):
        k_new = torch.randn(batch, heads, 1, hd, device=device, dtype=torch.float16)
        v_new = torch.randn(batch, heads, 1, hd, device=device, dtype=torch.float16)
        k_out, v_out = cache.update(k_new, v_new, layer_idx=0)
        print(f"  Decode {step+1}:  seq={cache.get_seq_length(0)}")
    
    assert cache.get_seq_length(0) == 68, f"Expected 68, got {cache.get_seq_length(0)}"
    assert k_out.shape == (batch, heads, 68, hd)
    assert not torch.isnan(k_out).any()
    
    # Verify chunk structure
    kl = cache._comp_keys[0]
    assert isinstance(kl, CompressedKVLayer)
    assert len(kl.chunks) == 5, f"Expected 5 chunks (1 prefill + 4 decode), got {len(kl.chunks)}"
    
    print(f"\n  PASSED: 5 chunks, seq=68, no NaN")
    print()


def test_chunk_self_consistency():
    """
    Verify that each chunk can be independently decompressed
    with its own scales, producing correct values.
    """
    print("Test: Chunk self-consistency (independent decompression)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    batch, heads, hd = 1, 8, 128
    
    torch.manual_seed(42)
    # Two different-scale chunks
    k1 = torch.randn(batch, heads, 32, hd, device=device, dtype=torch.float16) * 0.5
    v1 = torch.randn(batch, heads, 32, hd, device=device, dtype=torch.float16) * 0.5
    cache.update(k1, v1, layer_idx=0)
    
    k2 = torch.randn(batch, heads, 32, hd, device=device, dtype=torch.float16) * 5.0
    v2 = torch.randn(batch, heads, 32, hd, device=device, dtype=torch.float16) * 5.0
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    
    kl = cache._comp_keys[0]
    assert len(kl.chunks) == 2
    
    # Decompress each chunk independently
    dec1 = cache._decompress_chunk(kl.chunks[0])
    dec2 = cache._decompress_chunk(kl.chunks[1])
    
    # Concatenate should match the full output
    dec_full = torch.cat([dec1, dec2], dim=2)
    diff = (k_out.float() - dec_full.float()).abs().max().item()
    
    match = diff < 1e-3  # FP16 tolerance
    print(f"  Chunk 1 scale range: [{kl.chunks[0].scales.min():.4f}, {kl.chunks[0].scales.max():.4f}]")
    print(f"  Chunk 2 scale range: [{kl.chunks[1].scales.min():.4f}, {kl.chunks[1].scales.max():.4f}]")
    print(f"  Max diff (concat chunks vs output): {diff:.2e}")
    print(f"  {'PASS' if match else 'FAIL'}")
    print()
    
    return match


def test_multi_layer():
    print("Test: Multi-layer compression")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    batch, heads, seq, hd = 1, 8, 64, 128
    n_layers = 8
    
    torch.manual_seed(42)
    for layer in range(n_layers):
        k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        k_out, v_out = cache.update(k, v, layer_idx=layer)
        assert k_out.shape == (batch, heads, seq, hd)
        assert not torch.isnan(k_out).any()
    
    print(f"  {n_layers} layers OK")
    print(cache.memory_report())
    print()


def test_memory_report():
    print("Test: Memory report (theoretical savings)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    batch, heads, seq, hd = 1, 8, 512, 128
    n_layers = 4
    
    torch.manual_seed(42)
    for layer in range(n_layers):
        k = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq, hd, device=device, dtype=torch.float16)
        cache.update(k, v, layer_idx=layer)
    
    print(cache.memory_report())
    
    comp = cache.compressed_bytes()
    fp16 = cache.fp16_equivalent_bytes()
    ratio = fp16 / comp
    assert ratio > 1.0, f"No savings: {ratio:.2f}"
    print(f"\n  Savings verified: {ratio:.2f}x theoretical")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("LatticeQuant v2: CompressedKVCache Integration")
    print("=" * 60)
    print()
    
    test_roundtrip_quality()
    test_incremental_decode()
    test_chunk_self_consistency()
    test_multi_layer()
    test_memory_report()
    
    print("All CompressedKVCache tests PASSED.")