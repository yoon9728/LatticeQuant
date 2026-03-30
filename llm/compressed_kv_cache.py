"""
LatticeQuant v2 — CompressedKVCache (Layer-wise Decompress)
============================================================
DynamicCache subclass with REAL VRAM savings.

Only ONE layer's FP16 KV is decompressed at a time. When layer i+1's
update() is called, layer i's FP16 is freed. Compressed data for ALL
layers stays resident.

VRAM = compressed all layers + 1 layer FP16
  Llama-3.1-8B seq=2048: ~133 MB + ~8 MB = ~141 MB vs 256 MB baseline

Dependencies:
  - compact_storage.py (pack_e8, unpack_e8, check_representable)
  - e8_quantizer.py (encode_e8, compute_scale)
"""

import torch
import gc
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e8_quantizer import encode_e8, compute_scale
from compact_storage import pack_e8, unpack_e8, check_representable
from transformers.cache_utils import DynamicCache


@dataclass
class CompressedChunk:
    packed: torch.Tensor
    fallback: torch.Tensor
    repr_mask: torch.Tensor
    scales: torch.Tensor
    bits_per_dim: int
    orig_shape: tuple
    blocks_per_head: int
    num_blocks: int
    n_repr: int
    n_oor: int
    repr_indices: torch.Tensor = None
    repr_head_ids: torch.Tensor = None
    repr_cumsum: torch.Tensor = None
    fb_cumsum: torch.Tensor = None


class CompressedKVLayer:
    def __init__(self):
        self.chunks: List[CompressedChunk] = []

    @property
    def is_empty(self):
        return len(self.chunks) == 0

    @property
    def total_seq(self):
        return sum(c.orig_shape[2] for c in self.chunks) if self.chunks else 0

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
            for attr in [c.repr_cumsum, c.fb_cumsum, c.repr_head_ids]:
                if attr is not None:
                    total += attr.nelement() * attr.element_size()
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

    def append_chunk(self, chunk):
        self.chunks.append(chunk)


class UncompressibleKVLayer:
    def __init__(self, tensor):
        self.data = tensor.to(torch.float16)
    @property
    def is_empty(self):
        return False
    @property
    def total_bytes(self):
        return self.data.nelement() * self.data.element_size()
    @property
    def fp16_equivalent_bytes(self):
        return self.total_bytes


_PLACEHOLDER = None
def _get_placeholder(device):
    global _PLACEHOLDER
    if _PLACEHOLDER is None or _PLACEHOLDER.device != device:
        _PLACEHOLDER = torch.zeros(1, 1, 0, 1, dtype=torch.float16, device=device)
    return _PLACEHOLDER


class CompressedKVCache(DynamicCache):
    def __init__(self, bits_per_dim: int = 4):
        super().__init__()
        assert bits_per_dim in (3, 4, 5)
        self.bits = bits_per_dim
        if not hasattr(self, 'key_cache'):
            self.key_cache = []
        if not hasattr(self, 'value_cache'):
            self.value_cache = []
        self._comp_keys: List = []
        self._comp_values: List = []
        self.total_blocks = 0
        self.total_repr = 0
        self.total_oor = 0
        self._last_decompressed_layer = -1

    def _compress_tensor(self, tensor):
            device = tensor.device
            batch, heads, seq, hd = tensor.shape
            if hd % 8 != 0:
                return 'uncompressible', tensor.to(torch.float16)

            t = tensor.float()
            bph = seq * (hd // 8)
            n_heads = batch * heads
            t_heads = t.reshape(n_heads, bph, 8)
            sigma2 = (t_heads ** 2).mean(dim=(1, 2))
            per_head_scale = torch.zeros(n_heads, dtype=torch.float32, device=device)

            all_packed, all_fallback, all_masks = [], [], []
            total_repr, total_oor = 0, 0
            decompressed = torch.zeros_like(t_heads)

            for h in range(n_heads):
                blocks = t_heads[h]
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
                n_r, n_o = in_range.sum().item(), (~in_range).sum().item()
                if n_r > 0:
                    all_packed.append(pack_e8(q[in_range], self.bits))
                if n_o > 0:
                    all_fallback.append((q[~in_range] * scale).half())
                all_masks.append(in_range)
                decompressed[h] = q * scale
                total_repr += n_r
                total_oor += n_o

            self.total_blocks += n_heads * bph
            self.total_repr += total_repr
            self.total_oor += total_oor

            full_mask = torch.cat(all_masks, dim=0)

            chunk = CompressedChunk(
                packed=torch.cat(all_packed, dim=0) if all_packed else
                    torch.zeros(0, self.bits, dtype=torch.uint8, device=device),
                fallback=torch.cat(all_fallback, dim=0) if all_fallback else
                        torch.zeros(0, 8, dtype=torch.float16, device=device),
                repr_mask=full_mask, scales=per_head_scale,
                bits_per_dim=self.bits, orig_shape=tensor.shape,
                blocks_per_head=bph, num_blocks=n_heads * bph,
                n_repr=total_repr, n_oor=total_oor,
                repr_indices=None, repr_head_ids=None,
                repr_cumsum=None, fb_cumsum=None,
            )
            return chunk, decompressed.reshape(tensor.shape).to(torch.float16)

    def _decompress_chunk(self, chunk):
        device = chunk.repr_mask.device
        batch, heads, seq, hd = chunk.orig_shape
        bph = chunk.blocks_per_head
        n_heads = batch * heads
        result = torch.zeros(n_heads, bph, 8, dtype=torch.float32, device=device)
        repr_cursor, fb_cursor = 0, 0
        for h in range(n_heads):
            start, end = h * bph, (h + 1) * bph
            mask_h = chunk.repr_mask[start:end]
            scale = chunk.scales[h].item()
            n_r, n_o = mask_h.sum().item(), bph - mask_h.sum().item()
            if n_r > 0:
                q_h = unpack_e8(chunk.packed[repr_cursor:repr_cursor + n_r], chunk.bits_per_dim)
                result[h, mask_h] = q_h * scale
                repr_cursor += n_r
            if n_o > 0:
                result[h, ~mask_h] = chunk.fallback[fb_cursor:fb_cursor + n_o].float()
                fb_cursor += n_o
        return result.reshape(chunk.orig_shape).to(torch.float16)

    def _free_layer(self, layer_idx):
        if layer_idx < len(self.key_cache):
            dev = self.key_cache[layer_idx].device if self.key_cache[layer_idx] is not None else 'cuda:0'
            self.key_cache[layer_idx] = _get_placeholder(dev)
            self.value_cache[layer_idx] = _get_placeholder(dev)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        device = key_states.device

        # Free previous layer's FP16 to save VRAM
        if self._last_decompressed_layer >= 0 and self._last_decompressed_layer != layer_idx:
            self._free_layer(self._last_decompressed_layer)

        comp_k, k_dec_new = self._compress_tensor(key_states)
        comp_v, v_dec_new = self._compress_tensor(value_states)

        while len(self._comp_keys) <= layer_idx:
            self._comp_keys.append(None)
            self._comp_values.append(None)
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(_get_placeholder(device))
            self.value_cache.append(_get_placeholder(device))

        if comp_k == 'uncompressible':
            self._comp_keys[layer_idx] = UncompressibleKVLayer(key_states)
            self._comp_values[layer_idx] = UncompressibleKVLayer(value_states)
            self.key_cache[layer_idx] = k_dec_new
            self.value_cache[layer_idx] = v_dec_new
            self._last_decompressed_layer = layer_idx
            return k_dec_new, v_dec_new

        if self._comp_keys[layer_idx] is None:
            self._comp_keys[layer_idx] = CompressedKVLayer()
            self._comp_values[layer_idx] = CompressedKVLayer()

        self._comp_keys[layer_idx].append_chunk(comp_k)
        self._comp_values[layer_idx].append_chunk(comp_v)

        kl = self._comp_keys[layer_idx]
        vl = self._comp_values[layer_idx]

        if len(kl.chunks) == 1:
            k_full, v_full = k_dec_new, v_dec_new
        else:
            past_k = [self._decompress_chunk(c) for c in kl.chunks[:-1]]
            past_v = [self._decompress_chunk(c) for c in vl.chunks[:-1]]
            k_full = torch.cat(past_k + [k_dec_new], dim=2)
            v_full = torch.cat(past_v + [v_dec_new], dim=2)

        self.key_cache[layer_idx] = k_full
        self.value_cache[layer_idx] = v_full
        self._last_decompressed_layer = layer_idx
        return k_full, v_full

    def compressed_bytes(self):
        return sum(l.total_bytes for l in self._comp_keys + self._comp_values if l is not None)

    def fp16_equivalent_bytes(self):
        return sum(l.fp16_equivalent_bytes for l in self._comp_keys + self._comp_values if l is not None)

    def decompressed_resident_bytes(self):
        total = 0
        for i in range(len(self.key_cache)):
            for t in [self.key_cache[i], self.value_cache[i]]:
                if t is not None:
                    total += t.nelement() * t.element_size()
        return total

    def actual_vram_bytes(self):
        return self.compressed_bytes() + self.decompressed_resident_bytes()

    def memory_report(self):
        comp = self.compressed_bytes()
        fp16 = self.fp16_equivalent_bytes()
        decomp = self.decompressed_resident_bytes()
        actual = self.actual_vram_bytes()
        ratio_th = fp16 / max(comp, 1)
        ratio_act = fp16 / max(actual, 1)
        repr_pct = self.total_repr / max(self.total_blocks, 1) * 100
        oor_pct = self.total_oor / max(self.total_blocks, 1) * 100
        n_layers = sum(1 for x in self._comp_keys if x is not None)
        lines = [
            f"CompressedKVCache Report (layer-wise decompress)",
            f"  Bits/dim:     {self.bits}",
            f"  Layers:       {n_layers}",
            f"  Blocks:       {self.total_blocks:,}",
            f"  Repr:         {self.total_repr:,} ({repr_pct:.1f}%) packed",
            f"  OOR:          {self.total_oor:,} ({oor_pct:.1f}%) fallback",
            f"  ---",
            f"  Compressed:        {comp:,} B ({comp/1024/1024:.2f} MB)",
            f"  Decompressed now:  {decomp:,} B ({decomp/1024/1024:.2f} MB)",
            f"  Actual VRAM (KV):  {actual:,} B ({actual/1024/1024:.2f} MB)",
            f"  FP16 baseline:     {fp16:,} B ({fp16/1024/1024:.2f} MB)",
            f"  ---",
            f"  Theoretical ratio: {ratio_th:.2f}x",
            f"  Actual ratio:      {ratio_act:.2f}x",
        ]
        return "\n".join(lines)

    def get_seq_length(self, layer_idx=0):
        # Use compressed metadata, not resident FP16 (which may be freed)
        if layer_idx < len(self._comp_keys) and self._comp_keys[layer_idx] is not None:
            layer = self._comp_keys[layer_idx]
            if isinstance(layer, CompressedKVLayer):
                return layer.total_seq
            elif isinstance(layer, UncompressibleKVLayer):
                return layer.data.shape[2]
        return 0


# ============================================================
# Tests
# ============================================================

def test_roundtrip_quality():
    print("Test: Roundtrip quality")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for bits in [3, 4, 5]:
        cache = CompressedKVCache(bits_per_dim=bits)
        torch.manual_seed(42)
        k = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float16)
        k_out, _ = cache.update(k, v, layer_idx=0)
        mse = ((k.float() - k_out.float()) ** 2).mean().item()
        rel = mse / (k.float() ** 2).mean().item()
        assert not torch.isnan(k_out).any()
        print(f"  {bits}b: MSE={mse:.6f} rel={rel:.4f}")
    print()

def test_incremental_decode():
    print("Test: Incremental decode")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    torch.manual_seed(42)
    k = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
    cache.update(k, v, layer_idx=0)
    for _ in range(4):
        k_new = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        v_new = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        k_out, _ = cache.update(k_new, v_new, layer_idx=0)
    assert cache.get_seq_length(0) == 68 and not torch.isnan(k_out).any()
    print(f"  seq=68, no NaN -> PASS\n")

def test_layer_wise_vram():
    print("Test: Layer-wise VRAM (~1 layer decompressed)")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    n_layers = 8
    torch.manual_seed(42)
    for layer in range(n_layers):
        k = torch.randn(1, 8, 256, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 256, 128, device=device, dtype=torch.float16)
        cache.update(k, v, layer_idx=layer)

    decomp = cache.decompressed_resident_bytes()
    one_layer = 1 * 8 * 256 * 128 * 2 * 2
    all_layers = one_layer * n_layers
    assert decomp < all_layers * 0.3
    print(f"  Decompressed: {decomp:,} B vs all-layers {all_layers:,} B")
    print(f"  PASS: ~1/{n_layers} resident\n")
    print(cache.memory_report())
    print()

def test_actual_vram():
    if not torch.cuda.is_available():
        return
    print("Test: resident and peak VRAM comparison")
    print("=" * 60)
    device = 'cuda'
    n_layers = 8

    # Baseline
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    base = DynamicCache()
    if not hasattr(base, 'key_cache'):
        base.key_cache = []; base.value_cache = []
    torch.manual_seed(42)
    for layer in range(n_layers):
        k = torch.randn(1, 8, 512, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 512, 128, device=device, dtype=torch.float16)
        base.update(k, v, layer)
    mem_base = torch.cuda.memory_allocated() - mem0
    peak_base = torch.cuda.max_memory_allocated() - mem0
    del base; torch.cuda.empty_cache(); gc.collect()

    # Compressed
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    comp = CompressedKVCache(bits_per_dim=4)
    torch.manual_seed(42)
    for layer in range(n_layers):
        k = torch.randn(1, 8, 512, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 512, 128, device=device, dtype=torch.float16)
        comp.update(k, v, layer)
    mem_comp = torch.cuda.memory_allocated() - mem0
    peak_comp = torch.cuda.max_memory_allocated() - mem0

    print(f"  Config: {n_layers}L x 8H x 512T x 128D")
    print(f"  Baseline:    resident={mem_base/1e6:.1f}MB  peak={peak_base/1e6:.1f}MB")
    print(f"  Compressed:  resident={mem_comp/1e6:.1f}MB  peak={peak_comp/1e6:.1f}MB")
    print(f"  Resident ratio: {mem_base/max(mem_comp,1):.2f}x")
    print(f"  Peak ratio:     {peak_base/max(peak_comp,1):.2f}x")
    print()
    print(comp.memory_report())
    del comp; torch.cuda.empty_cache()
    print()

def test_multi_layer_decode():
    print("Test: Multi-layer decode step")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    n_layers = 4
    torch.manual_seed(42)
    for layer in range(n_layers):
        cache.update(
            torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16),
            torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16),
            layer_idx=layer)
    for layer in range(n_layers):
        k_out, _ = cache.update(
            torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16),
            torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16),
            layer_idx=layer)
        assert k_out.shape == (1, 8, 65, 128) and not torch.isnan(k_out).any()
    print(f"  4 layers, 64+1=65 tokens -> PASS\n")

def test_rebuild_correctness():
    """Verify prefill prefix is preserved after layer-wise free/rebuild."""
    print("Test: Rebuild correctness (prefill prefix preserved)")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layers = 4
   
    cache_comp = CompressedKVCache(bits_per_dim=4)
    
    torch.manual_seed(42)
    all_k_ref = []
    for layer in range(n_layers):
        k = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
        k_out, v_out = cache_comp.update(k, v, layer_idx=layer)
        all_k_ref.append(k_out.clone())  # save while layer is active

    # Decode step through all layers
    decode_refs = []
    for layer in range(n_layers):
        k_new = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        v_new = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        k_out, _ = cache_comp.update(k_new, v_new, layer_idx=layer)
        decode_refs.append(k_out.clone())

    # Verify: decode output should be prefill + new token concatenated
    for layer in range(n_layers):
        expected_seq = 65
        actual_seq = decode_refs[layer].shape[2]
        assert actual_seq == expected_seq, f"Layer {layer}: seq {actual_seq} != {expected_seq}"
        
        # First 64 tokens should match prefill reference
        prefill_part = decode_refs[layer][:, :, :64, :]
        diff = (prefill_part.float() - all_k_ref[layer].float()).abs().max().item()
        assert diff < 1e-3, f"Layer {layer}: prefill mismatch {diff}"
        
        print(f"  Layer {layer}: seq={actual_seq}, prefill diff={diff:.2e} -> OK")
    
    print("  PASS: all layers match reference\n")

if __name__ == '__main__':
    print("=" * 60)
    print("LatticeQuant v2: CompressedKVCache (Layer-wise Decompress)")
    print("=" * 60)
    print()
    test_roundtrip_quality()
    test_incremental_decode()
    test_layer_wise_vram()
    test_actual_vram()
    test_multi_layer_decode()
    test_rebuild_correctness()
    print("All tests PASSED.")    
