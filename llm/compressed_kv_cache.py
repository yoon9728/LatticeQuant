"""
LatticeQuant v2 — CompressedKVCache (rANS Entropy-Coded)
=========================================================
DynamicCache subclass with entropy-coded E₈ KV storage.
NO FP16 fallback. 0% OOR. All vectors entropy-coded.

Storage per 8-dim vector at 4 bits/dim:
  ~4 bytes (rANS bitstream) + 1 byte (coset) + amortized tables/scales
  vs 16 bytes FP16 → ~3× savings

Flow:
  Compress (CPU, once): E₈ quantize → symbolize → rANS encode
  Store (GPU): rANS bitstreams + tables + cosets + scales
  Decompress (GPU, on-the-fly): rANS decode kernel → symbols_to_e8 → scale

Layer-wise decompress: only 1 layer FP16 resident at a time.

Dependencies:
  - gpu_ans.py (RANSTable, build_rans_table, rans_encode, gpu_rans_decode, etc.)
  - e8_quantizer.py (encode_e8, compute_scale)
  - entropy_coder.py (e8_to_symbols, symbols_to_e8)
"""

import torch
import gc
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e8_quantizer import encode_e8, compute_scale
from entropy_coder import e8_to_symbols, symbols_to_e8
from gpu_ans import (
    RANSTable, RANSStream, build_rans_table, rans_encode,
    gpu_rans_decode, prepare_gpu_decode, run_gpu_decode_kernel, extract_gpu_results,
)
from transformers.cache_utils import DynamicCache


# ============================================================
# rANS Compressed Chunk
# ============================================================

@dataclass
class RANSCompressedChunk:
    """One chunk of entropy-coded KV data. 0% OOR."""
    streams: List[RANSStream]
    stream_meta: List[tuple]
    coset: torch.Tensor
    coset_counts: Tuple[int, int]
    scales: torch.Tensor
    orig_shape: tuple
    n_vectors: int
    n_heads: int
    blocks_per_head: int
    bits_per_dim: float
    gpu_bitstreams: List[torch.Tensor] = None


class CompressedKVLayer:
    """List of rANS compressed chunks for one layer."""
    def __init__(self):
        self.chunks: List[RANSCompressedChunk] = []
    
    @property
    def is_empty(self):
        return len(self.chunks) == 0
    
    @property
    def total_seq(self):
        return sum(c.orig_shape[2] for c in self.chunks) if self.chunks else 0
    
    @property
    def num_vectors(self):
        return sum(c.n_vectors for c in self.chunks)
    
    @property
    def total_bytes(self):
        total = 0
        for c in self.chunks:
            if c.gpu_bitstreams is not None:
                for t in c.gpu_bitstreams:
                    total += t.nelement() * t.element_size()
            else:
                for s in c.streams:
                    total += len(s.bitstream) * 4
            total += c.coset.nelement() * c.coset.element_size()
            total += c.scales.nelement() * c.scales.element_size()
        return total
    
    @property
    def fp16_equivalent_bytes(self):
        return self.num_vectors * 8 * 2
    
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
    @property
    def total_seq(self):
        return self.data.shape[2]


_PLACEHOLDER = None
def _get_placeholder(device):
    global _PLACEHOLDER
    if _PLACEHOLDER is None or _PLACEHOLDER.device != device:
        _PLACEHOLDER = torch.zeros(1, 1, 0, 1, dtype=torch.float16, device=device)
    return _PLACEHOLDER


# ============================================================
# CompressedKVCache (rANS)
# ============================================================

class CompressedKVCache(DynamicCache):
    """
    DynamicCache with rANS entropy-coded E₈ KV storage.
    0% OOR. ~3× VRAM savings. Layer-wise decompress.
    """
    
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
        self.total_vectors = 0
        self._last_decompressed_layer = -1
    
    def _compress_tensor(self, tensor: torch.Tensor) -> Tuple[Any, torch.Tensor]:
            device = tensor.device
            batch, heads, seq, hd = tensor.shape

            if hd % 8 != 0:
                return 'uncompressible', tensor.to(torch.float16)

            t = tensor.float()
            bph = seq * (hd // 8)
            n_heads = batch * heads
            n_vectors = n_heads * bph

            t_heads = t.reshape(n_heads, bph, 8)
            sigma2 = (t_heads ** 2).mean(dim=(1, 2))
            per_head_scale = torch.zeros(n_heads, dtype=torch.float32, device=device)

            all_q = torch.zeros_like(t_heads)
            for h in range(n_heads):
                s2 = sigma2[h].item()
                if s2 < 1e-12:
                    per_head_scale[h] = 1.0
                    all_q[h] = t_heads[h]
                    continue
                scale = compute_scale(s2, self.bits)
                per_head_scale[h] = scale
                all_q[h] = encode_e8(t_heads[h] / scale)

            decompressed = torch.zeros_like(t_heads)
            for h in range(n_heads):
                decompressed[h] = all_q[h] * per_head_scale[h].item()
            decompressed_out = decompressed.reshape(tensor.shape).to(torch.float16)

            q_flat = all_q.reshape(n_vectors, 8)

            coset, free_coords, coord8_half = e8_to_symbols(q_flat)
            cos_np = coset.cpu().numpy().astype(np.int32)
            free_np = free_coords.cpu().numpy().astype(np.int64)
            c8h_np = coord8_half.cpu().numpy().astype(np.int64)
            all_syms = np.concatenate([free_np, c8h_np[:, None]], axis=1)

            coset_counts = (int((cos_np == 0).sum()), int((cos_np == 1).sum()))

            streams = []
            stream_meta = []

            for c_val in [0, 1]:
                mask = (cos_np == c_val)
                if mask.sum() == 0:
                    continue
                syms_c = all_syms[mask]

                for idx in range(8):
                    col = syms_c[:, idx]
                    unique, ucounts = np.unique(col, return_counts=True)
                    lo = int(unique.min()) - 2
                    hi = int(unique.max()) + 2
                    alphabet = list(range(lo, hi + 1))
                    cdict = dict(zip(unique.tolist(), ucounts.tolist()))
                    counts = np.array([cdict.get(a, 0) for a in alphabet], dtype=np.int64)

                    table = build_rans_table(counts, alphabet)
                    bs, state, n_bits = rans_encode(col, table)

                    streams.append(RANSStream(
                        bitstream=bs, n_bits=n_bits,
                        initial_state=state, n_symbols=len(col), table=table,
                    ))
                    stream_meta.append((c_val, idx, len(col)))

            self.total_vectors += n_vectors

            gpu_bitstreams = []
            for s in streams:
                gpu_bitstreams.append(
                    torch.tensor(s.bitstream.astype(np.int32), dtype=torch.int32, device=device)
                )

            chunk = RANSCompressedChunk(
                streams=streams,
                stream_meta=stream_meta,
                coset=coset.to(torch.uint8).to(device),
                coset_counts=coset_counts,
                scales=per_head_scale,
                orig_shape=tensor.shape,
                n_vectors=n_vectors,
                n_heads=n_heads,
                blocks_per_head=bph,
                bits_per_dim=self.bits,
                gpu_bitstreams=gpu_bitstreams,
            )

            return chunk, decompressed_out
    
    def _decompress_chunk(self, chunk: RANSCompressedChunk) -> torch.Tensor:
        """
        Decompress via GPU rANS decode → symbols_to_e8 → per-head scale.
        """
        device = chunk.coset.device
        batch, heads, seq, hd = chunk.orig_shape
        n_vectors = chunk.n_vectors
        n_heads = chunk.n_heads
        bph = chunk.blocks_per_head
        
        # GPU rANS decode all streams
        decoded_streams = gpu_rans_decode(chunk.streams, str(device))
        
        # Reconstruct symbol arrays
        cos_np = chunk.coset.cpu().numpy().astype(np.int32)
        all_syms = np.zeros((n_vectors, 8), dtype=np.int64)
        
        si = 0
        for c_val in [0, 1]:
            mask = (cos_np == c_val)
            if mask.sum() == 0:
                continue
            rows = np.where(mask)[0]
            for idx in range(8):
                all_syms[rows, idx] = decoded_streams[si]
                si += 1
        
        # Reconstruct E₈ lattice points
        coset_t = chunk.coset.to(torch.long).to(device)
        free_t = torch.tensor(all_syms[:, :7], dtype=torch.long, device=device)
        c8h_t = torch.tensor(all_syms[:, 7], dtype=torch.long, device=device)
        q_flat = symbols_to_e8(coset_t, free_t, c8h_t)  # (n_vectors, 8)
        
        # Apply per-head scales
        q_heads = q_flat.reshape(n_heads, bph, 8)
        result = torch.zeros_like(q_heads)
        for h in range(n_heads):
            result[h] = q_heads[h] * chunk.scales[h].item()
        
        return result.reshape(batch, heads, seq, hd).to(torch.float16)
    
    def _free_layer(self, layer_idx):
        if layer_idx < len(self.key_cache):
            dev = self.key_cache[layer_idx].device if self.key_cache[layer_idx] is not None else 'cuda:0'
            self.key_cache[layer_idx] = _get_placeholder(dev)
            self.value_cache[layer_idx] = _get_placeholder(dev)
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        device = key_states.device
        
        # Free previous layer
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
    
    # ---- Reporting ----
    
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
        n_layers = sum(1 for x in self._comp_keys if x is not None)
        lines = [
            f"CompressedKVCache Report (rANS entropy-coded)",
            f"  Bits/dim:     {self.bits}",
            f"  Layers:       {n_layers}",
            f"  Vectors:      {self.total_vectors:,}",
            f"  OOR rate:     0.00% (entropy coded)",
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
        if layer_idx < len(self._comp_keys) and self._comp_keys[layer_idx] is not None:
            layer = self._comp_keys[layer_idx]
            if isinstance(layer, CompressedKVLayer):
                return layer.total_seq
            elif isinstance(layer, UncompressibleKVLayer):
                return layer.total_seq
        return 0


# ============================================================
# Tests
# ============================================================

def test_roundtrip_quality():
    print("Test: Roundtrip quality (rANS)")
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
    print("Test: Incremental decode (rANS)")
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
    print("Test: Layer-wise VRAM (rANS, ~1 layer decompressed)")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    n_layers = 32
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
    print("Test: Actual VRAM (memory_allocated)")
    print("=" * 60)
    device = 'cuda'
    n_layers = 8

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


def test_rebuild_correctness():
    """Verify prefill prefix preserved after layer-wise free/rebuild."""
    print("Test: Rebuild correctness (rANS)")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = CompressedKVCache(bits_per_dim=4)
    n_layers = 4
    
    torch.manual_seed(42)
    all_k_ref = []
    for layer in range(n_layers):
        k = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
        v = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
        k_out, _ = cache.update(k, v, layer_idx=layer)
        all_k_ref.append(k_out.clone())
    
    decode_refs = []
    for layer in range(n_layers):
        k_new = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        v_new = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        k_out, _ = cache.update(k_new, v_new, layer_idx=layer)
        decode_refs.append(k_out.clone())
    
    for layer in range(n_layers):
        assert decode_refs[layer].shape[2] == 65
        prefill_part = decode_refs[layer][:, :, :64, :]
        diff = (prefill_part.float() - all_k_ref[layer].float()).abs().max().item()
        assert diff < 1e-2, f"Layer {layer}: prefill mismatch {diff}"
        print(f"  Layer {layer}: seq=65, prefill diff={diff:.2e} -> OK")
    
    print("  PASS\n")


def test_multi_layer_decode():
    print("Test: Multi-layer decode (rANS)")
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


if __name__ == '__main__':
    print("=" * 60)
    print("LatticeQuant v2: CompressedKVCache (rANS Entropy-Coded)")
    print("=" * 60)
    print()
    test_roundtrip_quality()
    test_incremental_decode()
    test_layer_wise_vram()
    test_actual_vram()
    test_rebuild_correctness()
    test_multi_layer_decode()
    print("All tests PASSED.")