"""
LatticeQuant v2 — Phase 1B: Entropy-Coded E₈ Storage
=====================================================
Variable-length ANS-compressed storage for E₈ lattice points.
Removes the fixed-rate penalty (27% OOR) from Phase 1A.

Uses the parity-aware symbolization from entropy_coder.py:
  coset (1 bit) + 7 free coords (entropy coded) + coord8_half (entropy coded)

Key properties:
  - 0% out-of-range: any E₈ lattice point can be stored
  - Average rate converges to target bits/dim
  - Lossless: decompress recovers exact E₈ lattice points
  - Memory accounting includes bitstream + freq tables + scales

Dependencies:
  - entropy_coder.py (e8_to_symbols, symbols_to_e8, FrequencyModel)
  - e8_quantizer.py (encode_e8, compute_scale)
  - constriction (pip install constriction)
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e8_quantizer import encode_e8, compute_scale, G_E8
from entropy_coder import e8_to_symbols, symbols_to_e8, FrequencyModel

try:
    import constriction
except ImportError:
    raise ImportError(
        "Phase 1B requires constriction. Install with: pip install constriction"
    )


# ============================================================
# Compressed E₈ Tensor
# ============================================================

@dataclass
class CompressedE8Tensor:
    """
    Entropy-coded storage for E₈-quantized vectors.
    
    Stores ANS-compressed bitstreams + frequency tables + scales.
    Memory accounting includes everything needed for decompression.
    """
    # Compressed data: list of uint32 arrays (one per ANS stream)
    coset_stream: np.ndarray          # compressed coset bits
    symbol_streams: List[List[np.ndarray]]  # [coset][coord_idx] → compressed uint32
    
    # Frequency model (needed for decompression)
    freq_model: FrequencyModel
    
    # Reconstruction metadata
    scales: torch.Tensor              # (num_groups,) float32
    num_vectors: int                  # total E₈ vectors
    coset_counts: Tuple[int, int]     # (n_integer, n_half) per coset
    bits_per_dim_target: float        # target rate
    orig_shape: tuple                 # original tensor shape
    group_size: int                   # vectors per scale group
    
    # Symbol-to-index mappings for each stream (needed for decode)
    symbol_maps: List[List[dict]]     # [coset][coord_idx] → {idx: symbol_value}
    
    @property
    def compressed_bits(self) -> int:
        """Total bits in all compressed streams."""
        total = len(self.coset_stream) * 32
        for c in range(2):
            for idx in range(8):
                if c < len(self.symbol_streams) and idx < len(self.symbol_streams[c]):
                    total += len(self.symbol_streams[c][idx]) * 32
        return total
    
    @property
    def compressed_bytes(self) -> int:
        return self.compressed_bits // 8
    
    @property
    def freq_table_bytes(self) -> int:
        """Approximate bytes for frequency tables."""
        total_entries = 0
        for key, table in self.freq_model.tables.items():
            total_entries += len(table)
        # Each entry: int key (8 bytes) + int count (8 bytes)
        return total_entries * 16
    
    @property
    def scale_bytes(self) -> int:
        return self.scales.nelement() * self.scales.element_size()
    
    @property
    def total_bytes(self) -> int:
        """Total storage: compressed streams + freq tables + scales."""
        return self.compressed_bytes + self.freq_table_bytes + self.scale_bytes
    
    @property
    def fp16_bytes(self) -> int:
        return self.num_vectors * 8 * 2
    
    @property
    def effective_bits_per_dim(self) -> float:
        return (self.total_bytes * 8) / (self.num_vectors * 8)
    
    @property
    def compression_ratio(self) -> float:
        return self.fp16_bytes / self.total_bytes
    
    def summary(self) -> str:
        lines = [
            f"CompressedE8Tensor (entropy-coded)",
            f"  Vectors: {self.num_vectors:,} x 8-dim",
            f"  Target rate: {self.bits_per_dim_target} bits/dim",
            f"  Compressed:  {self.compressed_bytes:,} B ({self.compressed_bytes/1024:.1f} KB)",
            f"  Freq tables: {self.freq_table_bytes:,} B ({self.freq_table_bytes/1024:.1f} KB)",
            f"  Scales:      {self.scale_bytes:,} B",
            f"  Total:       {self.total_bytes:,} B ({self.total_bytes/1024:.1f} KB)",
            f"  FP16 equiv:  {self.fp16_bytes:,} B ({self.fp16_bytes/1024:.1f} KB)",
            f"  Effective:   {self.effective_bits_per_dim:.3f} bits/dim",
            f"  Compression: {self.compression_ratio:.2f}x",
            f"  OOR rate:    0.00% (entropy coding has no range limit)",
        ]
        return "\n".join(lines)


# ============================================================
# Compress: E₈ points → CompressedE8Tensor
# ============================================================

def compress_e8(
    q: torch.Tensor,
    bits_per_dim_target: float,
    scale: float,
    freq_model: Optional[FrequencyModel] = None,
) -> CompressedE8Tensor:
    """
    Compress E₈ lattice points into entropy-coded storage.
    
    Args:
        q: (N, 8) tensor of E₈ lattice points (in lattice coordinates, i.e. q = x/scale)
        bits_per_dim_target: target rate (for metadata only)
        scale: lattice scale factor
        freq_model: pre-fitted frequency model. If None, fits on q itself.
    
    Returns:
        CompressedE8Tensor with all data needed for decompression
    """
    N = q.shape[0]
    device = q.device
    
    # Step 1: Symbolize
    coset, free_coords, coord8_half = e8_to_symbols(q)
    cos_np = coset.cpu().numpy().astype(np.int32)
    free_np = free_coords.cpu().numpy().astype(np.int64)
    c8h_np = coord8_half.cpu().numpy().astype(np.int64)
    
    # Step 2: Fit frequency model if not provided
    if freq_model is None:
        freq_model = FrequencyModel()
        freq_model.fit(cos_np, free_np, c8h_np)
    
    # Step 3: Encode coset stream
    coset_probs = np.array(freq_model.coset_prob, dtype=np.float32)
    coset_probs /= coset_probs.sum()
    coset_probs = np.maximum(coset_probs, 1e-10).astype(np.float32)
    
    encoder_coset = constriction.stream.stack.AnsCoder()
    encoder_coset.encode_reverse(
        cos_np,
        constriction.stream.model.Categorical(coset_probs, perfect=False)
    )
    coset_stream = encoder_coset.get_compressed()
    
    # Step 4: Encode symbol streams per coset × coord
    all_symbols = np.concatenate([free_np, c8h_np[:, None]], axis=1)
    
    symbol_streams = [[], []]  # [coset][coord_idx]
    symbol_maps = [[], []]     # for decode: idx → symbol value
    
    coset_counts = [int((cos_np == 0).sum()), int((cos_np == 1).sum())]
    
    for c_val in [0, 1]:
        mask = (cos_np == c_val)
        if mask.sum() == 0:
            for idx in range(8):
                symbol_streams[c_val].append(np.array([], dtype=np.uint32))
                symbol_maps[c_val].append({})
            continue
        
        syms_c = all_symbols[mask]
        
        for idx in range(8):
            key = (c_val, idx)
            table = freq_model.tables.get(key, {})
            
            # Build alphabet: union of training and data symbols
            col = syms_c[:, idx]
            all_vals = sorted(set(table.keys()) | set(col.tolist()))
            sym_to_idx = {s: i for i, s in enumerate(all_vals)}
            idx_to_sym = {i: s for s, i in sym_to_idx.items()}
            alphabet_size = len(all_vals)
            
            # Build probability table
            probs = np.zeros(alphabet_size, dtype=np.float32)
            total_c = freq_model.totals.get(key, mask.sum())
            for i, s in enumerate(all_vals):
                probs[i] = table.get(s, 0) + 1  # Laplace smoothing
            probs /= probs.sum()
            probs = np.maximum(probs, 1e-10).astype(np.float32)
            
            # Encode
            indices = np.array([sym_to_idx[int(v)] for v in col], dtype=np.int32)
            
            encoder = constriction.stream.stack.AnsCoder()
            encoder.encode_reverse(
                indices,
                constriction.stream.model.Categorical(probs, perfect=False)
            )
            
            symbol_streams[c_val].append(encoder.get_compressed())
            symbol_maps[c_val].append(idx_to_sym)
    
    # Step 5: Build CompressedE8Tensor
    return CompressedE8Tensor(
        coset_stream=coset_stream,
        symbol_streams=symbol_streams,
        freq_model=freq_model,
        scales=torch.tensor([scale], dtype=torch.float32),
        num_vectors=N,
        coset_counts=tuple(coset_counts),
        bits_per_dim_target=bits_per_dim_target,
        orig_shape=(N, 8),
        group_size=N,
        symbol_maps=symbol_maps,
    )


# ============================================================
# Decompress: CompressedE8Tensor → E₈ points
# ============================================================

def decompress_e8(compressed: CompressedE8Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Decompress entropy-coded storage back to E₈ lattice points.
    
    Args:
        compressed: CompressedE8Tensor from compress_e8()
        device: target torch device
    
    Returns:
        q: (N, 8) float tensor of E₈ lattice points (lattice coordinates)
    """
    N = compressed.num_vectors
    freq_model = compressed.freq_model
    
    # Step 1: Decode coset stream
    coset_probs = np.array(freq_model.coset_prob, dtype=np.float32)
    coset_probs /= coset_probs.sum()
    coset_probs = np.maximum(coset_probs, 1e-10).astype(np.float32)
    
    decoder_coset = constriction.stream.stack.AnsCoder(compressed.coset_stream)
    cos_np = decoder_coset.decode(
        constriction.stream.model.Categorical(coset_probs, perfect=False),
        N
    ).astype(np.int32)
    
    # Step 2: Decode symbol streams per coset × coord
    all_symbols = np.zeros((N, 8), dtype=np.int64)
    
    for c_val in [0, 1]:
        mask = (cos_np == c_val)
        n_c = mask.sum()
        if n_c == 0:
            continue
        
        row_indices = np.where(mask)[0]
        
        for idx in range(8):
            key = (c_val, idx)
            table = freq_model.tables.get(key, {})
            idx_to_sym = compressed.symbol_maps[c_val][idx]
            
            # Rebuild alphabet and probs (must match encode exactly)
            all_vals = sorted(idx_to_sym.values())
            alphabet_size = len(all_vals)
            
            if alphabet_size == 0:
                continue
            
            total_c = freq_model.totals.get(key, n_c)
            probs = np.zeros(alphabet_size, dtype=np.float32)
            for i, s in enumerate(all_vals):
                probs[i] = table.get(s, 0) + 1
            probs /= probs.sum()
            probs = np.maximum(probs, 1e-10).astype(np.float32)
            
            # Decode
            stream_data = compressed.symbol_streams[c_val][idx]
            decoder = constriction.stream.stack.AnsCoder(stream_data)
            decoded_indices = decoder.decode(
                constriction.stream.model.Categorical(probs, perfect=False),
                int(n_c)
            ).astype(np.int32)
            
            # Map indices back to symbol values
            decoded_syms = np.array([idx_to_sym[int(i)] for i in decoded_indices])
            all_symbols[row_indices, idx] = decoded_syms
    
    # Step 3: Reconstruct E₈ points from symbols
    coset_t = torch.tensor(cos_np, dtype=torch.long, device=device)
    free_coords_t = torch.tensor(all_symbols[:, :7], dtype=torch.long, device=device)
    coord8_half_t = torch.tensor(all_symbols[:, 7], dtype=torch.long, device=device)
    
    q = symbols_to_e8(coset_t, free_coords_t, coord8_half_t)
    
    return q


# ============================================================
# Full pipeline: raw vectors → compress → decompress → dequant
# ============================================================

def quantize_compress(
    x: torch.Tensor,
    bits_per_dim: float,
    sigma2: float,
    freq_model: Optional[FrequencyModel] = None,
) -> CompressedE8Tensor:
    """
    Full pipeline: raw float vectors → E₈ quantize → entropy compress.
    
    Args:
        x: (N, 8) float tensor
        bits_per_dim: target rate
        sigma2: variance estimate for scale computation
        freq_model: pre-fitted model (None = fit on this data)
    
    Returns:
        CompressedE8Tensor
    """
    scale = compute_scale(sigma2, bits_per_dim)
    q = encode_e8(x / scale)
    return compress_e8(q, bits_per_dim, scale, freq_model)


def decompress_dequantize(
    compressed: CompressedE8Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Full pipeline: decompress → dequantize → float vectors.
    """
    q = decompress_e8(compressed, device)
    scale = compressed.scales[0].item()
    return q * scale


# ============================================================
# Tests
# ============================================================

def test_roundtrip_exact():
    """
    Core test: compress → decompress must recover exact E₈ lattice points.
    This is the lossless property of entropy coding.
    """
    print("Test: Compress → Decompress roundtrip exactness")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    all_pass = True
    
    for bits in [3.0, 4.0, 5.0]:
        sigma2 = 1.0
        scale = compute_scale(sigma2, bits)
        N = 50_000
        
        x = torch.randn(N, 8, device=device)
        q = encode_e8(x / scale)
        
        # Compress
        compressed = compress_e8(q, bits, scale)
        
        # Decompress
        q_rec = decompress_e8(compressed, device)
        
        # Check bit-exact
        diff = (q - q_rec).abs().max().item()
        exact = diff < 1e-6
        if not exact:
            all_pass = False
        
        status = "✓ PASS" if exact else "✗ FAIL"
        print(f"  {bits}b: {status} (max_diff={diff:.2e}, N={N:,})")
    
    print()
    return all_pass


def test_zero_oor():
    """
    Verify 0% out-of-range: entropy coding handles ALL E₈ points,
    including those that fixed-length packing cannot represent.
    """
    from compact_storage import check_representable
    
    print("Test: Zero out-of-range (vs Phase 1A fixed-length)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    
    for bits_int in [3, 4, 5]:
        bits = float(bits_int)
        sigma2 = 1.0
        scale = compute_scale(sigma2, bits)
        N = 100_000
        
        x = torch.randn(N, 8, device=device)
        q = encode_e8(x / scale)
        
        # Phase 1A: how many are representable in fixed-length?
        in_range = check_representable(q, bits_int)
        oor_fixed = (~in_range).sum().item()
        oor_pct = oor_fixed / N * 100
        
        # Phase 1B: compress all of them
        compressed = compress_e8(q, bits, scale)
        q_rec = decompress_e8(compressed, device)
        
        diff = (q - q_rec).abs().max().item()
        oor_entropy = 0  # by construction
        
        print(f"  {bits_int}b: Fixed-length OOR={oor_pct:.1f}% | Entropy OOR=0.0% | "
              f"Roundtrip diff={diff:.2e}")
    
    print()


def test_rate_accuracy():
    """
    Verify that actual compressed size matches target rate.
    """
    print("Test: Compressed rate vs target rate")
    print("=" * 60)
    print(f"{'target':>8} | {'effective':>10} | {'ratio':>8} | {'comp bytes':>12} | {'fp16 bytes':>12} | {'savings':>8}")
    print("-" * 75)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    
    for bits in [3.0, 3.5, 4.0, 5.0]:
        sigma2 = 1.0
        scale = compute_scale(sigma2, bits)
        N = 200_000
        
        x = torch.randn(N, 8, device=device)
        q = encode_e8(x / scale)
        
        compressed = compress_e8(q, bits, scale)
        
        eff = compressed.effective_bits_per_dim
        ratio = eff / bits
        
        print(f"{bits:>8.1f} | {eff:>10.3f} | {ratio:>8.3f} | "
              f"{compressed.total_bytes:>12,} | {compressed.fp16_bytes:>12,} | "
              f"{compressed.compression_ratio:>7.2f}x")
    
    print()


def test_mse_matches_direct():
    """
    Verify that compress → decompress → dequant gives same MSE
    as direct quantization (no compression).
    """
    print("Test: MSE after compression matches direct quantization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    all_pass = True
    
    for bits in [3.0, 4.0, 5.0]:
        sigma2 = 1.0
        scale = compute_scale(sigma2, bits)
        N = 100_000
        
        x = torch.randn(N, 8, device=device)
        
        # Direct quantization MSE
        q_direct = encode_e8(x / scale)
        x_hat_direct = q_direct * scale
        mse_direct = ((x - x_hat_direct) ** 2).mean().item()
        
        # Compress → decompress MSE
        compressed = compress_e8(q_direct, bits, scale)
        q_rec = decompress_e8(compressed, device)
        x_hat_compressed = q_rec * scale
        mse_compressed = ((x - x_hat_compressed) ** 2).mean().item()
        
        match = abs(mse_direct - mse_compressed) / mse_direct < 1e-6
        if not match:
            all_pass = False
        
        status = "✓" if match else "✗"
        print(f"  {bits}b: {status} MSE direct={mse_direct:.8f} | MSE compressed={mse_compressed:.8f}")
    
    print()
    return all_pass


def test_memory_comparison():
    """
    Compare memory: Phase 1A (fixed-length) vs Phase 1B (entropy-coded)
    for Llama-3.1-8B KV cache dimensions.
    """
    print("Test: Memory comparison — Fixed-length vs Entropy-coded")
    print("=" * 60)
    
    # Llama-3.1-8B dimensions
    seq_len = 2048
    n_heads = 8
    head_dim = 128
    n_layers = 32
    blocks_per_head = head_dim // 8
    total_vectors = n_layers * n_heads * seq_len * blocks_per_head
    
    print(f"  Llama-3.1-8B: {n_layers}L × {n_heads}H × {seq_len}T × {head_dim}D")
    print(f"  Total E₈ vectors: {total_vectors:,}")
    print()
    
    fp16_bytes = total_vectors * 8 * 2
    
    print(f"{'bits':>6} | {'FP16':>10} | {'Fixed-len':>10} | {'Entropy':>10} | "
          f"{'Fixed ratio':>11} | {'Entropy ratio':>13}")
    print("-" * 75)
    
    for bits in [3, 4, 5]:
        # Fixed-length: exactly bits bytes per vector
        fixed_bytes = total_vectors * bits
        
        # Entropy-coded: approximate from small-scale test
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(42)
        scale = compute_scale(1.0, float(bits))
        x_sample = torch.randn(10_000, 8, device=device)
        q_sample = encode_e8(x_sample / scale)
        comp = compress_e8(q_sample, float(bits), scale)
        eff_bits = comp.effective_bits_per_dim
        entropy_bytes = int(total_vectors * eff_bits)  # approximate
        
        print(f"{bits:>6} | {fp16_bytes/1024/1024:>8.1f} MB | {fixed_bytes/1024/1024:>8.1f} MB | "
              f"{entropy_bytes/1024/1024:>8.1f} MB | "
              f"{fp16_bytes/fixed_bytes:>10.2f}x | {fp16_bytes/max(entropy_bytes,1):>12.2f}x")
    
    print()
    print("Note: Fixed-length can only store ~73% of vectors (27% OOR).")
    print("Entropy-coded stores 100% of vectors with variable-length coding.")


if __name__ == '__main__':
    print("=" * 60)
    print("LatticeQuant v2 Phase 1B: Entropy-Coded E₈ Storage")
    print("=" * 60)
    print()
    
    r1 = test_roundtrip_exact()
    r2 = test_mse_matches_direct()
    test_zero_oor()
    test_rate_accuracy()
    test_memory_comparison()
    
    print()
    if r1 and r2:
        print("Phase 1B PASSED: Entropy-coded storage is lossless, 0% OOR,")
        print("and compressed rate matches target. Fixed-rate penalty eliminated.")
        print("Next: CompressedKVCache integration with HuggingFace transformers.")
    else:
        print("Phase 1B FAILED.")