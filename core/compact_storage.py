"""
LatticeQuant v2 — Phase 1A: Compact E₈ Storage
================================================
Pack E₈ lattice points into fixed-length byte representation.

Key idea: E₈ = D₈ ∪ (D₈ + ½), and D₈ has even-sum constraint.
The even-sum constraint means the 8th coordinate's LSB is
determined by the first 7 coordinates → saves 1 bit per vector.

Bit layout (b bits/dim, 8b bits per vector):
  [coset(1) | coord0(b) | coord1(b) | ... | coord6(b) | coord7_upper(b-1)]
  Total: 1 + 7b + (b-1) = 8b bits ✓

Storage: uint8 tensor of shape (N, b) — exactly b bytes per vector.

Theoretical storage savings vs float16:
  b=3: 3 bytes vs 16 bytes = 5.33×
  b=4: 4 bytes vs 16 bytes = 4.00×
  b=5: 5 bytes vs 16 bytes = 3.20×

Contract:
  pack_e8() is STRICT: input must be valid E₈ points with coordinates
  in representable range. Out-of-range or malformed input raises errors.
  The caller is responsible for ensuring inputs are representable.

Dependencies: e8_quantizer.py (encode_e8, compute_scale)
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# ============================================================
# E₈ Validation
# ============================================================

def validate_e8(q: torch.Tensor, atol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate that q contains well-formed E₈ lattice points.
    
    E₈ = D₈ ∪ (D₈ + ½) where D₈ = {x ∈ Z⁸ : Σx even}.
    
    Checks:
      1. All 8 coordinates are consistently integer or half-integer
      2. The shifted integer coordinates have even sum
    
    Args:
        q: (N, 8) tensor of candidate E₈ points
        atol: tolerance for integer/half-integer check
    
    Returns:
        is_valid: (N,) bool tensor — True if valid E₈ point
        coset: (N,) long tensor — 0 for integer coset, 1 for half-integer
    """
    # Check integer coset: all coords within atol of nearest integer
    frac = q - q.round()
    int_residual = frac.abs().max(dim=-1).values
    is_integer = int_residual < atol
    
    # Check half-integer coset: all coords within atol of n + 0.5
    half_frac = (q - 0.5) - (q - 0.5).round()
    half_residual = half_frac.abs().max(dim=-1).values
    is_half = half_residual < atol
    
    # Valid if integer or half-integer (tolerance prevents overlap)
    coset = is_half.long()  # 0 = integer, 1 = half-integer
    is_valid_coset = is_integer | is_half
    
    # Check even-sum constraint on the integer representation
    int_coords = q.clone()
    int_coords[is_half] -= 0.5
    int_coords = int_coords.round().long()
    parity = int_coords.sum(dim=-1) % 2
    is_even_sum = (parity == 0)
    
    is_valid = is_valid_coset & is_even_sum
    
    return is_valid, coset


# ============================================================
# Core Packing: E₈ lattice point → packed bytes (STRICT)
# ============================================================

def pack_e8(q: torch.Tensor, bits_per_dim: int) -> torch.Tensor:
    """
    Pack valid, in-range E₈ lattice points into compact byte representation.
    
    STRICT: raises ValueError if any input is not a valid E₈ point
    or has coordinates outside the representable range [-R, R-1]
    where R = 2^(bits_per_dim - 1).
    
    Args:
        q: (N, 8) tensor of E₈ lattice points
        bits_per_dim: integer bits per dimension (3, 4, or 5)
    
    Returns:
        packed: (N, bits_per_dim) uint8 tensor
    """
    assert bits_per_dim in (3, 4, 5), f"Only integer bit rates 3,4,5 supported, got {bits_per_dim}"
    assert q.shape[-1] == 8, f"Expected 8-dim vectors, got {q.shape[-1]}"
    
    N = q.shape[0]
    device = q.device
    
    # Step 1: Validate E₈
    is_valid, coset = validate_e8(q)
    n_invalid = (~is_valid).sum().item()
    if n_invalid > 0:
        raise ValueError(
            f"{n_invalid}/{N} inputs are not valid E₈ points. "
            f"All coordinates must be consistently integer or half-integer "
            f"with even sum on the integer representation."
        )
    
    # Step 2: Convert to integer representation
    int_coords = q.clone()
    is_half = coset.bool()
    int_coords[is_half] -= 0.5
    int_coords = int_coords.round().long()
    
    # Step 3: Check representable range
    R = 2 ** (bits_per_dim - 1)
    out_of_range = (int_coords < -R) | (int_coords > R - 1)
    n_oor = out_of_range.any(dim=-1).sum().item()
    if n_oor > 0:
        max_abs = int_coords.abs().max().item()
        raise ValueError(
            f"{n_oor}/{N} vectors have coordinates outside [{-R}, {R-1}]. "
            f"Max |coord| = {max_abs}. "
            f"At {bits_per_dim} bits/dim, lattice coordinates must fit in "
            f"{bits_per_dim}-bit signed range. "
            f"Use check_representable() to filter, use more bits, "
            f"or handle non-representable vectors in a separate path."
        )
    
    # Step 4: Convert to unsigned representation
    unsigned = (int_coords + R).long()  # range [0, 2R)
    
    # Step 5: Separate coord7 into upper bits (parity saves LSB)
    coord7_upper = unsigned[:, 7] >> 1
    
    # Step 6: Bit-pack into a single integer per vector
    packed_int = coset.long()
    for i in range(7):
        packed_int = (packed_int << bits_per_dim) | unsigned[:, i]
    packed_int = (packed_int << (bits_per_dim - 1)) | coord7_upper
    
    # Step 7: Convert to byte tensor
    packed_bytes = _int_to_bytes(packed_int, bits_per_dim, device)
    
    return packed_bytes


def unpack_e8(packed_bytes: torch.Tensor, bits_per_dim: int) -> torch.Tensor:
    """
    Unpack compact byte representation back to E₈ lattice points.
    
    Guaranteed to produce valid E₈ points (correct coset + even sum)
    because coord7 LSB is reconstructed from the parity constraint.
    
    Args:
        packed_bytes: (N, bits_per_dim) uint8 tensor
        bits_per_dim: integer bits per dimension (3, 4, or 5)
    
    Returns:
        q: (N, 8) float32 tensor of E₈ lattice points
    """
    assert bits_per_dim in (3, 4, 5)
    assert packed_bytes.dtype == torch.uint8
    assert packed_bytes.ndim == 2
    assert packed_bytes.shape[1] == bits_per_dim, \
        f"Expected {bits_per_dim} bytes per vector, got {packed_bytes.shape[1]}"
    
    device = packed_bytes.device
    packed_int = _bytes_to_int(packed_bytes, device)
    
    # Step 2: Extract fields (reverse order of packing)
    R = 2 ** (bits_per_dim - 1)
    coord_mask = (1 << bits_per_dim) - 1
    upper_mask = (1 << (bits_per_dim - 1)) - 1
    
    # Extract coord7 upper bits
    coord7_upper = (packed_int & upper_mask).long()
    packed_int = packed_int >> (bits_per_dim - 1)
    
    # Extract coords 6..0 (in reverse order)
    unsigned = torch.zeros(packed_bytes.shape[0], 7, dtype=torch.long, device=device)
    for i in range(6, -1, -1):
        unsigned[:, i] = (packed_int & coord_mask).long()
        packed_int = packed_int >> bits_per_dim
    
    # Extract coset bit
    coset = (packed_int & 1).long()
    
    # Step 3: Convert coords 0..6 to signed
    signed_coords = unsigned - R  # (N, 7)
    
    # Step 4: Recover coord7 LSB from even-sum constraint
    # D₈ requires: sum of all 8 integer coords ≡ 0 (mod 2)
    # coord7_signed = coord7_upper * 2 + lsb - R
    # Need: sum(signed[0..6]) + coord7_upper*2 + lsb - R ≡ 0 (mod 2)
    # Since coord7_upper*2 and R=2^(b-1) are both even (b ≥ 2):
    # lsb ≡ sum(signed[0..6]) (mod 2)
    partial_sum = signed_coords.sum(dim=-1)
    lsb = partial_sum % 2
    
    # Reconstruct coord7
    unsigned_7 = coord7_upper * 2 + lsb
    signed_7 = unsigned_7 - R
    
    # Step 5: Assemble all 8 signed coordinates
    int_coords = torch.cat([signed_coords, signed_7.unsqueeze(-1)], dim=-1)
    
    # Step 6: Apply coset offset
    result = int_coords.float()
    is_half = coset.bool()
    result[is_half] += 0.5
    
    return result


# ============================================================
# Representable range query
# ============================================================

def representable_range(bits_per_dim: int) -> Tuple[int, int]:
    """Return the signed integer range representable at given bit rate."""
    R = 2 ** (bits_per_dim - 1)
    return -R, R - 1


def check_representable(q: torch.Tensor, bits_per_dim: int) -> torch.Tensor:
    """
    Check which E₈ points are representable at the given bit rate.
    
    Returns:
        in_range: (N,) bool tensor — True if all coords fit in range
    """
    is_valid, coset = validate_e8(q)
    int_coords = q.clone()
    int_coords[coset.bool()] -= 0.5
    int_coords = int_coords.round().long()
    
    lo, hi = representable_range(bits_per_dim)
    in_range = is_valid & (int_coords >= lo).all(dim=-1) & (int_coords <= hi).all(dim=-1)
    return in_range


# ============================================================
# Byte conversion helpers
# ============================================================

def _int_to_bytes(packed_int: torch.Tensor, num_bytes: int, device: torch.device) -> torch.Tensor:
    """Convert packed integer tensor to uint8 byte tensor, big-endian."""
    N = packed_int.shape[0]
    result = torch.zeros(N, num_bytes, dtype=torch.uint8, device=device)
    temp = packed_int.clone()
    for i in range(num_bytes - 1, -1, -1):
        result[:, i] = (temp & 0xFF).to(torch.uint8)
        temp = temp >> 8
    return result


def _bytes_to_int(byte_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert uint8 byte tensor back to integer tensor, big-endian."""
    result = torch.zeros(byte_tensor.shape[0], dtype=torch.int64, device=device)
    for i in range(byte_tensor.shape[1]):
        result = (result << 8) | byte_tensor[:, i].long()
    return result


# ============================================================
# CompactE8Tensor: metadata + packed data
# ============================================================

@dataclass
class CompactE8Tensor:
    """
    Compact storage for a tensor of E₈-quantized vectors.
    
    Memory accounting is THEORETICAL storage footprint:
    raw bytes of packed data + scale floats. Does not include
    Python object overhead, tensor metadata, or allocator
    fragmentation. Paper-grade VRAM measurement should use
    torch.cuda.memory_allocated() separately.
    """
    packed: torch.Tensor        # (num_vectors, bits_per_dim) uint8
    scales: torch.Tensor        # (num_groups,) float32 — one scale per group
    bits_per_dim: int
    orig_shape: tuple
    group_size: int
    num_vectors: int
    
    @property
    def packed_bytes(self) -> int:
        return self.packed.nelement() * self.packed.element_size()
    
    @property
    def scale_bytes(self) -> int:
        return self.scales.nelement() * self.scales.element_size()
    
    @property
    def total_bytes(self) -> int:
        """Theoretical storage footprint (packed data + scales)."""
        return self.packed_bytes + self.scale_bytes
    
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
            f"CompactE8Tensor (theoretical storage)",
            f"  Vectors: {self.num_vectors:,} x 8-dim",
            f"  Nominal rate: {self.bits_per_dim} bits/dim",
            f"  Effective rate: {self.effective_bits_per_dim:.3f} bits/dim (incl. scales)",
            f"  Packed: {self.packed_bytes:,} B | Scales: {self.scale_bytes:,} B | Total: {self.total_bytes:,} B",
            f"  FP16 equiv: {self.fp16_bytes:,} B | Compression: {self.compression_ratio:.2f}x",
        ]
        return "\n".join(lines)


# ============================================================
# High-level API
# ============================================================

def quantize_and_pack(
    x: torch.Tensor,
    bits_per_dim: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize with E₈ and return packed bytes + representability mask.
    Only representable vectors are packed. Caller decides how to handle OOR.
    
    Returns:
        packed: (M, bits_per_dim) uint8 — only representable vectors
        in_range: (N,) bool — True where vector was packed
    """
    from e8_quantizer import encode_e8
    
    q = encode_e8(x / scale)
    in_range = check_representable(q, bits_per_dim)
    
    if in_range.sum() == 0:
        return torch.zeros(0, bits_per_dim, dtype=torch.uint8, device=x.device), in_range
    
    packed = pack_e8(q[in_range], bits_per_dim)
    return packed, in_range


def unpack_and_dequantize(
    packed: torch.Tensor,
    bits_per_dim: int,
    scale: float,
) -> torch.Tensor:
    """Unpack and dequantize back to float vectors."""
    q = unpack_e8(packed, bits_per_dim)
    return q * scale


# ============================================================
# Tests
# ============================================================

def test_handcrafted():
    """
    Unit tests with handcrafted E₈ points:
    - Integer coset (D₈)
    - Half-integer coset (D₈ + ½)
    - Boundary values at range edges
    - Parity recovery for coord7
    """
    print("Test: Handcrafted E₈ points")
    print("=" * 60)
    
    all_pass = True
    
    for bits in [3, 4, 5]:
        R = 2 ** (bits - 1)
        lo, hi = -R, R - 1
        
        # Build test cases: (label, 8-coord list)
        cases = []
        
        # --- Integer coset ---
        cases.append(("zero", [0,0,0,0,0,0,0,0]))
        cases.append(("two +1s", [1,1,0,0,0,0,0,0]))
        cases.append(("-1,+1", [-1,1,0,0,0,0,0,0]))
        cases.append(("four +1s", [1,1,1,1,0,0,0,0]))
        cases.append(("all +1s (sum=8)", [1,1,1,1,1,1,1,1]))
        cases.append(("c7 odd recovery", [1,0,0,0,0,0,0,1]))  # sum=2 even, c7=1 odd
        cases.append(("c7 even recovery", [2,0,0,0,0,0,0,0]))  # sum=2 even, c7=0 even
        cases.append(("negative pair", [-2,-2,0,0,0,0,0,0]))
        cases.append((f"boundary max [{hi},{hi-2},0..]", [hi, hi-2, 0,0,0,0,0,0]))  # ensure even sum
        cases.append((f"boundary min [{lo},{lo},0..]", [lo, lo, 0,0,0,0,0,0]))
        cases.append((f"mixed extreme [{lo},{hi-1},0..]", [lo, hi-1, 0,0,0,0,0,0]))
        
        # --- Half-integer coset ---
        cases.append(("all +0.5", [0.5]*8))  # shifted sum=0, even
        cases.append(("mixed ±0.5", [0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,-0.5]))
        cases.append(("half +1.5,-0.5", [1.5,-0.5,0.5,0.5,0.5,0.5,0.5,0.5]))  # shifted sum=4
        cases.append(("all -0.5", [-0.5]*8))  # shifted sum=-8, even
        
        # Filter to valid + representable
        test_q = []
        test_labels = []
        for label, coords in cases:
            t = torch.tensor([coords], dtype=torch.float32)
            valid, _ = validate_e8(t)
            if not valid.all():
                continue
            ir = check_representable(t, bits)
            if ir.all():
                test_q.append(coords)
                test_labels.append(label)
        
        if not test_q:
            print(f"  {bits}b: no representable cases in range [{lo},{hi}]")
            continue
        
        q = torch.tensor(test_q, dtype=torch.float32)
        packed = pack_e8(q, bits)
        q_rec = unpack_e8(packed, bits)
        
        diff = (q - q_rec).abs().max().item()
        passed = diff < 1e-6
        
        rec_valid, _ = validate_e8(q_rec)
        e8_ok = rec_valid.all().item()
        
        if not (passed and e8_ok):
            all_pass = False
        
        status = "✓" if (passed and e8_ok) else "✗"
        print(f"  {bits}b: {status} {len(test_q)} cases, max_diff={diff:.2e}, valid_E8={e8_ok}")
        
        if not passed:
            for i in range(len(test_q)):
                d = (q[i] - q_rec[i]).abs().max().item()
                if d > 1e-6:
                    print(f"    FAIL [{test_labels[i]}]: {q[i].tolist()} -> {q_rec[i].tolist()}")
    
    print()
    return all_pass


def test_strict_rejects_bad_input():
    """
    Test that pack_e8 rejects:
    - Non-E₈ points (odd sum, mixed coset)
    - Out-of-range coordinates
    """
    print("Test: Strict rejection of bad input")
    print("=" * 60)
    
    all_pass = True
    bits = 4
    R = 2 ** (bits - 1)
    
    bad_cases = [
        ("odd sum integer", torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])),  # sum=1 odd
        ("mixed coset", torch.tensor([[1.0, 0.5, 0, 0, 0, 0, 0, 0]])),     # int + half
        ("out of range large", torch.tensor([[float(R+5), float(-R-5), 0, 0, 0, 0, 0, 0]])),
        ("boundary +1 over", torch.tensor([[float(R), float(R), 0, 0, 0, 0, 0, 0]])),  # R = hi+1
        ("boundary -1 under", torch.tensor([[float(-R-1), float(-R-1), 0, 0, 0, 0, 0, 0]])),  # lo-1
    ]
    
    for label, q in bad_cases:
        try:
            pack_e8(q, bits)
            print(f"  ✗ [{label}]: should have raised ValueError but didn't")
            all_pass = False
        except ValueError as e:
            print(f"  ✓ [{label}]: correctly rejected")
    
    print()
    return all_pass


def test_roundtrip_gaussian():
    """
    Integration test: Gaussian → encode_e8 → filter representable → pack → unpack.
    Also measures OOR rate = fixed-rate penalty manifestation.
    """
    from e8_quantizer import encode_e8, compute_scale
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Test: Gaussian roundtrip on {device}")
    print("=" * 60)
    
    torch.manual_seed(42)
    all_pass = True
    
    for bits in [3, 4, 5]:
        sigma2 = 1.0
        scale = compute_scale(sigma2, bits)
        x = torch.randn(100_000, 8, device=device)
        q = encode_e8(x / scale)
        
        in_range = check_representable(q, bits)
        n_repr = in_range.sum().item()
        n_oor = (~in_range).sum().item()
        oor_pct = n_oor / len(q) * 100
        
        if n_repr == 0:
            print(f"  {bits}b: no representable vectors (OOR={oor_pct:.1f}%)")
            continue
        
        packed = pack_e8(q[in_range], bits)
        q_rec = unpack_e8(packed, bits)
        
        diff = (q[in_range] - q_rec).abs().max().item()
        exact = diff < 1e-6
        
        rec_valid, _ = validate_e8(q_rec)
        e8_ok = rec_valid.all().item()
        
        if not (exact and e8_ok):
            all_pass = False
        
        lo, hi = representable_range(bits)
        status = "✓ PASS" if (exact and e8_ok) else "✗ FAIL"
        
        print(f"  {bits}b: {status}")
        print(f"    Range: [{lo}, {hi}]")
        print(f"    Representable: {n_repr:,}/{len(q):,} ({n_repr/len(q)*100:.1f}%)")
        print(f"    Out-of-range:  {n_oor:,} ({oor_pct:.1f}%) <- fixed-rate penalty")
        print(f"    Max diff: {diff:.2e} | All valid E8: {e8_ok}")
        print()
    
    return all_pass


def test_oor_vs_bitwidth():
    """
    Measure OOR rate as function of bits/dim at Theorem 2 optimal scale.
    Shows the fixed-rate penalty that motivates Phase 1B (entropy coding).
    """
    from e8_quantizer import encode_e8, compute_scale
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Test: Out-of-range rate vs bit width")
    print("=" * 60)
    print(f"{'bits':>6} | {'Range':>12} | {'OOR rate':>10} | {'Max |coord|':>12}")
    print("-" * 50)
    
    torch.manual_seed(42)
    
    for bits in [3, 4, 5, 6, 7, 8]:
        sigma2 = 1.0
        scale = compute_scale(sigma2, bits)
        x = torch.randn(200_000, 8, device=device)
        q = encode_e8(x / scale)
        
        _, coset = validate_e8(q)
        int_c = q.clone()
        int_c[coset.bool()] -= 0.5
        int_c = int_c.round().long()
        
        max_abs = int_c.abs().max().item()
        R = 2 ** (bits - 1)
        oor = (int_c < -R).any(dim=-1) | (int_c > R - 1).any(dim=-1)
        oor_pct = oor.float().mean().item() * 100
        
        print(f"{bits:>6} | [{-R:>4}, {R-1:>4}] | {oor_pct:>9.2f}% | {max_abs:>12}")
    
    print()
    print("Fixed-rate penalty: at optimal scale, some coordinates exceed the range.")
    print("Phase 1B (entropy coding) removes this constraint entirely.")


def test_memory_accounting():
    """Theoretical memory accounting for Llama-3.1-8B KV cache."""
    print("Test: Memory Accounting (Llama-3.1-8B)")
    print("=" * 60)
    
    seq_len = 2048
    n_heads = 8
    head_dim = 128
    n_layers = 32
    blocks_per_head = head_dim // 8
    
    print(f"  seq={seq_len}, layers={n_layers}, kv_heads={n_heads}, head_dim={head_dim}")
    
    for bits in [3, 4, 5]:
        total_vectors = n_layers * n_heads * seq_len * blocks_per_head
        n_scales = n_layers * n_heads
        
        ct = CompactE8Tensor(
            packed=torch.zeros(total_vectors, bits, dtype=torch.uint8),
            scales=torch.zeros(n_scales, dtype=torch.float32),
            bits_per_dim=bits,
            orig_shape=(n_layers, n_heads, seq_len, head_dim),
            group_size=seq_len * blocks_per_head,
            num_vectors=total_vectors,
        )
        
        kv_comp = ct.total_bytes * 2
        kv_fp16 = ct.fp16_bytes * 2
        
        print(f"  {bits}b: {kv_comp/1024/1024:.1f} MB vs {kv_fp16/1024/1024:.1f} MB FP16 "
              f"= {kv_fp16/kv_comp:.2f}x ({ct.effective_bits_per_dim:.3f} eff bits/dim)")


if __name__ == '__main__':
    print("=" * 60)
    print("LatticeQuant v2 Phase 1A: Compact E₈ Storage")
    print("=" * 60)
    print()
    
    r1 = test_handcrafted()
    r2 = test_strict_rejects_bad_input()
    r3 = test_roundtrip_gaussian()
    test_oor_vs_bitwidth()
    print()
    test_memory_accounting()
    
    print()
    if r1 and r2 and r3:
        print("Phase 1A PASSED: pack/unpack bit-exact for all representable E₈ points.")
        print("Strict mode correctly rejects malformed and out-of-range input.")
        print("Next: Phase 1B (entropy-coded storage) removes fixed-rate OOR penalty.")
    else:
        print("Phase 1A FAILED.")