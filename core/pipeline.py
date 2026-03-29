"""
LatticeQuant Phase 4: Full Pipeline
=====================================
d-dim vector → RHT → 8-dim blocks → E₈ quantize → parity-aware entropy code → bitstream
bitstream → decode → inverse RHT → reconstructed vector

Components:
  1. Randomized Hadamard Transform (RHT): v → w = (1/√d) H D_ε v
  2. Block partitioning: w ∈ R^d → K blocks of R^8 (K = d/8)
  3. Per-block E₈ quantization at scale a_b
  4. Parity-aware entropy coding (7 free coords + coord8_half)
  5. Inverse pipeline for decoding
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e8_quantizer import encode_e8, quantize_e8, compute_scale, theoretical_mse, G_E8
from entropy_coder import (
    e8_to_symbols, symbols_to_e8, FrequencyModel,
    measure_ideal_code_length, measure_real_ans
)


# ============================================================
# Randomized Hadamard Transform
# ============================================================

def hadamard_matrix(n: int) -> torch.Tensor:
    """
    Construct n×n Hadamard matrix via Sylvester construction.
    n must be a power of 2.
    Returns unnormalized H (entries ±1). Normalize by 1/√n.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform in O(d log d).
    Input: x of shape (..., d) where d is power of 2.
    Output: H @ x / sqrt(d), same shape.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"

    result = x.clone()
    h = 1
    while h < d:
        # Butterfly operation
        for i in range(0, d, h * 2):
            a = result[..., i:i+h].clone()
            b = result[..., i+h:i+2*h].clone()
            result[..., i:i+h] = a + b
            result[..., i+h:i+2*h] = a - b
        h *= 2

    return result / np.sqrt(d)


def inverse_fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Fast WHT. Since H is symmetric and H^{-1} = H/d,
    inverse is just another forward transform (self-inverse up to scaling).
    forward: y = Hx/√d, inverse: x = Hy/√d (same operation).
    """
    return fast_hadamard_transform(x)


class RHT:
    """
    Randomized Hadamard Transform.
    w = (1/√d) H D_ε v, where D_ε = diag(±1) random signs.
    
    The sign vector ε must be shared between encoder and decoder.
    """

    def __init__(self, d: int, seed: int = 0):
        assert d % 8 == 0, f"d must be divisible by 8, got {d}"
        self.d = d
        # Generate random signs
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.signs = torch.where(
            torch.rand(d, generator=gen) < 0.5,
            torch.ones(d), -torch.ones(d)
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v → w = (1/√d) H D_ε v"""
        device = v.device
        signs = self.signs.to(device)
        # Apply random signs
        x = v * signs
        # Fast Hadamard
        w = fast_hadamard_transform(x)
        return w

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        """w → v = D_ε H (√d · w) / d = D_ε (1/√d) H w"""
        device = w.device
        signs = self.signs.to(device)
        # Inverse Hadamard (same as forward for WHT)
        x = inverse_fast_hadamard_transform(w)
        # Undo random signs
        v = x * signs
        return v


# ============================================================
# Block Partitioning
# ============================================================

def to_blocks(w: torch.Tensor) -> torch.Tensor:
    """
    Reshape d-dim vector(s) into 8-dim blocks.
    Input: (..., d) where d % 8 == 0
    Output: (..., d/8, 8)
    """
    *batch, d = w.shape
    assert d % 8 == 0
    return w.reshape(*batch, d // 8, 8)


def from_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """
    Reshape 8-dim blocks back to d-dim vector(s).
    Input: (..., K, 8)
    Output: (..., K*8)
    """
    *batch, K, eight = blocks.shape
    assert eight == 8
    return blocks.reshape(*batch, K * 8)


# ============================================================
# Full Encoder / Decoder
# ============================================================

class LatticeQuantEncoder:
    """
    Full LatticeQuant encoder.
    v ∈ R^d → compressed representation (symbols + entropy model).
    """

    def __init__(self, d: int, bits_per_dim: float, rht_seed: int = 0):
        self.d = d
        self.bits = bits_per_dim
        self.rht = RHT(d, seed=rht_seed)
        self.model = None  # frequency model, fit from data

    def encode(self, v: torch.Tensor) -> dict:
        """
        Encode a batch of d-dim vectors.
        Input: v of shape (N, d)
        Output: dict with quantized reconstruction and coding info
        """
        N = v.shape[0]
        device = v.device

        # Step 1: RHT
        w = self.rht.forward(v)

        # Step 2: Estimate σ² from data
        sigma2 = (w ** 2).mean().item()

        # Step 3: Compute scale
        scale = compute_scale(sigma2, self.bits)

        # Step 4: Block partition + quantize
        blocks = to_blocks(w)  # (N, K, 8)
        K = blocks.shape[1]
        blocks_flat = blocks.reshape(N * K, 8)  # (N*K, 8)

        q_flat = quantize_e8(blocks_flat, scale)  # quantized blocks
        w_hat = from_blocks(q_flat.reshape(N, K, 8))  # (N, d)

        # Step 5: Inverse RHT for reconstruction
        v_hat = self.rht.inverse(w_hat)

        # Step 6: Symbolize for entropy coding
        lattice_points = q_flat / scale  # E₈ lattice points
        coset, free_coords, coord8_half = e8_to_symbols(lattice_points)

        return {
            'v_hat': v_hat,           # reconstructed vectors
            'w': w,                    # RHT output (for analysis)
            'w_hat': w_hat,            # quantized RHT output
            'scale': scale,
            'sigma2': sigma2,
            'coset': coset,
            'free_coords': free_coords,
            'coord8_half': coord8_half,
            'N': N,
            'K': K,
        }

    def measure_rate(self, enc: dict, train_ratio: float = 0.5) -> dict:
        """
        Measure actual coding rate using train/test split.
        """
        coset = enc['coset'].cpu().numpy()
        free = enc['free_coords'].cpu().numpy()
        c8h = enc['coord8_half'].cpu().numpy()
        total = len(coset)
        split = int(total * train_ratio)

        # Fit model on train portion
        model = FrequencyModel()
        model.fit(coset[:split], free[:split], c8h[:split])
        self.model = model

        # Measure on test portion
        ideal_test = measure_ideal_code_length(
            model, coset[split:], free[split:], c8h[split:])

        # ANS measurement
        ans_result = measure_real_ans(
            model, coset[split:], free[split:], c8h[split:])

        return {
            'ideal_test_bpd': ideal_test['bits_per_dim'],
            'ans_test_bpd': ans_result.get('bits_per_dim'),
        }


# ============================================================
# Pipeline Test
# ============================================================

def test_pipeline():
    """
    End-to-end pipeline test:
    1. Generate random vectors (simulating KV cache)
    2. Encode with LatticeQuant
    3. Measure MSE and rate
    4. Compare with theory
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    print("Phase 4: Full Pipeline Test")
    print("=" * 80)

    # Test dimensions (must be divisible by 8 and power of 2 for RHT)
    # Common KV head dimensions: 64, 128
    d = 128
    n_vectors = 10000
    bits_per_dim = 4.0
    sigma2_true = 1.0

    print(f"Config: d={d}, N={n_vectors:,}, target={bits_per_dim} bits/dim")
    print(f"True σ² = {sigma2_true}")
    print()

    # Generate source vectors
    torch.manual_seed(42)
    v = torch.randn(n_vectors, d, device=device) * np.sqrt(sigma2_true)

    # Verify norm preservation
    norms_before = (v ** 2).sum(dim=-1)

    # Encode
    encoder = LatticeQuantEncoder(d, bits_per_dim, rht_seed=42)
    t0 = time.time()
    enc = encoder.encode(v)
    t_encode = time.time() - t0

    v_hat = enc['v_hat']

    # Verify norm preservation through RHT
    norms_w = (enc['w'] ** 2).sum(dim=-1)
    norm_err = (norms_before - norms_w).abs().max().item()
    print(f"RHT norm preservation error: {norm_err:.2e}")

    # MSE in original space
    mse_original = ((v - v_hat) ** 2).mean().item()

    # MSE in RHT space
    mse_rht = ((enc['w'] - enc['w_hat']) ** 2).mean().item()

    # Theory
    sigma2_est = enc['sigma2']
    theory_mse_val = theoretical_mse(sigma2_est, bits_per_dim)
    d_gauss = sigma2_est * (4 ** (-bits_per_dim))

    print(f"Estimated σ² = {sigma2_est:.6f}")
    print(f"Scale a = {enc['scale']:.6f}")
    print()

    print("--- MSE Results ---")
    print(f"  MSE (original space): {mse_original:.8f}")
    print(f"  MSE (RHT space):      {mse_rht:.8f}")
    print(f"  Theory MSE:            {theory_mse_val:.8f}")
    print(f"  MSE ratio (orig):      {mse_original / theory_mse_val:.4f}")
    print(f"  MSE ratio (RHT):       {mse_rht / theory_mse_val:.4f}")
    print(f"  Gap (orig): D/D* =     {mse_original / d_gauss:.4f}")
    print(f"  Gap (RHT):  D/D* =     {mse_rht / d_gauss:.4f}")
    print(f"  Theory gap:            {2*np.pi*np.e*G_E8:.4f}")
    print()

    # Rate measurement
    print("--- Rate Results ---")
    t0 = time.time()
    rate = encoder.measure_rate(enc)
    t_rate = time.time() - t0

    print(f"  Ideal test rate:  {rate['ideal_test_bpd']:.4f} bits/dim (target: {bits_per_dim})")
    if rate['ans_test_bpd']:
        print(f"  ANS test rate:    {rate['ans_test_bpd']:.4f} bits/dim")
    print(f"  Rate/target:      {rate['ideal_test_bpd']/bits_per_dim:.4f}")
    print()

    # Cosine similarity (important for LLM quality)
    cos_sim = torch.nn.functional.cosine_similarity(v, v_hat, dim=-1)
    print("--- Reconstruction Quality ---")
    print(f"  Cosine similarity: mean={cos_sim.mean().item():.6f}, "
          f"min={cos_sim.min().item():.6f}")
    print(f"  Max absolute error: {(v - v_hat).abs().max().item():.6f}")
    print()

    # Timing
    print("--- Timing ---")
    print(f"  Encode: {t_encode:.3f}s ({n_vectors/t_encode:,.0f} vectors/sec)")
    print(f"  Rate measurement: {t_rate:.3f}s")
    print()

    # Multi-bitrate test
    print("=" * 80)
    print("Multi-bitrate test (d=128, N=10000)")
    print("-" * 80)
    print(f"{'bits':>6} | {'MSE gap':>9} | {'rate b/d':>9} | {'rate/tgt':>9} | "
          f"{'cos_sim':>9} | {'status':>8}")
    print("-" * 80)

    for bits in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        enc_i = LatticeQuantEncoder(d, bits, rht_seed=42)
        enc_data = enc_i.encode(v)

        mse_i = ((v - enc_data['v_hat']) ** 2).mean().item()
        d_gauss_i = enc_data['sigma2'] * (4 ** (-bits))
        gap_i = mse_i / d_gauss_i

        rate_i = enc_i.measure_rate(enc_data)
        rate_bpd = rate_i['ideal_test_bpd']
        rate_ratio = rate_bpd / bits

        cos_i = torch.nn.functional.cosine_similarity(
            v, enc_data['v_hat'], dim=-1).mean().item()

        rate_ok = rate_bpd <= bits * 1.02
        mse_ok = abs(gap_i - 2*np.pi*np.e*G_E8) / (2*np.pi*np.e*G_E8) < 0.01
        status = "✓ BOTH" if (rate_ok and mse_ok) else ("✓ MSE" if mse_ok else "—")

        print(f"{bits:>6.1f} | {gap_i:>9.4f} | {rate_bpd:>9.4f} | "
              f"{rate_ratio:>9.4f} | {cos_i:>9.6f} | {status:>8}")

    print("=" * 80)
    print()
    print("Key:")
    print("  MSE gap = D / D*_Gauss (theory: 1.2246)")
    print("  rate/tgt ≤ 1.0 means target bitrate achieved")
    print("  ✓ BOTH = rate ≤ target AND gap ≈ 1.224")


if __name__ == '__main__':
    test_pipeline()