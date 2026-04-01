"""
E₈ Lattice Quantizer
=====================
E₈ nearest-neighbor encoding + Gaussian validation.

E₈ = D₈ ∪ (D₈ + [1/2]^8), where D₈ = {x ∈ Z^8 : sum(x) even}.
Nearest-neighbor decoding via Conway-Sloane algorithm.

Reference: NestQuant (cookiedoth/nestquant) e8.py for algorithm structure.
"""

import torch
import numpy as np
import time

# ============================================================
# E₈ Nearest-Neighbor Encoder
# ============================================================

def encode_d8(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest point in D₈ lattice.
    D₈ = {z ∈ Z^8 : sum(z) is even}.
    
    Algorithm: round each coordinate independently,
    then if sum is odd, flip the coordinate with smallest rounding cost.
    """
    # Round to nearest integer
    x_round = torch.round(x)
    
    # Check parity: sum must be even
    parity = x_round.sum(dim=-1) % 2  # 0 = even (good), 1 = odd (need fix)
    
    # Cost of flipping each coordinate
    residual = x - x_round  # how far we moved
    flip_cost = 1 - 2 * torch.abs(residual)  # cost increase if we flip
    # For coordinates rounded down (residual > 0), flipping up costs less when |residual| is large
    # We want to flip the coordinate with minimum cost increase
    
    # Among odd-parity rows, find cheapest coordinate to flip
    # Add small tiebreaker to avoid ambiguity
    EPS = 1e-7
    tiebreaker = torch.arange(8, device=x.device, dtype=x.dtype) * EPS
    flip_cost_adj = flip_cost + tiebreaker.unsqueeze(0)
    
    flip_idx = torch.argmin(flip_cost_adj, dim=-1)  # (N,)
    
    # Apply fix only to odd-parity rows
    needs_fix = (parity != 0)  # (N,)
    if needs_fix.any():
        rows = torch.where(needs_fix)[0]
        cols = flip_idx[needs_fix]
        # Flip: if we rounded to z, flip to z ± 1 (whichever is the other nearest integer)
        sign = torch.sign(residual[rows, cols])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        x_round[rows, cols] += sign
    
    return x_round


def encode_e8(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest point in E₈ lattice.
    E₈ = D₈ ∪ (D₈ + 1/2), i.e., union of D₈ and D₈ shifted by (1/2,...,1/2).
    
    Algorithm: find nearest D₈ point to x, and nearest D₈ point to x - 1/2,
    then return whichever is closer.
    
    Input: x of shape (N, 8)
    Output: nearest E₈ point, shape (N, 8)
    """
    # Candidate 1: nearest D₈ point
    c1 = encode_d8(x)
    
    # Candidate 2: nearest point in D₈ + 1/2
    c2 = encode_d8(x - 0.5) + 0.5
    
    # Pick closer candidate
    dist1 = ((x - c1) ** 2).sum(dim=-1)
    dist2 = ((x - c2) ** 2).sum(dim=-1)
    
    use_c1 = (dist1 <= dist2).unsqueeze(-1)
    return torch.where(use_c1, c1, c2)


# ============================================================
# Scaled Quantizer
# ============================================================

def quantize_e8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Quantize x using scaled E₈ lattice.
    x → Q(x/scale) * scale
    
    Input: x of shape (..., 8)
    Output: quantized x, same shape
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, 8)
    
    # Normalize by scale
    x_norm = x_flat / scale
    
    # Find nearest E₈ point
    q = encode_e8(x_norm)
    
    # Rescale
    x_hat = q * scale
    
    return x_hat.reshape(orig_shape)


def dequantize_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict:
    """Compute per-dimension MSE and other error metrics."""
    err = x - x_hat
    mse_total = (err ** 2).mean().item()
    mse_per_dim = (err ** 2).reshape(-1, 8).mean().item()  # same as total for 8-dim
    max_err = err.abs().max().item()
    return {
        'mse_per_dim': mse_per_dim,
        'mse_total': mse_total,
        'max_abs_err': max_err,
    }


# ============================================================
# Scale Selection (Theorem 2)
# ============================================================

# E₈ constants
G_E8 = 0.0717  # normalized second moment (Conway-Sloane)
VOL_E8_UNIT = 1.0  # volume of E₈ Voronoi cell at unit scale
# For E₈, the fundamental volume (determinant of generator matrix) = 1
# so Vol(V) = det(Lambda) = 1 for the standard E₈.
# At scale a: Vol(V_a) = a^8 * Vol(V_unit) = a^8

def compute_scale(sigma2: float, bits_per_dim: float) -> float:
    """
    Compute optimal lattice scale a_b from Theorem 2.
    
    From the proof: Vol(V_a)^{2/n} = 2*pi*e*sigma^2 * 4^{-b}
    For E₈ (n=8): Vol(V_a) = a^8, so a^2 = 2*pi*e*sigma^2 * 4^{-b}
    Hence: a = sqrt(2*pi*e*sigma^2 * 4^{-b})
    """
    target_vol_2n = 2 * np.pi * np.e * sigma2 * (4 ** (-bits_per_dim))
    # Vol(V_a)^{2/8} = a^2 = target_vol_2n
    a = np.sqrt(target_vol_2n)
    return a


def theoretical_mse(sigma2: float, bits_per_dim: float) -> float:
    """
    Theoretical per-dimension MSE from Theorem 2.
    D = G(E₈) * 2*pi*e * sigma^2 * 4^{-b}
    """
    return G_E8 * 2 * np.pi * np.e * sigma2 * (4 ** (-bits_per_dim))


# ============================================================
# Gaussian Sanity Check (Experiment A0)
# ============================================================

def sanity_check_gaussian():
    """
    Gaussian sanity check: verify MSE matches theory.
    Generate i.i.d. N(0, sigma^2) 8-dim blocks, quantize, measure MSE.
    Compare with theoretical prediction G(E₈) * 2πe * σ² * 4^{-b}.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"G(E₈) = {G_E8}")
    print(f"2πe·G(E₈) = {2 * np.pi * np.e * G_E8:.4f}")
    print()
    
    sigma2 = 1.0  # unit variance for clean testing
    n_blocks = 1_000_000  # 1M blocks = 8M samples → very tight estimate
    
    print(f"σ² = {sigma2}")
    print(f"Number of 8-dim blocks: {n_blocks:,}")
    print(f"Total samples: {n_blocks * 8:,}")
    print("=" * 70)
    print(f"{'bits/dim':>10} | {'Theory MSE':>12} | {'Measured MSE':>12} | "
          f"{'Ratio':>8} | {'Error%':>8} | {'Theory Gap':>10} | {'Meas Gap':>10}")
    print("-" * 70)
    
    results = []
    
    for bits in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        # Theoretical values
        theory_mse = theoretical_mse(sigma2, bits)
        theory_gap = G_E8 * 2 * np.pi * np.e  # = D / D_Gauss* at leading order
        
        # Compute scale
        scale = compute_scale(sigma2, bits)
        
        # Generate Gaussian data
        torch.manual_seed(42)
        x = torch.randn(n_blocks, 8, device=device) * np.sqrt(sigma2)
        
        # Quantize
        t0 = time.time()
        x_hat = quantize_e8(x, scale)
        t1 = time.time()
        
        # Measure MSE
        mse = ((x - x_hat) ** 2).mean().item()
        
        # Gaussian RDF
        d_gauss = sigma2 * (4 ** (-bits))
        
        # Measured gap
        meas_gap = mse / d_gauss
        
        ratio = mse / theory_mse
        error_pct = (ratio - 1) * 100
        
        print(f"{bits:>10.1f} | {theory_mse:>12.6f} | {mse:>12.6f} | "
              f"{ratio:>8.4f} | {error_pct:>+7.2f}% | {theory_gap:>10.4f} | {meas_gap:>10.4f}")
        
        results.append({
            'bits': bits,
            'scale': scale,
            'theory_mse': theory_mse,
            'measured_mse': mse,
            'ratio': ratio,
            'error_pct': error_pct,
            'theory_gap': theory_gap,
            'measured_gap': meas_gap,
            'time_sec': t1 - t0,
        })
    
    print("=" * 70)
    print()
    print("Interpretation:")
    print(f"  Theory gap = 2πe·G(E₈) = {theory_gap:.4f} (should be ~1.224)")
    print(f"  If Ratio ≈ 1.00 at high bits: theory matches practice")
    print(f"  At low bits: finite-rate effects cause deviation (expected)")
    print()
    
    # Convergence check at highest bitrate
    best = results[-1]
    print(f"  Highest rate ({best['bits']} bits/dim):")
    print(f"    Theory MSE = {best['theory_mse']:.8f}")
    print(f"    Measured MSE = {best['measured_mse']:.8f}")
    print(f"    Ratio = {best['ratio']:.4f} (target: 1.0000)")
    print(f"    Measured gap = {best['measured_gap']:.4f} (target: {theory_gap:.4f})")
    
    return results


# ============================================================
# Verification: E₈ lattice properties
# ============================================================

def verify_e8_properties():
    """Verify basic E₈ encoding properties."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("E₈ Encoder Verification")
    print("=" * 50)
    
    # Test 1: E₈ points should have integer or half-integer coords
    torch.manual_seed(0)
    x = torch.randn(10000, 8, device=device)
    q = encode_e8(x)
    
    # Check: all coords are integer or all are half-integer (per row)
    is_integer = (q - torch.round(q)).abs().max(dim=-1).values < 1e-6
    is_half = ((q - 0.5) - torch.round(q - 0.5)).abs().max(dim=-1).values < 1e-6
    valid = is_integer | is_half
    print(f"  All outputs valid E₈ points: {valid.all().item()}")
    
    # Check: integer coords have even sum
    int_rows = q[is_integer]
    if len(int_rows) > 0:
        sums = int_rows.sum(dim=-1)
        even = (sums % 2 == 0).all().item()
        print(f"  Integer coset: all sums even: {even}")
    
    # Check: half-integer coords have even sum
    half_rows = q[is_half & ~is_integer]
    if len(half_rows) > 0:
        sums = (half_rows - 0.5).sum(dim=-1)
        even = (sums % 2 == 0).all().item()
        print(f"  Half-integer coset: all (coord-0.5) sums even: {even}")
    
    # Test 2: Nearest-neighbor property (no closer E₈ point exists)
    # Sample a few and brute-force check against small E₈ neighborhood
    print(f"  Nearest-neighbor check: (sampling 100 points)...")
    x_test = torch.randn(100, 8, device=device) * 0.5
    q_test = encode_e8(x_test)
    dist_ours = ((x_test - q_test) ** 2).sum(dim=-1)
    
    # Check that no neighbor is closer
    # E₈ kissing number = 240, but we just check a few perturbations
    nn_correct = 0
    for i in range(100):
        # Try all ±1 perturbations of the found point
        found_closer = False
        for dim in range(8):
            for delta in [-1, 1, -0.5, 0.5]:
                neighbor = q_test[i].clone()
                neighbor[dim] += delta
                # Check if neighbor is valid E₈
                n_int = (neighbor - torch.round(neighbor)).abs().max() < 1e-6
                n_half = ((neighbor - 0.5) - torch.round(neighbor - 0.5)).abs().max() < 1e-6
                if n_int:
                    if neighbor.sum() % 2 != 0:
                        continue
                elif n_half:
                    if (neighbor - 0.5).sum() % 2 != 0:
                        continue
                else:
                    continue
                d = ((x_test[i] - neighbor) ** 2).sum()
                if d < dist_ours[i] - 1e-6:
                    found_closer = True
                    break
            if found_closer:
                break
        if not found_closer:
            nn_correct += 1
    
    print(f"  Nearest-neighbor correct: {nn_correct}/100")
    
    # Test 3: Throughput
    torch.manual_seed(42)
    x_bench = torch.randn(1_000_000, 8, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    _ = encode_e8(x_bench)
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    throughput = 1_000_000 / (t1 - t0)
    print(f"  Throughput: {throughput:,.0f} blocks/sec ({throughput * 8:,.0f} elements/sec)")
    
    print()


if __name__ == '__main__':
    verify_e8_properties()
    print()
    results = sanity_check_gaussian()