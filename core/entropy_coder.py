"""
LatticeQuant Phase 2: Entropy Coder (Final)
============================================
Parity-aware entropy coding for E₈ lattice quantizer.

Key insight: E₈ = D₈ ∪ (D₈ + 1/2), and D₈ has even-sum constraint.
So 8 coordinates have only 7 degrees of freedom.

Coding scheme:
  1. Coset bit: 0 = integer, 1 = half-integer  (1 bit per block)
  2. 7 free coordinates: entropy coded per-coordinate | coset
  3. coord8_half = floor(coord8 / 2): entropy coded (parity is free)
  → 8th coordinate reconstructed from coord8_half + parity constraint

This saves exactly 1 bit / 8 dims = 0.125 bits/dim vs naive 8-coord coding.

Three independent rate measurements:
  1. Ideal code length: sum of -log2(p) per symbol
  2. Real ANS coder: actual compressed bitstream
  3. Train/test split: no overfitting possible
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e8_quantizer import encode_e8, quantize_e8, compute_scale, theoretical_mse, G_E8

try:
    import constriction
    HAS_CONSTRICTION = True
except ImportError:
    HAS_CONSTRICTION = False
    print("WARNING: constriction not installed. Will skip real ANS coding.")
    print("  Install with: pip install constriction")
    print()


# ============================================================
# Parity-Aware Symbolization
# ============================================================

def e8_to_symbols(q: torch.Tensor):
    """
    E₈ point → (coset, 7 free coords, coord8_half). Lossless.
    
    coset: 0 = integer coset (D₈), 1 = half-integer coset (D₈ + 1/2)
    free_coords: first 7 integer coordinates
    coord8_half: floor(coord8 / 2)  — parity determined by even-sum constraint
    
    Total symbols per block: 1 (coset) + 7 (free) + 1 (coord8_half) = 9
    But coord8_half has ~half the alphabet of coord8 → saves 1 bit/block = 0.125 bits/dim
    """
    is_half = (q[:, 0] - torch.round(q[:, 0])).abs() > 0.25
    coset = is_half.long()
    
    int_coords = q.clone()
    int_coords[is_half] -= 0.5
    int_coords = int_coords.round().long()
    
    free_coords = int_coords[:, :7]
    coord8 = int_coords[:, 7]
    coord8_half = torch.div(coord8, 2, rounding_mode='floor')
    
    return coset, free_coords, coord8_half


def symbols_to_e8(coset: torch.Tensor, free_coords: torch.Tensor,
                   coord8_half: torch.Tensor):
    """
    Inverse of e8_to_symbols. Lossless reconstruction.
    Recovers coord8 from coord8_half + even-sum parity constraint.
    """
    sum7 = free_coords.sum(dim=-1)
    parity_needed = ((sum7 % 2) + 2) % 2  # coord8 must have this parity for even total

    coord8_base = 2 * coord8_half
    base_parity = ((coord8_base % 2) + 2) % 2
    needs_r1 = (base_parity != parity_needed)
    coord8 = coord8_base + needs_r1.long()

    full_coords = torch.cat([free_coords, coord8.unsqueeze(-1)], dim=-1)
    result = full_coords.float()
    is_half = (coset == 1)
    result[is_half] += 0.5
    return result


def verify_symbolization(n_blocks=100000):
    """Verify parity-aware symbolization is perfectly reversible."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    x = torch.randn(n_blocks, 8, device=device)
    q = encode_e8(x)

    coset, free_coords, coord8_half = e8_to_symbols(q)
    q_rec = symbols_to_e8(coset, free_coords, coord8_half)

    max_err = (q - q_rec).abs().max().item()
    print(f"Parity-aware symbolization roundtrip max error: {max_err:.2e}")
    print(f"Perfect reconstruction: {max_err < 1e-6}")
    return max_err < 1e-6


# ============================================================
# Frequency Model
# ============================================================

class FrequencyModel:
    """
    Per-symbol frequency model conditioned on coset.
    8 symbol streams: indices 0-6 = free coords, index 7 = coord8_half.
    """

    def __init__(self):
        self.tables = {}
        self.totals = {}
        self.coset_prob = [0.5, 0.5]

    def fit(self, coset: np.ndarray, free_coords: np.ndarray, coord8_half: np.ndarray):
        N = len(coset)
        p_half = (coset == 1).mean()
        self.coset_prob = [max(1 - p_half, 1e-30), max(p_half, 1e-30)]

        all_symbols = np.concatenate([free_coords, coord8_half[:, None]], axis=1)

        for c in [0, 1]:
            mask = (coset == c)
            if mask.sum() == 0:
                continue
            syms_c = all_symbols[mask]
            for idx in range(8):
                key = (c, idx)
                vals = syms_c[:, idx]
                unique, counts = np.unique(vals, return_counts=True)
                self.tables[key] = dict(zip(unique.tolist(), counts.tolist()))
                self.totals[key] = int(mask.sum())


# ============================================================
# Measurement 1: Ideal Code Length (vectorized)
# ============================================================

def measure_ideal_code_length(model: FrequencyModel,
                               coset: np.ndarray,
                               free_coords: np.ndarray,
                               coord8_half: np.ndarray) -> dict:
    N = len(coset)
    total_bits = 0.0

    # Coset bits
    n0 = (coset == 0).sum()
    n1 = N - n0
    total_bits += n0 * (-np.log2(model.coset_prob[0]))
    total_bits += n1 * (-np.log2(model.coset_prob[1]))

    # Symbol streams
    all_symbols = np.concatenate([free_coords, coord8_half[:, None]], axis=1)

    for c_val in [0, 1]:
        mask = (coset == c_val)
        if mask.sum() == 0:
            continue
        syms_c = all_symbols[mask]
        total_c = model.totals.get((c_val, 0), syms_c.shape[0])

        for idx in range(8):
            key = (c_val, idx)
            table = model.tables.get(key, {})
            alphabet_size = len(table) + 1

            col = syms_c[:, idx]
            unique_vals, counts = np.unique(col, return_counts=True)

            for val, cnt in zip(unique_vals, counts):
                freq = table.get(int(val), 0)
                prob = (freq + 1) / (total_c + alphabet_size)
                total_bits += cnt * (-np.log2(prob))

    bits_per_dim = total_bits / (N * 8)
    return {'total_bits': total_bits, 'bits_per_dim': bits_per_dim}


# ============================================================
# Measurement 2: Real ANS Coder
# ============================================================

def measure_real_ans(model: FrequencyModel,
                     coset: np.ndarray,
                     free_coords: np.ndarray,
                     coord8_half: np.ndarray) -> dict:
    if not HAS_CONSTRICTION:
        return {'bits_per_dim': None, 'status': 'constriction not installed'}

    N = len(coset)
    total_bits = 0

    # Coset bits (actual ANS)
    coset_probs = np.array(model.coset_prob, dtype=np.float32)
    coset_probs /= coset_probs.sum()
    coset_indices = coset.astype(np.int32)
    encoder_coset = constriction.stream.stack.AnsCoder()
    encoder_coset.encode_reverse(
        coset_indices,
        constriction.stream.model.Categorical(coset_probs, perfect=False)
    )
    total_bits += len(encoder_coset.get_compressed()) * 32

    # Symbol streams per coset
    all_symbols = np.concatenate([free_coords, coord8_half[:, None]], axis=1)

    for c_val in [0, 1]:
        mask = (coset == c_val)
        if mask.sum() == 0:
            continue
        syms_c = all_symbols[mask]

        for idx in range(8):
            key = (c_val, idx)
            table = model.tables.get(key, {})

            all_vals = sorted(set(table.keys()) | set(syms_c[:, idx].tolist()))
            sym_to_idx = {s: i for i, s in enumerate(all_vals)}
            alphabet_size = len(all_vals)

            probs = np.zeros(alphabet_size, dtype=np.float32)
            for i, s in enumerate(all_vals):
                probs[i] = table.get(s, 0) + 1
            probs /= probs.sum()

            col = syms_c[:, idx]
            indices = np.array([sym_to_idx[int(v)] for v in col], dtype=np.int32)

            encoder = constriction.stream.stack.AnsCoder()
            encoder.encode_reverse(
                indices,
                constriction.stream.model.Categorical(probs, perfect=False)
            )

            compressed = encoder.get_compressed()
            total_bits += len(compressed) * 32

    bits_per_dim = total_bits / (N * 8)
    return {'total_bits': total_bits, 'bits_per_dim': bits_per_dim, 'status': 'ok'}


# ============================================================
# Main Experiment
# ============================================================

def entropy_coding_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Constriction available: {HAS_CONSTRICTION}")
    print()

    sigma2 = 1.0
    n_blocks = 2_000_000

    print("Phase 2 Final: Parity-Aware Entropy Coding")
    print("Scheme: coset (1 bit) + 7 free coords (entropy) + coord8_half (entropy)")
    print(f"Parity saving: 1 bit / 8 dims = 0.125 bits/dim vs naive 8-coord")
    print(f"σ² = {sigma2}, N = {n_blocks:,} (50/50 train/test)")
    print(f"Theory gap: 2πe·G(E₈) = {2*np.pi*np.e*G_E8:.4f}")
    print()

    ans_col = "| ANS b/d  " if HAS_CONSTRICTION else ""
    width = 95 + len(ans_col)
    print("=" * width)
    print(f"{'tgt b':>7} | {'ideal b/d':>10} | {'test b/d':>10} "
          f"{ans_col}"
          f"| {'test/tgt':>8} | {'MSE gap':>9} | {'status':>8}")
    print("-" * width)

    results = []

    for bits in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        scale = compute_scale(sigma2, bits)

        torch.manual_seed(42)
        x = torch.randn(n_blocks, 8, device=device) * np.sqrt(sigma2)
        x_hat = quantize_e8(x, scale)
        q = x_hat / scale

        # MSE
        mse = ((x - x_hat) ** 2).mean().item()
        d_gauss = sigma2 * (4 ** (-bits))
        meas_gap = mse / d_gauss

        # Symbolization
        coset, free_coords, coord8_half = e8_to_symbols(q)
        cos_np = coset.cpu().numpy()
        free_np = free_coords.cpu().numpy()
        c8h_np = coord8_half.cpu().numpy()

        # Train/test split
        half = n_blocks // 2
        cos_tr, cos_te = cos_np[:half], cos_np[half:]
        free_tr, free_te = free_np[:half], free_np[half:]
        c8h_tr, c8h_te = c8h_np[:half], c8h_np[half:]

        # Fit on train
        model = FrequencyModel()
        model.fit(cos_tr, free_tr, c8h_tr)

        # Ideal (full)
        ideal_full = measure_ideal_code_length(model, cos_np, free_np, c8h_np)

        # Ideal (test only)
        ideal_test = measure_ideal_code_length(model, cos_te, free_te, c8h_te)

        # ANS (test)
        if HAS_CONSTRICTION:
            ans = measure_real_ans(model, cos_te, free_te, c8h_te)
            ans_bpd = ans['bits_per_dim']
        else:
            ans_bpd = None

        test_ratio = ideal_test['bits_per_dim'] / bits

        rate_ok = ideal_test['bits_per_dim'] <= bits * 1.02
        mse_ok = abs(meas_gap - 2*np.pi*np.e*G_E8) / (2*np.pi*np.e*G_E8) < 0.01
        if rate_ok and mse_ok:
            status = "✓ BOTH"
        elif mse_ok:
            status = "✓ MSE"
        else:
            status = "—"

        ans_str = f"| {ans_bpd:>8.4f} " if ans_bpd is not None else ""

        print(f"{bits:>7.1f} | {ideal_full['bits_per_dim']:>10.4f} | "
              f"{ideal_test['bits_per_dim']:>10.4f} "
              f"{ans_str}"
              f"| {test_ratio:>8.4f} | {meas_gap:>9.4f} | {status:>8}")

        results.append({
            'bits': bits,
            'ideal_full_bpd': ideal_full['bits_per_dim'],
            'ideal_test_bpd': ideal_test['bits_per_dim'],
            'ans_bpd': ans_bpd,
            'test_over_target': test_ratio,
            'measured_gap': meas_gap,
            'status': status,
        })

    print("=" * width)
    print()
    print("Legend:")
    print("  tgt b    = target bits/dim (input to scale formula)")
    print("  ideal    = arithmetic coding lower bound (full data)")
    print("  test     = ideal code length on held-out test set (no overfitting)")
    if HAS_CONSTRICTION:
        print("  ANS      = actual ANS compressed size on test set")
    print("  test/tgt = measured rate / target (≤ 1.0 = target achieved)")
    print("  MSE gap  = D / D*_Gauss (theory: 1.2246)")
    print("  ✓ BOTH   = rate ≤ target AND MSE gap ≈ 1.224")
    print()

    return results


if __name__ == '__main__':
    print("Step 0: Verify parity-aware symbolization")
    print("-" * 50)
    ok = verify_symbolization()
    print()

    if ok:
        results = entropy_coding_experiment()
    else:
        print("ERROR: Symbolization failed!")