"""
Theorem 5 Consistency Check
=============================
Verifies internal consistency of the oracle spectral gap
decomposition (Theorem 5):

    D_iso / D_oracle = [AM(sigma^2)/GM(sigma^2)] x [AM(sigma_tilde^2)/GM(sigma_tilde^2)]
                     = attention_spectral_gap x value_spectral_gap

This is a synthetic consistency check: both sides are computed
from the same quantities, so agreement confirms that the
theorem-derived formulas are internally consistent in the
implementation.  It does NOT test the theorem against an
independent measured quantity.

Design choices for spectral controllability:
  - A = U @ diag(sigma) @ W^T is used directly as the operator,
    WITHOUT softmax normalization.  Theorem 5 is stated for
    A with given SVD; softmax would destroy the prescribed
    singular values while adding no theoretical content.
  - V is constructed as V = W @ V_tilde, where V_tilde has
    a prescribed per-direction energy profile.  This gives
    exact control over sigma_tilde_j^2 = E[||V_tilde[j,:]||^2]/d,
    which is the quantity in the theorem's value spectral gap.
  - Sweep 1 (vary A, fix V_tilde): changing A changes W,
    so V = W @ V_tilde changes, but sigma_tilde^2 stays exactly
    fixed.  This gives true factor independence.
  - Sweep 2 (fix A, vary V_tilde): A and hence sigma^2 are fixed;
    only sigma_tilde^2 changes.

Usage:
  python allocation/thm5_validate.py
  python allocation/thm5_validate.py --T 64 --d 32 --n-sweep 20
"""

import numpy as np
import json
import argparse
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def am_gm_ratio(x):
    """AM/GM ratio of positive array."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    if len(x) == 0:
        return 1.0
    return x.mean() / np.exp(np.log(x).mean())


def make_operator(T, sigma_spectrum, rng):
    """
    Construct A = U @ diag(sigma) @ W^T with prescribed singular values.

    No softmax normalization -- exact spectral control.
    Returns A, W (right singular vectors), sigma^2 (squared singular values).
    """
    U, _ = np.linalg.qr(rng.randn(T, T))
    W, _ = np.linalg.qr(rng.randn(T, T))
    sigma = np.array(sigma_spectrum, dtype=np.float64)
    A = U @ np.diag(sigma) @ W.T
    return A, W, sigma ** 2


def make_V_in_W_basis(W, energy_profile, d, rng):
    """
    Construct V = W @ V_tilde where V_tilde has prescribed
    per-direction energy sigma_tilde_j^2.

    V_tilde[j,:] ~ N(0, energy_j * I_d), so
    E[||V_tilde[j,:]||^2]/d = energy_j exactly (in expectation).
    """
    T = W.shape[0]
    scales = np.sqrt(np.array(energy_profile, dtype=np.float64))
    V_tilde = rng.randn(T, d) * scales[:, None]
    V = W @ V_tilde
    return V, V_tilde


def compute_distortions(sigma_sq, V_tilde, rate):
    """
    Compute isotropic and oracle distortions using Theorem 5 formulas.

    Parameters:
        sigma_sq: (T,) squared singular values of A
        V_tilde: (T, d) value matrix in W-basis
        rate: bits per dimension
    """
    c = 1.2236  # doesn't matter for ratio

    # Per-direction energy in W-basis
    sigma2_tilde = (V_tilde ** 2).mean(axis=1)  # (T,)

    # Isotropic: c * 4^{-R} * AM(sigma^2) * AM(sigma_tilde^2)
    D_iso = c * (4.0 ** (-rate)) * sigma_sq.mean() * sigma2_tilde.mean()

    # Oracle: c * 4^{-R} * GM(sigma^2) * GM(sigma_tilde^2)
    s_pos = sigma_sq[sigma_sq > 1e-30]
    t_pos = sigma2_tilde[sigma2_tilde > 1e-30]
    gm_s = np.exp(np.log(s_pos).mean()) if len(s_pos) > 0 else 0
    gm_t = np.exp(np.log(t_pos).mean()) if len(t_pos) > 0 else 0
    D_oracle = c * (4.0 ** (-rate)) * gm_s * gm_t

    # Gap decomposition
    attn_gap = am_gm_ratio(sigma_sq)
    value_gap = am_gm_ratio(sigma2_tilde)
    predicted = attn_gap * value_gap
    actual = D_iso / D_oracle if D_oracle > 0 else float('inf')

    return {
        'D_iso': D_iso,
        'D_oracle': D_oracle,
        'attn_gap': attn_gap,
        'value_gap': value_gap,
        'predicted_total_gap': predicted,
        'actual_total_gap': actual,
        'ratio': actual / predicted if predicted > 0 else float('inf'),
    }


# ============================================================
# Spectrum profiles
# ============================================================

def spectrum_flat(T):
    return np.ones(T)

def spectrum_peaked(T, peak_frac=0.125, peak_ratio=20.0):
    s = np.ones(T)
    n_peak = max(1, int(T * peak_frac))
    s[:n_peak] = peak_ratio
    return s

def energy_flat(T):
    return np.ones(T)

def energy_skewed(T, peak_frac=0.125, peak_ratio=10.0):
    e = np.ones(T)
    n_peak = max(1, int(T * peak_frac))
    e[:n_peak] = peak_ratio
    e[n_peak:] = 0.3
    return e


# ============================================================
# (A) 2x2 Control Grid
# ============================================================

def run_2x2_grid(T=32, d=16, rate=4.0, seed=42):
    rng = np.random.RandomState(seed)

    a_spectra = {
        'flat':   spectrum_flat(T),
        'peaked': spectrum_peaked(T),
    }
    v_profiles = {
        'flat':   energy_flat(T),
        'skewed': energy_skewed(T),
    }

    results = {}
    for a_name, sigma_vals in a_spectra.items():
        A, W, sigma_sq = make_operator(T, sigma_vals, rng)

        for v_name, e_profile in v_profiles.items():
            V, V_tilde = make_V_in_W_basis(W, e_profile, d, rng)
            res = compute_distortions(sigma_sq, V_tilde, rate)
            res['a_condition'] = a_name
            res['v_condition'] = v_name
            results[(a_name, v_name)] = res

    return results


# ============================================================
# (B) Factor Independence Sweep
# ============================================================

def run_factor_sweep(T=32, d=16, rate=4.0, n_sweep=20, seed=42):
    rng = np.random.RandomState(seed)

    # ---- Sweep 1: Vary A-spectrum, fix V_tilde ----
    # V_tilde is fixed, so sigma_tilde^2 is exactly constant.
    # V = W @ V_tilde changes with each A (because W changes),
    # but value_gap depends only on sigma_tilde^2, which doesn't.
    V_tilde_fixed = rng.randn(T, d)  # flat energy
    sweep_attn = []

    for i in range(n_sweep):
        concentration = i / (n_sweep - 1)
        s = np.ones(T)
        n_peak = max(1, int(T * 0.125))
        s[:n_peak] = 1.0 + concentration * 19.0
        sigma_sq = s ** 2

        A, W, _ = make_operator(T, s, rng)
        res = compute_distortions(sigma_sq, V_tilde_fixed, rate)
        res['concentration'] = concentration
        sweep_attn.append(res)

    # ---- Sweep 2: Fix A, vary V_tilde energy ----
    s_moderate = spectrum_peaked(T, peak_frac=0.25, peak_ratio=5.0)
    A_fixed, W_fixed, sigma_sq_fixed = make_operator(T, s_moderate, rng)

    sweep_value = []
    for i in range(n_sweep):
        concentration = i / (n_sweep - 1)
        e = np.ones(T)
        n_peak = max(1, int(T * 0.125))
        e[:n_peak] = 1.0 + concentration * 19.0
        e[n_peak:] = 1.0 - concentration * 0.7

        V_tilde_i = rng.randn(T, d) * np.sqrt(e)[:, None]
        res = compute_distortions(sigma_sq_fixed, V_tilde_i, rate)
        res['concentration'] = concentration
        sweep_value.append(res)

    return sweep_attn, sweep_value


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Theorem 5 synthetic consistency check')
    parser.add_argument('--T', type=int, default=32)
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--rate', type=float, default=4.0)
    parser.add_argument('--n-sweep', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    print(f"{'':=<70}")
    print(f"  Theorem 5: Synthetic Consistency Check")
    print(f"  (Oracle Spectral Gap Decomposition)")
    print(f"  T={args.T}, d={args.d}, rate={args.rate} bits/dim")
    print(f"  Operator: A = U diag(s) W^T (no softmax)")
    print(f"  Values: V = W @ V_tilde (energy controlled in W-basis)")
    print(f"{'':=<70}")

    # ---- (A) 2x2 Grid ----
    print(f"\n  (A) 2x2 Control Grid")
    print(f"  {'A-spectrum':<12} | {'V-energy':<12} | {'attn_gap':>10} | "
          f"{'value_gap':>10} | {'predicted':>10} | {'actual':>10} | {'ratio':>8}")
    print(f"  {'-'*82}")

    grid = run_2x2_grid(args.T, args.d, args.rate, args.seed)
    for (a_cond, v_cond), res in sorted(grid.items()):
        print(f"  {a_cond:<12} | {v_cond:<12} | {res['attn_gap']:>10.4f} | "
              f"{res['value_gap']:>10.4f} | {res['predicted_total_gap']:>10.4f} | "
              f"{res['actual_total_gap']:>10.4f} | {res['ratio']:>8.6f}")

    ratios = [res['ratio'] for res in grid.values()]
    max_dev = max(abs(r - 1) for r in ratios)
    print(f"\n  Max deviation from factorization: {max_dev:.2e}")

    # ---- (B) Sweep ----
    print(f"\n  (B) Factor Independence Sweep ({args.n_sweep} points each)")
    sweep_attn, sweep_value = run_factor_sweep(
        args.T, args.d, args.rate, args.n_sweep, args.seed)

    print(f"\n  Sweep 1: Vary A-spectrum (V_tilde fixed)")
    print(f"  {'conc':>6} | {'attn_gap':>10} | {'value_gap':>10} | {'total_gap':>10} | {'ratio':>8}")
    print(f"  {'-'*54}")
    step = max(1, len(sweep_attn) // 5)
    for res in sweep_attn[::step]:
        print(f"  {res['concentration']:>6.2f} | {res['attn_gap']:>10.4f} | "
              f"{res['value_gap']:>10.4f} | {res['actual_total_gap']:>10.4f} | "
              f"{res['ratio']:>8.6f}")

    print(f"\n  Sweep 2: Vary V-energy (A fixed)")
    print(f"  {'conc':>6} | {'attn_gap':>10} | {'value_gap':>10} | {'total_gap':>10} | {'ratio':>8}")
    print(f"  {'-'*54}")
    step = max(1, len(sweep_value) // 5)
    for res in sweep_value[::step]:
        print(f"  {res['concentration']:>6.2f} | {res['attn_gap']:>10.4f} | "
              f"{res['value_gap']:>10.4f} | {res['actual_total_gap']:>10.4f} | "
              f"{res['ratio']:>8.6f}")

    # ---- Factor independence metrics ----
    vg_sweep1 = [r['value_gap'] for r in sweep_attn]
    ag_sweep2 = [r['attn_gap'] for r in sweep_value]
    vg_cv1 = np.std(vg_sweep1) / np.mean(vg_sweep1) if np.mean(vg_sweep1) > 0 else 0
    ag_cv2 = np.std(ag_sweep2) / np.mean(ag_sweep2) if np.mean(ag_sweep2) > 0 else 0

    all_ratios = [r['ratio'] for r in sweep_attn + sweep_value]
    max_dev_sweep = max(abs(r - 1.0) for r in all_ratios)

    print(f"\n  Factor independence:")
    print(f"    Sweep 1 (vary A): value_gap CV = {vg_cv1:.2e} "
          f"({'exactly constant' if vg_cv1 < 1e-10 else 'stable' if vg_cv1 < 0.01 else 'varies'})")
    print(f"    Sweep 2 (fix A):  attn_gap CV = {ag_cv2:.2e} "
          f"({'exactly constant' if ag_cv2 < 1e-10 else 'stable' if ag_cv2 < 0.01 else 'varies'})")
    print(f"    Max factorization deviation: {max_dev_sweep:.2e}")

    print(f"\n  Summary:")
    print(f"    Factorization is internally consistent")
    print(f"    (max deviation {max_dev_sweep:.1e} across all conditions).")
    if vg_cv1 < 1e-10:
        print(f"    Sweep 1: value_gap exactly constant (V_tilde fixed).")
    if ag_cv2 < 1e-10:
        print(f"    Sweep 2: attn_gap exactly constant (A fixed).")
    print(f"    Note: both sides computed from same quantities;")
    print(f"    this confirms formula consistency, not an independent test.")

    # ---- Save ----
    if args.output_dir is None:
        args.output_dir = 'results'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    save_data = {
        'T': args.T, 'd': args.d, 'rate': args.rate,
        'grid': {f"{k[0]}_{k[1]}": {
            'attn_gap': v['attn_gap'], 'value_gap': v['value_gap'],
            'predicted': v['predicted_total_gap'],
            'actual': v['actual_total_gap'], 'ratio': v['ratio'],
        } for k, v in grid.items()},
        'sweep_attn': [{'concentration': r['concentration'],
            'attn_gap': r['attn_gap'], 'value_gap': r['value_gap'],
            'total_gap': r['actual_total_gap'], 'ratio': r['ratio'],
        } for r in sweep_attn],
        'sweep_value': [{'concentration': r['concentration'],
            'attn_gap': r['attn_gap'], 'value_gap': r['value_gap'],
            'total_gap': r['actual_total_gap'], 'ratio': r['ratio'],
        } for r in sweep_value],
    }
    save_path = Path(args.output_dir) / f'thm5_synthetic_T{args.T}.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved → {save_path}")


if __name__ == '__main__':
    main()