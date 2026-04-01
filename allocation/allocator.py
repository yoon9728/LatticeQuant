"""
LatticeQuant v3 — Attention-Aware Bit Allocation (P5)
======================================================
Implements the water-filling optimal allocation from Theorem 3.

Given per-layer sensitivity (η_K, η_V, σ²_K, σ²_V) and optional
propagation factors (Γ_l), computes per-layer per-component (K/V)
bit allocations that minimise total attention-output distortion.

Closed-form (unclamped):
    b*_{X,l} = b̄ + ½ log₂(w_{X,l} / G)

where:
    w_{X,l} = Γ_l · η_{X,l} · σ²_{X,l}     (effective weight)
    G       = GM(w)                           (geometric mean of all weights)
    b̄       = B_total / (2L)                  (uniform budget per slot)

Gain over uniform allocation = AM(w) / GM(w).
This is an *unclamped continuous surrogate* gain under the high-rate
distortion objective — not directly comparable to PPL improvement.

Clamping to [b_min, b_max] with iterative budget redistribution:
    1. Compute unclamped allocation.
    2. Fix components that violate bounds at the bound value.
    3. Redistribute remaining budget among free components.
    4. Repeat until convergence (typically 2–3 iterations).

Usage:
  python allocation/allocator.py \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json \\
      --budget 4.0

  python allocation/allocator.py \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json \\
      --propagation results/propagation_Llama-3.1-8B.json \\
      --budget 4.0 --b-min 3 --b-max 5
"""

import json
import math
import argparse
import numpy as np
from typing import Optional
from pathlib import Path


# ============================================================
# Core allocation
# ============================================================

def compute_weights(
    sensitivity: dict,
    propagation: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute effective weights w_{X,l} = Γ_l · η_{X,l} · σ²_{X,l}.

    Returns array of shape (n_layers, 2) where [:, 0] = w_K, [:, 1] = w_V.
    """
    layers = sensitivity['layers']
    n_layers = len(layers)

    # Γ defaults to 1.0 for all layers if propagation data unavailable
    if propagation is not None:
        n_prop = len(propagation['layers'])
        if n_prop != n_layers:
            raise ValueError(
                f"Layer count mismatch: sensitivity has {n_layers} layers "
                f"but propagation has {n_prop}.  Are these from the same model?"
            )
        gammas = np.array([lr['gamma'] for lr in propagation['layers']])
    else:
        gammas = np.ones(n_layers)

    weights = np.zeros((n_layers, 2))
    for l, lr in enumerate(layers):
        # w_K, w_V are already GQA-aware grouped products:
        #   w_X = mean_g [η_X,g · σ²_X,g]
        # If legacy JSON (no grouped w_K), fall back to scalar product.
        w_K = lr.get('w_K', lr['eta_K'] * lr['sigma2_K'])
        w_V = lr.get('w_V', lr['eta_V'] * lr['sigma2_V'])
        weights[l, 0] = gammas[l] * w_K
        weights[l, 1] = gammas[l] * w_V

    return weights


def water_filling(
    weights: np.ndarray,
    budget: float = 4.0,
    b_min: float = 2.0,
    b_max: float = 8.0,
    max_iter: int = 20,
) -> dict:
    """
    Water-filling bit allocation with clamping.

    Parameters
    ----------
    weights : (n_layers, 2) array of effective weights w_{K,l}, w_{V,l}
    budget  : average bits per dim across all 2L slots
    b_min   : minimum bits per component (clamp floor)
    b_max   : maximum bits per component (clamp ceiling)

    Returns
    -------
    dict with:
        bits      : (n_layers, 2) array — allocated bits [b_K, b_V] per layer
        gain      : AM/GM ratio of weights (unclamped continuous surrogate gain;
                    NOT directly comparable to PPL improvement)
        budget_actual : realised average bits after clamping
        n_iter    : number of clamping iterations
    """
    n_layers = weights.shape[0]
    n_slots = 2 * n_layers  # one slot per (layer, K/V)
    flat_w = weights.flatten()  # (2L,)

    # ---- AM / GM gain (on all weights, before clamping) ----
    positive = flat_w > 0
    if not positive.all():
        import warnings
        n_bad = (~positive).sum()
        warnings.warn(
            f"{n_bad} of {n_slots} weight(s) are ≤ 0 "
            f"(min={flat_w.min():.4e}).  Replacing with smallest positive "
            f"weight for log computation.  This may indicate a numerical "
            f"issue in sensitivity/propagation extraction."
        )
        min_pos = flat_w[positive].min() if positive.any() else 1e-30
        flat_w = np.where(positive, flat_w, min_pos)

    log_w = np.log2(flat_w)
    am = flat_w.mean()
    gm = 2.0 ** log_w.mean()
    gain = am / gm  # ≥ 1 by AM-GM inequality

    # ---- Iterative clamped water-filling ----
    bits = np.full(n_slots, budget)  # start uniform
    fixed = np.zeros(n_slots, dtype=bool)

    for iteration in range(max_iter):
        free = ~fixed
        n_free = free.sum()
        if n_free == 0:
            break

        # Budget available for free slots
        budget_used_by_fixed = bits[fixed].sum() if fixed.any() else 0.0
        budget_for_free = budget * n_slots - budget_used_by_fixed

        # Water-filling among free slots
        free_w = flat_w[free]
        log_free_w = np.log2(free_w)
        gm_free = 2.0 ** log_free_w.mean()
        b_bar_free = budget_for_free / n_free

        alloc_free = b_bar_free + 0.5 * (log_free_w - np.log2(gm_free))

        # Check for bound violations
        newly_fixed = np.zeros(n_free, dtype=bool)
        clamped_free = alloc_free.copy()

        lo_mask = alloc_free < b_min
        hi_mask = alloc_free > b_max
        if lo_mask.any():
            clamped_free[lo_mask] = b_min
            newly_fixed[lo_mask] = True
        if hi_mask.any():
            clamped_free[hi_mask] = b_max
            newly_fixed[hi_mask] = True

        # Write back
        free_indices = np.where(free)[0]
        bits[free_indices] = clamped_free
        for i, fi in enumerate(free_indices):
            if newly_fixed[i]:
                fixed[fi] = True

        if not newly_fixed.any():
            break  # converged

    budget_actual = bits.mean()
    bits_2d = bits.reshape(n_layers, 2)

    return {
        'bits': bits_2d,
        'gain': float(gain),
        'budget_actual': float(budget_actual),
        'n_iter': iteration + 1,
    }


# ============================================================
# Discretisation (objective-aware greedy rounding)
# ============================================================

def discretise(
    bits_continuous: np.ndarray,
    weights: np.ndarray,
    budget: float,
    allowed: tuple = (3, 4, 5),
) -> np.ndarray:
    """
    Round continuous allocation to allowed integer rates, preserving
    total budget as closely as possible.

    Strategy: objective-aware greedy.
      1. Snap all slots to floor (nearest allowed ≤ continuous).
      2. Compute marginal distortion reduction for upgrading each slot:
             ΔD_i = w_i · (2^{-2·floor_i} - 2^{-2·ceil_i})
      3. Upgrade slots in descending ΔD order until budget is met.

    This prioritises upgrades that reduce total distortion most,
    unlike fractional-part rounding which ignores weight magnitudes.

    Parameters
    ----------
    bits_continuous : (n_layers, 2) continuous optimal allocation
    weights         : (n_layers, 2) effective weights (same as allocator input)
    budget          : target average bits per dim
    allowed         : tuple of allowed integer bitrates
    """
    n_layers, n_comp = bits_continuous.shape
    flat_b = bits_continuous.flatten()
    flat_w = weights.flatten()
    allowed = sorted(allowed)
    n_slots = len(flat_b)

    def snap_floor(x):
        for i in range(len(allowed) - 1, -1, -1):
            if allowed[i] <= x:
                return allowed[i]
        return allowed[0]

    def snap_ceil(x):
        for a in allowed:
            if a >= x:
                return a
        return allowed[-1]

    floored = np.array([snap_floor(v) for v in flat_b])
    ceiled = np.array([snap_ceil(v) for v in flat_b])
    step = ceiled - floored  # 0 if already at an allowed value, else 1 or 2

    total_target = budget * n_slots
    total_floored = floored.sum()
    deficit = total_target - total_floored

    # Marginal distortion reduction per upgrade: ΔD = w · (2^{-2·floor} − 2^{-2·ceil})
    marginal_gain = np.where(
        step > 0,
        flat_w * (np.power(2.0, -2.0 * floored) - np.power(2.0, -2.0 * ceiled)),
        0.0,
    )

    # Sort by marginal gain descending — upgrade most beneficial first
    order = np.argsort(-marginal_gain)

    result = floored.copy()
    remaining = deficit
    for idx in order:
        if remaining <= 0:
            break
        s = step[idx]
        if s > 0 and s <= remaining + 0.5:
            result[idx] = ceiled[idx]
            remaining -= s

    return result.reshape(n_layers, n_comp)


# ============================================================
# High-level API
# ============================================================

def allocate(
    sensitivity_path: str,
    propagation_path: Optional[str] = None,
    budget: float = 4.0,
    b_min: float = 2.0,
    b_max: float = 8.0,
    discrete: bool = False,
    allowed_bits: tuple = (3, 4, 5),
) -> dict:
    """
    Load data, run allocation, return results.

    Returns dict with:
        weights       : per-layer [w_K, w_V]
        continuous     : per-layer [b_K, b_V] (continuous optimum)
        discrete       : per-layer [b_K, b_V] (rounded, if discrete=True)
        gain           : AM/GM ratio
        budget_actual  : actual mean bits
        uniform_budget : the input budget (for reference)
    """
    with open(sensitivity_path) as f:
        sensitivity = json.load(f)

    propagation = None
    if propagation_path is not None:
        with open(propagation_path) as f:
            propagation = json.load(f)

    weights = compute_weights(sensitivity, propagation)
    result = water_filling(weights, budget=budget, b_min=b_min, b_max=b_max)

    out = {
        'model': sensitivity.get('model', 'unknown'),
        'n_layers': len(sensitivity['layers']),
        'uniform_budget': budget,
        'gamma_source': propagation_path if propagation_path else 'default (Γ=1)',
        'gain_am_gm': result['gain'],
        'budget_actual': result['budget_actual'],
        'n_iter': result['n_iter'],
        'b_min': b_min,
        'b_max': b_max,
        'layers': [],
    }

    for l in range(len(sensitivity['layers'])):
        entry = {
            'layer': l,
            'w_K': float(weights[l, 0]),
            'w_V': float(weights[l, 1]),
            'b_K': float(result['bits'][l, 0]),
            'b_V': float(result['bits'][l, 1]),
        }
        out['layers'].append(entry)

    if discrete:
        disc = discretise(result['bits'], weights, budget, allowed=allowed_bits)
        for l, entry in enumerate(out['layers']):
            entry['b_K_disc'] = int(disc[l, 0])
            entry['b_V_disc'] = int(disc[l, 1])
        out['budget_discrete'] = float(disc.mean())
        out['discrete_method'] = 'objective_aware_greedy'
        out['allowed_bits'] = list(allowed_bits)

    return out


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LatticeQuant v3: Attention-aware bit allocation')
    parser.add_argument('--sensitivity', type=str, required=True,
                        help='Path to sensitivity JSON')
    parser.add_argument('--propagation', type=str, default=None,
                        help='Path to propagation JSON (default: Γ=1)')
    parser.add_argument('--budget', type=float, default=4.0,
                        help='Average bits per dim (default: 4.0)')
    parser.add_argument('--b-min', type=float, default=2.0)
    parser.add_argument('--b-max', type=float, default=8.0)
    parser.add_argument('--discrete', action='store_true',
                        help='Also compute discrete allocation')
    parser.add_argument('--allowed-bits', type=int, nargs='+', default=[3, 4, 5],
                        help='Allowed discrete bitrates (default: 3 4 5)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    result = allocate(
        args.sensitivity,
        args.propagation,
        budget=args.budget,
        b_min=args.b_min,
        b_max=args.b_max,
        discrete=args.discrete,
        allowed_bits=tuple(sorted(args.allowed_bits)),
    )

    # ---- Print summary ----
    print(f"Model: {result['model']}")
    print(f"Budget: {result['uniform_budget']:.1f} bits/dim (uniform)")
    print(f"AM/GM gain (continuous surrogate): {result['gain_am_gm']:.4f}×")
    print(f"Actual mean: {result['budget_actual']:.3f} bits/dim")
    print(f"Clamping: [{args.b_min}, {args.b_max}], converged in {result['n_iter']} iter")
    print(f"Γ source: {result['gamma_source']}")

    print(f"\n{'Layer':>6} | {'w_K':>10} | {'w_V':>10} | "
          f"{'b_K':>6} | {'b_V':>6} | {'Δ_K':>6} | {'Δ_V':>6}", end='')
    if args.discrete:
        print(f" | {'d_K':>4} | {'d_V':>4}", end='')
    print()
    print("-" * (72 + (14 if args.discrete else 0)))

    for lr in result['layers']:
        dk = lr['b_K'] - result['uniform_budget']
        dv = lr['b_V'] - result['uniform_budget']
        line = (f"{lr['layer']:>6} | {lr['w_K']:>10.4e} | {lr['w_V']:>10.4e} | "
                f"{lr['b_K']:>6.2f} | {lr['b_V']:>6.2f} | "
                f"{dk:>+5.2f} | {dv:>+5.2f}")
        if args.discrete:
            line += f" | {lr.get('b_K_disc', ''):>4} | {lr.get('b_V_disc', ''):>4}"
        print(line)

    if args.discrete:
        print(f"\nDiscrete budget: {result['budget_discrete']:.3f} bits/dim")

    # ---- Save ----
    if args.output_dir is None:
        args.output_dir = str(Path(args.sensitivity).parent)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model_short = result['model'].split('/')[-1]
    save_path = Path(args.output_dir) / f'allocation_{model_short}_{args.budget:.0f}b.json'
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {save_path}")


if __name__ == '__main__':
    main()