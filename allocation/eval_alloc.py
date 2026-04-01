"""
Uniform vs Optimal PPL Evaluation
===================================
Compares perplexity under uniform bitrate vs attention-aware optimal
allocation from Theorem 3's water-filling solution.

Key design:
  - VariableRateKVCache subclasses CompressedKVCache, overriding
    update() to set self.bits per-layer and per-component (K vs V)
    before each _compress_tensor call.  No modification to
    compressed_kv_cache.py required.
  - PPL evaluation reuses the sliding-window approach from e2e_eval.py.
  - eval_only_no_entropy=True for speed (lossless, doesn't affect PPL).

Usage:
  # Single budget with pre-computed allocation
  python allocation/eval_alloc.py \\
      --model meta-llama/Llama-3.1-8B \\
      --allocation results/allocation_Llama-3.1-8B_4b.json

  # Multiple budgets (computes allocation on the fly)
  python allocation/eval_alloc.py \\
      --model meta-llama/Llama-3.1-8B \\
      --sensitivity results/sensitivity_Llama-3.1-8B.json \\
      --propagation results/propagation_Llama-3.1-8B.json \\
      --budgets 3 4 5
"""

import torch
import numpy as np
import time
import json
import argparse
import gc
from typing import Optional, List, Dict
from pathlib import Path

import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, '..'))
sys.path.insert(0, os.path.join(_this_dir, '..', 'llm'))
sys.path.insert(0, os.path.join(_this_dir, '..', 'core'))

from allocator import allocate
from compressed_kv_cache import CompressedKVCache
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset


# ============================================================
# Variable-rate KV cache (per-layer K/V bitrate)
# ============================================================

class VariableRateKVCache(CompressedKVCache):
    """
    CompressedKVCache with per-layer, per-component (K/V) bitrate.

    bits_schedule: list of dicts, one per layer:
        [{'K': 3, 'V': 5}, {'K': 4, 'V': 4}, ...]

    Implementation: before each _compress_tensor call, self.bits
    is set to the appropriate rate.  This works because
    _compress_tensor reads self.bits and nothing else.
    """

    def __init__(self, bits_schedule: List[Dict[str, int]],
                 eval_only_no_entropy: bool = True):
        # EXPERIMENTAL OVERRIDE: skip CompressedKVCache.__init__'s
        # assert bits_per_dim in (3,4,5) by calling DynamicCache.__init__
        # directly.  This is intentional — we set self.bits per-call in
        # update(), not once at construction.  Fragile if parent changes.
        super(CompressedKVCache, self).__init__()  # DynamicCache.__init__
        self.bits = 4  # dummy default, overridden per _compress_tensor call
        self.eval_only_no_entropy = eval_only_no_entropy
        self.bits_schedule = bits_schedule
        if not hasattr(self, 'key_cache'):
            self.key_cache = []
        if not hasattr(self, 'value_cache'):
            self.value_cache = []
        self._comp_keys = []
        self._comp_values = []
        self.total_vectors = 0
        self._model_dtype = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override: use per-layer K/V bitrate from schedule."""
        if layer_idx < len(self.bits_schedule):
            schedule = self.bits_schedule[layer_idx]
        else:
            # Fallback for layers beyond schedule (shouldn't happen)
            schedule = {'K': 4, 'V': 4}

        device = key_states.device
        if self._model_dtype is None:
            self._model_dtype = key_states.dtype

        # Compress K with K-specific bits
        self.bits = schedule['K']
        comp_k, k_dec_new = self._compress_tensor(key_states)

        # Compress V with V-specific bits
        self.bits = schedule['V']
        comp_v, v_dec_new = self._compress_tensor(value_states)

        # --- Storage bookkeeping (same as parent) ---
        while len(self._comp_keys) <= layer_idx:
            self._comp_keys.append(None)
            self._comp_values.append(None)

        from compressed_kv_cache import _get_placeholder, CompressedKVLayer
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(_get_placeholder(device, self._model_dtype))
            self.value_cache.append(_get_placeholder(device, self._model_dtype))

        # Note: 'uncompressible' only triggers when hd % 8 != 0.
        # Since K and V share the same head_dim, they are always
        # both compressible or both uncompressible simultaneously.
        if comp_k == 'uncompressible':
            from compressed_kv_cache import UncompressibleKVLayer
            self._comp_keys[layer_idx] = UncompressibleKVLayer(key_states)
            self._comp_values[layer_idx] = UncompressibleKVLayer(value_states)
            self.key_cache[layer_idx] = k_dec_new
            self.value_cache[layer_idx] = v_dec_new
            return k_dec_new, v_dec_new

        if self._comp_keys[layer_idx] is None:
            self._comp_keys[layer_idx] = CompressedKVLayer()
            self._comp_values[layer_idx] = CompressedKVLayer()

        self._comp_keys[layer_idx].append_chunk(comp_k)
        self._comp_values[layer_idx].append_chunk(comp_v)

        prev_k = self.key_cache[layer_idx]
        if prev_k.shape[2] > 0:
            k_full = torch.cat([prev_k, k_dec_new], dim=2)
            v_full = torch.cat([self.value_cache[layer_idx], v_dec_new], dim=2)
        else:
            k_full = k_dec_new
            v_full = v_dec_new

        self.key_cache[layer_idx] = k_full
        self.value_cache[layer_idx] = v_full
        return k_full, v_full


# ============================================================
# PPL evaluation (reuses e2e_eval sliding window pattern)
# ============================================================

def evaluate_ppl(
    model,
    tokenizer,
    cache_factory,
    label: str,
    max_length: int = 2048,
    stride: int = 512,
) -> dict:
    """
    Perplexity on wikitext2 with sliding window.

    cache_factory: callable() -> cache instance (called per window)
    """
    print(f"\n  [{label}] Evaluating PPL...")

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    print(f"    Dataset: {seq_len:,} tokens, stride={stride}, max_len={max_length}")

    nlls = []
    n_tokens = 0
    t_start = time.time()
    prev_end = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        chunk_ids = input_ids[:, begin:end].to('cuda:0')

        cache = cache_factory()

        with torch.no_grad():
            outputs = model(chunk_ids, past_key_values=cache, use_cache=True)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk_ids[:, 1:].contiguous()

        eval_start = max(0, chunk_ids.size(1) - trg_len - 1)
        eval_logits = shift_logits[:, eval_start:, :]
        eval_labels = shift_labels[:, eval_start:]

        loss = torch.nn.CrossEntropyLoss(reduction='sum')(
            eval_logits.reshape(-1, eval_logits.size(-1)),
            eval_labels.reshape(-1),
        )

        nlls.append(loss.item())
        n_tokens += eval_labels.numel()
        prev_end = end

        if end >= seq_len - 1:
            break

        if len(nlls) % 20 == 0:
            print(f"      {n_tokens:,} tokens, running PPL="
                  f"{np.exp(sum(nlls)/n_tokens):.2f}")

        del cache, outputs, logits

    t_elapsed = time.time() - t_start
    ppl = np.exp(sum(nlls) / n_tokens)
    print(f"    PPL={ppl:.4f}, {n_tokens:,} tokens, {t_elapsed:.0f}s")

    return {'label': label, 'ppl': ppl, 'n_tokens': n_tokens,
            'eval_time_sec': t_elapsed}


# ============================================================
# Allocation → bits_schedule converter
# ============================================================

def allocation_to_schedule(alloc: dict) -> List[Dict[str, int]]:
    """Convert allocator output to bits_schedule for VariableRateKVCache."""
    schedule = []
    for lr in alloc['layers']:
        schedule.append({
            'K': lr.get('b_K_disc', round(lr['b_K'])),
            'V': lr.get('b_V_disc', round(lr['b_V'])),
        })
    return schedule


def schedule_avg(schedule: List[Dict[str, int]]) -> float:
    """Compute average bits/dim from a schedule."""
    k_bits = [s['K'] for s in schedule]
    v_bits = [s['V'] for s in schedule]
    return (sum(k_bits) + sum(v_bits)) / (2 * len(schedule))


def schedule_summary(schedule: List[Dict[str, int]]) -> str:
    """Pretty-print a bits schedule."""
    k_bits = [s['K'] for s in schedule]
    v_bits = [s['V'] for s in schedule]
    avg = schedule_avg(schedule)
    return (f"avg={avg:.3f}b, K=[{min(k_bits)}-{max(k_bits)}], "
            f"V=[{min(v_bits)}-{max(v_bits)}]")


# ============================================================
# Main experiment
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Theorem 3 evaluation (uniform vs optimal PPL)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')

    # Option A: pre-computed allocation
    parser.add_argument('--allocation', type=str, default=None,
                        help='Path to allocator output JSON (single budget)')

    # Option B: compute allocation on the fly
    parser.add_argument('--sensitivity', type=str, default=None)
    parser.add_argument('--propagation', type=str, default=None)
    parser.add_argument('--budgets', type=float, nargs='+', default=[4.0],
                        help='Average bits/dim budgets to evaluate')

    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip FP16 KV baseline (saves time if already measured)')
    parser.add_argument('--optimal-only', action='store_true',
                        help='Skip uniform, only run optimal (use when uniform results exist)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    # ---- Load model ----
    print(f"Loading {args.model} (8-bit weights)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map='cuda:0')
    model.eval()
    print(f"  Loaded: {torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM")

    results = {'model': args.model, 'experiments': []}

    # ---- Baseline: no KV quantization ----
    if not args.skip_baseline:
        r = evaluate_ppl(model, tokenizer,
                         cache_factory=lambda: DynamicCache(),
                         label='baseline (FP16 KV)',
                         max_length=args.max_length)
        results['baseline_ppl'] = r['ppl']
        results['experiments'].append(r)
        gc.collect(); torch.cuda.empty_cache()

    # ---- Build allocations ----
    if args.allocation:
        # Single pre-computed allocation
        with open(args.allocation) as f:
            alloc = json.load(f)
        budgets_and_allocs = [(alloc['uniform_budget'], alloc)]
    elif args.sensitivity:
        # Compute allocations for each budget
        budgets_and_allocs = []
        for budget in args.budgets:
            alloc = allocate(
                args.sensitivity, args.propagation,
                budget=budget, b_min=3.0, b_max=5.0,
                discrete=True, allowed_bits=(3, 4, 5),
            )
            budgets_and_allocs.append((budget, alloc))
    else:
        parser.error("Provide either --allocation or --sensitivity")

    # ---- For each budget: uniform vs optimal ----
    # Uniform baseline uses CompressedKVCache which only supports
    # discrete rates {3, 4, 5}.  Non-integer budgets are skipped.
    for budget, alloc in budgets_and_allocs:
        if min(abs(budget - x) for x in (3.0, 4.0, 5.0)) > 1e-6:
            print(f"\n  SKIP budget {budget}: uniform baseline requires "
                  f"integer rate in {{3, 4, 5}}.")
            continue
        budget_int = int(round(budget))

        # Actual discrete budget from allocator
        actual_disc_budget = alloc.get('budget_discrete', budget)

        print(f"\n{'':=<70}")
        print(f"  Budget: {budget} bits/dim (discrete: {actual_disc_budget:.3f})")
        print(f"  AM/GM gain (surrogate): {alloc['gain_am_gm']:.4f}x")
        print(f"{'':=<70}")

        # ---- Uniform allocation ----
        r_uniform = None
        if not args.optimal_only:
            gc.collect(); torch.cuda.empty_cache()
            r_uniform = evaluate_ppl(
                model, tokenizer,
                cache_factory=lambda b=budget_int: CompressedKVCache(
                    bits_per_dim=b, eval_only_no_entropy=True),
                label=f'uniform {budget_int}b',
                max_length=args.max_length,
            )
            results['experiments'].append(r_uniform)
            r_uniform['allocation_type'] = 'uniform'
            r_uniform['budget_requested'] = budget
            r_uniform['budget_actual'] = float(budget_int)
        else:
            print(f"\n  [uniform {budget_int}b] SKIPPED (--optimal-only)")

        # ---- Optimal allocation ----
        gc.collect(); torch.cuda.empty_cache()
        schedule = allocation_to_schedule(alloc)
        sched_avg = schedule_avg(schedule)
        print(f"  Optimal schedule: {schedule_summary(schedule)}")
        if abs(sched_avg - actual_disc_budget) > 0.01:
            print(f"    ⚠ schedule avg ({sched_avg:.3f}) != "
                  f"allocator budget_discrete ({actual_disc_budget:.3f})")

        r_optimal = evaluate_ppl(
            model, tokenizer,
            cache_factory=lambda s=schedule: VariableRateKVCache(
                bits_schedule=s, eval_only_no_entropy=True),
            label=f'optimal {budget_int}b',
            max_length=args.max_length,
        )
        results['experiments'].append(r_optimal)

        # ---- Structured metadata ----
        r_optimal['allocation_type'] = 'optimal'
        r_optimal['budget_requested'] = budget
        r_optimal['budget_actual'] = sched_avg  # from actual schedule, not allocator metadata
        r_optimal['schedule'] = schedule
        r_optimal['am_gm_gain'] = alloc['gain_am_gm']

        # ---- Comparison ----
        if r_uniform is not None:
            budget_gap = actual_disc_budget - float(budget_int)
            delta = r_optimal['ppl'] - r_uniform['ppl']
            rel = (r_optimal['ppl'] / r_uniform['ppl'] - 1) * 100
            print(f"\n  Comparison @ nominal {budget_int}b:")
            print(f"    Uniform:  PPL = {r_uniform['ppl']:.4f}  (actual {budget_int}.000b)")
            print(f"    Optimal:  PPL = {r_optimal['ppl']:.4f}  (actual {actual_disc_budget:.3f}b)")
            if abs(budget_gap) > 0.01:
                print(f"    ⚠ Budget gap: {budget_gap:+.3f} bits "
                      f"(optimal uses {'more' if budget_gap > 0 else 'fewer'} bits)")
            print(f"    Delta:    {delta:+.4f} ({rel:+.2f}%)")
        else:
            print(f"\n  Optimal {budget_int}b: PPL = {r_optimal['ppl']:.4f}  "
                  f"(actual {actual_disc_budget:.3f}b)")

    # ---- Summary table ----
    print(f"\n{'':=<70}")
    print(f"  SUMMARY: {args.model}")
    print(f"{'':=<70}")
    baseline = results.get('baseline_ppl', 0)
    print(f"  {'Config':<20} | {'PPL':>8} | {'actual b':>8} | {'vs base':>10} | {'vs unif':>10}")
    print(f"  {'-'*64}")

    if baseline:
        # Baseline uses unquantized FP16 KV cache = 16 bits/dim
        print(f"  {'baseline':<20} | {baseline:>8.2f} | {'16.0':>8} | {'':>10} | {'':>10}")

    for r in results['experiments']:
        if 'baseline' in r['label']:
            continue
        vs_base = f"{(r['ppl']/baseline - 1)*100:+.2f}%" if baseline else ''
        actual_b = r.get('budget_actual', 0)

        # Match optimal to its uniform counterpart via budget_requested
        vs_uni = ''
        if r.get('allocation_type') == 'optimal':
            req = r.get('budget_requested', 0)
            for r2 in results['experiments']:
                if (r2.get('allocation_type') == 'uniform'
                        and abs(r2.get('budget_requested', -1) - req) < 1e-6):
                    vs_uni = f"{(r['ppl']/r2['ppl'] - 1)*100:+.2f}%"
                    break

        print(f"  {r['label']:<20} | {r['ppl']:>8.2f} | {actual_b:>8.3f} | "
              f"{vs_base:>10} | {vs_uni:>10}")

    # ---- Save ----
    if args.output_dir is None:
        args.output_dir = str(Path(_this_dir).parent / 'results')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model_short = args.model.split('/')[-1]
    budgets_str = '_'.join(str(int(b)) for b, _ in budgets_and_allocs)
    save_path = Path(args.output_dir) / f'thm3_{model_short}_{budgets_str}b.json'

    # Make JSON-serializable
    save_data = {
        'model': args.model,
        'baseline_ppl': results.get('baseline_ppl'),
        'baseline_measured_this_run': not args.skip_baseline,
        'experiments': [],
    }
    for r in results['experiments']:
        entry = {k: v for k, v in r.items() if k != 'schedule'}
        if 'schedule' in r:
            entry['schedule'] = r['schedule']
        save_data['experiments'].append(entry)

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved → {save_path}")


if __name__ == '__main__':
    main()