"""
LatticeQuant v2 — End-to-End Evaluation
=========================================
Three separate measurements, each with clear semantics:

1. evaluate_ppl()       — Perplexity via sliding window with compressed KV
2. measure_memory()     — VRAM measurement (memory_allocated + accounting)
3. measure_throughput() — Autoregressive decode tok/s

All measurements use 8-bit weight quantization (BitsAndBytes).
Baseline is "8bit-weight + uncompressed KV", not FP16.

Compatible with CompressedKVCache v6 (rANS entropy-coded, optimized side info).

Usage:
  python llm/e2e_eval.py --model meta-llama/Llama-3.1-8B --all
  python llm/e2e_eval.py --model meta-llama/Llama-3.1-8B --ppl-only --all
  python llm/e2e_eval.py --model meta-llama/Llama-3.1-8B --skip-ppl --all
  python llm/e2e_eval.py --model meta-llama/Llama-3.1-8B --bits 4
  python llm/e2e_eval.py --model Qwen/Qwen2.5-7B --all
"""

import torch
import numpy as np
import time
import json
import argparse
import gc
from typing import Optional, List
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset

from compressed_kv_cache import CompressedKVCache


# ============================================================
# Model loading
# ============================================================

def load_model(model_name: str):
    """Load model with 8-bit weight quantization."""
    print(f"Loading {model_name} (8-bit weights)...")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='cuda:0',
    )
    model.eval()

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {mem:.1f} GB VRAM")
    return model, tokenizer


def get_model_kv_config(model):
    """Extract n_layers, n_kv_heads, head_dim from model config."""
    config = model.config
    n_layers = config.num_hidden_layers

    n_kv_heads = getattr(config, 'num_key_value_heads',
                         getattr(config, 'num_attention_heads', None))

    head_dim = getattr(config, 'head_dim', None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    return n_layers, n_kv_heads, head_dim


# ============================================================
# 1. Perplexity (sliding window with compressed KV)
# ============================================================

def evaluate_ppl(
    model,
    tokenizer,
    bits_per_dim: Optional[int] = None,
    max_length: int = 2048,
    stride: int = 512,
) -> dict:
    """
    Perplexity on wikitext2 with sliding window.

    Each window does a full forward with the cache, so attention
    reads quantized-dequantized KV. PPL captures quality impact.

    Uses eval_only_no_entropy=True to skip rANS encoding (lossless,
    does not affect PPL, but saves ~10-30x wall time).
    """
    label = f"{bits_per_dim}b" if bits_per_dim else "baseline"
    print(f"\n  [{label}] Evaluating PPL...")

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids  # CPU
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

        if bits_per_dim is not None:
            cache = CompressedKVCache(bits_per_dim=bits_per_dim, eval_only_no_entropy=True)
        else:
            cache = DynamicCache()

        with torch.no_grad():
            outputs = model(
                chunk_ids,
                past_key_values=cache,
                use_cache=True,
            )

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk_ids[:, 1:].contiguous()

        # Only evaluate non-overlap tokens
        eval_start = max(0, chunk_ids.size(1) - trg_len - 1)
        eval_logits = shift_logits[:, eval_start:, :]
        eval_labels = shift_labels[:, eval_start:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(
            eval_logits.reshape(-1, eval_logits.size(-1)),
            eval_labels.reshape(-1),
        )

        nlls.append(loss.item())
        n_tokens += eval_labels.numel()

        prev_end = end
        if end >= seq_len - 1:
            break

        if len(nlls) % 20 == 0:
            print(f"      {n_tokens:,} tokens, running PPL={np.exp(sum(nlls)/n_tokens):.2f}")

        del cache, outputs, logits

    t_elapsed = time.time() - t_start
    ppl = np.exp(sum(nlls) / n_tokens)

    print(f"    PPL={ppl:.4f}, {n_tokens:,} tokens, {t_elapsed:.0f}s")

    return {
        'label': label,
        'bits': bits_per_dim if bits_per_dim else 'baseline',
        'ppl': ppl,
        'n_tokens': n_tokens,
        'eval_time_sec': t_elapsed,
    }


# ============================================================
# 2. Memory measurement (VRAM via memory_allocated + accounting)
# ============================================================

def measure_memory(
    model,
    tokenizer,
    bits_per_dim: Optional[int] = None,
    seq_len: int = 2048,
) -> dict:
    """
    Measure KV cache VRAM at a specific sequence length.

    Strategy: run forward to populate cache, then delete outputs
    and clear activation buffers before measuring. This isolates
    the KV cache VRAM from intermediate activations.
    """
    label = f"{bits_per_dim}b" if bits_per_dim else "baseline"
    n_layers, n_kv_heads, head_dim = get_model_kv_config(model)
    print(f"\n  [{label}] Measuring memory (seq={seq_len}, {n_layers}L×{n_kv_heads}H×{head_dim}D)...")

    # Theoretical FP16 KV size: 2(K+V) × layers × heads × seq × hd × 2bytes
    fp16_theoretical = 2 * n_layers * n_kv_heads * seq_len * head_dim * 2

    torch.manual_seed(42)
    dummy_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device='cuda:0')

    # --- Measure model-only baseline ---
    gc.collect()
    torch.cuda.empty_cache()
    mem_model = torch.cuda.memory_allocated()

    # --- Run forward to populate cache ---
    if bits_per_dim is not None:
        cache = CompressedKVCache(bits_per_dim=bits_per_dim)
    else:
        cache = DynamicCache()

    with torch.no_grad():
        outputs = model(dummy_ids, past_key_values=cache, use_cache=True)

    # --- Delete outputs to free activation buffers ---
    del outputs, dummy_ids
    gc.collect()
    torch.cuda.empty_cache()

    # --- Now memory_allocated = model + KV cache only ---
    mem_with_kv = torch.cuda.memory_allocated()
    kv_vram = mem_with_kv - mem_model

    # --- Accounting ---
    if bits_per_dim is not None:
        comp_bytes = cache.compressed_bytes()
        fp16_bytes = cache.fp16_equivalent_bytes()
        ratio_accounting = fp16_bytes / max(comp_bytes, 1)

        # Measure compressed-only (free decompressed layers)
        for li in range(len(cache._comp_keys)):
            cache._free_layer(li)
        gc.collect()
        torch.cuda.empty_cache()
        mem_compressed_only = torch.cuda.memory_allocated() - mem_model
    else:
        comp_bytes = fp16_theoretical
        fp16_bytes = fp16_theoretical
        ratio_accounting = 1.0
        mem_compressed_only = kv_vram

    ratio_vram = fp16_theoretical / max(kv_vram, 1)

    del cache
    gc.collect()
    torch.cuda.empty_cache()

    # Effective bits/dim
    if bits_per_dim is not None:
        total_dims = fp16_theoretical // 2  # total FP16 values
        eff_bits = (comp_bytes * 8) / max(total_dims, 1)
    else:
        eff_bits = 16.0

    print(f"    VRAM (KV only):          {kv_vram/1e6:.1f} MB")
    print(f"    FP16 theoretical:        {fp16_theoretical/1e6:.1f} MB")
    print(f"    VRAM ratio:              {ratio_vram:.2f}x")
    if bits_per_dim is not None:
        print(f"    Compressed-only VRAM:    {mem_compressed_only/1e6:.1f} MB")
        print(f"    Accounting ratio:        {ratio_accounting:.2f}x")
        print(f"    Effective bits/dim:      {eff_bits:.3f}")

    return {
        'label': label,
        'bits': bits_per_dim if bits_per_dim else 'baseline',
        'seq_len': seq_len,
        'kv_vram_bytes': kv_vram,
        'compressed_only_bytes': mem_compressed_only,
        'compressed_bytes': comp_bytes,
        'fp16_bytes': fp16_theoretical,
        'compression_ratio': ratio_vram,
        'accounting_ratio': ratio_accounting,
        'effective_bits_per_dim': eff_bits,
    }


# ============================================================
# 3. Decode throughput (autoregressive generation)
# ============================================================

def measure_throughput(
    model,
    tokenizer,
    bits_per_dim: Optional[int] = None,
    prefill_len: int = 128,
    decode_tokens: int = 64,
    warmup: int = 3,
    repeats: int = 10,
) -> dict:
    """
    Measure autoregressive decode throughput.

    Prefill with prefill_len tokens, then generate decode_tokens
    one at a time. Measures decode phase only.

    Note: compressed path uses CPU rANS encoding per step.
    For paper-grade throughput, integrate FusedTritonKVCache.
    """
    label = f"{bits_per_dim}b" if bits_per_dim else "baseline"
    print(f"\n  [{label}] Measuring throughput (prefill={prefill_len}, decode={decode_tokens})...")

    torch.manual_seed(42)
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, prefill_len), device='cuda:0')

    timings = []

    for trial in range(warmup + repeats):
        if bits_per_dim is not None:
            cache = CompressedKVCache(bits_per_dim=bits_per_dim, eval_only_no_entropy=True)
        else:
            cache = DynamicCache()

        # Prefill
        with torch.no_grad():
            outputs = model(prompt_ids, past_key_values=cache, use_cache=True)

        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        # Decode
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            for _ in range(decode_tokens):
                outputs = model(
                    next_token,
                    past_key_values=cache,
                    use_cache=True,
                )
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()
        t1 = time.time()

        if trial >= warmup:
            timings.append(t1 - t0)

        del cache, outputs
        torch.cuda.empty_cache()

    avg_time = np.mean(timings)
    med_time = np.median(timings)
    tok_per_sec = decode_tokens / avg_time
    ms_per_tok = avg_time / decode_tokens * 1000
    med_ms_per_tok = med_time / decode_tokens * 1000

    print(f"    {tok_per_sec:.1f} tok/s, {ms_per_tok:.1f} ms/tok (avg), {med_ms_per_tok:.1f} ms/tok (med)")

    return {
        'label': label,
        'bits': bits_per_dim if bits_per_dim else 'baseline',
        'prefill_len': prefill_len,
        'decode_tokens': decode_tokens,
        'decode_time_sec': avg_time,
        'tok_per_sec': tok_per_sec,
        'ms_per_tok': ms_per_tok,
        'med_ms_per_tok': med_ms_per_tok,
    }


# ============================================================
# Full evaluation suite
# ============================================================

def run_suite(model_name: str, bits_list: List[int], max_length: int = 2048,
              ppl_only: bool = False, skip_ppl: bool = False):
    """Run all measurements and produce summary table."""

    model, tokenizer = load_model(model_name)
    n_layers, n_kv_heads, head_dim = get_model_kv_config(model)

    print()
    print("=" * 85)
    print(f"LatticeQuant v2 Evaluation: {model_name}")
    print(f"  Weight quantization: 8-bit (BitsAndBytes)")
    print(f"  KV cache: E₈ lattice quantization (LatticeQuant)")
    print(f"  Model config: {n_layers}L × {n_kv_heads} KV heads × {head_dim}D")
    print("=" * 85)

    all_results = {'model': model_name, 'ppl': {}, 'memory': {}, 'throughput': {}}

    # --- PPL ---
    if not skip_ppl:
        print("\n[1/3] Perplexity Evaluation")
        print("-" * 40)

        ppl_baseline = evaluate_ppl(model, tokenizer, bits_per_dim=None, max_length=max_length)
        all_results['ppl']['baseline'] = ppl_baseline

        for bits in bits_list:
            gc.collect()
            torch.cuda.empty_cache()
            r = evaluate_ppl(model, tokenizer, bits_per_dim=bits, max_length=max_length)
            all_results['ppl'][f'{bits}b'] = r

        if ppl_only:
            print()
            print("=" * 60)
            print(f"PPL Summary: {model_name}")
            print("-" * 60)
            baseline_ppl = ppl_baseline['ppl']
            print(f"  {'Config':>12} | {'PPL':>8} | {'Δ%':>8}")
            print(f"  {'-'*35}")
            for cfg in ['baseline'] + [f'{b}b' for b in bits_list]:
                p = all_results['ppl'][cfg]
                delta = (p['ppl'] / baseline_ppl - 1) * 100
                label = '8bit+FP16kv' if cfg == 'baseline' else cfg
                print(f"  {label:>12} | {p['ppl']:>8.2f} | {delta:>+7.2f}%")
            print("=" * 60)
            _save_results(all_results, model_name)
            return all_results
    else:
        print("\n[1/3] Perplexity — SKIPPED")

    # --- Memory ---
    print("\n[2/3] Memory Measurement (seq_len=2048)")
    print("-" * 40)

    mem_baseline = measure_memory(model, tokenizer, bits_per_dim=None, seq_len=2048)
    all_results['memory']['baseline'] = mem_baseline

    for bits in bits_list:
        gc.collect()
        torch.cuda.empty_cache()
        r = measure_memory(model, tokenizer, bits_per_dim=bits, seq_len=2048)
        all_results['memory'][f'{bits}b'] = r

    # --- Throughput ---
    print("\n[3/3] Decode Throughput (prefill=128, decode=64)")
    print("-" * 40)

    tp_baseline = measure_throughput(model, tokenizer, bits_per_dim=None)
    all_results['throughput']['baseline'] = tp_baseline

    for bits in bits_list:
        gc.collect()
        torch.cuda.empty_cache()
        r = measure_throughput(model, tokenizer, bits_per_dim=bits)
        all_results['throughput'][f'{bits}b'] = r

    # --- Summary ---
    print()
    print("=" * 85)
    print(f"SUMMARY: {model_name} (8-bit weights, {n_layers}L×{n_kv_heads}H×{head_dim}D)")
    print("=" * 85)

    baseline_ppl = all_results['ppl'].get('baseline', {}).get('ppl', 0)

    print(f"\n{'Config':>12} | {'PPL':>8} | {'Δ%':>7} | "
          f"{'KV MB':>7} | {'Ratio':>6} | {'eff b/d':>7} | {'tok/s':>7} | {'ms/tok':>7}")
    print(f"{'-'*85}")

    configs = ['baseline'] + [f'{b}b' for b in bits_list]

    for cfg in configs:
        p = all_results['ppl'].get(cfg, {})
        m = all_results['memory'].get(cfg, {})
        t = all_results['throughput'].get(cfg, {})

        ppl = p.get('ppl', 0)
        delta = (ppl / baseline_ppl - 1) * 100 if baseline_ppl > 0 else 0
        kv_mb = m.get('kv_vram_bytes', 0) / 1024 / 1024
        ratio = m.get('compression_ratio', 1.0)
        eff_bits = m.get('effective_bits_per_dim', 16.0)
        tps = t.get('tok_per_sec', 0)
        mpt = t.get('ms_per_tok', 0)

        ratio_str = f"{ratio:.2f}x" if ratio > 1 else "1.00x"
        eff_str = f"{eff_bits:.1f}" if eff_bits < 16 else "16.0"
        label = cfg if cfg != 'baseline' else '8bit+FP16kv'

        ppl_str = f"{ppl:>8.2f}" if ppl > 0 else "    skip"
        delta_str = f"{delta:>+6.1f}%" if ppl > 0 else "   skip"

        print(f"{label:>12} | {ppl_str} | {delta_str} | "
              f"{kv_mb:>6.1f} | {ratio_str:>6} | {eff_str:>7} | {tps:>7.1f} | {mpt:>7.1f}")

    print("=" * 85)

    _save_results(all_results, model_name)
    return all_results


def _save_results(all_results: dict, model_name: str):
    """Save results to JSON."""
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)

    save_data = {
        'model': model_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    for section in ['ppl', 'memory', 'throughput']:
        for key, val in all_results[section].items():
            for k, v in val.items():
                save_data[f'{section}_{key}_{k}'] = v

    model_short = model_name.split('/')[-1]
    save_path = out_dir / f'e2e_{model_short}.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='LatticeQuant v2 E2E Evaluation')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--bits', type=int, default=None)
    parser.add_argument('--all', action='store_true', help='Evaluate 3, 4, 5 bits')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--ppl-only', action='store_true',
                        help='Only measure PPL (skip memory/throughput)')
    parser.add_argument('--skip-ppl', action='store_true',
                        help='Skip PPL, run memory + throughput only')

    args = parser.parse_args()

    if args.all:
        bits_list = [3, 4, 5]
    elif args.bits:
        bits_list = [args.bits]
    else:
        bits_list = [4]

    run_suite(args.model, bits_list, args.max_length,
              ppl_only=args.ppl_only, skip_ppl=args.skip_ppl)


if __name__ == '__main__':
    main()