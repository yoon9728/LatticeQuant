"""
LatticeQuant v2: Long-Context Evaluation
==========================================
Run on Colab Pro (A100 40GB).
Measures PPL at seq_len = 2048, 4096, 8192, 16384, 32768.

Setup on Colab:
  !pip install transformers datasets bitsandbytes accelerate torch
  # Upload core/ and llm/ folders, or clone repo
"""

import torch, sys, os, time, json, argparse
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, 'llm')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from compressed_kv_cache import CompressedKVCache


def eval_ppl(model, tokenizer, max_length=2048, stride=512, 
             bits=None, device='cuda:0'):
    """Evaluate perplexity with optional KV compression."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(dataset['text'])
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    seq_len = input_ids.size(1)
    
    nlls, n_tokens, prev_end = [], 0, 0
    
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        chunk = input_ids[:, begin:end].to(device)
        
        if bits is not None:
            cache = CompressedKVCache(bits_per_dim=bits, eval_only_no_entropy=True)
        else:
            cache = DynamicCache()
        
        with torch.no_grad():
            out = model(chunk, past_key_values=cache, use_cache=True)
        
        logits = out.logits[:, :-1, :].contiguous()
        labels = chunk[:, 1:].contiguous()
        eval_start = max(0, chunk.size(1) - trg_len - 1)
        
        loss = torch.nn.CrossEntropyLoss(reduction='sum')(
            logits[:, eval_start:].reshape(-1, logits.size(-1)),
            labels[:, eval_start:].reshape(-1))
        
        nlls.append(loss.item())
        n_tokens += labels[:, eval_start:].numel()
        prev_end = end
        if end >= seq_len - 1:
            break
        del cache, out
        torch.cuda.empty_cache()
    
    return np.exp(sum(nlls) / n_tokens)


def measure_memory(model, tokenizer, seq_len, bits=None, device='cuda:0'):
    """Measure KV cache VRAM at given seq_len."""
    ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    
    if bits is not None:
        cache = CompressedKVCache(bits_per_dim=bits, eval_only_no_entropy=True)
    else:
        cache = DynamicCache()
    
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    
    mem_after = torch.cuda.memory_allocated()
    kv_mem = (mem_after - mem_before) / 1e6  # MB
    
    # For compressed cache, also get compressed-only size
    comp_mb = None
    if bits is not None and hasattr(cache, 'compressed_bytes'):
        try:
            comp_mb = cache.compressed_bytes() / 1e6
        except:
            pass
    
    del cache, ids
    torch.cuda.empty_cache()
    return kv_mem, comp_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--seq-lengths', nargs='+', type=int, 
                        default=[2048, 4096, 8192])
    parser.add_argument('--bits', nargs='+', type=int, default=[4])
    parser.add_argument('--skip-ppl', action='store_true')
    parser.add_argument('--skip-memory', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map='cuda:0')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get model config
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', 
                         config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    print(f"Config: {n_layers}L × {n_kv_heads}H × {head_dim}D")
    print(f"Seq lengths: {args.seq_lengths}")
    print(f"Bit configs: baseline + {args.bits}")
    print("=" * 70)
    
    results = {}
    
    # === PPL across seq lengths ===
    if not args.skip_ppl:
        print("\n[PPL Evaluation]")
        print("-" * 70)
        for max_len in args.seq_lengths:
            stride = min(512, max_len // 4)
            print(f"\n  seq_len={max_len}, stride={stride}")
            
            # Baseline
            ppl_base = eval_ppl(model, tokenizer, max_length=max_len, 
                               stride=stride, bits=None)
            print(f"    baseline: PPL={ppl_base:.4f}")
            results[f'ppl_baseline_seq{max_len}'] = ppl_base
            
            # Compressed
            for b in args.bits:
                ppl_comp = eval_ppl(model, tokenizer, max_length=max_len, 
                                   stride=stride, bits=b)
                delta = (ppl_comp / ppl_base - 1) * 100
                print(f"    {b}b: PPL={ppl_comp:.4f} ({delta:+.2f}%)")
                results[f'ppl_{b}b_seq{max_len}'] = ppl_comp
                results[f'delta_{b}b_seq{max_len}'] = delta
    
    # === Memory across seq lengths ===
    if not args.skip_memory:
        print("\n[Memory Measurement]")
        print("-" * 70)
        for seq_len in args.seq_lengths:
            print(f"\n  seq_len={seq_len}")
            
            # FP16 theoretical
            fp16_mb = 2 * n_layers * n_kv_heads * seq_len * head_dim * 2 / 1e6
            print(f"    FP16 theoretical: {fp16_mb:.1f} MB")
            
            # Baseline measured
            mem_base, _ = measure_memory(model, tokenizer, seq_len, 
                                        bits=None)
            print(f"    baseline VRAM: {mem_base:.1f} MB")
            
            for b in args.bits:
                mem_comp, comp_mb = measure_memory(model, tokenizer, 
                                                   seq_len, bits=b)
                ratio = fp16_mb / comp_mb if comp_mb else 0
                print(f"    {b}b VRAM: {mem_comp:.1f} MB"
                      f" (compressed-only: {comp_mb:.1f} MB, "
                      f"{ratio:.2f}x)" if comp_mb else "")
                results[f'mem_{b}b_seq{seq_len}'] = comp_mb
                results[f'ratio_{b}b_seq{seq_len}'] = ratio
    
    # === Summary table ===
    print("\n" + "=" * 70)
    print("SUMMARY: Long-Context Scaling")
    print("=" * 70)
    
    if not args.skip_ppl:
        print(f"\n{'seq_len':>10} | {'baseline':>10} | ", end="")
        for b in args.bits:
            print(f"{b}b PPL | {b}b Δ%  | ", end="")
        print()
        print("-" * (20 + 20 * len(args.bits)))
        for max_len in args.seq_lengths:
            base = results.get(f'ppl_baseline_seq{max_len}', 0)
            print(f"{max_len:>10} | {base:>10.4f} | ", end="")
            for b in args.bits:
                ppl = results.get(f'ppl_{b}b_seq{max_len}', 0)
                delta = results.get(f'delta_{b}b_seq{max_len}', 0)
                print(f"{ppl:>7.4f} | {delta:>+6.2f}% | ", end="")
            print()
    
    # Save
    model_name = args.model.split('/')[-1]
    out_path = f'results/long_context_{model_name}.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()