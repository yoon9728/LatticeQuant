"""
Sequential Corruption Diagnostic
==================================
Track how quantization error accumulates layer-by-layer
in actual quantized inference (not single-layer injection).

Three conditions per model:
  - K+V: both keys and values quantized
  - K-only: only keys quantized, values clean
  - V-only: only values quantized, keys clean

For each condition, measures:
  - Per-layer hidden state divergence from clean baseline
  - Where the divergence explodes (if it does)
  - Relative error growth rate

Uses simple per-channel uniform quantization (not E8) as proxy.
The point is to observe the propagation pattern, not test a
specific quantizer. If the pattern differs between Llama and Qwen,
the cause is architectural, not quantizer-specific.

Usage:
  python -m ddt.sequential_corruption --model meta-llama/Llama-3.1-8B
  python -m ddt.sequential_corruption --model Qwen/Qwen2.5-7B
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_uniform(x, bits=4):
    """Per-token symmetric uniform quantization.
    x shape: (B, T, d) from k_proj/v_proj output.
    Scale computed per token (across feature dim)."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    max_val = 2 ** (bits - 1) - 1
    x_q = (x / scale * max_val).round().clamp(-max_val, max_val) * scale / max_val
    return x_q


class KVQuantHook:
    """Hook that intercepts and quantizes KV cache during forward pass."""

    def __init__(self, model, mode='kv', bits=4):
        """
        mode: 'kv' (both), 'k' (keys only), 'v' (values only), 'clean' (no quant)
        """
        self.model = model
        self.mode = mode
        self.bits = bits
        self.hooks = []
        self.layer_errors = {}  # layer -> {'k_mse': float, 'v_mse': float}

    def register(self):
        layers = list(self.model.model.layers)
        for idx, layer in enumerate(layers):
            attn = layer.self_attn

            def make_hook(layer_idx):
                def hook(module, args, kwargs, output):
                    # output = (attn_output, attn_weights, past_key_value)
                    # past_key_value contains the KV cache
                    # We need to intercept BEFORE attention computation
                    # But hooks fire AFTER forward...
                    #
                    # Alternative approach: hook on k_proj and v_proj outputs
                    pass
                return hook

            # Actually, we need to modify the KV BEFORE attention.
            # Use forward_pre_hook on self_attn to modify input,
            # or better: hook on k_proj and v_proj individually

        # Simpler approach: monkey-patch the attention forward
        for idx, layer in enumerate(layers):
            attn = layer.self_attn
            original_forward = attn.forward

            def make_patched(layer_idx, orig_fwd, parent_self):
                def patched_forward(*args, **kwargs):
                    # Call original to get KV
                    output = orig_fwd(*args, **kwargs)
                    # Can't modify KV after the fact...
                    return output
                return patched_forward

        # Best approach: hook on k_proj and v_proj output tensors
        # using register_forward_hook on the Linear layers
        for idx, layer in enumerate(layers):
            attn = layer.self_attn

            if self.mode in ('kv', 'k'):
                def make_k_hook(li):
                    def hook(module, input, output):
                        q_out = quantize_uniform(output, self.bits)
                        k_mse = ((output - q_out) ** 2).mean().item()
                        if li not in self.layer_errors:
                            self.layer_errors[li] = {}
                        self.layer_errors[li]['k_mse'] = k_mse
                        return q_out
                    return hook
                h = attn.k_proj.register_forward_hook(make_k_hook(idx))
                self.hooks.append(h)

            if self.mode in ('kv', 'v'):
                def make_v_hook(li):
                    def hook(module, input, output):
                        q_out = quantize_uniform(output, self.bits)
                        v_mse = ((output - q_out) ** 2).mean().item()
                        if li not in self.layer_errors:
                            self.layer_errors[li] = {}
                        self.layer_errors[li]['v_mse'] = v_mse
                        return q_out
                    return hook
                h = attn.v_proj.register_forward_hook(make_v_hook(idx))
                self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.layer_errors.clear()


class HiddenStateTracker:
    """Capture hidden states at every layer output."""

    def __init__(self, model):
        self.model = model
        self.states = {}
        self.hooks = []

    def register(self):
        for idx, layer in enumerate(self.model.model.layers):
            def make_hook(li):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.states[li] = output[0].detach().float()
                    else:
                        self.states[li] = output.detach().float()
                return hook
            h = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(h)

    def clear(self):
        self.states.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_condition(model, input_ids, mode, bits, clean_states):
    """Run one condition and compute per-layer divergence."""
    tracker = HiddenStateTracker(model)
    tracker.register()

    quant_hook = KVQuantHook(model, mode=mode, bits=bits)
    if mode != 'clean':
        quant_hook.register()

    with torch.no_grad():
        output = model(input_ids, use_cache=False, output_attentions=False)

    logits = output.logits
    # Compute loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).item()
    ppl = np.exp(loss) if loss < 20 else float('inf')

    # Per-layer divergence
    divergence = []
    for li in sorted(tracker.states.keys()):
        if li in clean_states:
            clean = clean_states[li]
            quant = tracker.states[li]
            mse = ((clean - quant) ** 2).mean().item()
            clean_norm = (clean ** 2).mean().item()
            rel_mse = mse / max(clean_norm, 1e-20)
            divergence.append({
                'layer': li,
                'mse': mse,
                'rel_mse': rel_mse,
                'clean_norm': clean_norm,
            })

    tracker.remove()
    quant_hook.remove()

    return {
        'loss': loss,
        'ppl': ppl,
        'divergence': divergence,
        'kv_errors': dict(quant_hook.layer_errors) if mode != 'clean' else {},
    }


def run_model(model_name, bits=4, seq_len=512):
    """Run all conditions for one model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  {model_name} @ {bits}bit")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Calibration tokens
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)
    print(f"  Tokens: {input_ids.shape[1]}")

    # Clean baseline
    print("  Running clean baseline...")
    tracker = HiddenStateTracker(model)
    tracker.register()
    with torch.no_grad():
        clean_out = model(input_ids, use_cache=False)
    clean_states = dict(tracker.states)
    tracker.remove()

    clean_logits = clean_out.logits
    shift_logits = clean_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    clean_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).item()
    clean_ppl = np.exp(clean_loss)
    print(f"  Clean: loss={clean_loss:.4f}, PPL={clean_ppl:.2f}")

    # Three conditions
    results = {'clean_ppl': clean_ppl, 'clean_loss': clean_loss}

    for mode in ['kv', 'k', 'v']:
        label = {'kv': 'K+V', 'k': 'K-only', 'v': 'V-only'}[mode]
        print(f"\n  Running {label} quantized...")
        r = run_condition(model, input_ids, mode, bits, clean_states)
        results[mode] = r
        print(f"  {label}: loss={r['loss']:.4f}, PPL={r['ppl']:.2f}")

        # Print divergence profile
        print(f"  {'Layer':>5} {'relMSE':>12} {'|Δh|/|h|':>12}")
        for d in r['divergence']:
            sqrt_rel = np.sqrt(d['rel_mse'])
            bar = '#' * min(int(sqrt_rel * 100), 50)
            print(f"  {d['layer']:5d} {d['rel_mse']:12.6f} {sqrt_rel:12.6f}  {bar}")

    # Summary comparison
    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {model_name}")
    print(f"  {'='*50}")
    print(f"  Clean PPL:  {clean_ppl:.2f}")
    print(f"  K+V PPL:    {results['kv']['ppl']:.2f}  "
          f"(+{(results['kv']['ppl']/clean_ppl - 1)*100:.1f}%)")
    print(f"  K-only PPL: {results['k']['ppl']:.2f}  "
          f"(+{(results['k']['ppl']/clean_ppl - 1)*100:.1f}%)")
    print(f"  V-only PPL: {results['v']['ppl']:.2f}  "
          f"(+{(results['v']['ppl']/clean_ppl - 1)*100:.1f}%)")

    # Find explosion layer (if any)
    for mode in ['kv', 'k', 'v']:
        divs = results[mode]['divergence']
        if len(divs) >= 2:
            max_jump = 0
            max_jump_layer = -1
            for i in range(1, len(divs)):
                prev = max(divs[i-1]['rel_mse'], 1e-20)
                curr = divs[i]['rel_mse']
                jump = curr / prev
                if jump > max_jump:
                    max_jump = jump
                    max_jump_layer = divs[i]['layer']
            label = {'kv': 'K+V', 'k': 'K-only', 'v': 'V-only'}[mode]
            print(f"  {label} max jump: {max_jump:.1f}x at layer {max_jump_layer}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sequential corruption diagnostic")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/ddt")
    args = parser.parse_args()

    results = run_model(args.model, args.bits, args.seq_len)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = os.path.join(args.output_dir,
                        f"sequential_corruption_{tag}_{args.bits}bit.json")

    save = {
        'model': args.model,
        'bits': args.bits,
        'clean_ppl': results['clean_ppl'],
        'kv_ppl': results['kv']['ppl'],
        'k_ppl': results['k']['ppl'],
        'v_ppl': results['v']['ppl'],
        'kv_divergence': results['kv']['divergence'],
        'k_divergence': results['k']['divergence'],
        'v_divergence': results['v']['divergence'],
    }
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()