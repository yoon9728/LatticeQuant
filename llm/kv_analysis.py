"""
LatticeQuant: KV Cache Quantization Analysis
==============================================
Captures KV cache tensors from a real LLM and measures
E₈ lattice quantization quality (MSE gap, cosine similarity).

This script is for offline analysis only.
For perplexity evaluation, use perplexity_eval_v2.py.

Usage:
  python llm/kv_analysis.py --model meta-llama/Llama-3.2-1B
  python llm/kv_analysis.py --model meta-llama/Llama-3.2-1B --bits 3.0
"""

import torch
import numpy as np
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from e8_quantizer import encode_e8, compute_scale, G_E8


# ============================================================
# KV Cache Capture
# ============================================================

def load_model(model_name: str, device: str = 'cuda'):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == 'cuda':
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_gb:.1f} GB")

    if '8B' in model_name or '8b' in model_name:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Parameters: {param_count:.2f}B")
    print(f"Model loaded successfully")
    print()

    return model, tokenizer


def capture_kv_cache(model, tokenizer, text: str, device: str = 'cuda') -> list:
    """
    Run forward pass and capture KV cache.
    Returns: list of (key, value) tensors per layer, head_dim
    """
    inputs = tokenizer(text, return_tensors='pt').to(device)
    seq_len = inputs['input_ids'].shape[1]
    print(f"Input sequence length: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv_raw = outputs.past_key_values

    if hasattr(past_kv_raw, 'key_cache'):
        num_layers = len(past_kv_raw.key_cache)
        past_kv = [(past_kv_raw.key_cache[i], past_kv_raw.value_cache[i])
                    for i in range(num_layers)]
    else:
        past_kv = list(past_kv_raw)
        num_layers = len(past_kv)

    num_heads = past_kv[0][0].shape[1]
    head_dim = past_kv[0][0].shape[3]

    print(f"Captured KV cache:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Seq len: {seq_len}")
    print(f"  Total KV vectors: {num_layers * 2 * num_heads * seq_len:,}")
    print()

    return past_kv, head_dim


# ============================================================
# Offline KV Quantization Analysis
# ============================================================

def analyze_kv_quantization(past_kv, head_dim: int, bits_per_dim: float):
    """
    Apply LatticeQuant to captured KV tensors and measure quality.
    """
    num_layers = len(past_kv)

    print(f"Analyzing KV quantization at {bits_per_dim} bits/dim")
    print("-" * 65)

    all_gaps = []
    all_cos = []
    layer_results = []

    for layer_idx in range(num_layers):
        key = past_kv[layer_idx][0]
        value = past_kv[layer_idx][1]

        for kv_name, tensor in [('K', key), ('V', value)]:
            batch, num_heads, seq_len, hd = tensor.shape
            vectors = tensor.float().reshape(-1, hd)

            assert hd % 8 == 0, f"head_dim {hd} not divisible by 8"

            sigma2 = (vectors ** 2).mean().item()

            blocks = vectors.reshape(-1, 8)
            scale = compute_scale(sigma2, bits_per_dim)
            blocks_norm = blocks / scale
            q = encode_e8(blocks_norm)
            blocks_hat = q * scale
            vectors_hat = blocks_hat.reshape(vectors.shape)

            mse = ((vectors - vectors_hat) ** 2).mean().item()
            d_gauss = sigma2 * (4 ** (-bits_per_dim))
            gap = mse / d_gauss if d_gauss > 0 else float('inf')

            cos_sim = torch.nn.functional.cosine_similarity(
                vectors, vectors_hat, dim=-1).mean().item()

            all_gaps.append(gap)
            all_cos.append(cos_sim)

            layer_results.append({
                'layer': layer_idx,
                'type': kv_name,
                'sigma2': sigma2,
                'mse': mse,
                'gap': gap,
                'cos_sim': cos_sim,
            })

    print(f"{'Layer':>6} | {'Type':>4} | {'sigma2':>10} | {'MSE':>12} | "
          f"{'Gap':>8} | {'cos_sim':>9}")
    print("-" * 65)

    for r in layer_results:
        print(f"{r['layer']:>6} | {r['type']:>4} | {r['sigma2']:>10.6f} | "
              f"{r['mse']:>12.8f} | {r['gap']:>8.4f} | {r['cos_sim']:>9.6f}")

    print("-" * 65)
    mean_gap = np.mean(all_gaps)
    mean_cos = np.mean(all_cos)
    print(f"{'Mean':>6} | {'':>4} | {'':>10} | {'':>12} | "
          f"{mean_gap:>8.4f} | {mean_cos:>9.6f}")
    print()

    theory_gap = 2 * np.pi * np.e * G_E8
    print(f"Theory gap: 2*pi*e*G(E8) = {theory_gap:.4f}")
    print(f"Mean measured gap:        {mean_gap:.4f}")
    print(f"Gap deviation:            {abs(mean_gap - theory_gap) / theory_gap * 100:.2f}%")
    print(f"Mean cosine similarity:   {mean_cos:.6f}")
    print()

    return layer_results


# ============================================================
# Multi-bitrate Analysis
# ============================================================

def multi_bitrate_analysis(past_kv, head_dim: int):
    """Run quantization analysis at multiple bitrates."""
    print("=" * 80)
    print("Multi-bitrate KV Cache Quantization Analysis")
    print("=" * 80)
    print()

    theory_gap = 2 * np.pi * np.e * G_E8

    print(f"{'bits':>6} | {'mean gap':>9} | {'mean cos':>9} | "
          f"{'min cos':>9} | {'sigma2 range':>20} | {'status':>8}")
    print("-" * 80)

    for bits in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        num_layers = len(past_kv)
        all_gaps = []
        all_cos = []
        all_sigma2 = []

        for layer_idx in range(num_layers):
            for tensor in [past_kv[layer_idx][0], past_kv[layer_idx][1]]:
                vectors = tensor.float().reshape(-1, tensor.shape[-1])
                sigma2 = (vectors ** 2).mean().item()
                all_sigma2.append(sigma2)

                blocks = vectors.reshape(-1, 8)
                scale = compute_scale(sigma2, bits)
                blocks_norm = blocks / scale
                q = encode_e8(blocks_norm)
                blocks_hat = q * scale
                vectors_hat = blocks_hat.reshape(vectors.shape)

                mse = ((vectors - vectors_hat) ** 2).mean().item()
                d_gauss = sigma2 * (4 ** (-bits))
                gap = mse / d_gauss if d_gauss > 0 else float('inf')

                cos = torch.nn.functional.cosine_similarity(
                    vectors, vectors_hat, dim=-1).mean().item()

                all_gaps.append(gap)
                all_cos.append(cos)

        mean_gap = np.mean(all_gaps)
        mean_cos = np.mean(all_cos)
        min_cos = np.min(all_cos)
        sigma2_range = f"[{np.min(all_sigma2):.4f}, {np.max(all_sigma2):.4f}]"

        status = "ok" if abs(mean_gap - theory_gap) / theory_gap < 0.05 else "~"

        print(f"{bits:>6.1f} | {mean_gap:>9.4f} | {mean_cos:>9.6f} | "
              f"{min_cos:>9.6f} | {sigma2_range:>20} | {status:>8}")

    print("=" * 80)
    print(f"Theory gap: {theory_gap:.4f}")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LatticeQuant KV Cache Analysis')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='HuggingFace model name')
    parser.add_argument('--bits', type=float, default=4.0,
                        help='Target bits per dimension')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("LatticeQuant: KV Cache Quantization Analysis")
    print("=" * 80)
    print()

    model, tokenizer = load_model(args.model, args.device)

    sample_text = (
        "The Randomized Hadamard Transform is a powerful tool in signal processing "
        "and machine learning. It provides a fast, norm-preserving rotation that "
        "approximately Gaussianizes the coordinate distribution of input vectors. "
        "This property is particularly useful for quantization, where the goal is "
        "to represent continuous values with a finite number of discrete levels. "
        "Lattice quantization extends scalar quantization to higher dimensions, "
        "exploiting the geometric structure of lattices like E8 to achieve better "
        "packing efficiency. The E8 lattice, also known as the Gosset lattice, "
        "is an 8-dimensional lattice with remarkable properties including the "
        "densest known sphere packing in 8 dimensions and a normalized second "
        "moment of approximately 0.0717. When combined with entropy coding, "
        "E8 lattice quantization achieves a Gaussian-normalized distortion gap "
        "of only 1.224, compared to 2.72 for scalar fixed-length quantization. "
        "This represents a fundamental improvement in the rate-distortion tradeoff "
        "that is particularly relevant for compressing key-value caches in large "
        "language models during inference."
    )

    print("--- Step 1: Capture KV Cache ---")
    past_kv, head_dim = capture_kv_cache(model, tokenizer, sample_text, args.device)

    print(f"--- Step 2: Detailed Analysis at {args.bits} bits/dim ---")
    analyze_kv_quantization(past_kv, head_dim, args.bits)

    print("--- Step 3: Multi-bitrate Sweep ---")
    multi_bitrate_analysis(past_kv, head_dim)

    del past_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Analysis complete.")


if __name__ == '__main__':
    main()