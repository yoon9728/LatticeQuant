"""
CABA E₈ scale 디버깅: 4b vs 5b MSE 직접 비교.

5b PPL(101) > 4b PPL(67.7) 원인 진단.
E₈ quantizer 자체가 5b에서 MSE가 낮아지는지 확인.
"""

import torch
import numpy as np
import math
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from e8_quantizer import encode_e8


def diagnose_e8_scale(bits_list=[3, 4, 5, 6]):
    """Synthetic Gaussian에서 E₈ MSE vs bits 확인."""
    print("=" * 60)
    print("Test 1: Synthetic Gaussian (σ²=1.0)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(10000, 8)
    sigma2 = 1.0

    for b in bits_list:
        a2 = 2 * math.pi * math.e * sigma2 * (4.0 ** (-b))
        a = math.sqrt(a2)
        x_scaled = x / a
        q = encode_e8(x_scaled)
        x_hat = q * a
        mse = ((x - x_hat) ** 2).mean().item()
        theory = 0.0717 * 2 * math.pi * math.e * sigma2 * (4.0 ** (-b))

        print(f"  {b}b: scale={a:.6f}, MSE={mse:.6f}, theory={theory:.6f}, "
              f"ratio={mse/theory:.3f}, "
              f"|x/a| range=[{(x_scaled).abs().min():.1f}, {(x_scaled).abs().max():.1f}]")

    print()


def diagnose_e8_varying_sigma(bits_list=[4, 5]):
    """다양한 σ² 범위에서 4b vs 5b 비교."""
    print("=" * 60)
    print("Test 2: Varying σ² (per-block scale)")
    print("=" * 60)

    torch.manual_seed(42)

    for sigma2 in [100.0, 1.0, 0.01, 0.001, 0.0001, 1e-6]:
        x = torch.randn(1000, 8) * math.sqrt(sigma2)
        results = {}

        for b in bits_list:
            a2 = 2 * math.pi * math.e * sigma2 * (4.0 ** (-b))
            a = math.sqrt(a2 + 1e-30)
            x_scaled = x / a
            q = encode_e8(x_scaled)
            x_hat = q * a
            mse = ((x - x_hat) ** 2).mean().item()
            rel_mse = mse / (sigma2 + 1e-30)
            results[b] = {"mse": mse, "rel_mse": rel_mse, "scale": a}

        r4, r5 = results[4], results[5]
        ratio = r5["mse"] / (r4["mse"] + 1e-30)
        status = "OK" if ratio < 0.5 else "BAD" if ratio > 0.8 else "WARN"

        print(f"  σ²={sigma2:.1e}: "
              f"4b MSE={r4['mse']:.2e} (scale={r4['scale']:.2e}), "
              f"5b MSE={r5['mse']:.2e} (scale={r5['scale']:.2e}), "
              f"5b/4b={ratio:.3f} [{status}]")

    print()


def diagnose_per_block_pathology():
    """Per-block scale에서 near-zero block이 5b를 망가뜨리는지 확인."""
    print("=" * 60)
    print("Test 3: Mixed blocks (simulating sorted Qwen KV)")
    print("=" * 60)

    torch.manual_seed(42)

    # Simulate sorted blocks: 첫 4개는 high variance, 나머지 12개는 low variance
    n_blocks = 16
    sigmas = [10.0] * 4 + [1.0] * 4 + [0.1] * 4 + [0.001] * 4

    for b in [4, 5]:
        total_mse = 0
        block_mses = []
        for i, s in enumerate(sigmas):
            x = torch.randn(100, 8) * s
            sigma2 = (x ** 2).mean().item()  # actual per-block σ²
            a2 = 2 * math.pi * math.e * sigma2 * (4.0 ** (-b))
            a = math.sqrt(a2 + 1e-30)

            x_scaled = x / a
            q = encode_e8(x_scaled)
            x_hat = q * a
            mse = ((x - x_hat) ** 2).mean().item()
            block_mses.append(mse)
            total_mse += mse

        avg_mse = total_mse / n_blocks
        print(f"\n  {b}b: avg MSE = {avg_mse:.6f}")
        for i, (s, m) in enumerate(zip(sigmas, block_mses)):
            flag = " <<<" if b == 5 and i > 0 and block_mses[i] > block_mses[i-1] * 10 else ""
            print(f"    block {i:>2} (σ={s:>8.3f}): MSE={m:.2e}{flag}")

    print()


def diagnose_actual_kv():
    """실제 Qwen KV 데이터에서 4b vs 5b MSE 비교."""
    print("=" * 60)
    print("Test 4: Actual Qwen KV (if available)")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_8bit=True)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B", quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()

        text = "The transformer architecture has revolutionized natural language processing. " * 20
        inputs = tok(text, return_tensors="pt", max_length=512, truncation=True).to("cuda:0")

        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        past = out.past_key_values
        if hasattr(past, 'key_cache'):
            k0 = past.key_cache[0][0, 0].float().cpu()  # layer 0, head 0
        else:
            k0 = list(past)[0][0][0, 0].float().cpu()

        print(f"  K[0,0] shape: {k0.shape}")
        print(f"  K[0,0] σ² range: [{(k0**2).mean(dim=0).min():.4f}, {(k0**2).mean(dim=0).max():.4f}]")
        print(f"  K[0,0] AM/GM per block:")

        # Sort dimensions (CABA sorted)
        dim_sigma2 = (k0 ** 2).mean(dim=0)  # per-dim σ²
        perm = torch.argsort(dim_sigma2)
        k_sorted = k0[:, perm]

        n_blocks = k_sorted.shape[1] // 8
        for b in [4, 5]:
            total_mse = 0
            block_details = []
            for i in range(n_blocks):
                block = k_sorted[:, i*8:(i+1)*8]  # (seq, 8)
                sigma2 = (block ** 2).mean().item()
                a2 = 2 * math.pi * math.e * sigma2 * (4.0 ** (-b))
                a = math.sqrt(a2 + 1e-30)
                x_scaled = block / a
                q = encode_e8(x_scaled)
                x_hat = q * a
                mse = ((block - x_hat) ** 2).mean().item()
                rel = mse / (sigma2 + 1e-30)
                total_mse += mse
                block_details.append((sigma2, mse, rel, (x_scaled.abs().max().item())))

            print(f"\n  {b}b total MSE: {total_mse/n_blocks:.6f}")
            for i, (s2, mse, rel, xmax) in enumerate(block_details):
                print(f"    block {i:>2}: σ²={s2:.2e}, MSE={mse:.2e}, relMSE={rel:.4f}, |x/a|_max={xmax:.1f}")

    except Exception as e:
        print(f"  Skipped (model not available): {e}")


if __name__ == "__main__":
    diagnose_e8_scale()
    diagnose_e8_varying_sigma()
    diagnose_per_block_pathology()
    diagnose_actual_kv()