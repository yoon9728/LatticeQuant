"""
CABA (Cascade-Aware Block Assignment) — Phase 1: Dimension Analysis

Per-dimension second moment profiling + variance-sorted block assignment.

이 파일이 검증하는 것:
  - Qwen vs Llama의 block-wise anisotropy 차이
  - Variance-sorted permutation이 block heterogeneity를 얼마나 줄이는지

이 파일이 검증하지 않는 것:
  - per-dim propagation sensitivity γ_i (Phase 3에서 측정)
  - γ_i와 μ_i의 anti-correlation (Phase 3에서 검증)

즉 이 파일은 anti-correlation hypothesis의 variance side만 다룬다.

Usage:
    python -m allocation.caba_analysis \\
        --model Qwen/Qwen2.5-7B \\
        --output results/caba_qwen.json

    python -m allocation.caba_analysis \\
        --model meta-llama/Llama-3.1-8B \\
        --output results/caba_llama.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# Block-level AM/GM computation
# ============================================================

def compute_block_am_gm(second_moments: np.ndarray, block_size: int = 8) -> dict:
    """Block-wise AM/GM ratio 계산.

    Args:
        second_moments: per-dimension E[x²], shape (head_dim,).
        block_size: E₈ block dimension (8).

    Returns:
        mean_am_gm, max_am_gm, per_block ratios.
    """
    n_blocks = len(second_moments) // block_size
    ratios = []
    for i in range(n_blocks):
        block = second_moments[i * block_size : (i + 1) * block_size]
        block = np.maximum(block, 1e-30)
        am = np.mean(block)
        gm = np.exp(np.mean(np.log(block)))
        ratios.append(am / gm)
    return {
        "mean_am_gm": float(np.mean(ratios)),
        "max_am_gm": float(np.max(ratios)),
        "per_block": [float(x) for x in ratios],
    }


# ============================================================
# Permutation computation
# ============================================================

def compute_sorted_permutation(second_moments: np.ndarray) -> np.ndarray:
    """Variance-sorted permutation (within a single head).

    전역 정렬 후 consecutive 8개씩 같은 block에 배정.
    비슷한 second moment를 가진 dimension이 한 block에 모인다.

    이것은 "optimal assignment"가 아니라 variance-sorted heuristic이다.
    Objective (mean AM/GM vs worst-block AM/GM vs downstream distortion)가
    아직 확정되지 않았으므로, 이름을 optimal로 부르지 않는다.

    Args:
        second_moments: per-dimension E[x²], shape (head_dim,).

    Returns:
        permutation index array, shape (head_dim,).
    """
    return np.argsort(second_moments)


def analyze_block_quality(
    second_moments: np.ndarray,
    perm: np.ndarray,
    block_size: int = 8,
) -> dict:
    """Permutation 적용 후 block-wise AM/GM 분석."""
    permuted = second_moments[perm]
    return compute_block_am_gm(permuted, block_size)


# ============================================================
# KV second moment profiling
# ============================================================

@torch.no_grad()
def profile_kv_second_moment(
    model,
    tokenizer,
    text: str,
    max_length: int = 2048,
    device: str = "cuda",
) -> dict:
    """모델의 모든 layer/head에서 per-dimension KV second moment 측정.

    E[x²] (second moment, not centered variance)를 측정한다.
    KV quantization에서 scale proxy로 사용되며,
    sensitivity.py와 동일한 convention을 따른다.

    Note:
        결과는 한 sequence, 한 batch, 한 context distribution에서의 profile이다.
        모델의 본질적 고정 성질이 아니라, 입력 분포 조건부 profile이다.
        논문 figure용 수치는 여러 document를 평균하여 뽑아야 한다.

    Returns:
        {
            "layer_0": {
                "K": {"head_0": [E[x²]_d0, E[x²]_d1, ...], ...},
                "V": {"head_0": [E[x²]_d0, E[x²]_d1, ...], ...},
            },
            ...
        }
    """
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True
    ).to(device)

    # Use eager attention for compatibility with KV extraction in this script.
    outputs = model(**inputs, output_attentions=False, use_cache=True)

    # KV extraction — same pattern as llm/kv_analysis.py
    past_kv_raw = outputs.past_key_values
    if hasattr(past_kv_raw, 'key_cache'):
        num_layers = len(past_kv_raw.key_cache)
        past_kv = [(past_kv_raw.key_cache[i], past_kv_raw.value_cache[i])
                    for i in range(num_layers)]
    else:
        past_kv = list(past_kv_raw)
        num_layers = len(past_kv)

    print(f"  KV cache: {num_layers} layers, "
          f"{past_kv[0][0].shape[1]} heads, "
          f"head_dim={past_kv[0][0].shape[3]}")

    results = {}
    for layer_idx in range(num_layers):
        k = past_kv[layer_idx][0][0]  # (num_kv_heads, seq_len, head_dim)
        v = past_kv[layer_idx][1][0]

        layer_key = f"layer_{layer_idx}"
        results[layer_key] = {"K": {}, "V": {}}

        for head_idx in range(k.shape[0]):
            k_head = k[head_idx].float()  # (seq_len, head_dim)
            v_head = v[head_idx].float()

            # E[x²] — second moment, not centered variance
            k_sm = (k_head ** 2).mean(dim=0).cpu().numpy()  # (head_dim,)
            v_sm = (v_head ** 2).mean(dim=0).cpu().numpy()

            results[layer_key]["K"][f"head_{head_idx}"] = k_sm.tolist()
            results[layer_key]["V"][f"head_{head_idx}"] = v_sm.tolist()

    return results


# ============================================================
# CABA permutation computation + analysis
# ============================================================

def compute_caba_permutations(profile: dict, block_size: int = 8) -> dict:
    """전체 모델에 대한 sorted permutation 계산 + block AM/GM 개선 분석.

    Returns:
        Per layer/component/head:
            original_mean_am_gm, sorted_mean_am_gm, improvement_ratio, permutation.
        Summary:
            전체 평균 + K/V 분리 평균 + layer-wise worst offenders.
    """
    results = {}

    all_original = []
    all_sorted = []
    k_original, k_sorted = [], []
    v_original, v_sorted = [], []
    layer_worst = {}  # layer_key -> worst AM/GM in that layer

    for layer_key, kv_data in profile.items():
        results[layer_key] = {}
        layer_max_am_gm = 0.0

        for component in ["K", "V"]:
            results[layer_key][component] = {}
            for head_key, sm_list in kv_data[component].items():
                second_moments = np.array(sm_list)

                orig = compute_block_am_gm(second_moments, block_size)
                perm = compute_sorted_permutation(second_moments)
                sorted_result = analyze_block_quality(second_moments, perm, block_size)

                improvement = orig["mean_am_gm"] / max(sorted_result["mean_am_gm"], 1e-10)
                layer_max_am_gm = max(layer_max_am_gm, orig["max_am_gm"])

                all_original.append(orig["mean_am_gm"])
                all_sorted.append(sorted_result["mean_am_gm"])

                if component == "K":
                    k_original.append(orig["mean_am_gm"])
                    k_sorted.append(sorted_result["mean_am_gm"])
                else:
                    v_original.append(orig["mean_am_gm"])
                    v_sorted.append(sorted_result["mean_am_gm"])

                results[layer_key][component][head_key] = {
                    "original_mean_am_gm": orig["mean_am_gm"],
                    "original_max_am_gm": orig["max_am_gm"],
                    "sorted_mean_am_gm": sorted_result["mean_am_gm"],
                    "sorted_max_am_gm": sorted_result["max_am_gm"],
                    "improvement_ratio": float(improvement),
                    "permutation": perm.tolist(),
                }

        layer_worst[layer_key] = layer_max_am_gm

    # Layer-wise top offenders (sorted by worst block AM/GM)
    top_layers = sorted(layer_worst.items(), key=lambda x: x[1], reverse=True)[:5]

    results["summary"] = {
        "original_am_gm_mean": float(np.mean(all_original)),
        "original_am_gm_max": float(np.max(all_original)),
        "sorted_am_gm_mean": float(np.mean(all_sorted)),
        "sorted_am_gm_max": float(np.max(all_sorted)),
        "overall_improvement": float(
            np.mean(all_original) / max(np.mean(all_sorted), 1e-10)
        ),
        "K_original_am_gm_mean": float(np.mean(k_original)) if k_original else 0.0,
        "K_sorted_am_gm_mean": float(np.mean(k_sorted)) if k_sorted else 0.0,
        "V_original_am_gm_mean": float(np.mean(v_original)) if v_original else 0.0,
        "V_sorted_am_gm_mean": float(np.mean(v_sorted)) if v_sorted else 0.0,
        "top_offender_layers": [
            {"layer": lk, "worst_block_am_gm": float(v)} for lk, v in top_layers
        ],
    }

    return results


def find_worst_blocks(profile: dict, top_k: int = 10) -> list:
    """AM/GM이 가장 큰 block들 찾기 (문제 진단용)."""
    blocks = []
    for layer_key, kv_data in profile.items():
        if layer_key == "summary":
            continue
        for component in ["K", "V"]:
            for head_key, sm_list in kv_data[component].items():
                second_moments = np.array(sm_list)
                n_blocks = len(second_moments) // 8
                for b in range(n_blocks):
                    block = second_moments[b * 8 : (b + 1) * 8]
                    block = np.maximum(block, 1e-30)
                    am = np.mean(block)
                    gm = np.exp(np.mean(np.log(block)))
                    ratio = am / gm
                    blocks.append({
                        "layer": layer_key,
                        "component": component,
                        "head": head_key,
                        "block_idx": b,
                        "am_gm": float(ratio),
                        "max_sm": float(np.max(block)),
                        "min_sm": float(np.min(block)),
                        "sm_ratio": float(np.max(block) / max(np.min(block), 1e-30)),
                    })

    blocks.sort(key=lambda x: x["am_gm"], reverse=True)
    return blocks[:top_k]


# ============================================================
# Output
# ============================================================

def print_summary(caba_results: dict, worst_blocks: list):
    """결과 요약 출력."""
    s = caba_results["summary"]

    print("\n" + "=" * 60)
    print("CABA Phase 1 — Block Anisotropy Diagnosis")
    print("=" * 60)

    print(f"\nOriginal (consecutive blocks):")
    print(f"  Overall mean AM/GM: {s['original_am_gm_mean']:.2f}")
    print(f"  Overall max  AM/GM: {s['original_am_gm_max']:.2f}")
    print(f"  K mean AM/GM:       {s['K_original_am_gm_mean']:.2f}")
    print(f"  V mean AM/GM:       {s['V_original_am_gm_mean']:.2f}")

    print(f"\nAfter sorting (variance-sorted blocks):")
    print(f"  Overall mean AM/GM: {s['sorted_am_gm_mean']:.4f}")
    print(f"  Overall max  AM/GM: {s['sorted_am_gm_max']:.4f}")
    print(f"  K mean AM/GM:       {s['K_sorted_am_gm_mean']:.4f}")
    print(f"  V mean AM/GM:       {s['V_sorted_am_gm_mean']:.4f}")

    print(f"\nImprovement: {s['overall_improvement']:.1f}x reduction in mean AM/GM")

    print(f"\nTop offender layers:")
    for entry in s["top_offender_layers"]:
        print(f"  {entry['layer']}: worst block AM/GM = {entry['worst_block_am_gm']:.1f}")

    print(f"\n{'=' * 60}")
    print(f"Top {len(worst_blocks)} Worst Blocks (original assignment)")
    print(f"{'=' * 60}")
    for i, b in enumerate(worst_blocks):
        print(
            f"  {i+1:>2}. {b['layer']:<10} {b['component']} {b['head']:<8} "
            f"block {b['block_idx']:>2}: "
            f"AM/GM={b['am_gm']:>8.1f}, "
            f"E[x²] ratio={b['sm_ratio']:>8.0f}x"
        )


# ============================================================
# Main
# ============================================================

def load_calibration_text(tokenizer, max_length: int = 2048) -> str:
    """Wikitext-2 test split에서 calibration text 로드.

    Hand-crafted text 대신 standard benchmark text를 사용하여
    모델별 variance profile 비교의 신뢰도를 높인다.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # 빈 줄 제거, concatenate
        texts = [t for t in dataset["text"] if t.strip()]
        text = "\n".join(texts)
        # 충분한 길이 확보
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        total_tokens = tokens.input_ids.shape[1]
        print(f"  Calibration text: wikitext-2-test, {total_tokens} tokens available")
        return text
    except ImportError:
        print("  Warning: datasets library not found, using fallback text")
        print("  Install with: pip install datasets")
        # Fallback: generic English text, 반복으로 길이 확보
        fallback = (
            "The study of mathematics reveals deep connections between "
            "seemingly unrelated areas. In number theory, prime numbers "
            "exhibit patterns that remain mysterious despite centuries of "
            "research. Similarly, in physics, the search for a unified "
            "theory continues to drive both theoretical and experimental "
            "advances. Machine learning has emerged as a powerful tool "
            "for discovering patterns in large datasets. "
        )
        return fallback * 50


def main():
    parser = argparse.ArgumentParser(
        description="CABA Phase 1: Block anisotropy diagnosis"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: results/caba_{model_short}.json)",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--text-file", type=str, default=None,
        help="Custom calibration text file (default: wikitext-2)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=15,
                       help="Number of worst blocks to report")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        args.output = f"results/caba_{model_short}.json"

    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    # Load model
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # Use eager attention for compatibility with KV extraction in this script.
        attn_implementation="eager",
    )
    model.eval()

    # Calibration text
    print("\nLoading calibration text...")
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
        print(f"  Custom text from {args.text_file}")
    else:
        text = load_calibration_text(tokenizer, args.max_length)

    # Profile
    print(f"\nProfiling KV second moments (max_length={args.max_length})...")
    profile = profile_kv_second_moment(
        model, tokenizer, text, max_length=args.max_length, device=args.device
    )

    # Compute permutations + analysis
    print("Computing sorted permutations + block AM/GM analysis...")
    caba_results = compute_caba_permutations(profile)

    # Worst blocks
    print(f"Finding top-{args.top_k} worst blocks...")
    worst_blocks = find_worst_blocks(profile, top_k=args.top_k)

    # Print
    print_summary(caba_results, worst_blocks)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "model": args.model,
        "max_length": args.max_length,
        "metric": "second_moment (E[x^2], not centered variance)",
        "scope": "within-head permutation only",
        "summary": caba_results["summary"],
        "worst_blocks": worst_blocks,
        "permutations": {},
    }

    for layer_key in caba_results:
        if layer_key == "summary":
            continue
        save_data["permutations"][layer_key] = {}
        for comp in ["K", "V"]:
            save_data["permutations"][layer_key][comp] = {}
            for head_key, data in caba_results[layer_key][comp].items():
                save_data["permutations"][layer_key][comp][head_key] = {
                    "perm": data["permutation"],
                    "orig_am_gm": data["original_mean_am_gm"],
                    "sorted_am_gm": data["sorted_mean_am_gm"],
                    "improvement": data["improvement_ratio"],
                }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Full second-moment profile (for detailed analysis / figures)
    profile_path = output_path.with_suffix(".profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f)
    print(f"Full second-moment profile saved to {profile_path}")


if __name__ == "__main__":
    main()