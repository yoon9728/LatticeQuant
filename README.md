# LatticeQuant

**E₈ Lattice Quantization with Entropy Coding for LLM KV Cache Compression**

LatticeQuant is a research framework for KV cache compression in large language models, combining lattice quantization theory, directional distortion analysis, and attention-aware bit allocation. Developed as an independent research project demonstrating that conference-level ML research is achievable without institutional affiliation.

## Papers

1. **LatticeQuant** (Paper I) — E₈ lattice quantization with entropy coding reduces the Gaussian-normalized distortion gap from 2.72× (scalar) to 1.224×.
2. **LatticeQuant System** (Paper II) — End-to-end system: CompressedKVCache with rANS entropy coding, Triton dequant kernels, and fused merge. 4× compression at +1.9% PPL on Llama-3.1-8B.
3. **Directional Distortion Theory** (Paper III — DDT) — Why error *direction* matters more than error magnitude. Three theorems: master identity, MDS variance decomposition, isotropic uniqueness.
4. **Attention-Aware Allocation** (Paper IV, forthcoming) — Optimal layer-wise bit allocation via DDT-weighted water-filling. Reduces Qwen 4-bit PPL from 8,400 to 293.

## What Each Paper Does

### Paper I: Lattice Quantization

When LLMs process long texts, they store intermediate computations (the "KV cache") that grow linearly with context length. Compressing this cache is essential, and quantization — rounding numbers to fewer bits — is the standard approach. But how much quality do you lose?

We prove that **standard scalar quantization** (rounding each number independently, as used by most existing methods) wastes at least **2.72× more distortion** than information theory allows. This 2.72× gap decomposes into two independent penalties: a *cell-shape penalty* (1.42×, because scalar quantization uses square cells instead of optimal hexagonal-like shapes) and a *fixed-rate penalty* (1.91×, because fixed-length codes can't adapt to the data distribution).

Our solution: **E₈ lattice quantization with entropy coding**. E₈ is a mathematically optimal 8-dimensional packing — the same structure that appears in sphere packing and error-correcting codes. By quantizing 8 coordinates jointly instead of independently, and using entropy coding to adapt the code length, we reduce the gap to **1.224×** — within 22% of the information-theoretic limit. A key trick exploits E₈'s parity structure: the even-sum constraint in D₈ means only 7 of 8 coordinates are free, saving 0.125 bits/dim.

### Paper II: System

Theory alone doesn't help if the system is too slow. This paper builds a complete, deployable KV cache compression pipeline:

**CompressedKVCache** replaces the standard HuggingFace KV cache with an rANS entropy-coded E₈ storage backend. Each KV vector is quantized on write, compressed into a variable-length bitstream, and decompressed on read. Overhead is minimized through coset bit-packing (1 bit per vector instead of 8), shared coding tables across layers, and float16 scale factors.

**Triton GPU kernels** handle E₈ dequantization and fused merge operations entirely on GPU, avoiding CPU-GPU data transfer bottlenecks.

Result on Llama-3.1-8B: **4× memory reduction** with only **+1.9% perplexity increase**, at 5.7 tok/s decode throughput on a single RTX 4070 Ti. We also show 1.8× lower MSE than TurboQuant (ICLR 2026) at comparable compression ratios — a direct consequence of the lattice geometry advantage.

### Paper III: Directional Distortion Theory (DDT)

A surprising observation motivates this paper: **rearranging the order of dimensions** within quantization blocks can change perplexity by 50× without changing MSE. MSE treats all error directions equally, but the model doesn't — some directions matter far more than others.

DDT formalizes this with three theorems:

**Theorem A (Master Identity)** decomposes the loss change ΔL into a Taylor series: a first-order term (gradient · error), a second-order Hessian drift, and explicitly bounded higher-order remainders. This tells us *exactly* which components of the error contribute to quality degradation.

**Theorem B (Directional Risk)** shows that under independent dithering (a standard randomization technique), the variance of the first-order loss change decomposes *additively* across all layers and positions — no cross-terms. Mathematically, this is a martingale difference sequence. The practical consequence: each layer's contribution to quality loss can be analyzed independently, and the total risk is predicted analytically from the Crypto Lemma without any Monte Carlo sampling.

**Theorem C (Isotropic Safety)** answers: what is the *safest* error distribution when you don't know what the model will do with the cached values? (This is the "causal constraint" — at the time you compress a KV vector, you don't know what future queries will attend to it.) The answer: **isotropic error (Σ = σ²I) is the unique safe strategy**. Any anisotropic error creates alignment-dependent risk that can be catastrophically bad if the error aligns with the model's sensitive directions.

This reveals a duality with weight quantization: methods like QuIP and HIGGS rotate the *sensitivity landscape* to make it isotropic; KV cache methods should instead make the *error* isotropic. The key difference is that KV cache compression operates under a causal constraint that weight quantization doesn't face.

### Paper IV: Attention-Aware Allocation (forthcoming)

Not all layers need the same number of bits. DDT's directional risk provides a principled way to measure each layer's sensitivity, and the optimal strategy is **water-filling**: give more bits to sensitive layers, fewer to robust ones.

On Llama-3.1-8B, layers are already roughly balanced (AM/GM ratio = 1.36), so optimal allocation barely improves over uniform. But on Qwen2.5-7B, where a few layers are extremely sensitive (AM/GM = 122.7), allocation reduces 4-bit PPL from **8,400 to 293** — a 96.5% reduction. The theory explains exactly *why* some models catastrophically fail under uniform quantization and *how* to fix it.

## Experimental Results

### Perplexity (WikiText-2, E₈ + Entropy Coding)

| Model | FP16 | 5b | 4b | 3.5b | 3b |
|-------|------|-----|-----|------|-----|
| **Llama-3.1-8B** | 5.64 | 5.67 (+0.5%) | 5.75 (+1.9%) | 5.88 (+4.2%) | 6.19 (+9.8%) |
| **Llama-3.2-1B** | 8.62 | 8.74 (+1.3%) | 9.00 (+4.4%) | 9.44 (+9.5%) | 10.89 (+26%) |
| **Mistral-7B-v0.3** | 5.25 | — | 5.38 (+2.5%) | — | — |
| **Qwen2.5-7B** | 6.11 | 271 | 8,400 | — | 31,110 |

Qwen exhibits catastrophic PPL explosion — diagnosed by DDT as extreme K-path sensitivity concentration (AM/GM = 122.7 vs Llama's 1.36).

### End-to-End System (Llama-3.1-8B, 8-bit weights)

| Bitrate | PPL | Memory (compressed) | Compression | Throughput |
|---------|-----|---------------------|-------------|------------|
| FP16 | 5.64 | 268 MB | 1× | 11.0 tok/s |
| 5b | 5.66 | 94.9 MB | 3.21× | 5.7 tok/s |
| 4b | 5.74 | 66.7 MB | 4.03× | 5.7 tok/s |
| 3b | 6.19 | 50.0 MB | 5.37× | 5.7 tok/s |

### TurboQuant Comparison (4-bit, Llama-3.1-8B)

| Method | MSE | Compression |
|--------|-----|-------------|
| LatticeQuant (E₈ + EC) | 0.005 | 3.92× |
| TurboQuant (scalar) | 0.009 | 3.80× |

LatticeQuant achieves 1.8× lower MSE at comparable compression. At equal quality, LatticeQuant requires 0.575 bits/dim fewer — a mathematical guarantee from the lattice gap.

### DDT Experiments

**Theorem A — First-order predictor:**

| Model | Spearman ρ | tr(MΣ) ρ | MSE ρ | Regime |
|-------|-----------|----------|-------|--------|
| Llama-3.1-8B | **0.71** | 0.39 | 0.43 | Moderate (ΔL/L ≈ 17%) |
| Qwen2.5-7B | < 0.05 | — | — | Catastrophic (ΔL/L > 60%) |

K-path sensitivity: Qwen 92–96%, Llama 34–38%. Bias drift (centered/uncentered gap): Qwen 65–82%, Llama 10–17%.

**Theorem B — Variance additivity (Llama, N=500 dithered trials):**

| Mode | pos_add | pred | E[X] |z| |
|------|---------|------|------|
| baseline | 0.975 | 0.982 | 0.02 |
| sorted | 0.955 | 0.955 | 0.02 |
| random (×3) | 0.97–1.08 | 0.97–1.08 | < 0.05 |

Additivity and analytical Crypto Lemma prediction both ≈ 1.0 within Monte Carlo error.

**Theorem C — Isotropic safety (both models, 9 combinations):**

| M type | range/iso (Llama) | range/iso (Qwen) |
|--------|-------------------|-------------------|
| actual | 0.16–0.25 | 0.21–0.24 |
| rank-1 | 0.17–0.21 | 0.23–0.24 |
| near-isotropic | **0.01** | **0.01** |

Anisotropic M → basis-dependent risk; near-isotropic M → risk constant. Consistent across all (M, Σ) pairs.

### Allocation (Qwen Recovery)

| Model | Uniform 4b | Optimal 4b | Reduction |
|-------|-----------|-----------|-----------|
| Llama-3.1-8B | 5.75 | 5.81 | — (already near-optimal) |
| Qwen2.5-7B | 8,400 | 293 | **96.5%** |

Llama's AM/GM = 1.36 (layers already balanced) → allocation doesn't help. Qwen's AM/GM = 122.7 → massive gain from shifting bits to sensitive layers.

## Project Structure

```
LatticeQuant/
├── core/                           # E₈ quantizer + entropy coding
│   ├── e8_quantizer.py             # E₈ lattice nearest-neighbor encoder/decoder
│   ├── entropy_coder.py            # Parity-aware rANS entropy coding
│   ├── pipeline.py                 # RHT + block partition + full pipeline
│   ├── compact_storage.py          # Compact KV storage format
│   ├── entropy_storage.py          # Compressed storage with rANS
│   ├── gpu_ans.py                  # GPU-accelerated ANS coding
│   └── triton_dequant.py           # Triton dequantization kernel
│
├── llm/                            # LLM integration
│   ├── compressed_kv_cache.py      # CompressedKVCache (rANS-coded E₈)
│   ├── e2e_eval.py                 # PPL / memory / throughput evaluation
│   ├── perplexity_eval.py          # Perplexity evaluation
│   ├── kv_analysis.py              # KV cache distribution analysis
│   ├── fused_triton_merge.py       # Triton fused merge kernel
│   ├── triton_kv_integration.py    # Triton KV integration
│   └── long_context_eval.py        # Long context evaluation
│
├── allocation/                     # Attention-aware bit allocation
│   ├── sensitivity.py              # Per-layer sensitivity extraction (η, σ²)
│   ├── propagation.py              # Cross-layer propagation factor (Γ)
│   ├── allocator.py                # Water-filling optimal allocation
│   ├── eval_alloc.py               # Variable-rate PPL evaluation
│   ├── thm{1,2,4,5}_validate.py   # Theorem validation scripts
│   ├── caba_analysis.py            # Block assignment analysis
│   └── caba_eval.py                # Block assignment PPL evaluation
│
├── ddt/                            # Directional Distortion Theory
│   ├── caba_explain.py             # Theorem A: first-order predictor
│   ├── variance_additivity.py      # Theorem B: variance additivity (MDS)
│   └── isotropic_safety.py         # Theorem C: isotropic uniqueness
│
├── experiments/
│   └── rht_guarantee.py            # RHT theoretical guarantee
│
├── papers/                         # Paper sources and PDFs
├── results/                        # Experimental results (JSON)
├── README.md
├── LICENSE
└── requirements.txt
```

## Quick Start

```bash
# Environment
conda create -n latticequant python=3.12
conda activate latticequant
pip install torch transformers datasets bitsandbytes

# WSL2 (for Triton / GPU-accelerated operations)
# pip install triton
```

### Gaussian Validation

```python
from core.e8_quantizer import encode_e8, compute_scale
import torch

x = torch.randn(1000, 8)
sigma2 = (x ** 2).mean().item()
scale = compute_scale(sigma2, bits=4.0)
q = encode_e8(x / scale)
x_hat = q * scale

mse = ((x - x_hat) ** 2).mean().item()
print(f"MSE gap: {mse / (sigma2 * 4**(-4)):.4f}")  # Should be ~1.224
```

### Perplexity Evaluation

```bash
# Single bitrate
python llm/perplexity_eval.py --model meta-llama/Llama-3.1-8B --bits 4.0

# All bitrates
python llm/perplexity_eval.py --model meta-llama/Llama-3.1-8B --all

# End-to-end (PPL + memory + throughput)
python llm/e2e_eval.py --model meta-llama/Llama-3.1-8B --bits 4
```

### DDT Experiments

```bash
# Theorem A: first-order predictor (ranking validation)
python -m ddt.caba_explain \
    --model meta-llama/Llama-3.1-8B \
    --caba results/caba_llama_3.1_8b.json --bits 4

# Theorem B: variance additivity (dithered Z-lattice, 500 trials)
python -m ddt.variance_additivity \
    --model meta-llama/Llama-3.1-8B \
    --caba results/caba_llama_3.1_8b.json --n-trials 500

# Theorem C: isotropic safety (HLP bounds + uniqueness test)
python -m ddt.isotropic_safety \
    --model meta-llama/Llama-3.1-8B \
    --caba results/caba_llama_3.1_8b.json

# Qwen comparison (catastrophic regime)
python -m ddt.caba_explain \
    --model Qwen/Qwen2.5-7B \
    --caba results/caba_qwen2.5_7b.json --bits 4
```

### Allocation Experiments

```bash
# Sensitivity extraction
python -m allocation.sensitivity --model meta-llama/Llama-3.1-8B

# Propagation factors
python -m allocation.propagation --model meta-llama/Llama-3.1-8B

# Optimal allocation + PPL evaluation
python -m allocation.eval_alloc --model meta-llama/Llama-3.1-8B --budget 4.0
```

## Hardware

All experiments run on a single RTX 4070 Ti (16 GB). Models loaded in 8-bit via BitsAndBytes. No multi-GPU or cluster required.

## Citation

```bibtex
@article{yoon2026latticequant,
  title={LatticeQuant: Lattice Quantization Principles for KV Cache Compression},
  author={Yoon, Jeehyun},
  year={2026}
}

@article{yoon2026latticequant_system,
  title={LatticeQuant System: End-to-End E₈ KV Cache Compression},
  author={Yoon, Jeehyun},
  year={2026}
}

@article{yoon2026ddt,
  title={Directional Distortion Theory for KV Cache Quantization},
  author={Yoon, Jeehyun},
  year={2026}
}
```

## License

MIT License