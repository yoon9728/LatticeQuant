# LatticeQuant

**Lattice Quantization Principles for KV Cache Compression**

LatticeQuant establishes a general lattice-based quantization framework for KV cache compression in large language models. We prove that scalar fixed-length quantization (used by TurboQuant and related methods) suffers an inherent 2.72× Gaussian-normalized distortion gap, and show that E₈ lattice quantization with entropy coding reduces this to 1.224×.

## Key Results

- **Theorem 1 (Scalar Barrier):** Any scalar fixed-length quantizer has an asymptotic distortion gap ≥ 2.72×, decomposed into a cell-shape penalty (πe/6 ≈ 1.424) and a fixed-rate penalty (3√3/e ≈ 1.911).
- **Theorem 2 (Lattice Achievability):** Entropy-coded nearest-neighbor lattice quantization achieves a gap of 2πeG(Λ) for any lattice Λ. For E₈, this gives ≤ 1.224×.
- **Experiments:** On Llama-3.1-8B, 4-bit KV cache quantization incurs only +1.94% perplexity increase on WikiText-2. 5-bit is near-lossless at +0.51%.

## Project Structure

```
LatticeQuant/
├── core/
│   ├── e8_quantizer.py       # E₈ lattice nearest-neighbor quantizer
│   ├── entropy_coder.py       # Parity-aware ANS entropy coding
│   └── pipeline.py            # Quantization pipeline
├── llm/
│   ├── perplexity_eval.py  # Perplexity evaluation on WikiText-2
│   └── kv_analysis.py         # KV cache distribution analysis
├── results/                   # Experimental results
└── paper/
    └── latticequant_paper.pdf  # Paper source
```

## Quick Start

```bash
conda create -n latticequant python=3.12
conda activate latticequant
pip install torch transformers datasets bitsandbytes
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
d_gauss = sigma2 * 4**(-4)
print(f"MSE gap: {mse / d_gauss:.4f}")  # Should be ~1.224
```

### KV Cache Quantization (Llama-3.1-8B)

```bash
python llm/perplexity_eval.py --model meta-llama/Llama-3.1-8B --bits 4.0
```

## Results

| Model | Config | PPL | Δ | Compression |
|-------|--------|-----|---|-------------|
| Llama-3.1-8B | FP16 baseline | 5.64 | — | 1× |
| Llama-3.1-8B | 5.0b E₈ + EC | 5.67 | +0.51% | 3.2× |
| Llama-3.1-8B | 4.0b E₈ + EC | 5.75 | +1.94% | 4× |
| Llama-3.1-8B | 3.5b E₈ + EC | 5.88 | +4.23% | 4.6× |
| Llama-3.2-1B | FP16 baseline | 8.62 | — | 1× |
| Llama-3.2-1B | 4.0b E₈ + EC | 9.00 | +4.4% | 4× |

## Citation

```bibtex
@article{yoon2026latticequant,
  title={LatticeQuant: Lattice Quantization Principles for KV Cache Compression},
  author={Yoon, Jeehyun},
  year={2026}
}
```

## License

MIT License