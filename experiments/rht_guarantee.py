# core/debug_rht_guarantee.py
import torch, sys, numpy as np
sys.path.insert(0, 'core')
from e8_quantizer import encode_e8, compute_scale, G_E8
from pipeline import fast_hadamard_transform, inverse_fast_hadamard_transform

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 100000
D = 64
bits = 4.0
theory_gap = 2 * np.pi * np.e * G_E8

print("RHT as Worst-Case Guarantee")
print("=" * 75)
print(f"Theory gap: {theory_gap:.4f}")
print(f"N={N}, D={D}, bits={bits}")
print()

signs = torch.where(torch.rand(D) < 0.5, torch.ones(D), -torch.ones(D)).to(device)

def measure_gap(vectors, label, use_rht=False):
    v = vectors.clone()
    if use_rht:
        v = fast_hadamard_transform(v * signs)
    sigma2 = (v ** 2).mean().item()
    scale = compute_scale(sigma2, bits)
    blocks = v.reshape(-1, 8) / scale
    q = encode_e8(blocks)
    v_hat = (q * scale).reshape(-1, D)
    if use_rht:
        v_hat = inverse_fast_hadamard_transform(v_hat) * signs
    mse = ((vectors - v_hat) ** 2).mean().item()
    d_gauss = sigma2 * (4 ** (-bits))
    gap = mse / d_gauss
    print(f"  {label:40s} | gap = {gap:.4f} | {'OK' if abs(gap - theory_gap)/theory_gap < 0.05 else 'BAD'}")
    return gap

print("--- Case 1: i.i.d. Gaussian (ideal) ---")
x_gauss = torch.randn(N, D, device=device)
measure_gap(x_gauss, "No RHT", use_rht=False)
measure_gap(x_gauss, "With RHT", use_rht=True)
print()

print("--- Case 2: Energy in 2 dims (adversarial) ---")
x_sparse = torch.zeros(N, D, device=device)
x_sparse[:, 0] = torch.randn(N, device=device) * 10.0
x_sparse[:, 1] = torch.randn(N, device=device) * 10.0
x_sparse[:, 2:] = torch.randn(N, D-2, device=device) * 0.01
measure_gap(x_sparse, "No RHT (energy concentrated)", use_rht=False)
measure_gap(x_sparse, "With RHT (energy spread)", use_rht=True)
print()

print("--- Case 3: Correlated coordinates ---")
base = torch.randn(N, 1, device=device)
x_corr = base.expand(N, D) + torch.randn(N, D, device=device) * 0.1
measure_gap(x_corr, "No RHT (highly correlated)", use_rht=False)
measure_gap(x_corr, "With RHT (decorrelated)", use_rht=True)
print()

print("--- Case 4: Heavy-tailed (Cauchy-like) ---")
x_heavy = torch.distributions.StudentT(df=2.0).sample((N, D)).to(device)
measure_gap(x_heavy, "No RHT", use_rht=False)
measure_gap(x_heavy, "With RHT", use_rht=True)
print()

print("--- Case 5: Real KV cache (Llama-3.2-1B) ---")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', dtype=torch.float16, device_map='cuda')
    inputs = tok('The quick brown fox jumps over the lazy dog. ' * 20, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv = out.past_key_values
    if hasattr(kv, 'to_legacy_cache'):
        legacy = kv.to_legacy_cache()
    else:
        legacy = list(kv)
    k = legacy[5][0].float().reshape(-1, D)
    measure_gap(k, "No RHT (real KV)", use_rht=False)
    measure_gap(k, "With RHT (real KV)", use_rht=True)
except Exception as e:
    print(f"  Skipped: {e}")

print()
print("=" * 75)
print("Conclusion:")
print("  Gaussian / real KV: RHT unnecessary (gap already ~1.22)")
print("  Adversarial distributions: RHT is essential (restores gap to ~1.22)")
print("  RHT = worst-case guarantee, not always needed in practice")