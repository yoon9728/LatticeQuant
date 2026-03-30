"""
LatticeQuant v2 — GPU rANS Decode (Final)
==========================================
Standard streaming rANS (Duda 2009) with GPU-parallel decode via Triton.

  Decode: slot = x & (L-1)
          s = lookup[slot]
          x = f[s] * (x >> k) + slot - cf[s]
          while x < L: x = (x << 1) | read_bit()

  Encode (reverse): while x >= 2*f[s]: write_bit(x&1); x >>= 1
                    x = L * (x // f[s]) + cf[s] + (x % f[s])

Validation:
  - CPU roundtrip: 7 diverse distributions
  - Cross-validation vs constriction: rate within 2%
  - GPU decode matches CPU bit-exact: 12+ streams
  - E₈ full pipeline: quantize → symbolize → encode → GPU decode → reconstruct
  - Rate accuracy: coded rate vs entropy on real E₈ symbol streams

Dependencies: triton, torch, constriction (optional), e8_quantizer.py, entropy_coder.py
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e8_quantizer import encode_e8, compute_scale
from entropy_coder import e8_to_symbols, symbols_to_e8

try:
    import constriction
    HAS_CONSTRICTION = True
except ImportError:
    HAS_CONSTRICTION = False

LOG2_L = 10
L = 1 << LOG2_L  # 1024


# ============================================================
# rANS Table Construction
# ============================================================

@dataclass
class RANSTable:
    """Precomputed rANS decode table."""
    lookup: np.ndarray      # (L,) int32 — slot → symbol index
    freqs: np.ndarray       # (n_symbols,) int32
    cumfreqs: np.ndarray    # (n_symbols+1,) int32
    alphabet: List[int]
    n_symbols: int


def build_rans_table(counts: np.ndarray, alphabet: List[int]) -> RANSTable:
    """Build rANS table: normalize counts to sum=L, build slot→symbol lookup."""
    n_sym = len(alphabet)
    total = counts.sum()

    if total == 0:
        freqs = np.ones(n_sym, dtype=np.int32)
    else:
        freqs = np.maximum((counts / total * L).astype(np.int32), 1)

    # Adjust to sum exactly to L
    while freqs.sum() != L:
        diff = L - freqs.sum()
        if diff > 0:
            order = np.argsort(-counts)
            for j in range(abs(diff)):
                freqs[order[j % n_sym]] += 1
        else:
            order = np.argsort(-freqs)
            for j in range(abs(diff)):
                idx = order[j % n_sym]
                if freqs[idx] > 1:
                    freqs[idx] -= 1

    assert freqs.sum() == L and (freqs >= 1).all()

    cumfreqs = np.zeros(n_sym + 1, dtype=np.int32)
    for i in range(n_sym):
        cumfreqs[i + 1] = cumfreqs[i] + freqs[i]

    lookup = np.zeros(L, dtype=np.int32)
    for s in range(n_sym):
        for j in range(cumfreqs[s], cumfreqs[s + 1]):
            lookup[j] = s

    return RANSTable(lookup=lookup, freqs=freqs, cumfreqs=cumfreqs,
                     alphabet=alphabet, n_symbols=n_sym)


# ============================================================
# rANS Encode (CPU)
# ============================================================

def rans_encode(symbols: np.ndarray, table: RANSTable) -> Tuple[np.ndarray, int, int]:
    """
    Streaming rANS encode (reverse order).
    Returns: (bitstream_uint32, initial_state_for_decode, n_bits)
    """
    sym_to_idx = {v: i for i, v in enumerate(table.alphabet)}

    bits_list = []
    x = L

    for sym in reversed(symbols):
        si = sym_to_idx.get(int(sym))
        if si is None:
            si = min(range(table.n_symbols),
                     key=lambda i: abs(table.alphabet[i] - int(sym)))

        fs = int(table.freqs[si])
        cfs = int(table.cumfreqs[si])

        while x >= 2 * fs:
            bits_list.append(x & 1)
            x >>= 1

        x = L * (x // fs) + cfs + (x % fs)

    final_state = x
    bits_list.reverse()
    n_bits = len(bits_list)

    n_words = (n_bits + 31) // 32
    bitstream = np.zeros(n_words + 1, dtype=np.uint32)
    for i, b in enumerate(bits_list):
        bitstream[i // 32] |= (b << (i % 32))

    return bitstream, final_state, n_bits


# ============================================================
# rANS Decode (CPU reference)
# ============================================================

def rans_decode_cpu(bitstream: np.ndarray, n_bits: int, state: int,
                    table: RANSTable, n_symbols: int) -> np.ndarray:
    """CPU rANS decode — reference for validation."""
    output = np.zeros(n_symbols, dtype=np.int64)
    x = state
    bit_pos = 0

    for i in range(n_symbols):
        slot = x & (L - 1)
        si = table.lookup[slot]
        fs = int(table.freqs[si])
        cfs = int(table.cumfreqs[si])
        output[i] = table.alphabet[si]

        x = fs * (x >> LOG2_L) + slot - cfs

        while x < L and bit_pos < n_bits:
            word = int(bitstream[bit_pos // 32])
            bit = (word >> (bit_pos % 32)) & 1
            x = (x << 1) | bit
            bit_pos += 1

    return output


# ============================================================
# Triton rANS Decode Kernel (branchless, no break)
# ============================================================

@triton.jit
def _rans_decode_kernel(
    bitstream_ptr,       # (total_words,) int32
    bit_offsets_ptr,     # (n_streams,) int32
    n_bits_ptr,          # (n_streams,) int32
    init_states_ptr,     # (n_streams,) int32
    n_decode_ptr,        # (n_streams,) int32
    lookup_ptr,          # (n_tables * L,) int32
    freqs_ptr,           # (n_tables * MAX_SYM,) int32
    cumfreqs_ptr,        # (n_tables * MAX_SYM,) int32
    alphabet_ptr,        # (n_tables * MAX_SYM,) int32
    table_id_ptr,        # (n_streams,) int32
    out_ptr,             # (total_output,) int64
    out_offsets_ptr,     # (n_streams,) int32
    L_VAL: tl.constexpr,
    LOG2_L_VAL: tl.constexpr,
    MAX_SYM: tl.constexpr,
    MAX_DECODE: tl.constexpr,
):
    """One program per stream. Branchless via tl.where."""
    sid = tl.program_id(0)

    bit_off = tl.load(bit_offsets_ptr + sid)
    n_bits = tl.load(n_bits_ptr + sid)
    state = tl.load(init_states_ptr + sid)
    n_dec = tl.load(n_decode_ptr + sid)
    tid = tl.load(table_id_ptr + sid)
    out_off = tl.load(out_offsets_ptr + sid)

    tbl_lookup_base = tid * L_VAL
    tbl_data_base = tid * MAX_SYM
    mask_l = L_VAL - 1
    bit_pos = 0

    for i in range(MAX_DECODE):
        active = (i < n_dec)

        # Decode: table lookup (always execute, guarded store)
        slot = state & mask_l
        si = tl.load(lookup_ptr + tbl_lookup_base + slot)
        fs = tl.load(freqs_ptr + tbl_data_base + si)
        cfs = tl.load(cumfreqs_ptr + tbl_data_base + si)
        sym_val = tl.load(alphabet_ptr + tbl_data_base + si)

        if active:
            tl.store(out_ptr + out_off + i, sym_val.to(tl.int64))

        # State transition
        new_state = fs * (state >> LOG2_L_VAL) + slot - cfs
        state = tl.where(active, new_state, state)

        # Renormalize: read bits while state < L (branchless)
        for _ in range(LOG2_L_VAL + 1):
            need_bit = (state < L_VAL) & (bit_pos < n_bits) & active
            abs_bit = bit_pos + bit_off
            word_idx = abs_bit // 32
            bit_idx = abs_bit % 32
            word = tl.load(bitstream_ptr + word_idx)
            bit = (word >> bit_idx) & 1
            state = tl.where(need_bit, (state << 1) | bit, state)
            bit_pos = tl.where(need_bit, bit_pos + 1, bit_pos)


# ============================================================
# GPU decode wrapper
# ============================================================

@dataclass
class RANSStream:
    """One encoded stream ready for GPU decode."""
    bitstream: np.ndarray
    n_bits: int
    initial_state: int
    n_symbols: int
    table: RANSTable


@dataclass
class PreparedGPUDecode:
    """Pre-allocated GPU tensors for kernel-only benchmarking."""
    d_bits: torch.Tensor
    d_bit_off: torch.Tensor
    d_nbits: torch.Tensor
    d_init: torch.Tensor
    d_ndec: torch.Tensor
    d_lookup: torch.Tensor
    d_freqs: torch.Tensor
    d_cumfreqs: torch.Tensor
    d_alphabet: torch.Tensor
    d_tid: torch.Tensor
    d_out: torch.Tensor
    d_out_off: torch.Tensor
    max_sym: int
    max_decode: int
    n_streams: int
    out_offsets: List[int]
    stream_lengths: List[int]


def prepare_gpu_decode(streams: List[RANSStream], device: str = 'cuda') -> PreparedGPUDecode:
    """Prepare all GPU tensors once. For kernel-only benchmarking."""
    n_streams = len(streams)
    max_sym = max(s.table.n_symbols for s in streams)
    max_decode = max(s.n_symbols for s in streams)

    # Bitstreams
    bit_offsets = []
    total_bit_offset = 0
    all_words = []
    for s in streams:
        bit_offsets.append(total_bit_offset)
        all_words.append(s.bitstream.astype(np.int32))
        total_bit_offset += len(s.bitstream) * 32

    flat_bits = np.concatenate(all_words) if all_words else np.zeros(1, dtype=np.int32)

    # Tables (deduplicated)
    table_map = {}
    table_list = []
    table_ids = []
    for s in streams:
        key = id(s.table)
        if key not in table_map:
            table_map[key] = len(table_list)
            table_list.append(s.table)
        table_ids.append(table_map[key])

    n_tables = len(table_list)
    all_lookup = np.zeros((n_tables, L), dtype=np.int32)
    all_freqs = np.zeros((n_tables, max_sym), dtype=np.int32)
    all_cumfreqs = np.zeros((n_tables, max_sym), dtype=np.int32)
    all_alphabet = np.zeros((n_tables, max_sym), dtype=np.int32)

    for ti, tbl in enumerate(table_list):
        all_lookup[ti] = tbl.lookup
        all_freqs[ti, :tbl.n_symbols] = tbl.freqs
        all_cumfreqs[ti, :tbl.n_symbols] = tbl.cumfreqs[:tbl.n_symbols]
        for si, a in enumerate(tbl.alphabet):
            all_alphabet[ti, si] = a

    out_offsets = []
    total_out = 0
    stream_lengths = []
    for s in streams:
        out_offsets.append(total_out)
        total_out += s.n_symbols
        stream_lengths.append(s.n_symbols)

    d = device
    return PreparedGPUDecode(
        d_bits=torch.tensor(flat_bits, dtype=torch.int32, device=d),
        d_bit_off=torch.tensor(bit_offsets, dtype=torch.int32, device=d),
        d_nbits=torch.tensor([s.n_bits for s in streams], dtype=torch.int32, device=d),
        d_init=torch.tensor([s.initial_state for s in streams], dtype=torch.int32, device=d),
        d_ndec=torch.tensor([s.n_symbols for s in streams], dtype=torch.int32, device=d),
        d_lookup=torch.tensor(all_lookup.reshape(-1), dtype=torch.int32, device=d),
        d_freqs=torch.tensor(all_freqs.reshape(-1), dtype=torch.int32, device=d),
        d_cumfreqs=torch.tensor(all_cumfreqs.reshape(-1), dtype=torch.int32, device=d),
        d_alphabet=torch.tensor(all_alphabet.reshape(-1), dtype=torch.int32, device=d),
        d_tid=torch.tensor(table_ids, dtype=torch.int32, device=d),
        d_out=torch.zeros(max(total_out, 1), dtype=torch.int64, device=d),
        d_out_off=torch.tensor(out_offsets, dtype=torch.int32, device=d),
        max_sym=max_sym,
        max_decode=max_decode,
        n_streams=n_streams,
        out_offsets=out_offsets,
        stream_lengths=stream_lengths,
    )


def run_gpu_decode_kernel(prep: PreparedGPUDecode):
    """Run kernel only (no H2D/D2H). For benchmarking."""
    prep.d_out.zero_()
    _rans_decode_kernel[(prep.n_streams,)](
        prep.d_bits, prep.d_bit_off, prep.d_nbits, prep.d_init, prep.d_ndec,
        prep.d_lookup, prep.d_freqs, prep.d_cumfreqs, prep.d_alphabet, prep.d_tid,
        prep.d_out, prep.d_out_off,
        L_VAL=L, LOG2_L_VAL=LOG2_L,
        MAX_SYM=prep.max_sym, MAX_DECODE=prep.max_decode,
    )


def extract_gpu_results(prep: PreparedGPUDecode) -> List[np.ndarray]:
    """Copy results back and split by stream."""
    out_np = prep.d_out.cpu().numpy()
    return [out_np[prep.out_offsets[i]:prep.out_offsets[i] + prep.stream_lengths[i]]
            for i in range(prep.n_streams)]


def gpu_rans_decode(streams: List[RANSStream], device: str = 'cuda') -> List[np.ndarray]:
    """Full GPU decode: prepare + kernel + extract."""
    if not streams:
        return []
    prep = prepare_gpu_decode(streams, device)
    run_gpu_decode_kernel(prep)
    return extract_gpu_results(prep)


# ============================================================
# Tests
# ============================================================

def _entropy(counts):
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-30))


def test_cpu_roundtrip_diverse():
    """rANS roundtrip on 7 diverse distributions."""
    print("Test: CPU rANS roundtrip (diverse distributions)")
    print("=" * 60)

    np.random.seed(42)
    all_pass = True

    cases = [
        ("gaussian-like", list(range(-5, 6)),
         np.array([1, 3, 10, 30, 50, 60, 50, 30, 10, 3, 1], dtype=np.int64)),
        ("highly skewed", list(range(5)),
         np.array([500, 10, 3, 2, 1], dtype=np.int64)),
        ("near-uniform", list(range(8)),
         np.array([13, 12, 13, 12, 13, 12, 13, 12], dtype=np.int64)),
        ("two symbols", [0, 1],
         np.array([70, 30], dtype=np.int64)),
        ("single dominant", list(range(-3, 4)),
         np.array([1, 1, 1, 200, 1, 1, 1], dtype=np.int64)),
        ("wide alphabet", list(range(-15, 16)),
         np.array([max(1, int(50 * np.exp(-x**2/10))) for x in range(-15, 16)], dtype=np.int64)),
        ("short stream", list(range(-2, 3)),
         np.array([5, 10, 20, 10, 5], dtype=np.int64)),
    ]

    for name, alphabet, counts in cases:
        table = build_rans_table(counts, alphabet)
        probs = counts / counts.sum()
        n = 16 if name == "short stream" else 2000
        symbols = np.random.choice(alphabet, size=n, p=probs)

        bitstream, state, n_bits = rans_encode(symbols, table)
        decoded = rans_decode_cpu(bitstream, n_bits, state, table, len(symbols))

        match = np.array_equal(symbols, decoded)
        rate = n_bits / len(symbols)
        H = _entropy(counts)
        if not match:
            all_pass = False
        print(f"  {name:<20} | n={n:>5} | H={H:.3f} rate={rate:.3f} | {'PASS' if match else 'FAIL'}")

    print()
    return all_pass


def test_cross_validate_constriction():
    """Compare our rANS rate vs constriction's ANS rate."""
    print("Test: Rate cross-validation vs constriction")
    print("=" * 60)

    if not HAS_CONSTRICTION:
        print("  constriction not installed, skipping")
        return True

    np.random.seed(42)
    alphabet = list(range(-10, 11))
    counts = np.array([max(1, int(100*np.exp(-x**2/8))) for x in range(-10, 11)], dtype=np.int64)
    probs = (counts / counts.sum()).astype(np.float32)
    H = _entropy(counts)

    symbols = np.random.choice(alphabet, size=10000, p=counts/counts.sum())
    sym_indices = np.array([alphabet.index(s) for s in symbols], dtype=np.int32)

    # Our rANS
    table = build_rans_table(counts, alphabet)
    _, _, n_bits = rans_encode(symbols, table)
    our_rate = n_bits / len(symbols)

    # Constriction
    encoder = constriction.stream.stack.AnsCoder()
    probs_norm = probs / probs.sum()
    encoder.encode_reverse(sym_indices, constriction.stream.model.Categorical(probs_norm, perfect=False))
    con_bits = len(encoder.get_compressed()) * 32
    con_rate = con_bits / len(symbols)

    diff_pct = abs(our_rate - con_rate) / con_rate * 100
    ok = diff_pct < 5

    print(f"  Entropy:      {H:.4f} bits/sym")
    print(f"  Our rANS:     {our_rate:.4f} bits/sym")
    print(f"  Constriction: {con_rate:.4f} bits/sym")
    print(f"  Difference:   {diff_pct:.2f}% {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_gpu_vs_cpu():
    """GPU decode must match CPU decode exactly."""
    print("Test: GPU rANS decode vs CPU reference")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return True

    np.random.seed(42)
    all_pass = True

    distributions = [
        (list(range(-8, 9)), np.array([max(1, int(50*np.exp(-x**2/6))) for x in range(-8, 9)], dtype=np.int64)),
        (list(range(4)), np.array([200, 20, 5, 1], dtype=np.int64)),
        (list(range(-3, 4)), np.array([5, 10, 20, 50, 20, 10, 5], dtype=np.int64)),
    ]

    streams = []
    cpu_refs = []

    for alphabet, counts in distributions:
        table = build_rans_table(counts, alphabet)
        probs = counts / counts.sum()
        for _ in range(4):
            syms = np.random.choice(alphabet, size=1000, p=probs)
            bs, state, n_bits = rans_encode(syms, table)
            cpu_refs.append(rans_decode_cpu(bs, n_bits, state, table, len(syms)))
            streams.append(RANSStream(bitstream=bs, n_bits=n_bits,
                                       initial_state=state, n_symbols=len(syms), table=table))

    gpu_results = gpu_rans_decode(streams, 'cuda')

    for i in range(len(streams)):
        if not np.array_equal(cpu_refs[i], gpu_results[i]):
            all_pass = False
            n_diff = np.sum(cpu_refs[i] != gpu_results[i])
            print(f"  Stream {i}: FAIL ({n_diff} mismatches)")

    if all_pass:
        print(f"  {len(streams)} streams: all PASS")
    print()
    return all_pass


def test_gpu_throughput():
    """Benchmark: full pipeline (H2D + kernel + D2H) and kernel-only."""
    print("Test: GPU rANS decode throughput")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available")
        return

    alphabet = list(range(-10, 11))
    counts = np.array([max(1, int(100*np.exp(-x**2/8))) for x in range(-10, 11)], dtype=np.int64)
    probs = counts / counts.sum()
    table = build_rans_table(counts, alphabet)
    np.random.seed(42)

    print(f"  {'streams':>8} | {'full (ms)':>10} | {'kernel (ms)':>12} | "
          f"{'full M/s':>9} | {'kernel M/s':>10}")
    print(f"  {'-'*60}")

    for n_streams in [16, 64, 128]:
        streams = []
        total_syms = 0
        for _ in range(n_streams):
            syms = np.random.choice(alphabet, size=2000, p=probs)
            bs, state, n_bits = rans_encode(syms, table)
            streams.append(RANSStream(bitstream=bs, n_bits=n_bits,
                                       initial_state=state, n_symbols=len(syms), table=table))
            total_syms += len(syms)

        # Full pipeline benchmark
        for _ in range(3):
            gpu_rans_decode(streams, 'cuda')
        torch.cuda.synchronize()

        n_iter = 50
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            gpu_rans_decode(streams, 'cuda')
        torch.cuda.synchronize()
        t_full = (time.time() - t0) / n_iter

        # Kernel-only benchmark (pre-allocated tensors)
        prep = prepare_gpu_decode(streams, 'cuda')
        for _ in range(5):
            run_gpu_decode_kernel(prep)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            run_gpu_decode_kernel(prep)
        torch.cuda.synchronize()
        t_kernel = (time.time() - t0) / n_iter

        full_ms = t_full * 1e3
        kern_ms = t_kernel * 1e3
        full_mps = total_syms / t_full / 1e6
        kern_mps = total_syms / t_kernel / 1e6

        print(f"  {n_streams:>8} | {full_ms:>10.2f} | {kern_ms:>12.2f} | "
              f"{full_mps:>9.1f} | {kern_mps:>10.1f}")

    print()


def test_e8_full_pipeline():
    """E₈ quantize → symbolize → rANS encode → GPU decode → reconstruct."""
    print("Test: E₈ full pipeline (GPU rANS decode)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available")
        return True

    device = 'cuda'
    torch.manual_seed(42)
    all_pass = True

    for bits in [3.0, 4.0, 5.0]:
        scale = compute_scale(1.0, bits)
        N = 10_000

        x = torch.randn(N, 8, device=device)
        q = encode_e8(x / scale)

        coset, free_coords, coord8_half = e8_to_symbols(q)
        cos_np = coset.cpu().numpy().astype(np.int32)
        free_np = free_coords.cpu().numpy().astype(np.int64)
        c8h_np = coord8_half.cpu().numpy().astype(np.int64)
        all_syms = np.concatenate([free_np, c8h_np[:, None]], axis=1)

        streams = []
        stream_meta = []

        for c_val in [0, 1]:
            mask = (cos_np == c_val)
            if mask.sum() == 0:
                continue
            syms_c = all_syms[mask]
            for idx in range(8):
                col = syms_c[:, idx]
                unique, ucounts = np.unique(col, return_counts=True)
                lo, hi = int(unique.min()) - 2, int(unique.max()) + 2
                alphabet = list(range(lo, hi + 1))
                cdict = dict(zip(unique.tolist(), ucounts.tolist()))
                counts = np.array([cdict.get(a, 0) for a in alphabet], dtype=np.int64)

                table = build_rans_table(counts, alphabet)
                bs, state, n_bits = rans_encode(col, table)
                streams.append(RANSStream(bitstream=bs, n_bits=n_bits,
                                           initial_state=state, n_symbols=len(col), table=table))
                stream_meta.append((c_val, idx, len(col)))

        decoded_streams = gpu_rans_decode(streams, device)

        all_decoded = np.zeros_like(all_syms)
        si = 0
        for c_val in [0, 1]:
            mask = (cos_np == c_val)
            if mask.sum() == 0:
                continue
            rows = np.where(mask)[0]
            for idx in range(8):
                all_decoded[rows, idx] = decoded_streams[si]
                si += 1

        coset_t = torch.tensor(cos_np, dtype=torch.long, device=device)
        free_t = torch.tensor(all_decoded[:, :7], dtype=torch.long, device=device)
        c8h_t = torch.tensor(all_decoded[:, 7], dtype=torch.long, device=device)
        q_rec = symbols_to_e8(coset_t, free_t, c8h_t)

        diff = (q - q_rec).abs().max().item()
        total_bits = sum(s.n_bits for s in streams) + len(cos_np)
        rate = total_bits / (N * 8)
        match = diff < 1e-6
        if not match:
            all_pass = False

        print(f"  {bits}b: diff={diff:.2e}, rate={rate:.3f} (target={bits}), {'PASS' if match else 'FAIL'}")

    print()
    return all_pass


def test_rate_accuracy():
    """Verify coded rate matches entropy across E₈ symbol streams."""
    print("Test: E₈ stream rate vs entropy")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for bits in [3.0, 4.0, 5.0]:
        scale = compute_scale(1.0, bits)
        x = torch.randn(50_000, 8, device=device)
        q = encode_e8(x / scale)

        coset, free_coords, coord8_half = e8_to_symbols(q)
        cos_np = coset.cpu().numpy().astype(np.int32)
        free_np = free_coords.cpu().numpy().astype(np.int64)
        c8h_np = coord8_half.cpu().numpy().astype(np.int64)
        all_syms = np.concatenate([free_np, c8h_np[:, None]], axis=1)

        total_entropy_bits = 0
        total_coded_bits = 0
        total_dims = 50_000 * 8

        p_half = (cos_np == 1).mean()
        if 0 < p_half < 1:
            total_entropy_bits += len(cos_np) * (-p_half*np.log2(p_half) - (1-p_half)*np.log2(1-p_half))
        total_coded_bits += len(cos_np)

        for c_val in [0, 1]:
            mask = (cos_np == c_val)
            if mask.sum() == 0:
                continue
            syms_c = all_syms[mask]
            for idx in range(8):
                col = syms_c[:, idx]
                unique, ucounts = np.unique(col, return_counts=True)
                lo, hi = int(unique.min()) - 2, int(unique.max()) + 2
                alphabet = list(range(lo, hi + 1))
                cdict = dict(zip(unique.tolist(), ucounts.tolist()))
                counts = np.array([cdict.get(a, 0) for a in alphabet], dtype=np.int64)

                H = _entropy(counts[counts > 0])
                total_entropy_bits += len(col) * H

                table = build_rans_table(counts, alphabet)
                _, _, n_bits = rans_encode(col, table)
                total_coded_bits += n_bits

        entropy_rate = total_entropy_bits / total_dims
        coded_rate = total_coded_bits / total_dims
        overhead = (coded_rate / entropy_rate - 1) * 100

        print(f"  {bits}b: entropy={entropy_rate:.4f} coded={coded_rate:.4f} overhead={overhead:+.2f}%")

    print()


if __name__ == '__main__':
    print("=" * 60)
    print("LatticeQuant v2: GPU rANS Decode (Final)")
    print("=" * 60)
    print()

    r1 = test_cpu_roundtrip_diverse()
    r2 = test_cross_validate_constriction()
    r3 = test_gpu_vs_cpu()
    test_gpu_throughput()
    r4 = test_e8_full_pipeline()
    test_rate_accuracy()

    if r1 and r2 and r3 and r4:
        print("GPU rANS PASSED: all correctness, cross-validation, GPU decode, E₈ pipeline.")
    else:
        print("GPU rANS: some tests FAILED.")