"""
Architecture-Agnostic Treatment Pipeline
=========================================
Usage:
  python -m ddt.treat --model Qwen/Qwen2.5-7B
  python -m ddt.treat --model Qwen/Qwen2.5-7B --corpus c4
  python -m ddt.treat --model Qwen/Qwen2.5-7B --corpus ptb
  python -m ddt.treat --model tiiuae/falcon-7b
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name):
    kwargs = dict(dtype=torch.bfloat16, device_map="auto",
                  attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=False, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **kwargs)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def load_text(corpus, tokenizer, seq_len, device):
    from datasets import load_dataset
    if corpus == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif corpus == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        text = "\n\n".join(x["text"] for x, _ in zip(ds, range(200)))
    elif corpus == "ptb":
        ds = load_dataset("ptb-text-only/ptb_text_only", split="test")
        text = "\n\n".join(ds["sentence"])
    else:
        raise ValueError(f"Unknown corpus: {corpus}")
    input_ids = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=seq_len).input_ids.to(device)
    return input_ids


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 10:
            return mod
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def get_attn(layer):
    for name in ['self_attn', 'self_attention', 'attention', 'attn']:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Cannot find attn in {type(layer).__name__}: "
                     f"{[n for n, _ in layer.named_children()]}")


def get_config(model):
    cfg = model.config
    if hasattr(cfg, 'text_config'):
        cfg = cfg.text_config
    nh = getattr(cfg, 'num_attention_heads', None)
    nkv = getattr(cfg, 'num_key_value_heads', None)
    if nkv is None:
        if getattr(cfg, 'multi_query', False):
            nkv = 1
        elif getattr(cfg, 'num_kv_heads', None):
            nkv = cfg.num_kv_heads
        else:
            nkv = nh
    hd = getattr(cfg, 'head_dim', None)
    if hd is None:
        hd = getattr(cfg, 'hidden_size', 0) // max(nh, 1)
    return nh, nkv, hd


def find_proj(attn):
    if hasattr(attn, 'k_proj'):
        return attn.q_proj, attn.k_proj, attn.v_proj, 'separate'
    for name in ['qkv_proj', 'query_key_value', 'c_attn']:
        if hasattr(attn, name):
            return getattr(attn, name), None, None, 'fused'
    raise ValueError(f"Unknown proj in {type(attn).__name__}: "
                     f"{[n for n, _ in attn.named_children()]}")


def quantize(x, bits=4, mode='per_token'):
    if mode == 'per_channel':
        scale = x.abs().amax(dim=-2, keepdim=True).clamp(min=1e-10)
    else:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    mv = 2 ** (bits - 1) - 1
    return (x / scale * mv).round().clamp(-mv, mv) * scale / mv


def measure_sigma2(model, input_ids, bits=4, quant_mode='per_token'):
    layers = get_layers(model)
    nl = len(layers)
    nh, nkv, hd = get_config(model)
    hpk = nh // nkv
    q_dim, k_dim = nh * hd, nkv * hd

    captures = {}
    hooks = []
    for idx in range(nl):
        attn = get_attn(layers[idx])
        q_mod, k_mod, _, style = find_proj(attn)
        if style == 'separate':
            def mh(li, nm):
                def hook(m, i, o):
                    if li not in captures: captures[li] = {}
                    captures[li][nm] = o.detach().float()
                return hook
            hooks.append(q_mod.register_forward_hook(mh(idx, 'q')))
            hooks.append(k_mod.register_forward_hook(mh(idx, 'k')))
        else:
            def mfh(li, qd, kd):
                def hook(m, i, o):
                    if li not in captures: captures[li] = {}
                    flat = o.reshape(-1, o.shape[-1])
                    captures[li]['q'] = flat[:, :qd].detach().float()
                    captures[li]['k'] = flat[:, qd:qd+kd].detach().float()
                return hook
            hooks.append(q_mod.register_forward_hook(mfh(idx, q_dim, k_dim)))

    with torch.no_grad():
        out = model(input_ids, use_cache=False, output_attentions=True)
    for h in hooks:
        h.remove()

    aw = out.attentions if hasattr(out, 'attentions') and out.attentions else None

    results = []
    for li in range(nl):
        if li not in captures:
            results.append({'layer': li, 'sigma2_eff': 0.0, 'q_norm': 0.0,
                           'noise_factor': 0.0})
            continue
        q_raw = captures[li]['q'].reshape(-1, q_dim)
        k_raw = captures[li]['k'].reshape(-1, k_dim)
        T_l = q_raw.shape[0]
        k_q = quantize(k_raw.unsqueeze(0), bits, mode=quant_mode).squeeze(0)
        dk = k_q - k_raw
        qh = q_raw.view(T_l, nh, hd)
        dkh = dk.view(T_l, nkv, hd)

        s2l = []
        for kh in range(nkv):
            dh = dkh[:, kh, :]
            for qi in range(kh * hpk, min((kh+1) * hpk, nh)):
                d = (qh[:, qi, :] @ dh.T) / (hd ** 0.5)
                s2l.append(d.var(dim=-1).mean().item())
        s2a = np.mean(s2l)

        ne = T_l
        if aw and li < len(aw) and aw[li] is not None:
            A = aw[li].squeeze(0).float()
            eta = (A ** 2).sum(dim=-1).mean().item()
            ne = 1.0 / max(eta, 1e-10)
        s2e = s2a * max(1.0 - 1.0/ne, 0.0)
        qn = (qh ** 2).sum(dim=-1).mean().item() / hd
        results.append({'layer': li, 'sigma2_eff': s2e, 'q_norm': qn,
                       'noise_factor': s2a / max(qn, 1e-10)})
    return results


def opt_bits(s2, target=0.5):
    if s2 <= target: return 4
    return max(2, min(8, int(np.ceil(4 + np.log2(s2/target)/2))))


def evaluate(model, input_ids, k_cfg, v_bits=None):
    layers = get_layers(model)
    nl = len(layers)
    nh, nkv, hd = get_config(model)
    q_dim, k_dim, v_dim = nh*hd, nkv*hd, nkv*hd
    hooks = []
    for idx in range(nl):
        attn = get_attn(layers[idx])
        q_mod, k_mod, v_mod, style = find_proj(attn)
        if style == 'separate':
            if idx in k_cfg:
                kb, km = k_cfg[idx]
                def mk(b, m):
                    def hook(mod, i, o): return quantize(o, b, mode=m)
                    return hook
                hooks.append(k_mod.register_forward_hook(mk(kb, km)))
            if v_bits is not None:
                def mv(b):
                    def hook(mod, i, o): return quantize(o, b)
                    return hook
                hooks.append(v_mod.register_forward_hook(mv(v_bits)))
        else:
            if idx in k_cfg or v_bits is not None:
                kb = k_cfg[idx][0] if idx in k_cfg else 4
                km = k_cfg[idx][1] if idx in k_cfg else 'per_token'
                vb = v_bits
                def mf(qd, kd, vd, k_b, k_m, v_b):
                    def hook(mod, i, o):
                        s = o.shape
                        f = o.reshape(-1, s[-1])
                        q = f[:, :qd]
                        k = quantize(f[:, qd:qd+kd], k_b, mode=k_m)
                        v = f[:, qd+kd:qd+kd+vd]
                        if v_b is not None:
                            v = quantize(v, v_b)
                        return torch.cat([q, k, v], dim=-1).view(s)
                    return hook
                hooks.append(q_mod.register_forward_hook(
                    mf(q_dim, k_dim, v_dim, kb, km, vb)))

    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()
    logits = out.logits
    sl = logits[:, :-1, :].contiguous()
    lab = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1)).item()
    return np.exp(loss) if loss < 20 else float('inf')


def run_treatment(model_name, seq_len=512, target=0.5,
                  output_dir="results/ddt", corpus="wikitext"):
    print(f"\n{'='*70}")
    print(f"  Treatment: {model_name} [{corpus}]")
    print(f"{'='*70}")

    model, tok = load_model(model_name)
    layers = get_layers(model)
    nl = len(layers)
    nh, nkv, hd = get_config(model)
    _, _, _, style = find_proj(get_attn(layers[0]))
    print(f"  L={nl}, {nh}Q/{nkv}KV, d={hd}, proj={style}")

    device = next(model.parameters()).device
    input_ids = load_text(corpus, tok, seq_len, device)
    T = input_ids.shape[1]
    print(f"  T={T}, corpus={corpus}")

    print(f"\n  Diagnosing...")
    d_pt = measure_sigma2(model, input_ids, 4, 'per_token')
    d_pc = measure_sigma2(model, input_ids, 4, 'per_channel')

    print(f"\n  {'L':>3} {'s2_pt':>8} {'s2_pc':>8} {'red':>6} {'st':>4}")
    print(f"  {'-'*35}")
    for li in range(nl):
        sp = d_pt[li]['sigma2_eff']
        sc = d_pc[li]['sigma2_eff']
        r = sp / max(sc, 1e-10) if sc > 0.001 else float('inf')
        st = "CRIT" if sc > 1 else ("MOD" if sc > 0.1 else "safe")
        if sp > 0.3 or sc > 0.3:
            print(f"  {li:3d} {sp:8.2f} {sc:8.2f} {r:5.1f}x {st:>4}")

    cpt = sum(1 for r in d_pt if r['sigma2_eff'] > 1)
    cpc = sum(1 for r in d_pc if r['sigma2_eff'] > 1)
    print(f"\n  CRITICAL: {cpt} (pt) -> {cpc} (pc)")

    alloc = {li: (opt_bits(d_pc[li]['sigma2_eff'], target), 'per_channel')
             for li in range(nl)}
    avg_k = np.mean([b for b, _ in alloc.values()])

    print(f"\n  Evaluating...")
    clean = evaluate(model, input_ids, {})
    if clean > 10000:
        print(f"  X Clean PPL={clean:.0f} -- model not suitable")
        return

    u_pt = {l: (4, 'per_token') for l in range(nl)}
    u_pc = {l: (4, 'per_channel') for l in range(nl)}
    c1 = evaluate(model, input_ids, u_pt, 4)
    c2 = evaluate(model, input_ids, u_pc, 4)
    c3 = evaluate(model, input_ids, alloc)

    print(f"\n  {'':40s} {'PPL':>10s} {'K avg':>6s}")
    print(f"  {'-'*58}")
    print(f"  {'Clean':40s} {clean:10.2f}")
    print(f"  {'Per-token K4 + V4':40s} {c1:10.2f} {'4.00':>6s}")
    print(f"  {'Per-channel K4 + V4':40s} {c2:10.2f} {'4.00':>6s}")
    print(f"  {'Per-channel + optimal K':40s} {c3:10.2f} {avg_k:6.2f}")

    best = min(c2, c3)
    print(f"\n  {'='*58}")
    print(f"  Clean: {clean:.2f}")
    print(f"  Best:  {best:.2f}")
    if c1 > 100:
        print(f"  Recovery: {c1:.0f} -> {best:.2f} ({c1/best:.0f}x)")
    else:
        print(f"  Improvement: {c1:.2f} -> {best:.2f} "
              f"({(1-best/c1)*100:.1f}% better)")
    print(f"  vs Clean: +{(best/clean-1)*100:.1f}%")
    print(f"  {'='*58}")

    os.makedirs(output_dir, exist_ok=True)
    tag = model_name.replace("/", "_")
    save = {'model': model_name, 'corpus': corpus, 'clean': clean,
            'pt_k4_v4': c1, 'pc_k4_v4': c2, 'pc_opt_k': c3,
            'avg_k': avg_k, 'crit_pt': cpt, 'crit_pc': cpc,
            'sigma2_pt': {str(r['layer']): r['sigma2_eff'] for r in d_pt},
            'sigma2_pc': {str(r['layer']): r['sigma2_eff'] for r in d_pc}}
    with open(os.path.join(output_dir,
              f"treatment_{tag}_{corpus}.json"), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  Saved.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--target-sigma2", type=float, default=0.5)
    p.add_argument("--output-dir", default="results/ddt")
    p.add_argument("--corpus", default="wikitext",
                   choices=["wikitext", "c4", "ptb"])
    a = p.parse_args()
    run_treatment(a.model, a.seq_len, a.target_sigma2, a.output_dir, a.corpus)


if __name__ == "__main__":
    main()