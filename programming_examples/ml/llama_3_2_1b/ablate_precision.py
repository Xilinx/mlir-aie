"""Phase A precision ablation: find the MINIMUM set of inter-op carriers that
must leave int8 for the W8A8-dynamic mh chain to recover quality.

The mh chain (numpy_layer_mh_forward) is int8-everywhere and collapses to
0/20 top-1 (full-run hidden cos vs the fp32 oracle = 0.10). This harness runs
the same real weights + prompts through a configurable-precision single-layer
forward and reports, per ablation:
  - per-layer cosine(hidden, oracle) across all 16 layers (filled KV cache)
  - next-token top-1 agreement vs the fp16 oracle on a small prompt set

Matmul weights + inputs stay int8 per-token in EVERY variant (that's the W8
part, and today's dyn-rmsnorm + acttail work). Only the inter-op CARRIERS are
toggled:
  - residual:  int8@static-0.05  vs  fp32
  - attention: int8 KV + LUT softmax (mh flowkv)  vs  fp32 KV + exact softmax

Run:
  LLAMA_CHAIN_N=16 python ablate_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from numpy_llama_ref import (
    load_model,
    rmsnorm,
    rope_cos_sin,
    apply_rope,
    attention_full_gqa,
    silu,
    linear_int8,
    quant_act_per_token,
    lm_head_logits,
    forward_layer_int8,
    HEAD_DIM,
    N_HEADS,
    N_KV_GROUPS,
    Q_DIM,
    KV_DIM,
    N_LAYERS,
)
from numpy_layer_mh import gen_real_layer_mh
from generate import load_tokenizer

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The opposite of black is",
    "Two plus two equals",
    "The sun rises in the",
    "The Great Wall is in",
]


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


ACT_SCALE_STATIC = 0.05  # the mh chain's static residual format scale


def _maybe_int8_residual(x_f32, residual_mode):
    """Round-trip the residual through int8 to emulate the mh carrier.
    residual_mode: 'fp32' (no-op), 'int8static' (static 0.05), 'int8pertok'."""
    if residual_mode == "fp32":
        return x_f32
    if residual_mode == "int8static":
        q = np.clip(np.round(x_f32 / ACT_SCALE_STATIC), -128, 127)
        return (q * ACT_SCALE_STATIC).astype(np.float32)
    if residual_mode == "int8pertok":
        s = np.maximum(np.abs(x_f32).max(axis=-1, keepdims=True), 1e-12) / 127.0
        q = np.clip(np.round(x_f32 / s), -127, 127)
        return (q * s).astype(np.float32)
    raise ValueError(residual_mode)


def _attn(q, kc, vc, attn_mode):
    """attn_mode: 'fp32' (exact) or 'int8kv' (round-trip K/V through per-row
    int8 + still-fp32 softmax — isolates the KV-cache quantization)."""
    if attn_mode == "fp32":
        return attention_full_gqa(q, kc, vc)
    if attn_mode == "int8kv":
        def rt(c):
            s = np.maximum(np.abs(c).max(axis=-1, keepdims=True), 1e-12) / 127.0
            return (np.clip(np.round(c / s), -127, 127) * s).astype(np.float32)
        return attention_full_gqa(q, rt(kc), rt(vc))
    raise ValueError(attn_mode)


def layer_w8a8(x_f32, W, kc, vc, position, residual_mode="fp32", attn_mode="fp32"):
    """One decoder layer, W8A8-dynamic with configurable inter-op carriers.
    Matmul weights+inputs are ALWAYS int8 per-token (via linear_int8). Only
    the residual and KV carriers are toggled. Returns (x_out_f32, kc, vc).
    """
    M = x_f32.shape[0]
    x_f32 = _maybe_int8_residual(x_f32, residual_mode)
    h = rmsnorm(x_f32, W.gamma_in)
    q = linear_int8(h, W.wq_i8, W.wq_sc)
    k = linear_int8(h, W.wk_i8, W.wk_sc)
    v = linear_int8(h, W.wv_i8, W.wv_sc)
    pos = np.arange(position, position + M)
    cs, sn = rope_cos_sin(pos, HEAD_DIM)
    q = apply_rope(q.reshape(M, N_HEADS, HEAD_DIM), cs, sn).reshape(M, Q_DIM)
    k = apply_rope(k.reshape(M, N_KV_GROUPS, HEAD_DIM), cs, sn).reshape(M, KV_DIM)
    kc = k if kc.size == 0 else np.concatenate([kc, k], axis=0)
    vc = v if vc.size == 0 else np.concatenate([vc, v], axis=0)
    a = _attn(q, kc, vc, attn_mode)
    a = linear_int8(a, W.wo_i8, W.wo_sc)
    x_f32 = x_f32 + a
    x_f32 = _maybe_int8_residual(x_f32, residual_mode)
    h = rmsnorm(x_f32, W.gamma_post)
    g = linear_int8(h, W.wg_i8, W.wg_sc)
    u = linear_int8(h, W.wu_i8, W.wu_sc)
    s = silu(g) * u
    d = linear_int8(s, W.wd_i8, W.wd_sc)
    x_f32 = x_f32 + d
    return x_f32, kc, vc


def run_variant(model, enc, ref_toks, prompt_ids, residual_mode, attn_mode):
    from numpy_llama_ref import embed_tokens

    n_match = 0
    worst_cos = 1.0
    for ids, ref_tok in zip(prompt_ids, ref_toks):
        x = embed_tokens(model, ids)
        kc = [np.zeros((0, KV_DIM), np.float32) for _ in range(N_LAYERS)]
        vc = [np.zeros((0, KV_DIM), np.float32) for _ in range(N_LAYERS)]
        xo = x.copy()
        oc_k = [np.zeros((0, KV_DIM), np.float32) for _ in range(N_LAYERS)]
        oc_v = [np.zeros((0, KV_DIM), np.float32) for _ in range(N_LAYERS)]
        for L in range(N_LAYERS):
            x, kc[L], vc[L] = layer_w8a8(
                x, model.layers[L], kc[L], vc[L], 0, residual_mode, attn_mode
            )
            xo, oc_k[L], oc_v[L] = forward_layer_int8(
                xo, model.layers[L], oc_k[L], oc_v[L], position=0
            )
            worst_cos = min(worst_cos, cos(x[-1], xo[-1]))
        our_tok = int(np.argmax(lm_head_logits(model, x[-1][None, :])[0]))
        n_match += int(our_tok == ref_tok)
    return n_match, worst_cos


def main():
    data_dir = Path(__file__).parent / "data"
    print(f"loading model from {data_dir} ...", flush=True)
    model = load_model(data_dir)
    enc = load_tokenizer(data_dir / "tokenizer.model")
    from numpy_llama_ref import forward_full

    prompt_ids = [
        np.array([128000] + enc.encode(p), dtype=np.int64) for p in PROMPTS
    ]
    ref_toks = [int(np.argmax(forward_full(model, ids)[-1])) for ids in prompt_ids]
    n = len(PROMPTS)

    # Ablation grid: residual carrier x attention carrier.
    variants = [
        ("fp32", "fp32"),          # ceiling (W8A8 full)
        ("int8pertok", "fp32"),    # residual int8 per-token, exact attn
        ("int8static", "fp32"),    # residual int8 @0.05 (the mh bug), exact attn
        ("fp32", "int8kv"),        # fp32 residual, int8 KV cache
        ("int8pertok", "int8kv"),  # both int8 per-token (no static, no fp32 glue)
        ("int8static", "int8kv"),  # ~ the current mh chain
    ]
    print(f"\n{'residual':>12} {'attn':>8} {'top-1':>8} {'worst-cos':>12}")
    for rm, am in variants:
        nm, wc = run_variant(model, enc, ref_toks, prompt_ids, rm, am)
        print(f"{rm:>12} {am:>8} {nm:>4}/{n:<3} {wc:>12.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
