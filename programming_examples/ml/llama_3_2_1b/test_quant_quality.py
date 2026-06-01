"""Phase 6c.5a Track 2: PTQ quality validation.

Loads BF16 Llama 3.2 1B from safetensors, also loads our quantized INT8
.bin files (from gen_llama_data.py). Runs BOTH paths on the same input
and reports cosine similarity per layer + max-abs delta normalized.

This is the QUALITY check on the quantization scheme — not a
correctness check on any NPU code. Failing here means the quant scheme
needs tuning before we touch kernels.

Pass criteria (per cautious-eureka quality findings):
  - per-layer cos-sim >= 0.99   (per-token activation should clear this)
  - end-to-end token-overlap >= 4/5 in top-5 at last position
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

# Reuse the safetensors reader from the extractor.
from gen_llama_data import SafetensorsReader, VOCAB_SIZE, EMB_DIM, N_LAYERS, \
    N_HEADS, N_KV_GROUPS, HEAD_DIM, HIDDEN_DIM, Q_DIM, KV_DIM
# Reuse the INT8 numpy ref.
from numpy_llama_ref import (
    load_model, embed_tokens, forward_layer_int8, lm_head_logits,
    rmsnorm, rope_cos_sin, apply_rope, silu, attention_full_gqa,
    quant_act_per_token,
    ROPE_BASE, RMS_EPS,
)


# ---------------------------------------------------------------------------
# BF16 (fp32-compute) reference — same math as numpy_llama_ref but with
# direct fp32 weights from safetensors (no quantization).
# ---------------------------------------------------------------------------
def _load_w_f32(reader: SafetensorsReader, name: str) -> np.ndarray:
    return reader.load_f32(name)


def forward_layer_fp32(x_f32: np.ndarray, reader: SafetensorsReader, L: int,
                       k_cache: np.ndarray, v_cache: np.ndarray,
                       position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same layer math as the INT8 path, but with direct fp32 weights —
    no quant at all. This is the upper-bound reference."""
    M = x_f32.shape[0]
    p = f"model.layers.{L}."

    gamma_in   = reader.load_bf16(p + "input_layernorm.weight")
    gamma_post = reader.load_bf16(p + "post_attention_layernorm.weight")
    wq = _load_w_f32(reader, p + "self_attn.q_proj.weight")
    wk = _load_w_f32(reader, p + "self_attn.k_proj.weight")
    wv = _load_w_f32(reader, p + "self_attn.v_proj.weight")
    wo = _load_w_f32(reader, p + "self_attn.o_proj.weight")
    wg = _load_w_f32(reader, p + "mlp.gate_proj.weight")
    wu = _load_w_f32(reader, p + "mlp.up_proj.weight")
    wd = _load_w_f32(reader, p + "mlp.down_proj.weight")

    # Attention
    h = rmsnorm(x_f32, gamma_in)
    q = (h @ wq.T).astype(np.float32)
    k = (h @ wk.T).astype(np.float32)
    v = (h @ wv.T).astype(np.float32)
    pos = np.arange(position, position + M)
    cos, sin = rope_cos_sin(pos, HEAD_DIM)
    q = apply_rope(q.reshape(M, N_HEADS,    HEAD_DIM), cos, sin).reshape(M, Q_DIM)
    k = apply_rope(k.reshape(M, N_KV_GROUPS, HEAD_DIM), cos, sin).reshape(M, KV_DIM)

    if k_cache.size == 0:
        k_full = k; v_full = v
    else:
        k_full = np.concatenate([k_cache, k], axis=0)
        v_full = np.concatenate([v_cache, v], axis=0)

    a = attention_full_gqa(q, k_full, v_full)
    a = (a @ wo.T).astype(np.float32)
    x_f32 = x_f32 + a

    # FFN
    h = rmsnorm(x_f32, gamma_post)
    g = (h @ wg.T).astype(np.float32)
    u = (h @ wu.T).astype(np.float32)
    s = silu(g) * u
    d = (s @ wd.T).astype(np.float32)
    x_f32 = x_f32 + d

    return x_f32, k_full, v_full


# ---------------------------------------------------------------------------
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path(__file__).parent / "data")
    p.add_argument("--weights", type=Path,
                   default=Path("/scratch/roesti/models/llama_3.2_1b/model.safetensors"))
    p.add_argument("--M", type=int, default=4,
                   help="number of input tokens for the test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=N_LAYERS,
                   help="how many layers to test (default: all 16)")
    p.add_argument("--prompt", type=str, default="The capital of France is",
                   help="real prompt for the end-to-end check (uses tiktoken)")
    opts = p.parse_args()

    print(f"loading BF16 reference from {opts.weights}")
    reader = SafetensorsReader(opts.weights)
    print(f"loading INT8 model from {opts.data_dir}")
    model = load_model(opts.data_dir)

    rng = np.random.default_rng(opts.seed)
    # Realistic Gaussian residual stream as input (RMSNorm makes the
    # layer scale-invariant on input, so unit-Gaussian is fair).
    x0 = rng.standard_normal((opts.M, EMB_DIM)).astype(np.float32)

    # Per-layer A/B: feed IDENTICAL x_in into both paths and cos-sim outputs.
    print(f"\n--- Per-layer quality (M={opts.M}) ---")
    fail = 0
    for L in range(opts.layers):
        kc_f = np.zeros((0, KV_DIM), dtype=np.float32)
        vc_f = np.zeros((0, KV_DIM), dtype=np.float32)
        kc_i = np.zeros((0, KV_DIM), dtype=np.float32)
        vc_i = np.zeros((0, KV_DIM), dtype=np.float32)
        ref_out, _, _ = forward_layer_fp32(x0, reader, L, kc_f, vc_f, position=0)
        int8_out, _, _ = forward_layer_int8(x0, model.layers[L], kc_i, vc_i, position=0)
        cs = cos_sim(ref_out, int8_out)
        re = max_rel_err(int8_out, ref_out)
        ok = (cs >= 0.99)
        if not ok: fail += 1
        marker = "✓" if ok else "✗"
        print(f"  layer {L:02d}  cos_sim={cs:.6f}  max_rel_err={re:.3e}  {marker}")

    # End-to-end logits comparison on a REAL prompt via tiktoken.
    import base64, tiktoken
    mergeable = {}
    with open(opts.data_dir / "tokenizer.model") as f:
        for line in f:
            tok_b64, rank = line.strip().split(' ')
            mergeable[base64.b64decode(tok_b64)] = int(rank)
    PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
           r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    SPECIAL = {'<|begin_of_text|>': 128000, '<|end_of_text|>': 128001, '<|eot_id|>': 128009}
    enc = tiktoken.Encoding(name='llama3', pat_str=PAT,
                            mergeable_ranks=mergeable, special_tokens=SPECIAL)
    prompt_ids = [128000] + enc.encode(opts.prompt)   # BOS + prompt
    token_ids = np.array(prompt_ids, dtype=np.int64)
    print(f"\n--- End-to-end logits on prompt {opts.prompt!r} ({len(token_ids)} tokens) ---")
    # INT8 path
    int8_x = embed_tokens(model, token_ids)
    kc_i = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    vc_i = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    for L, W in enumerate(model.layers):
        int8_x, kc_i[L], vc_i[L] = forward_layer_int8(int8_x, W, kc_i[L], vc_i[L], 0)
    int8_logits = lm_head_logits(model, int8_x)
    # BF16/fp32 path (same embedding, just dequantized)
    ref_embed = reader.load_f32("model.embed_tokens.weight")
    ref_norm_w = reader.load_bf16("model.norm.weight")
    ref_x = ref_embed[token_ids].astype(np.float32)
    kc_f = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    vc_f = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    for L in range(N_LAYERS):
        ref_x, kc_f[L], vc_f[L] = forward_layer_fp32(ref_x, reader, L, kc_f[L], vc_f[L], 0)
    ref_logits = (rmsnorm(ref_x, ref_norm_w) @ ref_embed.T).astype(np.float32)

    cs_last = cos_sim(ref_logits[-1], int8_logits[-1])
    int8_top5 = set(np.argsort(int8_logits[-1])[-5:][::-1].tolist())
    ref_top5  = set(np.argsort(ref_logits[-1])[-5:][::-1].tolist())
    overlap = len(int8_top5 & ref_top5)
    int8_top1 = int(np.argmax(int8_logits[-1]))
    ref_top1  = int(np.argmax(ref_logits[-1]))
    print(f"  logits cos_sim (last pos): {cs_last:.6f}")
    print(f"  top-1 INT8: id={int8_top1} '{enc.decode([int8_top1])}'")
    print(f"  top-1 BF16: id={ref_top1} '{enc.decode([ref_top1])}'")
    print(f"  top-5 INT8: {sorted(int8_top5)} ({[enc.decode([t]) for t in int8_top5]})")
    print(f"  top-5 BF16: {sorted(ref_top5)} ({[enc.decode([t]) for t in ref_top5]})")
    print(f"  top-5 overlap: {overlap}/5")

    print()
    if fail == 0 and overlap >= 4:
        print(f"PASS  ({opts.layers}/{opts.layers} layers >= 0.99 cos-sim, top-5 overlap {overlap}/5)")
        return 0
    print(f"FAIL  ({opts.layers - fail}/{opts.layers} layers passed, top-5 overlap {overlap}/5)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
