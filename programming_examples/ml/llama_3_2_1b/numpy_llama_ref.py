#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The int8 recipe in numpy: Llama 3.2 1B's forward pass on gen_llama_data.py's
weights, with per-channel int8 weights, per-token int8 activations, int32
accumulation and fp32 dequant, silu, softmax and rmsnorm.

The device follows this recipe but not bit for bit (its reciprocal square
root, softmax exp and adder differ); llama_oracle.py is the oracle it
matches exactly. This is the quality reference the device is scored against.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from llama_config import (
    HD,
    HEAD_D,
    KV_DIM,
    N_HEADS_KV,
    N_HEADS_Q,
    N_LAYERS,
    Q_DIM,
    VOCAB,
    D,
)

ROPE_BASE = 500_000.0
# config.json's rope_scaling (rope_type "llama3").
ROPE_FACTOR = 32.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_ORIGINAL_CONTEXT = 8192
RMS_EPS = 1e-5


@dataclass
class LayerW:
    gamma_in: np.ndarray  # (D,) bf16
    gamma_post: np.ndarray  # (D,) bf16
    wq_i8: np.ndarray
    wq_sc: np.ndarray  # (Q_DIM, D),  (Q_DIM,)
    wk_i8: np.ndarray
    wk_sc: np.ndarray  # (KV_DIM, D), (KV_DIM,)
    wv_i8: np.ndarray
    wv_sc: np.ndarray  # (KV_DIM, D), (KV_DIM,)
    wo_i8: np.ndarray
    wo_sc: np.ndarray  # (D, Q_DIM),  (D,)
    wg_i8: np.ndarray
    wg_sc: np.ndarray  # (HD, D),     (HD,)
    wu_i8: np.ndarray
    wu_sc: np.ndarray  # (HD, D),     (HD,)
    wd_i8: np.ndarray
    wd_sc: np.ndarray  # (D, HD),     (D,)


@dataclass
class ModelW:
    embed_i8: np.ndarray
    embed_sc: np.ndarray  # (V, D), (V,)
    final_norm: np.ndarray  # (D,) bf16
    layers: list[LayerW]


def _load_qw(
    data_dir: Path, stem: str, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    q = np.frombuffer(
        (data_dir / f"{stem}.i8.bin").read_bytes(), dtype=np.int8
    ).reshape(shape)
    sc = np.frombuffer(
        (data_dir / f"{stem}.scales.f32.bin").read_bytes(), dtype=np.float32
    )
    assert sc.shape == (shape[0],), f"{stem}: scale shape {sc.shape} != ({shape[0]},)"
    return q, sc


def _load_bf16(data_dir: Path, stem: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(
        (data_dir / f"{stem}.bf16.bin").read_bytes(), dtype=ml_dtypes.bfloat16
    ).reshape(shape)


def load_model(data_dir: Path) -> ModelW:
    embed_i8, embed_sc = _load_qw(data_dir, "embed", (VOCAB, D))
    final_norm = _load_bf16(data_dir, "final_norm", (D,))
    layers = []
    for L in range(N_LAYERS):
        ld = data_dir / f"layer_{L:02d}"
        wq_i8, wq_sc = _load_qw(ld, "wq", (Q_DIM, D))
        wk_i8, wk_sc = _load_qw(ld, "wk", (KV_DIM, D))
        wv_i8, wv_sc = _load_qw(ld, "wv", (KV_DIM, D))
        wo_i8, wo_sc = _load_qw(ld, "wo", (D, Q_DIM))
        wg_i8, wg_sc = _load_qw(ld, "wg", (HD, D))
        wu_i8, wu_sc = _load_qw(ld, "wu", (HD, D))
        wd_i8, wd_sc = _load_qw(ld, "wd", (D, HD))
        layers.append(
            LayerW(
                gamma_in=_load_bf16(ld, "gamma_in", (D,)),
                gamma_post=_load_bf16(ld, "gamma_post", (D,)),
                wq_i8=wq_i8,
                wq_sc=wq_sc,
                wk_i8=wk_i8,
                wk_sc=wk_sc,
                wv_i8=wv_i8,
                wv_sc=wv_sc,
                wo_i8=wo_i8,
                wo_sc=wo_sc,
                wg_i8=wg_i8,
                wg_sc=wg_sc,
                wu_i8=wu_i8,
                wu_sc=wu_sc,
                wd_i8=wd_i8,
                wd_sc=wd_sc,
            )
        )
    return ModelW(
        embed_i8=embed_i8, embed_sc=embed_sc, final_norm=final_norm, layers=layers
    )


def load_tokenizer(tokenizer_path: Path):
    """Llama 3 tiktoken encoder from the data/ tokenizer.model ranks file."""
    # Optional dep, only needed to tokenize; not in the lint/CI env.
    import tiktoken  # pyright: ignore[reportMissingImports]

    mergeable = {}
    with open(tokenizer_path) as f:
        for line in f:
            tok_b64, rank = line.strip().split(" ")
            mergeable[base64.b64decode(tok_b64)] = int(rank)
    PAT = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    SPECIAL = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|eot_id|>": 128009,
    }
    return tiktoken.Encoding(
        name="llama3", pat_str=PAT, mergeable_ranks=mergeable, special_tokens=SPECIAL
    )


def quant_act_per_token(x_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization. x_f32 is (M, D); returns
    (q_i8, scale_f32) where scale has shape (M,)."""
    absmax = np.maximum(np.abs(x_f32).max(axis=1, keepdims=True), 1e-12)
    s = (absmax / 127.0).astype(np.float32)
    q = np.round(x_f32 / s).clip(-127, 127).astype(np.int8)
    return q, s.squeeze(1).astype(np.float32)


def linear_int8(x_f32: np.ndarray, w_i8: np.ndarray, w_sc: np.ndarray) -> np.ndarray:
    """y = x @ W.T with per-token x quant and per-row w scale, return fp32.

    Quantizes x per token, accumulates in int32 and dequantizes with
    x_scale * w_sc[n] per output.
    """
    if x_f32.ndim == 1:
        x_f32 = x_f32[None, :]  # (1, D) for uniform handling
        squeezed = True
    else:
        squeezed = False

    qx, x_sc = quant_act_per_token(x_f32)  # qx: (M, D) i8; x_sc: (M,) fp32
    # int32 accumulate via fp64 BLAS (exact for our magnitudes).
    acc = qx.astype(np.float64) @ w_i8.astype(np.float64).T  # (M, N) exact int32
    # Dequant: y[m, n] = acc[m, n] * x_sc[m] * w_sc[n]
    y = (acc.astype(np.float32) * x_sc[:, None] * w_sc[None, :]).astype(np.float32)
    return y[0] if squeezed else y


def rmsnorm(
    x_f32: np.ndarray, gamma_bf16: np.ndarray, eps: float = RMS_EPS
) -> np.ndarray:
    """RMSNorm in fp32 (gamma is bf16, cast on the fly). Returns fp32."""
    gamma = gamma_bf16.astype(np.float32)
    if x_f32.ndim == 1:
        rms = np.sqrt(np.mean(x_f32**2) + eps).astype(np.float32)
        return (x_f32 / rms * gamma).astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps).astype(np.float32)
    return (x_f32 / rms * gamma).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def rope_inv_freq(head_dim: int) -> np.ndarray:
    """Llama 3's RoPE frequencies, with its rope_scaling, as HF computes them.

    Args:
        head_dim: The head dimension.

    Returns:
        float32 (head_dim // 2,): each rotary pair's angle per position.
    """
    inv_freq = (
        1.0 / (ROPE_BASE ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    ).astype(np.float32)
    wavelen = (2 * np.pi / inv_freq).astype(np.float32)
    low_freq_wavelen = ROPE_ORIGINAL_CONTEXT / ROPE_LOW_FREQ_FACTOR
    high_freq_wavelen = ROPE_ORIGINAL_CONTEXT / ROPE_HIGH_FREQ_FACTOR
    scaled = np.where(wavelen > low_freq_wavelen, inv_freq / ROPE_FACTOR, inv_freq)
    smooth = (ROPE_ORIGINAL_CONTEXT / wavelen - ROPE_LOW_FREQ_FACTOR) / (
        ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR
    )
    smoothed = (1 - smooth) * scaled / ROPE_FACTOR + smooth * scaled
    medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return np.where(medium, smoothed, scaled).astype(np.float32)


def rope_cos_sin(positions: np.ndarray, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Llama-3 RoPE: cos/sin for each (position, dim-pair)."""
    angles = positions[:, None].astype(np.float32) * rope_inv_freq(head_dim)[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)
    return cos, sin  # (M, head_dim)


def rope_table(n_positions: int, head_dim: int) -> np.ndarray:
    """The device's RoPE table: per position, cos then sin, in bf16.

    Args:
        n_positions: Rows, for positions 0 .. n_positions - 1.
        head_dim: The head dimension.

    Returns:
        bf16 (n_positions, 2 * head_dim).
    """
    cos, sin = rope_cos_sin(np.arange(n_positions), head_dim)
    return np.concatenate([cos, sin], axis=-1).astype(ml_dtypes.bfloat16)


def apply_rope(qk: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """qk: (M, n_heads, head_dim); cos/sin: (M, head_dim)."""
    half = qk.shape[-1] // 2
    x1, x2 = qk[..., :half], qk[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    cs = cos[:, None, :]
    sn = sin[:, None, :]
    return (qk * cs + rotated * sn).astype(np.float32)


def attention_full_gqa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Full causal-mask softmax attention with GQA (8 KV groups, 32 Q heads).

    q: (M, Q_DIM)  k,v: (M_cache, KV_DIM)
    Returns: (M, Q_DIM) attention output (fp32 - to be requantized by
    the o_proj caller).
    """
    M = q.shape[0]
    Mc = k.shape[0]
    q = q.reshape(M, N_HEADS_Q, HEAD_D)
    k = k.reshape(Mc, N_HEADS_KV, HEAD_D)
    v = v.reshape(Mc, N_HEADS_KV, HEAD_D)
    scale = 1.0 / np.sqrt(HEAD_D)
    out = np.zeros((M, N_HEADS_Q, HEAD_D), dtype=np.float32)
    rep = N_HEADS_Q // N_HEADS_KV  # 4
    for h in range(N_HEADS_Q):
        kvh = h // rep
        s = (q[:, h, :] @ k[:, kvh, :].T) * scale  # (M, Mc)
        # Query row i sits at position Mc - M + i and sees keys up to it.
        s = np.where(
            np.arange(Mc)[None, :] > np.arange(Mc - M, Mc)[:, None], -np.inf, s
        )
        s_max = s.max(axis=-1, keepdims=True)
        e = np.exp(s - s_max)
        p = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        out[:, h, :] = p @ v[:, kvh, :]
    return out.reshape(M, Q_DIM)


def forward_layer_int8(
    x_f32: np.ndarray,
    W: LayerW,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    position: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One decoder layer. x_f32: (M, D). Returns (x_out_f32, k_new, v_new).

    For decode M=1 the caller appends k_new, v_new to its running KV cache.
    For prefill M>1 caller stores the full block. (This single-pass code
    handles either by stacking M onto the cache before attention.)
    """
    M = x_f32.shape[0]

    # ---- Attention ----
    h = rmsnorm(x_f32, W.gamma_in)
    q = linear_int8(h, W.wq_i8, W.wq_sc)
    k = linear_int8(h, W.wk_i8, W.wk_sc)
    v = linear_int8(h, W.wv_i8, W.wv_sc)
    pos = np.arange(position, position + M)
    cos, sin = rope_cos_sin(pos, HEAD_D)
    q = apply_rope(q.reshape(M, N_HEADS_Q, HEAD_D), cos, sin).reshape(M, Q_DIM)
    k = apply_rope(k.reshape(M, N_HEADS_KV, HEAD_D), cos, sin).reshape(M, KV_DIM)

    # Append this step's K/V to caches (caches may be empty pre-prefill).
    if k_cache.size == 0:
        k_full = k
        v_full = v
    else:
        k_full = np.concatenate([k_cache, k], axis=0)
        v_full = np.concatenate([v_cache, v], axis=0)

    a = attention_full_gqa(q, k_full, v_full)
    a = linear_int8(a, W.wo_i8, W.wo_sc)
    x_f32 = x_f32 + a

    # ---- FFN (SwiGLU) ----
    h = rmsnorm(x_f32, W.gamma_post)
    g = linear_int8(h, W.wg_i8, W.wg_sc)
    u = linear_int8(h, W.wu_i8, W.wu_sc)
    s = silu(g) * u
    d = linear_int8(s, W.wd_i8, W.wd_sc)
    x_f32 = x_f32 + d

    return x_f32, k_full, v_full


def embed_tokens(model: ModelW, token_ids: np.ndarray) -> np.ndarray:
    """token_ids: (M,) int. Returns (M, D) fp32 via dequant of the
    quantized embed table."""
    rows = model.embed_i8[token_ids].astype(np.float32)
    return rows * model.embed_sc[token_ids][:, None]


def lm_head_logits(model: ModelW, x_f32: np.ndarray) -> np.ndarray:
    """Final RMSNorm + tied lm_head projection. x_f32: (M, D) -> (M, V)."""
    h = rmsnorm(x_f32, model.final_norm)
    return linear_int8(h, model.embed_i8, model.embed_sc)


def forward_full(model: ModelW, token_ids: np.ndarray) -> np.ndarray:
    """Prefill: feed token_ids through embedding + all layers + final
    norm + lm_head. Returns logits of shape (M, V). For decode pass
    a single new token id (shape (1,)) and the caller's running KV
    cache."""
    x = embed_tokens(model, token_ids)
    # Empty caches; full-sequence prefill.
    kc = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    vc = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    for L, W in enumerate(model.layers):
        x, kc[L], vc[L] = forward_layer_int8(x, W, kc[L], vc[L], position=0)
    return lm_head_logits(model, x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    p.add_argument(
        "--layer", type=int, default=0, help="single layer to test (default 0)"
    )
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--full",
        action="store_true",
        help="run full N-layer + lm_head instead of one layer",
    )
    args = p.parse_args()

    print(f"loading INT8 weights from {args.data_dir}/ ...")
    model = load_model(args.data_dir)
    print(f"  embed: {model.embed_i8.shape} (scales {model.embed_sc.shape})")
    print(f"  final_norm: {model.final_norm.shape}")
    print(f"  layers: {len(model.layers)}")
    L0 = model.layers[0]
    print(f"  layer 00 wq: {L0.wq_i8.shape} scales {L0.wq_sc.shape}")
    print(f"  layer 00 wd: {L0.wd_i8.shape} scales {L0.wd_sc.shape}")

    rng = np.random.default_rng(args.seed)

    if args.full:
        # Run on the first few token ids as a sanity check (no tokenizer needed).
        token_ids = np.array([1, 2, 3, 4][: args.M], dtype=np.int64)
        print(f"\nrunning full forward on token_ids={token_ids.tolist()}")
        logits = forward_full(model, token_ids)
        print(f"  logits shape: {logits.shape}")
        last = logits[-1]
        top5 = np.argsort(last)[-5:][::-1]
        print(f"  top-5 token ids at last pos: {top5.tolist()}")
        print(f"  top-5 logit values: {last[top5].tolist()}")
    else:
        x_in = rng.standard_normal((args.M, D)).astype(np.float32)
        print(f"\nrunning ONE layer ({args.layer}) on random (M={args.M}, D={D}) input")
        W = model.layers[args.layer]
        kc = np.zeros((0, KV_DIM), dtype=np.float32)
        vc = np.zeros((0, KV_DIM), dtype=np.float32)
        out, _, _ = forward_layer_int8(x_in, W, kc, vc, position=0)
        print(
            f"  out shape: {out.shape}  norm: {np.linalg.norm(out):.3f}  "
            f"max|abs|: {np.abs(out).max():.3f}"
        )


if __name__ == "__main__":
    main()
