#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# The residual_dyn/residual_fp32 branches bind names read only under the same
# flag later; pyright can't correlate separate `if`s on one flag.
# pyright: reportPossiblyUnboundVariable=false
"""The decode's oracle: what aie2_decode.py computes, in numpy, word for word.

One Llama 3.2 1B GQA layer (32 Q heads, 8 KV heads, per-Q-head dynamic
scales) as the decode's kernels compute it, the block-wise attention of
llama_attn_block.cc, the one-stream sampler of llama_topk_sample.cc
(xoshiro128++, the exp table, top-k renormalized over exactly the survivors),
and a token through the layers and the lm_head. Scalar fp32 adds go through
aie.utils.aie_fp32's model of the AIE2P adder. numpy_llama_ref.py is the
INT8 recipe, a different reference.
"""

from __future__ import annotations

import numpy as np
from aie.iron.kernels.tables import exp_lut, silu_lut
from aie.utils.aie_fp32 import aie_add
from aie2_decode import KV_TB
from gen_llama_data import quant_int8_perchan_absmax
from llama_config import (
    ACT_SCALE,
    EXP_QUANT_SCALE,
    GATE_INV_OUT_SCALE,
    HD,
    HEAD_D,
    INV_ACT_SCALE,
    KV_DIM,
    N_HEADS_KV,
    N_HEADS_Q,
    Q_DIM,
    REP,
    SAMPLE_EXP_LOW,
    SILU_GATE_SCALE,
    D,
)
from ml_dtypes import bfloat16


def sw_invsqrt(a):
    """Mirrors the kernels' `sw_invsqrt`: Quake-III guess plus 2 Newton steps.

    Args:
        `a`: fp32 scalar.

    Returns:
        The fp32 inverse square root the device computes.
    """
    a = np.float32(a)
    x = (np.int32(0x5F3759DF) - (a.view(np.int32) >> np.int32(1))).view(np.float32)
    for _ in range(2):
        t = np.float32(np.float32(np.float32(np.float32(0.5) * a) * x) * x)
        x = np.float32(x * aie_add(np.float32(1.5), -t))
    return x


def round_to_i8(v):
    """Mirrors the kernels' `round_to_i8`: truncate `v ± 0.5`, saturate.

    Args:
        `v`: fp32 array.

    Returns:
        The int8 array the device computes.
    """
    v = np.asarray(v, np.float32)
    half = np.where(v >= 0, np.float32(0.5), np.float32(-0.5))
    return np.clip(np.trunc(aie_add(v, half)), -128, 127).astype(np.int8)


FLT_MIN = np.float32(np.finfo(np.float32).tiny)
# 2^-g on [0, 1] in Q29 for g in Q23, highest degree first; c0 = 1 exactly.
EXP2_NEG_Q29 = [58465, -684167, 5142538, -29791239, 128969442, -372130494, 536870912]


def mul_rne(a, b):
    """Mirrors `vec_mul_rne`: the IEEE fp32 product, zero below FLT_MIN.

    Args:
        `a`: fp32 scalar or array.
        `b`: fp32 scalar or array, broadcastable with `a`.

    Returns:
        The fp32 products, elementwise.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    exact = a.astype(np.float64) * b.astype(np.float64)
    tiny = (np.abs(a) < FLT_MIN) | (np.abs(b) < FLT_MIN) | (np.abs(exact) < FLT_MIN)
    return np.where(tiny, np.copysign(np.float32(0), exact), (a * b)).astype(np.float32)


def srs_rne(p, shift):
    """Mirrors an acc64 srs by `shift` under rounding_mode::conv_even.

    Args:
        `p`: int64 array.
        `shift`: bits to drop, at least 1.

    Returns:
        p / 2^shift rounded half to even, as int64.
    """
    q = p >> shift
    r = p & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    return q + ((r > half) | ((r == half) & ((q & 1) == 1))).astype(np.int64)


def exp_nonpos(x):
    """Mirrors the kernel's `vec_exp_nonpos` (e^x for x <= 0), lane-wise.

    With t = x * log2(e), k = trunc(t) and f = t - k in (-1, 0], e^x = 2^k *
    2^-g for g = -f rounded to Q23 by the vadd.f of 1 - f; 2^-g is a degree-6
    Q29 polynomial whose Horner products round half to even. 0 below t = -125.

    Args:
        `x`: fp32 scalar or array, at most 0.

    Returns:
        The fp32 values the device computes, shaped like `x`.
    """
    t = mul_rne(x, np.float32("1.442695022"))
    under = t < np.float32(-125.0)
    t = np.where(under, np.float32(0.0), t)
    k = np.trunc(t).astype(np.int32)
    f = aie_add(t, -k.astype(np.float32))
    g = np.asarray(aie_add(np.float32(1.0), -f), np.float32).view(np.int32)
    g = g.astype(np.int64) - 0x3F800000
    p = np.full(g.shape, EXP2_NEG_Q29[0], np.int64)
    for c in EXP2_NEG_Q29[1:]:
        p = srs_rne(p * g, 23) + c
    e = (p.astype(np.float64) / 2.0**29).astype(np.float32)
    e = (e.view(np.int32) + (k << 23)).view(np.float32)
    return np.where(under, np.float32(0.0), e).astype(np.float32)[()]


def attend_block(q_i8, q_scale, k_i8, v_i8, k_slot, v_slot):
    """One Q head over one block's rows, as `llama_attn_block` computes it.

    Args:
        `q_i8`: int8 (head_dim,).
        `q_scale`: fp32 scale of `q_i8`.
        `k_i8`: int8 (n, head_dim).
        `v_i8`: int8 (n, head_dim).
        `k_slot`: fp32 (n,) per-row K scales.
        `v_slot`: fp32 (n,) per-row V scales.

    Returns:
        (block max, block sum, normalized sv fp32 (head_dim,)).
    """
    dots = (k_i8.astype(np.int32) @ q_i8.astype(np.int32)).astype(np.float32)
    qk = mul_rne(mul_rne(np.float32(q_scale), k_slot), np.float32(0.125))
    scores = mul_rne(dots, qk)
    m = np.float32(scores.max())
    e = exp_nonpos(aie_add(scores, -m))
    s = np.float32(0.0)
    for ei in e:
        s = aie_add(s, ei)
    pv = mul_rne(mul_rne(e, np.float32(1) / np.float32(s)), v_slot)
    acc = np.zeros(q_i8.shape[0], np.float32)
    for r in range(len(e)):
        acc = aie_add(acc, mul_rne(v_i8[r].astype(np.float32), pv[r]))
    return m, s, acc


def fold(state, m_b, s_b, sv_b):
    """Folds a block (or a core's state) into `state` as the kernel's `fold`.

    Args:
        `state`: None or (max, sum, sv).
        `m_b`: fp32 max of the incoming part.
        `s_b`: fp32 sum of the incoming part.
        `sv_b`: fp32 (head_dim,) normalized sv of the incoming part.

    Returns:
        The new (max, sum, sv).
    """
    if state is None:
        return m_b, s_b, sv_b
    m, s, sv = state
    big = max(m, m_b)
    a = np.float32(s * exp_nonpos(aie_add(m, -big)))
    b = np.float32(s_b * exp_nonpos(aie_add(m_b, -big)))
    den = aie_add(a, b)
    inv = np.float32(1) / np.float32(den)
    wa = np.float32(a * inv)
    wb = np.float32(b * inv)
    return big, den, aie_add(mul_rne(sv, wa), mul_rne(sv_b, wb))


def core_ranges(n_blocks, n_cores):
    """The contiguous block range each core attends: [c*n//C, (c+1)*n//C).

    Args:
        `n_blocks`: blocks in the live cache.
        `n_cores`: cores sharing them.

    Returns:
        A (first, end) pair per core.
    """
    return [
        (c * n_blocks // n_cores, (c + 1) * n_blocks // n_cores) for c in range(n_cores)
    ]


def attn_block_attention(
    q_chunk_i8,
    q_scales,
    k_i8,
    v_i8,
    k_slot,
    v_slot,
    t,
    tb,
    n_cores,
    interleaved=False,
):
    """`llama_attn_block` over the cache's blocks, then `llama_attn_merge_pair_selfcal`.

    Args:
        `q_chunk_i8`: int8 (rep, head_dim).
        `q_scales`: fp32 (rep,).
        `k_i8`: int8 (t, head_dim).
        `v_i8`: int8 (t, head_dim).
        `k_slot`: fp32 (t,).
        `v_slot`: fp32 (t,).
        `t`: live rows.
        `tb`: rows per block.
        `n_cores`: cores the blocks are split across.
        `interleaved`: core c takes blocks c, c + n_cores, ... instead of a
            contiguous range (the array decode's split, where every core sees
            every block).

    Returns:
        (int8 (rep, head_dim) output, fp32 (rep,) sv_out_scales, fp32 (rep, head_dim) sv).
    """
    n_blocks = -(-t // tb)
    per_core = (
        [range(c, n_blocks, n_cores) for c in range(n_cores)]
        if interleaved
        else [range(first, end) for first, end in core_ranges(n_blocks, n_cores)]
    )
    rep = q_chunk_i8.shape[0]
    out = np.zeros(q_chunk_i8.shape, np.int8)
    scales = np.zeros(rep, np.float32)
    svs = np.zeros(q_chunk_i8.shape, np.float32)
    for h in range(rep):
        merged = None
        for blocks in per_core:
            state = None
            for b in blocks:
                rows = slice(b * tb, min(t, (b + 1) * tb))
                state = fold(
                    state,
                    *attend_block(
                        q_chunk_i8[h],
                        q_scales[h],
                        k_i8[rows],
                        v_i8[rows],
                        k_slot[rows],
                        v_slot[rows],
                    ),
                )
            if state is not None:
                merged = fold(merged, *state)
        if merged is None:
            raise ValueError("attention needs at least one live row")
        sv = merged[2]
        absmax = np.float32(max(float(np.abs(sv).max()), 1e-12))
        scale = np.float32(absmax * np.float32(1.0 / 127.0))
        out[h] = round_to_i8(mul_rne(sv, np.float32(1) / np.float32(scale)))
        scales[h] = scale
        svs[h] = sv
    return out, scales, svs


T = 128


def requant(fp, inv):
    return round_to_i8(np.asarray(fp, np.float32) * np.float32(inv))


def attn_quant_shifted(shifted: np.ndarray) -> np.ndarray:
    """The exp LUT index of max-shifted scores, as the kernels' quant_shifted."""
    # 1/scale in fp32, as the kernel takes it; fp64's 1/0.05 differs in the last bit.
    inv = np.float32(1.0) / np.float32(EXP_QUANT_SCALE)
    v = (shifted.astype(np.float32) * inv).astype(np.float32)
    half = np.where(v >= 0, np.float32(0.5), np.float32(-0.5))
    q = np.trunc(aie_add(v, half)).astype(np.int32)
    return np.clip(q, -128, 0)


def numpy_attention(
    q_i8, k_i8, v_i8, head_dim, t, q_scale, k_scale, v_scale, inv_out_scale, lut
):
    """LUT-softmax attention with one k and v scale per head, in fp32."""
    qs = np.float32(q_scale)
    ks = np.float32(k_scale)
    inv_sqrt = np.float32(0.125)
    qk_scale = ((qs * ks).astype(np.float32) * inv_sqrt).astype(np.float32)

    k_mat = k_i8.astype(np.int32).reshape(t, head_dim)
    dots = k_mat @ q_i8.astype(np.int32)
    scores = dots.astype(np.float32) * qk_scale

    max_s = np.float32(scores.max())
    shifted = aie_add(scores, -max_s)
    q = attn_quant_shifted(shifted)
    exp_v = lut[q + 128].astype(np.float32)

    # Sums run left to right, not in numpy's pairwise order.
    sum_e = np.float32(0.0)
    for i in range(t):
        sum_e = aie_add(sum_e, exp_v[i])
    inv_sum = np.float32(1) / np.float32(sum_e)
    probs = (exp_v * inv_sum).astype(np.float32)

    v_mat = v_i8.astype(np.int32).reshape(t, head_dim).astype(np.float32)
    acc = np.zeros(head_dim, dtype=np.float32)
    for i in range(t):
        acc = aie_add(acc, probs[i] * v_mat[i])
    vs = np.float32(v_scale)
    ios = np.float32(inv_out_scale)
    out_f = ((acc * vs).astype(np.float32) * ios).astype(np.float32)
    return round_to_i8(out_f)


def numpy_rmsnorm_int8_dyn(
    x_i8: np.ndarray,
    gamma_bf16: np.ndarray,
    act_scale_in: float | np.floating,
    eps: float = 1e-5,
):
    """Returns (y_i8: int8[D], scale_dyn: fp32), as llama_rmsnorm_int8_dyn_acttail."""
    sum_sq_i32 = int((x_i8.astype(np.int32) ** 2).sum())
    var = np.float32(
        np.float32(sum_sq_i32)
        * np.float32(act_scale_in)
        * np.float32(act_scale_in)
        / np.float32(len(x_i8))
    )
    inv_rms = sw_invsqrt(aie_add(var, np.float32(eps)))
    pre = np.float32(np.float32(act_scale_in) * np.float32(inv_rms))

    g_f = gamma_bf16.astype(np.float32)
    x_f = x_i8.astype(np.float32)
    y_f = (x_f * pre * g_f).astype(np.float32)
    absmax = float(np.abs(y_f).max())
    if absmax < 1e-12:
        absmax = 1e-12
    scale_dyn = np.float32(absmax * np.float32(1.0 / 127.0))
    inv_dyn = np.float32(1) / np.float32(scale_dyn)
    combined = np.float32(pre * inv_dyn)

    # The kernel recomputes the product rather than keep it between passes.
    out_f = (x_f * combined * g_f).astype(np.float32)
    y_i8 = round_to_i8(out_f)
    return y_i8, float(scale_dyn)


def numpy_rope_dyn(x_i8, cos_bf, sin_bf, n_heads, head_dim):
    """RoPE over `n_heads` int8 heads, as kernels/llama_rope_heads.h."""
    half = head_dim // 2
    x_f = x_i8.astype(np.float32).reshape(n_heads, head_dim)
    c = cos_bf.astype(np.float32)
    s = sin_bf.astype(np.float32)
    out = np.empty_like(x_f)
    out[:, :half] = aie_add(x_f[:, :half] * c[:half], -(x_f[:, half:] * s[:half]))
    out[:, half:] = aie_add(x_f[:, half:] * c[half:], x_f[:, :half] * s[half:])
    return round_to_i8(out.flatten())


def numpy_silu_mul(gate, up, lut, up_scale, inv_out_scale):
    """SiLU(gate) * up through the gate's LUT, as llama_silu_mul_int8."""
    silu_bf = lut[(gate.astype(np.int32) + 128)]
    s_f = silu_bf.astype(np.float32)
    u_f = up.astype(np.float32) * up_scale
    out_f = s_f * u_f * inv_out_scale
    return round_to_i8(out_f)


def _dyn_scale(fp):
    """Per-token dynamic scale, bit-identical to the device rescale-add kernel:
    absmax (fp32, 1e-12 floor) * (1.0f/127.0f). Using float64 /127.0 diverges
    by ~1 ULP and flips knife-edge requant roundings, so keep everything fp32."""
    absmax = np.float32(max(float(np.abs(fp).max()), 1e-12))
    return np.float32(absmax * np.float32(1.0 / 127.0))


def attention_perslot(
    q_i8, k_i8, v_i8, head_dim, t, q_scale, k_slot_scales, v_slot_scales, inv_out_scale
):
    """Attention with each cached position dequantized by its own k/v scale.

    fp32 softmax. q_i8: (head_dim,) int8; k_i8/v_i8: (t*head_dim,) int8;
    *_slot_scales: (t,). Returns int8[head_dim] at inv_out_scale.
    """
    qf = q_i8.astype(np.float32) * np.float32(q_scale)
    k_mat = k_i8.astype(np.float32).reshape(t, head_dim) * k_slot_scales[:t, None]
    v_mat = v_i8.astype(np.float32).reshape(t, head_dim) * v_slot_scales[:t, None]
    scale = np.float32(1.0) / np.sqrt(np.float32(head_dim))
    scores = (k_mat @ qf) * scale  # (t,)
    scores = scores - scores.max()
    e = np.exp(scores)
    probs = e / e.sum()
    sv = probs @ v_mat  # (head_dim,)
    return requant(sv, np.float32(inv_out_scale)), sv.astype(np.float32)


def i8_add_wrap(a, b):
    return (a.astype(np.int32) + b.astype(np.int32)).astype(np.int8)


def numpy_gemm_perchan(act_i8, x_scale, w_i8, w_scales, bias_i32, inv_out_scale):
    """Bit-exact with the perchan kernel.
    acc = w@a (int32) + bias
    fp = float(acc) * x_scale * w_scales[n]
    out = round_half_away_from_zero(fp * inv_out_scale).clip(-128,127)
    """
    acc = w_i8.astype(np.int32) @ act_i8.astype(np.int32) + bias_i32
    fp = acc.astype(np.float32) * np.float32(x_scale) * w_scales.astype(np.float32)
    scaled = fp * np.float32(inv_out_scale)
    rounded = np.where(
        scaled >= 0,
        np.floor(scaled + np.float32(0.5)),
        np.ceil(scaled - np.float32(0.5)),
    )
    return rounded.clip(-128, 127).astype(np.int8)


def compute_sv_fp(qr, kcache, vcache, head_dim, t, q_scale, k_scale, v_scale, lut):
    """Replicate numpy_attention's fp32 sv computation up to the final
    (* inv_out_scale + round) — returns the unrounded fp32 sv vector so
    the caller can derive sv_out_scale = max(|sv|)/127."""
    qs = np.float32(q_scale)
    ks = np.float32(k_scale)
    inv_sqrt = np.float32(0.125)
    qk_scale = ((qs * ks).astype(np.float32) * inv_sqrt).astype(np.float32)
    k_mat = kcache.astype(np.int32).reshape(t, head_dim)
    dots = k_mat @ qr.astype(np.int32)
    scores = dots.astype(np.float32) * qk_scale
    max_s = np.float32(scores.max())
    shifted = aie_add(scores, -max_s)
    q = attn_quant_shifted(shifted)
    exp_v = lut[q + 128].astype(np.float32)
    sum_e = np.float32(0.0)
    for i in range(t):
        sum_e = aie_add(sum_e, exp_v[i])
    inv_sum = np.float32(1) / np.float32(sum_e)
    probs = (exp_v * inv_sum).astype(np.float32)
    v_mat = vcache.astype(np.int32).reshape(t, head_dim).astype(np.float32)
    acc = np.zeros(head_dim, dtype=np.float32)
    for i in range(t):
        acc = aie_add(acc, probs[i] * v_mat[i])
    return (acc * np.float32(v_scale)).astype(np.float32)


def numpy_attention_perslot_flowkv(
    q_i8,
    k_i8,
    v_i8,
    head_dim,
    t,
    q_scale,
    k_slot_scales,
    v_slot_scales,
    inv_out_scale,
):
    """Per-cache-slot attention with the device's softmax, e =
    exp_nonpos(score - max), unlike the np.exp attention_perslot, which is
    the quality golden. Op order, as the device's:
      score[i] = dot_i32 * (q_scale*0.125 * k_slot[i])     (strict fp32)
      sv[j]   += (e*inv_sum) * v_slot[i] * v[i,j]          (strict fp32)
    q_i8: (head_dim,); k_i8/v_i8: (t*head_dim,); *_slot_scales: (>=t,).
    """
    qs = np.float32(q_scale)
    inv_sqrt = np.float32(0.125)

    k_mat = k_i8.astype(np.int32).reshape(t, head_dim)
    dots = (k_mat @ q_i8.astype(np.int32)).astype(np.int32)  # (t,)
    ksl = np.asarray(k_slot_scales, dtype=np.float32)[:t]
    # Scale order (q_scale * k_slot[i]) * 0.125 mirrors the kernel;
    # (q*0.125)*k diverges ~1 ULP.
    qk = ((qs * ksl).astype(np.float32) * inv_sqrt).astype(np.float32)
    scores = (dots.astype(np.float32) * qk).astype(np.float32)

    max_s = np.float32(scores.max())
    exp_v = exp_nonpos(aie_add(scores, -max_s))

    sum_e = np.float32(0.0)
    for i in range(t):
        sum_e = aie_add(sum_e, exp_v[i])
    inv_sum = np.float32(1) / np.float32(sum_e)
    probs = (exp_v * inv_sum).astype(np.float32)

    # sv: kernel accumulates acc[j] += p[i] * v_slot[i] * v[i,j], strictly
    # left-to-right per output channel.
    vsl = np.asarray(v_slot_scales, dtype=np.float32)[:t]
    v_mat = v_i8.astype(np.int32).reshape(t, head_dim).astype(np.float32)
    acc = np.zeros(head_dim, dtype=np.float32)
    for i in range(t):
        acc = aie_add(acc, (probs[i] * vsl[i]).astype(np.float32) * v_mat[i])
    out_f = (acc * np.float32(inv_out_scale)).astype(np.float32)
    return round_to_i8(out_f), acc.astype(np.float32)


def gen_real_layer_mh(L: int, data_dir, rng: np.random.Generator) -> dict:
    """Layer L's Llama 3.2 1B weights from data_dir, in the 8-KV-head layout,
    with empty KV caches."""
    from pathlib import Path

    ld = Path(data_dir) / f"layer_{L:02d}"

    def _i8(name, shape):
        return (
            np.frombuffer((ld / name).read_bytes(), dtype=np.int8).reshape(shape).copy()
        )

    def _f32(name, shape):
        return (
            np.frombuffer((ld / name).read_bytes(), dtype=np.float32)
            .reshape(shape)
            .copy()
        )

    def _bf16(name, shape):
        return (
            np.frombuffer((ld / name).read_bytes(), dtype=bfloat16)
            .reshape(shape)
            .copy()
        )

    wq_i8 = _i8("wq.i8.bin", (Q_DIM, D))
    wq_sc = _f32("wq.scales.f32.bin", (Q_DIM,))
    bq = np.zeros(Q_DIM, np.int32)

    wk_i8 = _i8("wk.i8.bin", (KV_DIM, D))
    wk_sc = _f32("wk.scales.f32.bin", (KV_DIM,))
    wv_i8 = _i8("wv.i8.bin", (KV_DIM, D))
    wv_sc = _f32("wv.scales.f32.bin", (KV_DIM,))

    wo_i8 = _i8("wo.i8.bin", (D, Q_DIM))
    wo_sc = _f32("wo.scales.f32.bin", (D,))
    bo = np.zeros(D, np.int32)

    wg_i8 = _i8("wg.i8.bin", (HD, D))
    wg_sc = _f32("wg.scales.f32.bin", (HD,))
    bg = np.zeros(HD, np.int32)
    wu_i8 = _i8("wu.i8.bin", (HD, D))
    wu_sc = _f32("wu.scales.f32.bin", (HD,))
    bu = np.zeros(HD, np.int32)
    wd_i8 = _i8("wd.i8.bin", (D, HD))
    wd_sc = _f32("wd.scales.f32.bin", (D,))
    bd = np.zeros(D, np.int32)

    gamma_in = _bf16("gamma_in.bf16.bin", (D,))
    gamma_post = _bf16("gamma_post.bf16.bin", (D,))

    half = HEAD_D // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    # numpy_layer_mh_forward(position=...) fills a slot, and its scales, a token.
    kcaches = [np.zeros(T * HEAD_D, dtype=np.int8) for _ in range(N_HEADS_KV)]
    vcaches = [np.zeros(T * HEAD_D, dtype=np.int8) for _ in range(N_HEADS_KV)]
    k_scales = np.full(N_HEADS_KV, 1e-6, dtype=np.float32)
    v_scales = np.full(N_HEADS_KV, 1e-6, dtype=np.float32)

    return dict(
        lut_exp=exp_lut(EXP_QUANT_SCALE).astype(np.float32),
        lut_silu=silu_lut(SILU_GATE_SCALE),
        wq_i8=wq_i8,
        wq_sc=wq_sc,
        bq=bq,
        wk_i8=wk_i8,
        wk_sc=wk_sc,
        wv_i8=wv_i8,
        wv_sc=wv_sc,
        wo_i8=wo_i8,
        wo_sc=wo_sc,
        bo=bo,
        wg_i8=wg_i8,
        wg_sc=wg_sc,
        bg=bg,
        wu_i8=wu_i8,
        wu_sc=wu_sc,
        bu=bu,
        wd_i8=wd_i8,
        wd_sc=wd_sc,
        bd=bd,
        cos=cos,
        sin=sin,
        kcaches=kcaches,
        vcaches=vcaches,
        k_scales=k_scales,
        v_scales=v_scales,
        gamma_in=gamma_in,
        gamma_post=gamma_post,
    )


def gen_layer_mh(rng: np.random.Generator) -> dict:
    """Random per-layer fixture at multi-head GQA shapes."""

    def random_w(out_dim, in_dim):
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
        row_scale = rng.uniform(0.1, 1.0, size=out_dim).astype(np.float32)
        return base * row_scale[:, None]

    wq_fp = random_w(Q_DIM, D)
    bq = np.zeros(Q_DIM, np.int32)
    wo_fp = random_w(D, Q_DIM)
    bo = np.zeros(D, np.int32)
    wg_fp = random_w(HD, D)
    bg = np.zeros(HD, np.int32)
    wu_fp = random_w(HD, D)
    bu = np.zeros(HD, np.int32)
    wd_fp = random_w(D, HD)
    bd = np.zeros(D, np.int32)
    wq_i8, wq_sc = quant_int8_perchan_absmax(wq_fp)
    wo_i8, wo_sc = quant_int8_perchan_absmax(wo_fp)
    wg_i8, wg_sc = quant_int8_perchan_absmax(wg_fp)
    wu_i8, wu_sc = quant_int8_perchan_absmax(wu_fp)
    wd_i8, wd_sc = quant_int8_perchan_absmax(wd_fp)

    # Rope: one (cos, sin) pair for the single decode position.
    half = HEAD_D // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    # Per-KV-head cache + scale.
    kcaches = [
        rng.integers(-32, 33, size=T * HEAD_D, dtype=np.int8) for _ in range(N_HEADS_KV)
    ]
    vcaches = [
        rng.integers(-32, 33, size=T * HEAD_D, dtype=np.int8) for _ in range(N_HEADS_KV)
    ]
    k_scales = np.full(N_HEADS_KV, 0.05, dtype=np.float32)
    v_scales = np.full(N_HEADS_KV, 0.05, dtype=np.float32)

    gamma_in = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)
    gamma_post = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(
        bfloat16
    )

    # wk/wv drawn LAST so the earlier rng consumers (kcaches/vcaches/gammas)
    # keep their draws.
    wk_fp = random_w(KV_DIM, D)
    wv_fp = random_w(KV_DIM, D)
    wk_i8, wk_sc = quant_int8_perchan_absmax(wk_fp)
    wv_i8, wv_sc = quant_int8_perchan_absmax(wv_fp)

    return dict(
        lut_exp=exp_lut(EXP_QUANT_SCALE).astype(np.float32),
        lut_silu=silu_lut(SILU_GATE_SCALE),
        wq_i8=wq_i8,
        wq_sc=wq_sc,
        bq=bq,
        wk_i8=wk_i8,
        wk_sc=wk_sc,
        wv_i8=wv_i8,
        wv_sc=wv_sc,
        wo_i8=wo_i8,
        wo_sc=wo_sc,
        bo=bo,
        wg_i8=wg_i8,
        wg_sc=wg_sc,
        bg=bg,
        wu_i8=wu_i8,
        wu_sc=wu_sc,
        bu=bu,
        wd_i8=wd_i8,
        wd_sc=wd_sc,
        bd=bd,
        cos=cos,
        sin=sin,
        kcaches=kcaches,
        vcaches=vcaches,
        k_scales=k_scales,
        v_scales=v_scales,
        gamma_in=gamma_in,
        gamma_post=gamma_post,
    )


def numpy_layer_mh_forward(
    x,
    layer,
    position=None,
    residual_fp32=False,
    residual_dyn=False,
    attn_perslot=None,
    attn_flowkv=False,
    attn_tb=None,
    attn_cores=1,
    attn_interleaved=False,
):
    """Multi-head GQA single-layer forward + per-Q-head scale calibration.

    Returns (x_out, scales) where scales is a dict packing every
    dynamic scale that the on-device design needs in its wblob/kvblob
    prefixes/tails.

    residual_fp32 (proof-of-concept): when True, the residual stream is
    carried in fp32 instead of int8@static-0.05, and each rmsnorm input is
    quantized per-token. This isolates whether the int8 residual format is
    the dominant quality killer (it is — int8@0.05 clips at +-6.35 while the
    residual grows to ~6.7 per-token by layer 15). All OTHER mh quantizations
    (per-Q-head, per-KV-head, silu LUT, af-concat, gate/up/down perchan) are
    unchanged, so a quality recovery here pins the residual as the cause.
    `x` is fp32 in and fp32 out when this flag is set.

    residual_dyn: the device's form. `x` is (int8[D], f32 scale, fp32[D]) in
    and out: the int8 form each norm reads and the fp32 residual both adds
    use, as the decode's stage rows hold them.

    attn_tb: attend in blocks of this many rows, as the growing decode's
    llama_attn_block does (attn_block_attention), split across
    `attn_cores` cores per KV head, contiguously or (`attn_interleaved`)
    block by block; needs `position`.
    """
    lut_exp = layer["lut_exp"]
    lut_silu = layer["lut_silu"]
    gamma_in = layer["gamma_in"]
    gamma_post = layer["gamma_post"]
    wq_i8 = layer["wq_i8"]
    wq_sc = layer["wq_sc"]
    bq = layer["bq"]
    wo_i8 = layer["wo_i8"]
    wo_sc = layer["wo_sc"]
    bo = layer["bo"]
    wg_i8 = layer["wg_i8"]
    wg_sc = layer["wg_sc"]
    bg = layer["bg"]
    wu_i8 = layer["wu_i8"]
    wu_sc = layer["wu_sc"]
    bu = layer["bu"]
    wd_i8 = layer["wd_i8"]
    wd_sc = layer["wd_sc"]
    bd = layer["bd"]
    cos = layer["cos"]
    sin = layer["sin"]
    kcaches = layer["kcaches"]
    vcaches = layer["vcaches"]
    k_scales = layer["k_scales"]
    v_scales = layer["v_scales"]

    # The residual entering the layer: int8 at the fixed ACT_SCALE (default),
    # fp32 (residual_fp32), or int8 with its own per-token scale (residual_dyn,
    # the device's form).
    if residual_dyn:
        # The norm's int8 form and its scale, and the fp32 residual the adds use.
        x_i8, res_scale, x_resid_fp = x
        x_i8 = np.asarray(x_i8, dtype=np.int8)
        rms_in_scale = float(res_scale)
    elif residual_fp32:
        x_resid_fp = x.astype(np.float32)
        res_scale = float(np.maximum(np.abs(x_resid_fp).max(), 1e-12)) / 127.0
        x_i8 = requant(x_resid_fp, np.float32(1.0) / np.float32(res_scale))
        rms_in_scale = res_scale
    else:
        x_i8 = x
        rms_in_scale = ACT_SCALE

    # 1) rmsnorm1 -> h1 (per-token dynamic output scale).
    h1, act_scale1 = numpy_rmsnorm_int8_dyn(x_i8, gamma_in, rms_in_scale)

    # 1b) With a position: K/V projection, rope_k, per-KV-head requant and the
    # cache append there; without one the given caches are used as they are.
    if position is not None:
        wk_i8 = layer["wk_i8"]
        wk_sc = layer["wk_sc"]
        wv_i8 = layer["wv_i8"]
        wv_sc = layer["wv_sc"]
        k_fp = (
            (wk_i8.astype(np.int32) @ h1.astype(np.int32)).astype(np.float32)
            * np.float32(act_scale1)
            * wk_sc.astype(np.float32)
        )  # (KV_DIM,)
        v_fp = (
            (wv_i8.astype(np.int32) @ h1.astype(np.int32)).astype(np.float32)
            * np.float32(act_scale1)
            * wv_sc.astype(np.float32)
        )
        # Apply rope to k per KV head (Llama half-split convention).
        # cos / sin are full HEAD_D (two copies of half), bf16; cast to fp32.
        cosf = np.asarray(cos, dtype=np.float32)
        sinf = np.asarray(sin, dtype=np.float32)
        half = HEAD_D // 2
        k_rope = np.empty_like(k_fp)
        for h in range(N_HEADS_KV):
            kh = k_fp[h * HEAD_D : (h + 1) * HEAD_D]
            # Copies: as views of kh, the first half's write would change x1.
            x1 = kh[:half].copy()
            x2 = kh[half:].copy()
            out = np.empty(HEAD_D, dtype=np.float32)
            out[:half] = aie_add(x1 * cosf[:half], -(x2 * sinf[:half]))
            out[half:] = aie_add(x2 * cosf[half:], x1 * sinf[half:])
            k_rope[h * HEAD_D : (h + 1) * HEAD_D] = out
        # One k and v scale per cached position; (N_HEADS_KV, T).
        k_scales_slot = layer.get("k_scales_slot")
        if k_scales_slot is None:
            k_scales_slot = np.full((N_HEADS_KV, T), 1e-6, dtype=np.float32)
            v_scales_slot = np.full((N_HEADS_KV, T), 1e-6, dtype=np.float32)
            layer["k_scales_slot"] = k_scales_slot
            layer["v_scales_slot"] = v_scales_slot
        else:
            v_scales_slot = layer["v_scales_slot"]

        # Per-KV-head dynamic quant + cache append at slot `position`.
        for h in range(N_HEADS_KV):
            kh = k_rope[h * HEAD_D : (h + 1) * HEAD_D]
            vh = v_fp[h * HEAD_D : (h + 1) * HEAD_D]
            ks = _dyn_scale(kh)
            vs = _dyn_scale(vh)
            k_scales[h] = np.float32(ks)
            v_scales[h] = np.float32(vs)
            k_scales_slot[h, position] = np.float32(ks)
            v_scales_slot[h, position] = np.float32(vs)
            kcaches[h][position * HEAD_D : (position + 1) * HEAD_D] = requant(
                kh, np.float32(1) / np.float32(ks)
            )
            vcaches[h][position * HEAD_D : (position + 1) * HEAD_D] = requant(
                vh, np.float32(1) / np.float32(vs)
            )

    # 2) q_proj (full Q_DIM=2048)
    fp_q_full = (
        (wq_i8.astype(np.int32) @ h1.astype(np.int32) + bq).astype(np.float32)
        * np.float32(act_scale1)
        * wq_sc.astype(np.float32)
    )

    # 3) Per-Q-head requant. qf shape (Q_DIM,).
    q_out_scales = np.zeros(N_HEADS_Q, dtype=np.float32)
    q_inv_outs = np.zeros(N_HEADS_Q, dtype=np.float32)
    qf = np.zeros(Q_DIM, dtype=np.int8)
    for h in range(N_HEADS_Q):
        slice_h = fp_q_full[h * HEAD_D : (h + 1) * HEAD_D]
        s = _dyn_scale(slice_h)
        q_out_scales[h] = np.float32(s)
        q_inv_outs[h] = np.float32(1) / np.float32(s)
        qf[h * HEAD_D : (h + 1) * HEAD_D] = requant(slice_h, q_inv_outs[h])

    # 4) rope (per-head). numpy_rope_dyn reshapes to (n_heads, head_dim) internally.
    qr = numpy_rope_dyn(qf, cos, sin, N_HEADS_Q, HEAD_D)

    # 5) Per-Q-head attention (kvh = h_q // REP), each head with its own
    #    sv_out_scale. With a position, only slots 0..position (causal);
    #    otherwise all T.
    t_used = (position + 1) if position is not None else T
    # attn_perslot selects the per-cache-slot-scale fp32-softmax attention;
    # it follows residual_dyn unless set.
    if attn_perslot is None:
        attn_perslot = residual_dyn
    sv_out_scales = np.zeros(N_HEADS_Q, dtype=np.float32)
    sv_inv_out_scales = np.zeros(N_HEADS_Q, dtype=np.float32)
    sv_i8_per_head = []
    if attn_tb is not None:
        for kvh in range(N_HEADS_KV):
            heads = slice(kvh * REP, (kvh + 1) * REP)
            out, scales, _ = attn_block_attention(
                qr[kvh * REP * HEAD_D : (kvh + 1) * REP * HEAD_D].reshape(REP, HEAD_D),
                q_out_scales[heads],
                kcaches[kvh][: t_used * HEAD_D].reshape(t_used, HEAD_D),
                vcaches[kvh][: t_used * HEAD_D].reshape(t_used, HEAD_D),
                layer["k_scales_slot"][kvh][:t_used],
                layer["v_scales_slot"][kvh][:t_used],
                t_used,
                attn_tb,
                attn_cores,
                attn_interleaved,
            )
            sv_out_scales[heads] = scales
            for r in range(REP):
                sv_inv_out_scales[kvh * REP + r] = np.float32(1) / np.float32(scales[r])
                sv_i8_per_head.append(out[r])
    else:
        for h_q in range(N_HEADS_Q):
            kvh = h_q // REP
            q_h = qr[h_q * HEAD_D : (h_q + 1) * HEAD_D]
            k_slice = kcaches[kvh][: t_used * HEAD_D]
            v_slice = vcaches[kvh][: t_used * HEAD_D]
            if attn_perslot or attn_flowkv:
                # attn_flowkv=True takes the device's softmax; False keeps the
                # fp32 np.exp softmax (the quality golden). Both use per-slot scales.
                ksl = layer["k_scales_slot"][kvh]
                vsl = layer["v_scales_slot"][kvh]
                attn_fn = (
                    numpy_attention_perslot_flowkv if attn_flowkv else attention_perslot
                )
                # First pass to derive this head's sv_out_scale.
                _, sv_fp_h = attn_fn(
                    q_h,
                    k_slice,
                    v_slice,
                    HEAD_D,
                    t_used,
                    float(q_out_scales[h_q]),
                    ksl,
                    vsl,
                    1.0,
                )
                s = _dyn_scale(sv_fp_h)
                sv_out_scales[h_q] = np.float32(s)
                sv_inv_out_scales[h_q] = np.float32(1) / np.float32(s)
                sv_i8, _ = attn_fn(
                    q_h,
                    k_slice,
                    v_slice,
                    HEAD_D,
                    t_used,
                    float(q_out_scales[h_q]),
                    ksl,
                    vsl,
                    float(sv_inv_out_scales[h_q]),
                )
                sv_i8_per_head.append(sv_i8)
                continue
            sv_fp_h = compute_sv_fp(
                q_h,
                k_slice,
                v_slice,
                HEAD_D,
                t_used,
                float(q_out_scales[h_q]),
                float(k_scales[kvh]),
                float(v_scales[kvh]),
                lut_exp,
            )
            s = float(np.maximum(np.abs(sv_fp_h).max(), 1e-12)) / 127.0
            sv_out_scales[h_q] = np.float32(s)
            sv_inv_out_scales[h_q] = np.float32(1.0) / np.float32(s)
            sv_i8 = numpy_attention(
                q_h,
                k_slice,
                v_slice,
                HEAD_D,
                t_used,
                float(q_out_scales[h_q]),
                float(k_scales[kvh]),
                float(v_scales[kvh]),
                float(sv_inv_out_scales[h_q]),
                lut_exp,
            )
            sv_i8_per_head.append(sv_i8)

    # 6) af-concat: per-head dequant (using each head's sv_out_scale),
    #    concat to Q_DIM fp32, then global requant with o_act_scale.
    af_fp = np.concatenate(
        [
            sv_i8_per_head[h].astype(np.float32) * np.float32(sv_out_scales[h])
            for h in range(N_HEADS_Q)
        ]
    )
    o_act_scale = float(_dyn_scale(af_fp))
    o_inv_act_scale = float(np.float32(1) / np.float32(o_act_scale))
    af = np.concatenate(
        [
            requant(
                sv_i8_per_head[h].astype(np.float32),
                np.float32(sv_out_scales[h]) * np.float32(o_inv_act_scale),
            )
            for h in range(N_HEADS_Q)
        ]
    )

    _cap = layer.get("_cap")
    if _cap is not None:
        _cap["h1_fp"] = h1.astype(np.float32) * np.float32(act_scale1)
        _cap["af_fp"] = af_fp.copy()

    # 7) o_proj: op_fp = acc * o_act_scale * wo_sc (o_act_scale already
    # varies per token, from af's own input).
    op_acc = wo_i8.astype(np.int32) @ af.astype(np.int32) + bo
    op_fp = (
        op_acc.astype(np.float32) * np.float32(o_act_scale) * wo_sc.astype(np.float32)
    )
    if residual_dyn:
        # Rescale-add: fp32 sum of (residual + o_proj), requant to a new
        # per-token residual scale. (Device: o_proj emits dynamic int8+scale,
        # rescale-add kernel reconstructs both and requants.)
        x1_fp = aie_add(x_resid_fp, op_fp)
        # As the device: absmax (1e-12 floor) * (1.0f/127.0f) in fp32; /127.0
        # in float64 is 1 ULP off and flips knife-edge requant roundings.
        x1_scale = _dyn_scale(x1_fp)
        x1_i8 = requant(x1_fp, np.float32(1) / np.float32(x1_scale))
        h2, act_scale2 = numpy_rmsnorm_int8_dyn(x1_i8, gamma_post, float(x1_scale))
    elif residual_fp32:
        x1 = x_resid_fp + op_fp  # fp32 residual
        x1_scale = float(np.maximum(np.abs(x1).max(), 1e-12)) / 127.0
        x1_i8 = requant(x1, np.float32(1.0) / np.float32(x1_scale))
        h2, act_scale2 = numpy_rmsnorm_int8_dyn(x1_i8, gamma_post, x1_scale)
    else:
        op = numpy_gemm_perchan(af, o_act_scale, wo_i8, wo_sc, bo, INV_ACT_SCALE)
        x1 = i8_add_wrap(op, x)
        h2, act_scale2 = numpy_rmsnorm_int8_dyn(x1, gamma_post, ACT_SCALE)

    if _cap is not None:
        _cap["x1"] = x1_fp.copy() if residual_dyn else np.asarray(x1, np.float32).copy()

    # 9) gate (closure-baked GATE_INV_OUT_SCALE = 1/SILU_GATE_SCALE)
    fp_gate = (
        (wg_i8.astype(np.int32) @ h2.astype(np.int32) + bg).astype(np.float32)
        * np.float32(act_scale2)
        * wg_sc.astype(np.float32)
    )
    gf = requant(fp_gate, GATE_INV_OUT_SCALE)

    # 10) up (dynamic out scale)
    fp_up = (
        (wu_i8.astype(np.int32) @ h2.astype(np.int32) + bu).astype(np.float32)
        * np.float32(act_scale2)
        * wu_sc.astype(np.float32)
    )
    up_out_scale = float(_dyn_scale(fp_up))
    up_inv_out = float(np.float32(1) / np.float32(up_out_scale))
    uf = requant(fp_up, up_inv_out)

    # 11) silu_mul
    silu_up_scale = up_out_scale
    s_gate_fp = lut_silu[gf.astype(np.int32) + 128].astype(np.float32)
    s_up_fp = uf.astype(np.float32) * np.float32(silu_up_scale)
    sf_fp = s_gate_fp * s_up_fp
    silu_out_scale = float(_dyn_scale(sf_fp))
    silu_inv_out_scale = float(np.float32(1) / np.float32(silu_out_scale))
    sf = numpy_silu_mul(gf, uf, lut_silu, silu_up_scale, silu_inv_out_scale)

    if _cap is not None:
        _cap["h2_fp"] = h2.astype(np.float32) * np.float32(act_scale2)
        _cap["gate_fp"] = fp_gate.copy()
        _cap["up_fp"] = fp_up.copy()
        _cap["sf_fp"] = sf_fp.copy()

    # 12) down + 13) residual add.
    down_act_scale = silu_out_scale
    if residual_dyn:
        df_acc = wd_i8.astype(np.int32) @ sf.astype(np.int32) + bd
        df_fp = (
            df_acc.astype(np.float32)
            * np.float32(down_act_scale)
            * wd_sc.astype(np.float32)
        )
        xo_fp = aie_add(x1_fp, df_fp)
        xo_scale = _dyn_scale(xo_fp)
        xo_i8 = requant(xo_fp, np.float32(1) / np.float32(xo_scale))
        x_out = (xo_i8, np.float32(xo_scale), xo_fp)
    elif residual_fp32:
        df_acc = wd_i8.astype(np.int32) @ sf.astype(np.int32) + bd
        df_fp = (
            df_acc.astype(np.float32)
            * np.float32(down_act_scale)
            * wd_sc.astype(np.float32)
        )
        x_out = x1 + df_fp  # fp32 residual out
    else:
        df = numpy_gemm_perchan(sf, down_act_scale, wd_i8, wd_sc, bd, INV_ACT_SCALE)
        # 13) add
        x_out = i8_add_wrap(df, x1)

    scales = dict(
        q_out_scales=q_out_scales,
        q_inv_outs=q_inv_outs,
        sv_out_scales=sv_out_scales,
        sv_inv_out_scales=sv_inv_out_scales,
        o_act_scale=o_act_scale,
        o_inv_act_scale=o_inv_act_scale,
        k_scales=k_scales,
        v_scales=v_scales,
        up_out_scale=up_out_scale,
        up_inv_out=up_inv_out,
        silu_up_scale=silu_up_scale,
        silu_inv_out_scale=silu_inv_out_scale,
        down_act_scale=down_act_scale,
    )
    return x_out, scales


def _u32(x):
    return np.uint32(x & 0xFFFFFFFF)


def _u64(x):
    return np.uint64(x & 0xFFFFFFFFFFFFFFFF)


def _rotl32(x: np.uint32, k: int) -> np.uint32:
    x = _u32(x)
    return _u32((int(x) << k) | (int(x) >> (32 - k)))


def xoshiro_seed(seed: int) -> list:
    """splitmix64-style expansion of one uint32 seed into 4 uint32 state."""
    x = _u64(int(seed) * 0x9E3779B97F4A7C15 + 1)
    s = []
    for _ in range(4):
        x = _u64(int(x) ^ (int(x) >> 30))
        x = _u64(int(x) * 0xBF58476D1CE4E5B9)
        x = _u64(int(x) ^ (int(x) >> 27))
        x = _u64(int(x) * 0x94D049BB133111EB)
        x = _u64(int(x) ^ (int(x) >> 31))
        s.append(_u32(int(x) ^ (int(x) >> 32)))
    return s


def xoshiro_next(s: list) -> tuple:
    result = _u32(int(_rotl32(_u32(int(s[0]) + int(s[3])), 7)) + int(s[0]))
    t = _u32(int(s[1]) << 9)
    s[2] = _u32(int(s[2]) ^ int(s[0]))
    s[3] = _u32(int(s[3]) ^ int(s[1]))
    s[1] = _u32(int(s[1]) ^ int(s[2]))
    s[0] = _u32(int(s[0]) ^ int(s[3]))
    s[2] = _u32(int(s[2]) ^ int(t))
    s[3] = _rotl32(s[3], 11)
    return result, s


def xoshiro_uniform(s: list) -> tuple:
    r, s = xoshiro_next(s)
    u = np.float32(np.float32(int(r) >> 8) * np.float32(1.0 / 16777216.0))
    return u, s


def sample_quant_shifted(shifted_f32: np.ndarray) -> np.ndarray:
    inv = np.float32(1.0) / np.float32(EXP_QUANT_SCALE)
    v = (shifted_f32.astype(np.float32) * inv).astype(np.float32)
    q = np.where(
        v >= 0, np.floor(v + np.float32(0.5)), np.ceil(v - np.float32(0.5))
    ).astype(np.int32)
    return np.clip(q, SAMPLE_EXP_LOW, 0)


def sample_lookup_exp(q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[(q - SAMPLE_EXP_LOW).astype(np.int32)].astype(np.float32)


def topk_set_reference(logits_f32: np.ndarray, k_set: int):
    """Mirror llama_lmhead_topk_insert's set: an unsorted set of capacity
    k_set streamed in index order, ordered by logit and then by lower index on
    equal logits. The set is the k_set largest under that order, ties
    included, whatever the streaming order. Returns the set's global indices
    in slot order, as the kernel holds them."""
    V = logits_f32.size
    set_idx: list[int] = []
    for g in range(V):
        lg = logits_f32[g]
        if len(set_idx) < k_set:
            set_idx.append(g)
            continue
        mi = 0
        for i in range(1, k_set):
            a, b = logits_f32[set_idx[i]], logits_f32[set_idx[mi]]
            if a < b or (a == b and set_idx[i] > set_idx[mi]):
                mi = i
        mv = logits_f32[set_idx[mi]]
        if lg > mv or (lg == mv and g < set_idx[mi]):
            set_idx[mi] = g
    return set_idx


def topk_sample_reference(
    logits_f32: np.ndarray,
    temperature: float,
    top_k: int,
    state: list,
    k_set: int = 64,
    lut: np.ndarray | None = None,
) -> tuple[int, list]:
    """The sampled global token id, and the generation's xoshiro state after it.

    k_set is the resident-set capacity (the kernel's kKset); sampling draws
    one uniform from `state` over the set's best top_k members (by logit, then
    lower id), and top_k 0 means the whole set. Greedy (temperature <= 0) is
    the argmax whatever top_k is, and leaves `state` as it was.

    Raises:
        ValueError: top_k is outside 0..k_set.
    """
    if not 0 <= top_k <= k_set:
        raise ValueError(f"top_k must be 0 (the whole set) or 1..{k_set}, got {top_k}")
    top_k = top_k or k_set
    logits = logits_f32.astype(np.float32)
    set_idx = topk_set_reference(logits, k_set)

    # Greedy: global argmax with first-occurrence (smallest gidx) tie-break.
    if temperature <= 0.0:
        wv = max(logits[g] for g in set_idx)
        winners = [g for g in set_idx if logits[g] == wv]
        return int(min(winners)), state

    if lut is None:
        lut = exp_lut(EXP_QUANT_SCALE, SAMPLE_EXP_LOW).astype(np.float32)
    inv_temp = np.float32(1) / np.float32(temperature)

    # The survivors: the set's best top_k by logit, then lower id.
    by_logit = sorted(set_idx, key=lambda g: (-logits[g], g))
    survivors = sorted(by_logit[:top_k])  # ascending global-index order

    zs = [np.float32(logits[g] * inv_temp) for g in survivors]
    max_z = np.float32(max(zs))

    sum_exp = np.float32(0.0)
    for z in zs:
        q = sample_quant_shifted(np.array([z - max_z], dtype=np.float32))[0]
        sum_exp = np.float32(sum_exp + sample_lookup_exp(np.array([q]), lut)[0])

    u_unit, state = xoshiro_uniform(list(state))
    u = np.float32(u_unit * sum_exp)

    c = np.float32(0.0)
    pick = survivors[-1]  # fallback if u falls off the end (fp roundoff)
    for gi, z in zip(survivors, zs):
        q = sample_quant_shifted(np.array([z - max_z], dtype=np.float32))[0]
        c = np.float32(c + sample_lookup_exp(np.array([q]), lut)[0])
        if c > u:
            pick = gi
            break
    return int(pick), state


# A greedy token other than the oracle's still passes as a near-tie, judged
# against the logit spread: the oracle does not yet mirror the vector fp32
# multiplies of rmsnorm, silu and up_requant, which leave rare 1-ulp
# differences that grow with depth.
TOL_GAP_STD = 0.15
TOL_RANK = 5


def chain_forward_logits(residual, layers, position, embed_i8, embed_sc, gamma):
    """One token's forward at `position`, attending [0, position].

    Appends the token's KV row to the layers' caches and sets each layer's
    `scales`.

    Args:
        residual: The layer-0 residual, (int8[D], f32 scale, fp32[D]).

    Returns:
        (fp32 logits over the vocab, the final residual in the same form).
    """
    x_i8, res_scale, x_fp = residual
    x_cur = (x_i8.copy(), np.float32(res_scale), x_fp.copy())
    for L in range(len(layers)):
        x_cur, scales = numpy_layer_mh_forward(
            x_cur,
            layers[L],
            position=position,
            residual_dyn=True,
            attn_perslot=True,
            attn_flowkv=True,
            attn_tb=KV_TB,
            attn_cores=2,
            attn_interleaved=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale, _ = x_cur
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(xo_i8, gamma, np.float32(xo_scale))
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    return logits, x_cur


def greedy_verdict(logits, token):
    """Judge the device's greedy `token` against the oracle's `logits`.

    Args:
        logits: The oracle's fp32 logits for the position.
        token: The token the device picked.

    Returns:
        (whether it passes, the oracle's argmax, a tag saying how).
    """
    ref = int(np.argmax(logits))
    if token == ref:
        return True, ref, "exact"
    gap = float(logits.max() - logits[token]) / float(logits.std())
    rank = int(np.where(np.argsort(logits)[::-1] == token)[0][0])
    return (
        gap <= TOL_GAP_STD and rank < TOL_RANK,
        ref,
        f"near-tie gap/std={gap:.3f} rank={rank}",
    )


def embed_seed_ref(row_i8, embed_sc):
    """A token's layer-0 residual, as llama_topk_sample.cc's residual_seed makes it.

    Args:
        row_i8: The token's int8 embed row.
        embed_sc: Its scale.

    Returns:
        (int8[D], f32 scale, fp32[D]): the row requantized for the first norm,
        and the row itself, row * embed_sc, for the residual adds.
    """
    amax = max(int(np.max(np.abs(row_i8.astype(np.int32)))), 1)
    inv = np.float32(np.float32(1) / np.float32(amax)) * np.float32(127)
    xin = np.empty(D, np.int8)
    for i in range(D):
        v = np.float32(np.float32(row_i8[i]) * inv)
        r = np.floor(v + 0.5) if v >= 0 else np.ceil(v - 0.5)
        xin[i] = np.int8(np.clip(r, -128, 127))
    scale = np.float32(
        np.float32(amax) * np.float32(embed_sc) * np.float32(1.0 / 127.0)
    )
    return xin, scale, row_i8.astype(np.float32) * np.float32(embed_sc)


def teacher_forced_logits(layers, lm, rope, tokens):
    """The oracle's next-token logits at each position of `tokens`.

    Each position's input is its token's embed seed, as the device makes it
    for a prompt token; the layers' caches grow as the device's do.

    Args:
        layers: The oracle's layers, their caches empty.
        lm: (embed_i8, embed_sc, final_norm gamma).
        rope: The RoPE table, a row per position.
        tokens: The token ids, from position 0.

    Returns:
        fp32 logits, (len(tokens), vocab): row p predicts tokens[p + 1].
    """
    embed_i8, embed_sc, gamma = lm
    rows = []
    for pos, tok in enumerate(tokens):
        seed = embed_seed_ref(embed_i8[tok], float(embed_sc[tok]))
        for layer in layers:
            layer["cos"] = rope[pos, :HEAD_D]
            layer["sin"] = rope[pos, HEAD_D:]
        logits, _ = chain_forward_logits(seed, layers, pos, embed_i8, embed_sc, gamma)
        rows.append(logits)
    return np.stack(rows)
