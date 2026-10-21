#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The decode's oracle: what aie2_decode.py computes, in numpy, word for word.

One Llama 3.2 1B GQA layer (32 Q heads, 8 KV heads, per-head and per-token
dynamic scales) as the decode's kernels compute it, the block-wise attention
of llama_attn_block.cc, the one-stream sampler of llama_topk_sample.cc
(xoshiro128++, top-k renormalized over exactly the survivors), and a token
through the layers and the lm_head. Scalar and
vector fp32 adds go through aie.utils.aie_fp32's model of the AIE2P adder.
numpy_llama_ref.py is the INT8 recipe, a different reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aie.utils.aie_fp32 import aie_add
from gen_llama_data import quant_int8_perchan_absmax
from llama_config import (
    HEAD_D,
    KV_TB,
    N_HEADS_KV,
    N_HEADS_Q,
    PROJECTIONS,
    REP,
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


def requant(fp, inv):
    """`round_to_i8(fp * inv)`, the product rounded once in fp32.

    Args:
        fp: fp32 array.
        inv: The fp32 reciprocal of its scale, broadcastable with `fp`.

    Returns:
        The int8 array.
    """
    return round_to_i8(np.asarray(fp, np.float32) * np.asarray(inv, np.float32))


def dyn_scale(fp, axis=None):
    """The kernels' per-token dynamic scale, max(absmax, 1e-12) * (1 / 127).

    Args:
        fp: fp32 array.
        axis: The axis to take the absmax over, or None for the whole array.

    Returns:
        fp32: the scale, one per entry along the other axes.
    """
    absmax = np.maximum(np.abs(fp).max(axis=axis), np.float32(1e-12))
    return (absmax.astype(np.float32) * np.float32(1.0 / 127.0)).astype(np.float32)


def rmsnorm_int8(x_i8, gamma_bf16, act_scale_in, eps=1e-5):
    """RMSNorm of an int8 row, requantized to int8, as llama_rmsnorm_int8_dyn_acttail.

    Args:
        x_i8: int8 (D,).
        gamma_bf16: bf16 (D,).
        act_scale_in: `x_i8`'s fp32 scale.
        eps: The norm's epsilon.

    Returns:
        (int8 (D,), its fp32 scale).
    """
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
    scale_dyn = dyn_scale(y_f)
    inv_dyn = np.float32(1) / np.float32(scale_dyn)
    combined = np.float32(pre * inv_dyn)

    # The kernel recomputes the product rather than keep it between passes.
    out_f = (x_f * combined * g_f).astype(np.float32)
    return round_to_i8(out_f), scale_dyn


def rope_q(q_fp, cos_bf, sin_bf):
    """RoPE over fp32 Q heads, as llama_gemv_array_qkv_comb_cs.

    Args:
        q_fp: fp32 (N_HEADS_Q * HEAD_D,).
        cos_bf: bf16 (HEAD_D,), the position's cos.
        sin_bf: bf16 (HEAD_D,), its sin.

    Returns:
        fp32 (N_HEADS_Q * HEAD_D,), each pair rotated in fp32.
    """
    half = HEAD_D // 2
    x = np.asarray(q_fp, np.float32).reshape(-1, HEAD_D)
    c = np.asarray(cos_bf, np.float32)
    s = np.asarray(sin_bf, np.float32)
    out = np.empty_like(x)
    out[:, :half] = aie_add(
        mul_rne(x[:, :half], c[:half]), -mul_rne(x[:, half:], s[:half])
    )
    out[:, half:] = aie_add(
        mul_rne(x[:, half:], c[half:]), mul_rne(x[:, :half], s[half:])
    )
    return out.ravel()


def silu_mul(gate, up):
    """SiLU(gate) * up in fp32, as llama_silu_gate then llama_silu_up.

    Args:
        gate: fp32 (HD,).
        up: fp32 (HD,).

    Returns:
        fp32 (HD,).
    """
    g = np.asarray(gate, np.float32)
    neg = np.signbit(g)
    e = np.asarray(exp_nonpos(np.where(neg, g, aie_add(np.float32(0), -g))), np.float32)
    d = aie_add(np.float32(1), e)
    y = aie_add(
        np.float32(24) / np.float32(17), -mul_rne(np.float32(8) / np.float32(17), d)
    )
    for _ in range(3):
        y = aie_add(y, mul_rne(y, aie_add(np.float32(1), -mul_rne(d, y))))
    s = np.where(neg, mul_rne(e, y), y)
    return mul_rne(mul_rne(g, s), np.asarray(up, np.float32))


def fold(state, m_b, s_b, sv_b):
    """Folds a block (or a core's state) into `state` as the kernel's `fold`.

    Args:
        `state`: None or (max, sum, sv).
        `m_b`: fp32 max of the incoming part, a scalar or per head.
        `s_b`: fp32 sum of the incoming part, likewise.
        `sv_b`: fp32 normalized sv of the incoming part, (..., head_dim).

    Returns:
        The new (max, sum, sv).
    """
    if state is None:
        return m_b, s_b, sv_b
    m, s, sv = state
    big = np.maximum(m, m_b)
    a = (s * exp_nonpos(aie_add(m, -big))).astype(np.float32)
    b = (s_b * exp_nonpos(aie_add(m_b, -big))).astype(np.float32)
    den = aie_add(a, b)
    inv = (np.float32(1) / den).astype(np.float32)
    wa = (a * inv).astype(np.float32)
    wb = (b * inv).astype(np.float32)
    return (
        big,
        den,
        aie_add(
            mul_rne(sv, np.expand_dims(wa, -1)), mul_rne(sv_b, np.expand_dims(wb, -1))
        ),
    )


def attention_blocks(q_i8, q_scales, k_i8, v_i8, k_slot, v_slot, t, tb):
    """Every Q head over the cache, as the decode's attention cores compute it.

    The heads of a KV head attend its blocks of `tb` rows as
    `llama_attn_block` does, every block at once (`attend_block` per head and
    block); two cores per KV head take the even and the odd blocks, each
    folding its blocks in order, and the second merges the first's state,
    then its own (`llama_attn_merge_pair_q`); each head's sv is then
    requantized by its absmax.

    Args:
        q_i8: int8 (N_HEADS_Q, HEAD_D).
        q_scales: fp32 (N_HEADS_Q,).
        k_i8: int8 (N_HEADS_KV, rows, HEAD_D), rows >= t.
        v_i8: int8 (N_HEADS_KV, rows, HEAD_D).
        k_slot: fp32 (N_HEADS_KV, rows), the rows' K scales.
        v_slot: fp32 (N_HEADS_KV, rows), their V scales.
        t: The live rows.
        tb: Rows per block.

    Returns:
        (int8 (N_HEADS_Q, HEAD_D) output, fp32 (N_HEADS_Q,) sv_out_scales).
    """
    nb = -(-t // tb)
    kvh = np.arange(N_HEADS_Q) // REP
    pad = nb * tb - t
    rows = np.arange(nb * tb).reshape(nb, tb)
    live = rows < t

    k, v, ks, vs = (
        np.pad(x[:, :t], [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        for x in (k_i8, v_i8, k_slot, v_slot)
    )
    k = k.astype(np.float64)
    v = v.reshape(N_HEADS_KV, nb, tb, HEAD_D).astype(np.float32)
    ks = ks.reshape(N_HEADS_KV, nb, tb)
    vs = vs.reshape(N_HEADS_KV, nb, tb)
    dots = np.einsum("hrd,hd->hr", k[kvh], q_i8.astype(np.float64))
    dots = dots.astype(np.float32).reshape(N_HEADS_Q, nb, tb)
    qk = mul_rne(
        mul_rne(np.asarray(q_scales, np.float32)[:, None, None], ks[kvh]),
        np.float32(0.125),
    )
    scores = mul_rne(dots, qk)
    m = np.where(live, scores, -np.inf).max(axis=2).astype(np.float32)
    e = np.where(live, exp_nonpos(aie_add(scores, -m[:, :, None])), np.float32(0))
    s = np.zeros((N_HEADS_Q, nb), np.float32)
    for r in range(tb):
        s = aie_add(s, e[:, :, r])
    pv = mul_rne(mul_rne(e, (np.float32(1) / s)[:, :, None]), vs[kvh])
    acc = np.zeros((N_HEADS_Q, nb, HEAD_D), np.float32)
    for r in range(tb):
        acc = aie_add(acc, mul_rne(v[kvh, :, r, :], pv[:, :, r, None]))

    merged = None
    for core in range(2):
        state = None
        for b in range(core, nb, 2):
            state = fold(state, m[:, b], s[:, b], acc[:, b])
        if state is not None:
            merged = fold(merged, *state)
    sv = merged[2]
    absmax = np.maximum(np.abs(sv).max(axis=1), np.float32(1e-12)).astype(np.float32)
    scales = (absmax * np.float32(1.0 / 127.0)).astype(np.float32)
    out = round_to_i8(mul_rne(sv, (np.float32(1) / scales)[:, None]))
    return out, scales


class Layer:
    """One decoder layer as the decode holds it: weights, norms and KV cache.

    Args:
        weights: Per projection (PROJECTIONS' letters), its int8 weights
            (rows, K) and their fp32 per-row scales.
        gamma_in: bf16 (D,), the attention norm's gamma.
        gamma_post: bf16 (D,), the FFN norm's gamma.
        t: KV cache rows.

    Attributes:
        kcache: int8 (N_HEADS_KV, t, HEAD_D), each row a token's roped K.
        vcache: int8 (N_HEADS_KV, t, HEAD_D).
        k_slot: fp32 (N_HEADS_KV, t), each row's K scale.
        v_slot: fp32 (N_HEADS_KV, t), each row's V scale.
    """

    def __init__(self, weights, gamma_in, gamma_post, t):
        self.weights = weights
        self.gamma_in = gamma_in
        self.gamma_post = gamma_post
        self.kcache = np.zeros((N_HEADS_KV, t, HEAD_D), np.int8)
        self.vcache = np.zeros((N_HEADS_KV, t, HEAD_D), np.int8)
        self.k_slot = np.zeros((N_HEADS_KV, t), np.float32)
        self.v_slot = np.zeros((N_HEADS_KV, t), np.float32)

    @classmethod
    def real(cls, index, data_dir, t):
        """Llama 3.2 1B's layer `index`, from gen_llama_data.py's `data_dir`.

        Args:
            index: The layer.
            data_dir: data/.
            t: KV cache rows.

        Returns:
            The layer, its cache empty.
        """
        ld = Path(data_dir) / f"layer_{index:02d}"
        weights = {
            p: (
                np.fromfile(ld / f"w{p}.i8.bin", np.int8).reshape(shape),
                np.fromfile(ld / f"w{p}.scales.f32.bin", np.float32),
            )
            for p, shape in PROJECTIONS.items()
        }
        return cls(
            weights,
            np.fromfile(ld / "gamma_in.bf16.bin", bfloat16),
            np.fromfile(ld / "gamma_post.bf16.bin", bfloat16),
            t,
        )

    @classmethod
    def random(cls, rng, t):
        """A seeded random layer at the model's shapes.

        Args:
            rng: The numpy Generator.
            t: KV cache rows.

        Returns:
            The layer, its cache empty.
        """
        weights = {}
        for p, (rows, k) in PROJECTIONS.items():
            base = rng.standard_normal((rows, k)).astype(np.float32) * 0.05
            row_scale = rng.uniform(0.1, 1.0, size=rows).astype(np.float32)
            weights[p] = quant_int8_perchan_absmax(base * row_scale[:, None])
        gammas = [
            (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)
            for _ in range(2)
        ]
        return cls(weights, *gammas, t)

    def project(self, p, act_i8, act_scale):
        """Projection `p` of an int8 activation, dequantized to fp32."""
        w_i8, w_sc = self.weights[p]
        acc = w_i8.astype(np.int32) @ act_i8.astype(np.int32)
        return acc.astype(np.float32) * np.float32(act_scale) * w_sc

    def forward(self, x, position, cos, sin):
        """The layer for one token at `position`, appending its KV row.

        Args:
            x: The residual entering it, (int8 (D,), its fp32 scale, fp32
                (D,)): the int8 form the norm reads and the fp32 form the
                adds use, as the decode's stage rows hold them.
            position: The token's position; it attends rows 0..position.
            cos: bf16 (HEAD_D,), RoPE's cos at `position`.
            sin: bf16 (HEAD_D,), its sin.

        Returns:
            The residual leaving it, in `x`'s form.
        """
        x_i8, res_scale, x_fp = x
        h1, act1 = rmsnorm_int8(np.asarray(x_i8, np.int8), self.gamma_in, res_scale)

        # K roped in fp32 (llama_kv_row.h), then K and V per head to int8,
        # appended at `position`.
        half = HEAD_D // 2
        cf = np.asarray(cos, np.float32)
        sf = np.asarray(sin, np.float32)
        kh = self.project("k", h1, act1).reshape(N_HEADS_KV, HEAD_D)
        k_rope = np.empty_like(kh)
        k_rope[:, :half] = aie_add(
            kh[:, :half] * cf[:half], -(kh[:, half:] * sf[:half])
        )
        k_rope[:, half:] = aie_add(kh[:, half:] * cf[half:], kh[:, :half] * sf[half:])
        vh = self.project("v", h1, act1).reshape(N_HEADS_KV, HEAD_D)
        for cache, slot, rows in (
            (self.kcache, self.k_slot, k_rope),
            (self.vcache, self.v_slot, vh),
        ):
            scales = dyn_scale(rows, axis=1)
            slot[:, position] = scales
            cache[:, position] = requant(rows, (np.float32(1) / scales)[:, None])

        # Q roped in fp32, then per head to int8 (llama_gemv_array_qkv_comb_cs).
        q = rope_q(self.project("q", h1, act1), cos, sin).reshape(N_HEADS_Q, HEAD_D)
        q_scales = dyn_scale(q, axis=1)
        q_i8 = round_to_i8(mul_rne(q, (np.float32(1) / q_scales)[:, None]))
        sv_i8, sv_scales = attention_blocks(
            q_i8,
            q_scales,
            self.kcache,
            self.vcache,
            self.k_slot,
            self.v_slot,
            position + 1,
            KV_TB,
        )

        # The heads' outputs concatenated and requantized together
        # (llama_af_concat_selfcal2), then o and the residual add.
        o_act_scale = dyn_scale(sv_i8.astype(np.float32) * sv_scales[:, None])
        o_inv = np.float32(1) / o_act_scale
        af = requant(sv_i8.astype(np.float32), (sv_scales * o_inv)[:, None]).ravel()
        x1_fp = aie_add(x_fp, self.project("o", af, o_act_scale))
        x1_scale = dyn_scale(x1_fp)
        h2, act2 = rmsnorm_int8(
            requant(x1_fp, np.float32(1) / x1_scale), self.gamma_post, x1_scale
        )

        # SiLU(gate) * up in fp32, requantized for down, then the residual add.
        sf_fp = silu_mul(self.project("g", h2, act2), self.project("u", h2, act2))
        silu_scale = dyn_scale(sf_fp)
        sf = round_to_i8(mul_rne(sf_fp, np.float32(1) / silu_scale))
        xo_fp = aie_add(x1_fp, self.project("d", sf, silu_scale))
        xo_scale = dyn_scale(xo_fp)
        return requant(xo_fp, np.float32(1) / xo_scale), xo_scale, xo_fp


def chain_forward_logits(residual, layers, position, rope, lm):
    """One token's forward at `position` through the layers and the lm_head.

    Appends the token's KV row to each layer's cache.

    Args:
        residual: The layer-0 residual, (int8 (D,), fp32 scale, fp32 (D,)).
        layers: The layers.
        position: The token's position.
        rope: bf16 (t, 2 * HEAD_D), numpy_llama_ref.rope_table.
        lm: (embed_i8, embed_sc, final norm gamma).

    Returns:
        (fp32 logits over the vocab, the final residual in `residual`'s form).
    """
    embed_i8, embed_sc, gamma = lm
    cos, sin = rope[position, :HEAD_D], rope[position, HEAD_D:]
    x = residual
    for layer in layers:
        x = layer.forward(x, position, cos, sin)
    normed_i8, norm_scale = rmsnorm_int8(x[0], gamma, x[1])
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    return logits, x


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
) -> tuple[int, list]:
    """The sampled global token id, and the generation's xoshiro state after it.

    k_set is the resident-set capacity (the kernel's kKset); sampling draws
    one uniform from `state` over the set's best top_k members (by logit, then
    lower id), weighted e^(z - max z) with z = logit / temperature, and top_k
    0 means the whole set. Greedy (temperature <= 0) is the argmax whatever
    top_k is, and leaves `state` as it was.

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

    inv_temp = np.float32(1) / np.float32(temperature)
    max_z = max(np.float32(logits[g] * inv_temp) for g in set_idx)
    # The survivors, in ascending global index, the order the kernel sums in.
    by_logit = sorted(set_idx, key=lambda g: (-logits[g], g))
    survivors = sorted(by_logit[:top_k])
    weights = exp_nonpos(
        aie_add((logits[survivors] * inv_temp).astype(np.float32), -max_z)
    )

    sum_exp = np.float32(0.0)
    for w in weights:
        sum_exp = aie_add(sum_exp, w)
    u_unit, state = xoshiro_uniform(list(state))
    u = np.float32(u_unit * sum_exp)

    c = np.float32(0.0)
    pick = survivors[-1]  # fallback if u falls off the end (fp roundoff)
    for gi, w in zip(survivors, weights):
        c = aie_add(c, w)
        if c > u:
            pick = gi
            break
    return int(pick), state


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
    embed_i8, embed_sc, _ = lm
    rows = []
    for pos, tok in enumerate(tokens):
        seed = embed_seed_ref(embed_i8[tok], float(embed_sc[tok]))
        rows.append(chain_forward_logits(seed, layers, pos, rope, lm)[0])
    return np.stack(rows)
