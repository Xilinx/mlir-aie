# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Numeric gates for mha.cc's softmax entry points, alone and as one round.

``partial_softmax`` and ``rescale_O`` read and write a carried float32
``scale_buffer``, so neither fits a kernel contract, whose outputs are zeroed on
core. Each is driven alone, which sees its raw weights and carried state, and
then all four steps run as one decode round, which checks that they agree, and
as one round over a long context, which checks that the carry stays accurate.

References use true ``np.exp2`` and state the mask over absolute positions;
nothing models the device's arithmetic. On AIE2P, ``aie::exp2`` is a linear
interpolant that overshoots by up to 6.15%, so raw weights get that envelope
(``_RTOL_EXP2``) and the round, where it divides out, gets bf16 rounding only.
On AIE2 mha.cc evaluates a cubic instead, and the weights are held to
``kernels.mha_softmax``'s tolerance. With ``accurate_exp2`` both take the same
polynomial and are held to one ulp. Each parametrized configuration is the
only one that catches some mask or carry bug.

Not covered: a key block that is all padding inside the causal region.
``partial_softmax`` returns early there and leaves the previous correction
factor in ``scale_buffer``; that is existing kernel behaviour, not frozen here.
"""

import ml_dtypes
import numpy as np
import pytest
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
    kernels,
)
from aie.iron.controlflow import range_
from aie.utils.compile.utils import resolve_target_arch
from aie.utils.verify import Tolerance, compare, nearly_equal

BF = np.dtype[bfloat16]
F32 = np.dtype[np.float32]
I32 = np.dtype[np.int32]

# The key block and the head, which mha.cc is compiled for. A round's query
# block is these 64 rows or a narrower multiple of 16.
_B = 64

# log2(e) / sqrt(d) as the bf16 scalar the kernel takes.
_INV_SCALE = float(bfloat16(np.log2(np.e) / np.sqrt(_B)))

# In bf16 steps at the top of each quantity's range. The worst case measures
# 1.26 steps; the quietest mutation moves 6.5.
_ATOL_ULP = 4

# init_scale_buffer's seed for the running max; -inf would make the first
# correction factor NaN.
_LOWEST = float(ml_dtypes.finfo(bfloat16).min)

# AIE2P's aie::exp2 evaluates 2**floor(u) * (1 + frac(u)), which overshoots by
# at most 6.15% (at frac(u) = 1/ln2 - 1); with a bf16 rounding on each side the
# envelope is 6.98%. Measured: weights 6.79%, row sums 6.66%.
_RTOL_EXP2 = 0.07

# Under -DEXP2_BF16_ACCURATE, a float32 row sum is off by its weights' bf16
# roundings and its correction factor's, at most 2^-8 of it.
_L_RTOL_ACCURATE = 2.0**-9


def _unblock(flat: np.ndarray, rows: int = _B, cols: int = _B) -> np.ndarray:
    """Return the 8x8-blocked O tile as (rows, cols)."""
    return (
        flat.reshape(rows // 8, cols // 8, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols)
    )


def _keep(
    q_block: int,
    kv_block: int,
    s_q_eff: int,
    s_kv_eff: int,
    b_q: int = _B,
    b_kv: int = _B,
    causal: bool = True,
    window: int = 0,
    packed: bool = False,
    offset: int = 0,
) -> np.ndarray:
    """Which (query, key) pairs of this block pair the band admits.

    The query at q keeps the keys from ``q - window`` to ``q`` when causal,
    else to ``q + window``; a window of 0 is none. Packed, every query is at
    the last key. The query block's rows sit ``offset`` positions past its
    start.
    """
    rows = q_block * b_q + np.arange(b_q)
    cols = kv_block * b_kv + np.arange(b_kv)
    at = np.full(b_q, s_kv_eff - 1) if packed else rows + offset
    ahead = cols[None, :] - at[:, None]
    keep = (rows[:, None] < s_q_eff) & (cols[None, :] < s_kv_eff)
    if causal or window:
        keep &= ahead <= (0 if causal else window)
    if window:
        keep &= ahead >= -window
    return keep


def _softmax_step(scores, keep, m_prev, l_prev):
    """One key block of the online softmax: (weights, running max, row sum).

    The running max is of the unscaled scores, as the kernel keeps it; each
    product with the bf16 scale is exact in float32.
    """
    s = np.float32(_INV_SCALE)
    scores = scores.astype(np.float32)
    m_new = np.where(keep, scores, -np.float32(np.inf)).max(axis=1)
    m_new = np.maximum(m_new, m_prev)
    top = (m_new * s)[:, None]
    p = np.exp2(np.where(keep, scores * s, top) - top)
    # The row sum accumulates the stored bf16 weights.
    p = np.where(keep, p.astype(bfloat16).astype(np.float32), np.float32(0))
    l_new = np.exp2(m_prev * s - m_new * s) * l_prev + p.sum(axis=1)
    return p, m_new, l_new


def _attention(scores, v, keep, *, inv_scale):
    """Masked attention, rounded to bf16 wherever the kernel stores.

    The running sum and output are float32, so only the result is rounded.
    """
    scaled = scores.astype(np.float32) * np.float32(inv_scale)
    live = keep.any(axis=1)
    peak = np.where(live, np.where(keep, scaled, -np.inf).max(axis=1), 0)
    p = np.exp2(scaled - peak[:, None])
    p = np.where(keep, p.astype(bfloat16).astype(np.float32), np.float32(0))
    y = p @ v.astype(np.float32)
    inv_l = np.float32(1.0) / np.where(live, p.sum(axis=1), np.float32(1))
    return (y * inv_l[:, None]).astype(bfloat16)


@jit
def softmax_blocks(
    a_in: In,
    p_out: Out,
    scale_out: Out,
    *,
    q_block: CompileTime[int] = 0,
    kv_first: CompileTime[int] = 0,
    kv_second: CompileTime[int] = 1,
    s_q_eff: CompileTime[int] = 64,
    s_kv_eff: CompileTime[int] = 64,
    accurate_exp2: CompileTime[bool] = False,
):
    """``partial_softmax`` over two key blocks sharing one ``scale_buffer``.

    The second block reads the state the first carried.
    """
    tile_ty = np.ndarray[(_B * _B,), BF]
    scale_ty = np.ndarray[(4 * _B,), F32]
    idx_ty = np.ndarray[(3,), I32]

    mha = kernels.mha(dim_m=_B, dim_k=_B, dim_n=_B, accurate_exp2=accurate_exp2)
    obj = mha.object_file
    softmax = obj.bind(
        "partial_softmax",
        [tile_ty, tile_ty, scale_ty, idx_ty, bfloat16, *([np.int32] * 4)],
    )
    init = obj.bind("init_scale_buffer", [scale_ty, np.int32])

    of_a = ObjectFifo(tile_ty, name="a", depth=2)
    of_p = ObjectFifo(tile_ty, name="p", depth=2)
    of_scale = ObjectFifo(scale_ty, name="scale", depth=1)

    # idx_buffer is (key block, query block, row offset).
    idx = [
        Buffer(
            idx_ty, name=f"idx{n}", initial_value=np.array([kv, q_block, 0], np.int32)
        )
        for n, kv in enumerate((kv_first, kv_second))
    ]

    def core(of_a, of_p, of_scale, softmax, init, idx_first, idx_second):
        scale = of_scale.acquire(1)
        init(scale, _B)
        for idx_buf in (idx_first, idx_second):
            softmax(
                of_a.acquire(1),
                of_p.acquire(1),
                scale,
                idx_buf,
                _INV_SCALE,
                _B,
                _B,
                s_q_eff,
                s_kv_eff,
            )
            of_p.release(1)
            of_a.release(1)
        of_scale.release(1)

    worker = Worker(
        core,
        fn_args=[of_a.cons(), of_p.prod(), of_scale.prod(), softmax, init, *idx],
    )

    host = [
        np.ndarray[(2 * _B * _B,), BF],
        np.ndarray[(2 * _B * _B,), BF],
        scale_ty,
    ]

    def sequence(a_h, p_h, scale_h, a_fifo, p_fifo, scale_fifo):
        a_fifo.fill(a_h)
        p_fifo.drain(p_h, wait=True)
        scale_fifo.drain(scale_h, wait=True)

    rt = Runtime(sequence, [*host, of_a.prod(), of_p.cons(), of_scale.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@jit
def rescale_tile(o_in: In, scale_in: In, o_out: Out):
    """``rescale_O``: the float32 O divided by the row sums into bf16."""
    acc_ty = np.ndarray[(_B * _B,), F32]
    tile_ty = np.ndarray[(_B * _B,), BF]
    scale_ty = np.ndarray[(4 * _B,), F32]
    idx_ty = np.ndarray[(3,), I32]

    mha = kernels.mha(dim_m=_B, dim_k=_B, dim_n=_B)
    rescale = mha.object_file.bind(
        "rescale_O", [acc_ty, tile_ty, scale_ty, np.int32, idx_ty]
    )

    of_o_in = ObjectFifo(acc_ty, name="o_in", depth=1)
    of_scale = ObjectFifo(scale_ty, name="scale", depth=1)
    of_o_out = ObjectFifo(tile_ty, name="o_out", depth=1)

    # rescale_O never reads its idx_buffer.
    idx = Buffer(idx_ty, name="idx", initial_value=np.array([0, 0, 0], np.int32))

    def core(of_o_in, of_scale, of_o_out, rescale, idx):
        rescale(of_o_in.acquire(1), of_o_out.acquire(1), of_scale.acquire(1), _B, idx)
        of_o_in.release(1)
        of_scale.release(1)
        of_o_out.release(1)

    worker = Worker(
        core,
        fn_args=[of_o_in.cons(), of_scale.cons(), of_o_out.prod(), rescale, idx],
    )

    def sequence(o_h, scale_h, out_h, in_fifo, scale_fifo, out_fifo):
        in_fifo.fill(o_h)
        scale_fifo.fill(scale_h)
        out_fifo.drain(out_h, wait=True)

    rt = Runtime(
        sequence,
        [acc_ty, scale_ty, tile_ty, of_o_in.prod(), of_scale.prod(), of_o_out.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _dyadic(rng, shape, scale, mag=16):
    """Small dyadic bf16, so host arithmetic on them is exact.

    A small ``mag`` keeps a float32 matmul of them exact too.
    """
    return (rng.integers(-mag, mag + 1, size=shape) * scale).astype(bfloat16)


def _softmax_case_data(q_block, kv_first, kv_second, s_q_eff, s_kv_eff):
    """Score tiles for one softmax case, and the masks that go with them."""
    rng = np.random.default_rng(20260923 + q_block * 17 + s_kv_eff)
    keeps = [
        _keep(q_block, kv, s_q_eff, s_kv_eff) for kv in (kv_first, kv_second)
    ]  # fmt: skip
    assert keeps[0].any() or keeps[1].any(), "a case with no live key gates nothing"
    # Hotter second-block scores move the correction factor well away from 1.
    scores = [_dyadic(rng, (_B, _B), 0.25), _dyadic(rng, (_B, _B), 0.75)]
    return scores, keeps


@pytest.mark.parametrize(
    "q_block,kv_first,kv_second,s_q_eff,s_kv_eff",
    [
        (1, 0, 1, 128, 128),  # unmasked block, then the diagonal
        (0, 0, 1, 64, 128),  # diagonal, then a block the kernel must skip
        (1, 0, 1, 100, 100),  # query and key tails both mid-vector
        (2, 1, 2, 192, 150),  # key tail alone
        (1, 0, 1, 100, 70),  # short key tail under padded query rows
    ],
)
@pytest.mark.parametrize("accurate_exp2", [False, True])
def test_partial_softmax_matches_masked_attention(
    q_block, kv_first, kv_second, s_q_eff, s_kv_eff, accurate_exp2
):
    scores, keeps = _softmax_case_data(
        q_block, kv_first, kv_second, s_q_eff, s_kv_eff
    )  # fmt: skip

    p_t = iron.zeros((2 * _B * _B,), dtype=bfloat16)
    scale_t = iron.zeros((4 * _B,), dtype=np.float32)
    softmax_blocks(
        iron.tensor(np.concatenate([s.reshape(-1) for s in scores]), dtype=bfloat16),
        p_t,
        scale_t,
        q_block=q_block,
        kv_first=kv_first,
        kv_second=kv_second,
        s_q_eff=s_q_eff,
        s_kv_eff=s_kv_eff,
        accurate_exp2=accurate_exp2,
    )
    got_p = p_t.numpy().astype(np.float32).reshape(2, _B, _B)
    got_scale = scale_t.numpy()

    m, sums = np.full(_B, _LOWEST, np.float32), np.zeros(_B, np.float32)
    ref_p = []
    for block_scores, keep in zip(scores, keeps):
        p, m, sums = _softmax_step(block_scores, keep, m, sums)
        ref_p.append(p)

    # The exp2 envelope with an absolute floor for weights near zero; padded
    # rows must be exact zeros. On aie2, the factory's bound (0.389% measured).
    # The accurate exp2 is held to one ulp, or to its flush to +0.
    aie2 = resolve_target_arch(iron.get_current_device()) == "aie2"
    tol = kernels.mha_softmax().contract_for().tolerance
    assert tol.rtol is not None and tol.atol is not None
    ulp = 2**-7
    if accurate_exp2:
        verdict = compare(
            p_t.numpy().reshape(2, _B, _B),
            np.stack(ref_p),
            Tolerance.bf16_ulps(1, atol=2.0**-125),
        )
        assert verdict, verdict.detail
    elif aie2:
        assert nearly_equal(got_p, np.stack(ref_p), rtol=tol.rtol, atol=tol.atol).all()
    else:
        np.testing.assert_allclose(
            got_p, np.stack(ref_p), rtol=_RTOL_EXP2, atol=_ATOL_ULP * ulp
        )

    # Carried state, on live rows only; padded rows' state is unspecified.
    live = keeps[0].any(axis=1) | keeps[1].any(axis=1)
    # The max is a score, kept as it came.
    np.testing.assert_array_equal(got_scale[:_B][live], m[live])
    # A sum of weights shares their envelope. On aie2 it is c * l_prev + sum(P)
    # with c an exp2 too, so it gets two weights' allowance (0.54% measured).
    got_l = got_scale[2 * _B : 3 * _B][live]
    if accurate_exp2:
        verdict = compare(got_l, sums[live], Tolerance.relative(_L_RTOL_ACCURATE))
        assert verdict, verdict.detail
    elif aie2:
        assert nearly_equal(got_l, sums[live], rtol=2 * tol.rtol, atol=tol.atol).all()
    else:
        l_ulp = 2**-7 * float(sums[live].max())
        np.testing.assert_allclose(
            got_l, sums[live], rtol=_RTOL_EXP2, atol=_ATOL_ULP * l_ulp
        )


@pytest.mark.parametrize(
    "l_lo,l_hi",
    [
        (1.0, 64.0),  # what a 64-key block produces
        (0.125, 1.0),  # sums below 1, where the reciprocal amplifies
        (0.03125, 512.0),  # three decades in one tile
    ],
)
def test_rescale_o_divides_rows_by_their_sums(l_lo, l_hi):
    rng = np.random.default_rng(20260923 + int(l_hi))
    o_blocked = _dyadic(rng, (_B * _B,), 0.5).astype(np.float32)
    # Sums vary within each 8-row block, so a wrong-row broadcast shows.
    sums = np.exp2(rng.uniform(np.log2(l_lo), np.log2(l_hi), _B)).astype(np.float32)
    scale = np.zeros(4 * _B, np.float32)
    scale[2 * _B : 3 * _B] = sums

    out_t = iron.zeros((_B * _B,), dtype=bfloat16)
    rescale_tile(
        iron.tensor(o_blocked, dtype=np.float32), iron.tensor(scale, dtype=np.float32), out_t
    )  # fmt: skip
    got = _unblock(out_t.numpy()).astype(np.float32)

    ref = _unblock(o_blocked) / sums[:, None]
    ulp = 2**-7 * float(np.abs(ref).max())
    np.testing.assert_allclose(got, ref, rtol=0, atol=_ATOL_ULP * ulp)


@jit
def mha_round(
    s_in: In,
    v_in: In,
    o_out: Out,
    *,
    q_block: CompileTime[int] = 0,
    n_kv: CompileTime[int] = 1,
    s_q_eff: CompileTime[int] = _B,
    s_kv_eff: CompileTime[int] = _B,
    b_q: CompileTime[int] = _B,
    b_kv: CompileTime[int] = _B,
    d: CompileTime[int] = _B,
    causal: CompileTime[bool] = True,
    window: CompileTime[int] = 0,
    packed: CompileTime[bool] = False,
    accurate_exp2: CompileTime[bool] = False,
    f32_scores: CompileTime[bool] = False,
    offset: CompileTime[int] = 0,
):
    """One round of mha.cc's steps from the mask on, over two cores.

    ``QK^T`` has its own device case and stays on the host, so the scores can
    be chosen. As IRON's MHA runs it, one core takes the softmax and sends
    its running state with each ``P``, and the other accumulates ``P V``. P
    is reblocked through a memtile on its way to ``matmul_PV``. The query
    block is ``b_q`` rows against a ``b_kv``-row key block, over a head
    dimension ``d``. With ``f32_scores`` the scores are float32. The query
    block's first row is ``offset`` positions past its start.
    """
    s_ty = np.ndarray[(b_q * b_kv,), F32 if f32_scores else BF]
    p_ty = np.ndarray[(b_q * b_kv,), BF]
    v_ty = np.ndarray[(b_kv * d,), BF]
    o_ty = np.ndarray[(b_q * d,), BF]
    acc_ty = np.ndarray[(b_q * d,), F32]
    scale_ty = np.ndarray[(4 * b_q,), F32]
    idx_ty = np.ndarray[(3,), I32]
    inv_scale = float(bfloat16(np.log2(np.e) / np.sqrt(d)))

    qkt = kernels.mha(
        dim_m=b_q,
        dim_k=d,
        dim_n=b_kv,
        accurate_exp2=accurate_exp2,
        causal=causal,
        window=window,
        packed=packed,
        f32_scores=f32_scores,
    )
    obj = qkt.object_file
    softmax = obj.bind(
        "partial_softmax",
        [s_ty, p_ty, scale_ty, idx_ty, bfloat16, *([np.int32] * 4)],
    )
    init = obj.bind("init_scale_buffer", [scale_ty, np.int32])
    pv = obj.bind(
        "matmul_PV",
        [p_ty, v_ty, acc_ty, scale_ty, np.int32, np.int32, idx_ty, np.int32],
    )
    rescale = obj.bind("rescale_O", [acc_ty, o_ty, scale_ty, np.int32, idx_ty])
    copy_scale = kernels.passthrough(4 * b_q, np.int32).object_file.bind(
        "passThroughLine", [scale_ty, scale_ty, np.int32]
    )

    of_s = ObjectFifo(s_ty, name="s", depth=2)
    # Past d = 64, V, O and the accumulator fill the PV core's L1 alone.
    of_v = ObjectFifo(v_ty, name="v", depth=2 if d == _B else 1)
    of_o = ObjectFifo(o_ty, name="o", depth=1)
    of_p = ObjectFifo(p_ty, name="p", depth=1)
    # The row-major P is read back in the mmul's 8x8 block order.
    of_pb = of_p.cons().forward(
        to_stream=TensorAccessPattern.full((b_q, b_kv)).tile((8, 8)), depth=1
    )
    of_scale = ObjectFifo(scale_ty, name="scale_out", depth=1)

    idx_bufs = {
        core: [
            Buffer(
                idx_ty,
                name=f"idx_{core}{k}",
                initial_value=np.array([k, q_block, offset], np.int32),
            )
            for k in range(n_kv)
        ]
        for core in ("softmax", "pv")
    }

    def softmax_core(of_s, of_p, of_scale, softmax, init, copy_scale, scale, *idx):
        init(scale, b_q)
        for k in range(n_kv):
            p = of_p.acquire(1)
            sent = of_scale.acquire(1)
            softmax(
                of_s.acquire(1),
                p,
                scale,
                idx[k],
                inv_scale,
                b_q,
                b_kv,
                s_q_eff,
                s_kv_eff,
            )
            copy_scale(scale, sent, 4 * b_q)
            of_s.release(1)
            of_p.release(1)
            of_scale.release(1)

    def pv_core(of_pb, of_v, of_scale, of_o, pv, rescale, acc, *idx):
        for k in range(n_kv):
            scale = of_scale.acquire(1)
            # The first key block starts the accumulator.
            pv(
                of_pb.acquire(1),
                of_v.acquire(1),
                acc,
                scale,
                b_q,
                int(k > 0),
                idx[k],
                s_kv_eff,
            )
            of_pb.release(1)
            of_v.release(1)
            if k == n_kv - 1:
                rescale(acc, of_o.acquire(1), scale, b_q, idx[0])
                of_o.release(1)
            of_scale.release(1)

    workers = [
        Worker(
            softmax_core,
            fn_args=[
                of_s.cons(),
                of_p.prod(),
                of_scale.prod(),
                softmax,
                init,
                copy_scale,
                Buffer(scale_ty, name="scale"),
                *idx_bufs["softmax"],
            ],
        ),
        Worker(
            pv_core,
            fn_args=[
                of_pb.cons(),
                of_v.cons(),
                of_scale.cons(),
                of_o.prod(),
                pv,
                rescale,
                Buffer(acc_ty, name="acc"),
                *idx_bufs["pv"],
            ],
        ),
    ]

    def sequence(s_h, v_h, o_h, sf, vf, of):
        sf.fill(s_h)
        vf.fill(v_h)
        of.drain(o_h, wait=True)

    host = [
        np.ndarray[(n_kv * b_q * b_kv,), F32 if f32_scores else BF],
        np.ndarray[(n_kv * b_kv * d,), BF],
        o_ty,
    ]
    rt = Runtime(sequence, [*host, of_s.prod(), of_v.prod(), of_o.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _round_case_data(
    q_block,
    n_kv,
    s_q_eff,
    s_kv_eff,
    b_q=_B,
    b_kv=_B,
    d=_B,
    causal=True,
    window=0,
    packed=False,
    offset=0,
):
    """Return one round's inputs, as the mathematical operands and as device buffers."""
    rng = np.random.default_rng(
        20260923 + 97 * q_block + n_kv + s_q_eff + s_kv_eff + 7 * offset
    )
    q = _dyadic(rng, (b_q, d), 0.25, mag=2)
    k = _dyadic(rng, (n_kv * b_kv, d), 0.125, mag=2)
    v = _dyadic(rng, (n_kv * b_kv, d), 0.25, mag=2)
    # Aim each query at its diagonal key, in the last block, so the running max
    # rises late and matmul_PV must rescale what earlier blocks accumulated.
    # Packed, row r is aimed r keys before the last.
    keep = np.concatenate(
        [
            _keep(
                q_block,
                kv,
                s_q_eff,
                s_kv_eff,
                b_q,
                b_kv,
                causal,
                window,
                packed,
                offset,
            )
            for kv in range(n_kv)
        ],
        axis=1,
    )
    for row in range(b_q):
        if keep[row].any():
            key = s_kv_eff - 1 - row if packed else q_block * b_q + offset + row
            k[key] = (q[row].astype(np.float32) * 0.5).astype(bfloat16)

    scores = (q.astype(np.float32) @ k.astype(np.float32).T).astype(bfloat16)
    # Keys from s_kv_eff on do not exist; the device's V rows for them hold
    # NaN, which must not reach O.
    stale = v.copy()
    stale[s_kv_eff:] = np.nan
    # Each block's row-major scores, and its blocked V.
    s_host = np.concatenate(
        [scores[:, b * b_kv : (b + 1) * b_kv].reshape(-1) for b in range(n_kv)]
    )
    v_host = np.concatenate(
        [
            TensorAccessPattern.full((b_kv, d))
            .tile((8, 8))
            .gather(stale[b * b_kv : (b + 1) * b_kv])
            for b in range(n_kv)
        ]
    )
    return scores, v, keep, s_host, v_host


def _skip_f32_off_aie2p(f32_scores):
    if f32_scores and resolve_target_arch(iron.get_current_device()) == "aie2":
        pytest.skip("mha.cc's float32 scores are written for AIE2P")


def _check_round(
    q_block, n_kv, s_q_eff, s_kv_eff, accurate_exp2, f32_scores=False, **shape
):
    """Run ``mha_round`` and hold it to masked attention over its band."""
    _skip_f32_off_aie2p(f32_scores)
    b_q, d = shape.get("b_q", _B), shape.get("d", _B)
    scores, v, keep, s_host, v_host = _round_case_data(
        q_block, n_kv, s_q_eff, s_kv_eff, **shape
    )

    o_t = iron.zeros((b_q * d,), dtype=bfloat16)
    mha_round(
        iron.tensor(s_host.astype(np.float32 if f32_scores else bfloat16)),
        iron.tensor(v_host, dtype=bfloat16),
        o_t,
        q_block=q_block,
        n_kv=n_kv,
        s_q_eff=s_q_eff,
        s_kv_eff=s_kv_eff,
        accurate_exp2=accurate_exp2,
        f32_scores=f32_scores,
        **shape,
    )
    got = _unblock(o_t.numpy(), b_q, d).astype(np.float32)

    inv_scale = float(bfloat16(np.log2(np.e) / np.sqrt(d)))
    ref = _attention(scores, v, keep, inv_scale=inv_scale).astype(np.float32)
    live = keep.any(axis=1)
    # Outputs are convex combinations of V rows, so measure in steps of max|v|.
    ulp = 2**-7 * float(np.abs(v.astype(np.float32)).max())
    np.testing.assert_allclose(got[live], ref[live], rtol=0, atol=_ATOL_ULP * ulp)
    # A row past s_q_eff attends over nothing; it must come out 0, not 0 * (1 / 0).
    np.testing.assert_array_equal(got[~live], 0)

    # The reference uses the kernel's bf16 scale; check it against the exact one.
    exact = _attention(scores, v, keep, inv_scale=np.log2(np.e) / np.sqrt(d)).astype(
        np.float32
    )
    np.testing.assert_allclose(ref[live], exact[live], rtol=0, atol=1 * ulp)


@pytest.mark.parametrize(
    "q_block,n_kv,s_q_eff,s_kv_eff,b_q",
    [
        (0, 1, _B, _B, _B),  # the pure diagonal
        (1, 2, 2 * _B, 2 * _B, _B),  # a real correction carried between blocks
        (1, 2, 100, 100, _B),  # query and key tails both mid-vector
        (0, 1, 100, 40, _B),  # key tail cutting before the diagonal for most rows
        (1, 3, 2 * _B, 3 * _B, _B),  # a block past the diagonal, which must drop
        (1, 1, _B, _B, 32),  # a narrower query block, its diagonal mid-block
        (3, 2, 4 * 32, 2 * _B, 32),  # carried across key blocks, then mid-block
        (1, 2, 2 * 32, 2 * _B, 32),  # a key block past every row, which must drop
        (3, 1, 4 * 16, _B, 16),  # the 16-row state walk
        (1, 1, _B, _B, _B),  # a query block wholly past s_q_eff
    ],
)
@pytest.mark.parametrize("accurate_exp2", [False, True])
@pytest.mark.parametrize("f32_scores", [False, True], ids=["bf16_scores", "f32_scores"])
def test_mha_round_matches_masked_attention(
    q_block, n_kv, s_q_eff, s_kv_eff, b_q, accurate_exp2, f32_scores
):
    _check_round(q_block, n_kv, s_q_eff, s_kv_eff, accurate_exp2, f32_scores, b_q=b_q)


# Self-attention, so every live row keeps a key of its own.
@pytest.mark.parametrize(
    "q_block,n_kv,s_eff,shape",
    [
        # Two query blocks to a key block, the diagonal mid-block, over d=256.
        (5, 3, 90, dict(b_q=16, b_kv=32, d=256)),
        # A window on both sides, cutting the first and last key blocks.
        (2, 3, 90, dict(b_q=16, b_kv=32, d=256, causal=False, window=32)),
        # Bidirectional: keys after the queries, the last block a key tail.
        (1, 2, 100, dict(b_q=32, d=128, causal=False)),
        # The row-group path's window: the first key block drops, the last
        # is kept past the diagonal.
        (2, 4, 4 * _B, dict(causal=False, window=_B)),
        # A causal sliding window, as a sliding layer of a decoder runs.
        (2, 3, 3 * _B, dict(window=_B)),
        # 16-key blocks, a row one 16-lane vector.
        (3, 6, 90, dict(b_q=16, b_kv=16, causal=False, window=16)),
        # Windows of no whole number of query blocks, as Gemma's 511: the last
        # row keeps no key of a block the others keep, and none before it.
        (2, 3, 3 * _B, dict(window=_B - 1)),
        (5, 3, 90, dict(b_q=16, b_kv=32, d=256, window=21)),
        # One decode step, a KV group's heads as the rows at the last key,
        # the window dropping the first key block and cutting the second.
        (0, 4, 200, dict(b_q=16, packed=True, window=100)),
        (0, 4, 100, dict(b_q=16, b_kv=32, packed=True, window=50)),
    ],
)
@pytest.mark.parametrize("accurate_exp2", [False, True])
@pytest.mark.parametrize("f32_scores", [False, True], ids=["bf16_scores", "f32_scores"])
def test_mha_round_keeps_its_band(
    q_block, n_kv, s_eff, shape, accurate_exp2, f32_scores
):
    _check_round(q_block, n_kv, s_eff, s_eff, accurate_exp2, f32_scores, **shape)


# Queries starting mid-block, as a prompt call from any position runs: the
# last valid row is the last key's, each row offset positions past its block's.
@pytest.mark.parametrize(
    "q_block,n_kv,s_q_eff,offset,shape",
    [
        # The diagonal mid-block, the query rows padded.
        (0, 1, 50, 14, {}),
        # Carried across key blocks to a last one cut partway.
        (1, 3, 2 * _B, 37, {}),
        # A narrower query block, on the row-group path.
        (3, 2, 4 * 32 - 5, 5, dict(b_q=32)),
        # A causal sliding window, its edge off the key blocks'.
        (2, 3, 3 * _B - 20, 20, dict(window=_B)),
        # 16-key blocks, a row one 16-lane vector.
        (3, 6, 60, 30, dict(b_q=16, b_kv=16, window=16)),
    ],
)
@pytest.mark.parametrize("accurate_exp2", [False, True])
def test_mha_round_starts_its_queries_mid_block(
    q_block, n_kv, s_q_eff, offset, shape, accurate_exp2
):
    _check_round(
        q_block, n_kv, s_q_eff, s_q_eff + offset, accurate_exp2, offset=offset, **shape
    )


# A bf16 score whose product with _INV_SCALE, 45695, is 127 above its own bf16
# rounding: past 2^15 bf16 steps by 256. A running max kept as that rounding
# made every key tied at the top weigh 2^127, and two of them overflow.
_PAST_2_15 = 252928.0


@pytest.mark.parametrize(
    "b_q,b_kv", [(_B, _B), (32, 32)], ids=["row_groups", "per_row"]
)
@pytest.mark.parametrize("accurate_exp2", [False, True])
@pytest.mark.parametrize("f32_scores", [False, True], ids=["bf16_scores", "f32_scores"])
def test_mha_round_keeps_scores_past_2_15_finite(b_q, b_kv, accurate_exp2, f32_scores):
    """Each row's top keys tie at a scaled score past 2^15, the rest far below."""
    _skip_f32_off_aie2p(f32_scores)
    scaled = np.float32(_PAST_2_15) * np.float32(_INV_SCALE)
    assert scaled - np.float32(bfloat16(scaled)) == 127
    q_block, s_eff = 1, 2 * b_q
    n_kv = s_eff // b_kv
    rng = np.random.default_rng(20261013)
    top = rng.random((b_q, n_kv * b_kv)) < 0.5
    top[:, 0] = True
    scores = np.where(top, _PAST_2_15, -_PAST_2_15).astype(bfloat16)
    v = _dyadic(rng, (n_kv * b_kv, _B), 0.25, mag=2)
    keep = np.concatenate(
        [_keep(q_block, kv, s_eff, s_eff, b_q, b_kv) for kv in range(n_kv)], axis=1
    )
    s_host = np.concatenate(
        [scores[:, b * b_kv : (b + 1) * b_kv].reshape(-1) for b in range(n_kv)]
    )
    v_host = np.concatenate(
        [
            TensorAccessPattern.full((b_kv, _B))
            .tile((8, 8))
            .gather(v[b * b_kv : (b + 1) * b_kv])
            for b in range(n_kv)
        ]
    )

    o_t = iron.zeros((b_q * _B,), dtype=bfloat16)
    mha_round(
        iron.tensor(s_host.astype(np.float32 if f32_scores else bfloat16)),
        iron.tensor(v_host, dtype=bfloat16),
        o_t,
        q_block=q_block,
        n_kv=n_kv,
        s_q_eff=s_eff,
        s_kv_eff=s_eff,
        b_q=b_q,
        b_kv=b_kv,
        accurate_exp2=accurate_exp2,
        f32_scores=f32_scores,
    )
    got = _unblock(o_t.numpy(), b_q, _B).astype(np.float32)

    # Each row is the mean of V over its kept top keys.
    ref = _attention(scores, v, keep, inv_scale=_INV_SCALE).astype(np.float32)
    ulp = 2**-7 * float(np.abs(v.astype(np.float32)).max())
    np.testing.assert_allclose(got, ref, rtol=0, atol=_ATOL_ULP * ulp)


# 2^17: bf16 steps by 1024 here, float32 by 2^-6.
_NEAR_TIES = 131072.0


@pytest.mark.parametrize("b_q,b_kv", [(_B, _B), (32, 32)], ids=["64_keys", "32_keys"])
@pytest.mark.parametrize("accurate_exp2", [False, True])
def test_mha_round_f32_scores_resolve_near_ties(b_q, b_kv, accurate_exp2):
    """Scores at 2^17 plus multiples of 4, which bf16 would round to one value."""
    _skip_f32_off_aie2p(True)
    q_block, s_eff = 1, 2 * b_q
    n_kv = s_eff // b_kv
    rng = np.random.default_rng(20261014)
    # Scaled, a step of 4 is 0.72: within bf16's 1024 they would all tie.
    scores = (_NEAR_TIES + 4 * rng.integers(0, 8, (b_q, n_kv * b_kv))).astype(
        np.float32
    )
    assert np.unique(scores.astype(bfloat16)).size == 1
    v = _dyadic(rng, (n_kv * b_kv, _B), 0.25, mag=2)
    keep = np.concatenate(
        [_keep(q_block, kv, s_eff, s_eff, b_q, b_kv) for kv in range(n_kv)], axis=1
    )
    s_host = np.concatenate(
        [scores[:, b * b_kv : (b + 1) * b_kv].reshape(-1) for b in range(n_kv)]
    )
    v_host = np.concatenate(
        [
            TensorAccessPattern.full((b_kv, _B))
            .tile((8, 8))
            .gather(v[b * b_kv : (b + 1) * b_kv])
            for b in range(n_kv)
        ]
    )

    o_t = iron.zeros((b_q * _B,), dtype=bfloat16)
    mha_round(
        iron.tensor(s_host, dtype=np.float32),
        iron.tensor(v_host, dtype=bfloat16),
        o_t,
        q_block=q_block,
        n_kv=n_kv,
        s_q_eff=s_eff,
        s_kv_eff=s_eff,
        b_q=b_q,
        b_kv=b_kv,
        accurate_exp2=accurate_exp2,
        f32_scores=True,
    )
    got = _unblock(o_t.numpy(), b_q, _B).astype(np.float32)

    ref = _attention(scores, v, keep, inv_scale=_INV_SCALE).astype(np.float32)
    ulp = 2**-7 * float(np.abs(v.astype(np.float32)).max())
    np.testing.assert_allclose(got, ref, rtol=0, atol=_ATOL_ULP * ulp)


@jit
def mha_long_round(
    s_in: In,
    v_in: In,
    o_out: Out,
    *,
    n_kv: CompileTime[int] = 2,
    accurate_exp2: CompileTime[bool] = False,
):
    """One query block over ``n_kv`` key blocks, the last its diagonal.

    As ``mha_round``, but the key blocks go by in loops over each core's
    ``idx_buffer``, so the context can be long.
    """
    assert n_kv >= 2, "the PV core's first and last blocks are separate calls"
    tile_ty = np.ndarray[(_B * _B,), BF]
    acc_ty = np.ndarray[(_B * _B,), F32]
    scale_ty = np.ndarray[(4 * _B,), F32]
    idx_ty = np.ndarray[(3,), I32]
    s_eff = n_kv * _B

    qkt = kernels.mha(dim_m=_B, dim_k=_B, dim_n=_B, accurate_exp2=accurate_exp2)
    obj = qkt.object_file
    softmax = obj.bind(
        "partial_softmax",
        [tile_ty, tile_ty, scale_ty, idx_ty, bfloat16, *([np.int32] * 4)],
    )
    init = obj.bind("init_scale_buffer", [scale_ty, np.int32])
    pv = obj.bind(
        "matmul_PV",
        [tile_ty, tile_ty, acc_ty, scale_ty, np.int32, np.int32, idx_ty, np.int32],
    )
    rescale = obj.bind("rescale_O", [acc_ty, tile_ty, scale_ty, np.int32, idx_ty])
    copy_scale = kernels.passthrough(4 * _B, np.int32).object_file.bind(
        "passThroughLine", [scale_ty, scale_ty, np.int32]
    )

    of_s = ObjectFifo(tile_ty, name="s", depth=2)
    of_v = ObjectFifo(tile_ty, name="v", depth=2)
    of_o = ObjectFifo(tile_ty, name="o", depth=1)
    of_p = ObjectFifo(tile_ty, name="p", depth=1)
    of_pb = of_p.cons().forward(
        to_stream=TensorAccessPattern.full((_B, _B)).tile((8, 8)), depth=1
    )
    of_scale = ObjectFifo(scale_ty, name="scale_out", depth=1)

    def softmax_core(of_s, of_p, of_scale, softmax, init, copy_scale, scale, idx):
        init(scale, _B)
        for _ in range_(n_kv):
            p = of_p.acquire(1)
            sent = of_scale.acquire(1)
            softmax(of_s.acquire(1), p, scale, idx, _INV_SCALE, _B, _B, s_eff, s_eff)
            copy_scale(scale, sent, 4 * _B)
            of_s.release(1)
            of_p.release(1)
            of_scale.release(1)
            idx[0] += 1

    def pv_core(of_pb, of_v, of_scale, of_o, pv, rescale, acc, idx):
        def block(first_iter, last=False):
            scale = of_scale.acquire(1)
            pv(
                of_pb.acquire(1),
                of_v.acquire(1),
                acc,
                scale,
                _B,
                first_iter,
                idx,
                s_eff,
            )
            of_pb.release(1)
            of_v.release(1)
            idx[0] += 1
            if last:
                rescale(acc, of_o.acquire(1), scale, _B, idx)
                of_o.release(1)
            of_scale.release(1)

        block(0)
        for _ in range_(n_kv - 2):
            block(1)
        block(1, last=True)

    workers = [
        Worker(
            softmax_core,
            fn_args=[
                of_s.cons(),
                of_p.prod(),
                of_scale.prod(),
                softmax,
                init,
                copy_scale,
                Buffer(scale_ty, name="scale"),
                Buffer(
                    idx_ty,
                    name="idx_softmax",
                    initial_value=np.array([0, n_kv - 1, 0], np.int32),
                ),
            ],
        ),
        Worker(
            pv_core,
            fn_args=[
                of_pb.cons(),
                of_v.cons(),
                of_scale.cons(),
                of_o.prod(),
                pv,
                rescale,
                Buffer(acc_ty, name="acc"),
                Buffer(
                    idx_ty,
                    name="idx_pv",
                    initial_value=np.array([0, n_kv - 1, 0], np.int32),
                ),
            ],
        ),
    ]

    def sequence(s_h, v_h, o_h, sf, vf, of):
        sf.fill(s_h)
        vf.fill(v_h)
        of.drain(o_h, wait=True)

    blocks_ty = np.ndarray[(n_kv * _B * _B,), BF]
    rt = Runtime(
        sequence,
        [blocks_ty, blocks_ty, tile_ty, of_s.prod(), of_v.prod(), of_o.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@pytest.mark.parametrize("accurate_exp2", [False, True])
def test_a_long_context_keeps_the_running_state_accurate(accurate_exp2):
    """The running sum and output gain a key block's worth at a time.

    Rounded to bf16 between blocks, a block's share is lost the more the
    larger the totals have grown. Every row's peak score is 0 and the rest
    put 2^x near -1 and -2, where the exp2 interpolant is within 0.07%, so
    the gate is the output's own rounding. V centred on 1 keeps the outputs
    far from 0.
    """
    n_kv = 64
    rng = np.random.default_rng(20261007)
    # bf16 scores that the kernel's scale takes to within 0.0014 of 0, -1, -2.
    levels = np.array([0, -5.53125, -11.0625], np.float32)
    scores = levels[rng.integers(0, 3, size=(_B, n_kv * _B))]
    scores[:, 0] = 0
    scores = scores.astype(bfloat16)
    v = (1 + _dyadic(rng, (n_kv * _B, _B), 0.25, mag=2).astype(np.float32)).astype(
        bfloat16
    )
    keep = np.concatenate(
        [_keep(n_kv - 1, kv, n_kv * _B, n_kv * _B) for kv in range(n_kv)], axis=1
    )
    s_host = np.concatenate(
        [scores[:, b * _B : (b + 1) * _B].reshape(-1) for b in range(n_kv)]
    )
    v_host = np.concatenate(
        [
            TensorAccessPattern.full((_B, _B))
            .tile((8, 8))
            .gather(v[b * _B : (b + 1) * _B])
            for b in range(n_kv)
        ]
    )

    o_t = iron.zeros((_B * _B,), dtype=bfloat16)
    mha_long_round(
        iron.tensor(s_host, dtype=bfloat16),
        iron.tensor(v_host, dtype=bfloat16),
        o_t,
        n_kv=n_kv,
        accurate_exp2=accurate_exp2,
    )
    got = _unblock(o_t.numpy()).astype(np.float32)

    ref = _attention(scores, v, keep, inv_scale=_INV_SCALE).astype(np.float32)
    ulp = 2**-7 * float(np.abs(ref).max())
    np.testing.assert_allclose(got, ref, rtol=0, atol=2 * ulp)
