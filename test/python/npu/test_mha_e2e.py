# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Numeric gates for mha.cc's softmax entry points, alone and as one round.

``partial_softmax`` and ``rescale_O`` read and write a carried
``scale_buffer``, so neither fits a kernel contract, whose outputs are zeroed on
core. Each is driven alone, which sees its raw weights and carried state, and
then all four steps run as one decode round, which checks that they agree.

References use true ``np.exp2`` and state the mask over absolute positions;
nothing models the device's arithmetic. On AIE2P, ``aie::exp2`` is a linear
interpolant that overshoots by up to 6.15%, so raw weights get that envelope
(``_RTOL_EXP2``) and the round, where it divides out, gets bf16 rounding only.
On AIE2 mha.cc evaluates a cubic instead, and the weights are held to
``kernels.mha_softmax``'s tolerance. Each parametrized configuration is the
only one that catches some mask or carry bug.

Not covered: a key block that is all padding inside the causal region.
``partial_softmax`` returns early there and leaves the previous correction
factor in ``scale_buffer``; that is existing kernel behaviour, not frozen here.
"""

from collections.abc import Sequence

import ml_dtypes
import numpy as np
import pytest
from ml_dtypes import bfloat16

import aie.iron as iron
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
from aie.utils.compile.utils import resolve_target_arch
from aie.utils.verify import nearly_equal

BF = np.dtype[bfloat16]
I32 = np.dtype[np.int32]

# mha.cc's P zeroing is compiled for 64, so every block and tile is 64.
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

# Reads row-major P back in the mmul's 8x8 block order.
_REBLOCK: list[Sequence[int]] = [(8, 512), (8, 8), (8, 64), (8, 1)]


def _block(mat: np.ndarray) -> np.ndarray:
    """(64, 64) -> mm.cc's 8x8-blocked operand order; its own inverse."""
    return mat.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(-1).copy()


def _unblock(flat: np.ndarray) -> np.ndarray:
    """The 8x8-blocked O tile back to (64, 64)."""
    return flat.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(_B, _B)


def _keep(q_block: int, kv_block: int, s_q_eff: int, s_kv_eff: int) -> np.ndarray:
    """Which (query, key) pairs of this block pair a causal mask admits."""
    rows = q_block * _B + np.arange(_B)
    cols = kv_block * _B + np.arange(_B)
    return (
        (cols[None, :] <= rows[:, None])
        & (rows[:, None] < s_q_eff)
        & (cols[None, :] < s_kv_eff)
    )


def _keep_round(q_block: int, n_kv: int, s_q_eff: int, s_kv_eff: int) -> np.ndarray:
    """``_keep`` over every key block a round streams, side by side."""
    return np.concatenate(
        [_keep(q_block, kv, s_q_eff, s_kv_eff) for kv in range(n_kv)], axis=1
    )


def _softmax_step(scores, keep, m_prev, l_prev):
    """One key block of the online softmax: (weights, running max, row sum)."""
    scaled = scores.astype(np.float32) * np.float32(_INV_SCALE)
    m_new = np.where(keep, scaled, -np.float32(np.inf)).max(axis=1)
    m_new = np.maximum(m_new, m_prev)
    p = np.exp2(np.where(keep, scaled, m_new[:, None]) - m_new[:, None])
    # The row sum accumulates the stored bf16 weights.
    p = np.where(keep, p.astype(bfloat16).astype(np.float32), np.float32(0))
    l_new = np.exp2(m_prev - m_new) * l_prev + p.sum(axis=1)
    return p, m_new, l_new


def _attention(scores, v, keep, *, inv_scale):
    """Masked attention, rounded to bf16 wherever the kernel stores."""
    scaled = scores.astype(np.float32) * np.float32(inv_scale)
    live = keep.any(axis=1)
    peak = np.where(live, np.where(keep, scaled, -np.inf).max(axis=1), 0)
    p = np.exp2(scaled - peak.astype(bfloat16).astype(np.float32)[:, None])
    p = np.where(keep, p.astype(bfloat16).astype(np.float32), np.float32(0))
    y = p @ v.astype(np.float32)
    inv_l = np.float32(1.0) / np.where(live, p.sum(axis=1), np.float32(1))
    return (
        y.astype(bfloat16).astype(np.float32)
        * inv_l.astype(bfloat16).astype(np.float32)[:, None]
    ).astype(bfloat16)


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
):
    """``partial_softmax`` over two key blocks sharing one ``scale_buffer``.

    The second block reads the state the first carried.
    """
    tile_ty = np.ndarray[(_B * _B,), BF]
    scale_ty = np.ndarray[(4 * _B,), BF]
    idx_ty = np.ndarray[(2,), I32]

    mha = kernels.mha(dim_m=_B, dim_k=_B, dim_n=_B)
    obj = mha.object_file
    softmax = obj.bind(
        "partial_softmax",
        [tile_ty, tile_ty, scale_ty, idx_ty, bfloat16, *([np.int32] * 4)],
    )
    init = obj.bind("init_scale_buffer", [scale_ty, np.int32])

    of_a = ObjectFifo(tile_ty, name="a", depth=2)
    of_p = ObjectFifo(tile_ty, name="p", depth=2)
    of_scale = ObjectFifo(scale_ty, name="scale", depth=1)

    # idx_buffer is (key block, query block).
    idx = [
        Buffer(idx_ty, name=f"idx{n}", initial_value=np.array([kv, q_block], np.int32))
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
        stack_size=kernels.mha_softmax().contract.stack_bytes,
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
    """``rescale_O`` on a copy of O, since it rescales in place."""
    tile_ty = np.ndarray[(_B * _B,), BF]
    scale_ty = np.ndarray[(4 * _B,), BF]
    idx_ty = np.ndarray[(2,), I32]

    mha = kernels.mha(dim_m=_B, dim_k=_B, dim_n=_B)
    rescale = mha.object_file.bind(
        "rescale_O", [tile_ty, scale_ty, np.int32, idx_ty]
    )  # fmt: skip
    copy = kernels.passthrough(_B * _B, np.int16).object_file.bind(
        "passThroughLine", [tile_ty, tile_ty, np.int32]
    )

    of_o_in = ObjectFifo(tile_ty, name="o_in", depth=2)
    of_scale = ObjectFifo(scale_ty, name="scale", depth=2)
    of_o_out = ObjectFifo(tile_ty, name="o_out", depth=2)

    # rescale_O never reads its idx_buffer.
    idx = Buffer(idx_ty, name="idx", initial_value=np.array([0, 0], np.int32))

    def core(of_o_in, of_scale, of_o_out, rescale, copy, idx):
        o = of_o_out.acquire(1)
        copy(of_o_in.acquire(1), o, _B * _B)
        of_o_in.release(1)
        rescale(o, of_scale.acquire(1), _B, idx)
        of_scale.release(1)
        of_o_out.release(1)

    worker = Worker(
        core,
        fn_args=[
            of_o_in.cons(),
            of_scale.cons(),
            of_o_out.prod(),
            rescale,
            copy,
            idx,
        ],
        stack_size=mha.contract.stack_bytes,
    )

    def sequence(o_h, scale_h, out_h, in_fifo, scale_fifo, out_fifo):
        in_fifo.fill(o_h)
        scale_fifo.fill(scale_h)
        out_fifo.drain(out_h, wait=True)

    rt = Runtime(
        sequence,
        [tile_ty, scale_ty, tile_ty, of_o_in.prod(), of_scale.prod(), of_o_out.cons()],
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
def test_partial_softmax_matches_masked_attention(
    q_block, kv_first, kv_second, s_q_eff, s_kv_eff
):
    scores, keeps = _softmax_case_data(
        q_block, kv_first, kv_second, s_q_eff, s_kv_eff
    )  # fmt: skip

    p_t = iron.zeros((2 * _B * _B,), dtype=bfloat16)
    scale_t = iron.zeros((4 * _B,), dtype=bfloat16)
    softmax_blocks(
        iron.tensor(np.concatenate([s.reshape(-1) for s in scores]), dtype=bfloat16),
        p_t,
        scale_t,
        q_block=q_block,
        kv_first=kv_first,
        kv_second=kv_second,
        s_q_eff=s_q_eff,
        s_kv_eff=s_kv_eff,
    )
    got_p = p_t.numpy().astype(np.float32).reshape(2, _B, _B)
    got_scale = scale_t.numpy().astype(np.float32)

    m, l = np.full(_B, _LOWEST, np.float32), np.zeros(_B, np.float32)
    ref_p = []
    for block_scores, keep in zip(scores, keeps):
        p, m, l = _softmax_step(block_scores, keep, m, l)
        ref_p.append(p)

    # The exp2 envelope with an absolute floor for weights near zero; padded
    # rows must be exact zeros. On aie2, the factory's bound (0.389% measured).
    aie2 = resolve_target_arch(iron.get_current_device()) == "aie2"
    tol = kernels.mha_softmax().contract.tolerance
    assert tol.rtol is not None and tol.atol is not None
    ulp = 2**-7
    if aie2:
        assert nearly_equal(got_p, np.stack(ref_p), rtol=tol.rtol, atol=tol.atol).all()
    else:
        np.testing.assert_allclose(
            got_p, np.stack(ref_p), rtol=_RTOL_EXP2, atol=_ATOL_ULP * ulp
        )

    # Carried state, on live rows only; padded rows' state is unspecified.
    live = keeps[0].any(axis=1) | keeps[1].any(axis=1)
    # The max is never exponentiated, so it gets no relative allowance.
    m_ulp = 2**-7 * float(np.abs(m[live]).max())
    np.testing.assert_allclose(
        got_scale[:_B][live], m[live], rtol=0, atol=_ATOL_ULP * m_ulp
    )
    # A sum of weights shares their envelope. On aie2 it is c * l_prev + sum(P)
    # with c an exp2 too, so it gets two weights' allowance (0.54% measured).
    got_l = got_scale[2 * _B : 3 * _B][live]
    if aie2:
        assert nearly_equal(got_l, l[live], rtol=2 * tol.rtol, atol=tol.atol).all()
    else:
        l_ulp = 2**-7 * float(l[live].max())
        np.testing.assert_allclose(
            got_l, l[live], rtol=_RTOL_EXP2, atol=_ATOL_ULP * l_ulp
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
    o_blocked = _dyadic(rng, (_B * _B,), 0.5)
    # Sums vary within each 8-row block, so a wrong-row broadcast shows.
    l = np.exp2(rng.uniform(np.log2(l_lo), np.log2(l_hi), _B)).astype(bfloat16)
    scale = np.zeros(4 * _B, bfloat16)
    scale[2 * _B : 3 * _B] = l

    out_t = iron.zeros((_B * _B,), dtype=bfloat16)
    rescale_tile(
        iron.tensor(o_blocked, dtype=bfloat16), iron.tensor(scale, dtype=bfloat16), out_t
    )  # fmt: skip
    got = _unblock(out_t.numpy()).astype(np.float32)

    ref = _unblock(o_blocked).astype(np.float32) / l.astype(np.float32)[:, None]
    ulp = 2**-7 * float(np.abs(ref).max())
    np.testing.assert_allclose(got, ref, rtol=0, atol=_ATOL_ULP * ulp)


@jit
def mha_round(
    sv_in: In,
    o_out: Out,
    *,
    q_block: CompileTime[int] = 0,
    n_kv: CompileTime[int] = 1,
    s_q_eff: CompileTime[int] = _B,
    s_kv_eff: CompileTime[int] = _B,
):
    """One decode round on one core: mha.cc's steps from the mask on.

    ``QK^T`` has its own device case and stays on the host, so the scores can
    be chosen. P is reblocked through a memtile on its way to ``matmul_PV``.
    """
    tile_ty = np.ndarray[(_B * _B,), BF]
    scale_ty = np.ndarray[(4 * _B,), BF]
    idx_ty = np.ndarray[(2,), I32]

    qkt = kernels.mha(dim_m=_B, dim_k=_B, dim_n=_B)
    obj = qkt.object_file
    softmax = obj.bind(
        "partial_softmax",
        [tile_ty, tile_ty, scale_ty, idx_ty, bfloat16, *([np.int32] * 4)],
    )
    init = obj.bind("init_scale_buffer", [scale_ty, np.int32])
    pv = obj.bind(
        "matmul_PV", [tile_ty, tile_ty, tile_ty, scale_ty, np.int32, np.int32, idx_ty]
    )
    rescale = obj.bind("rescale_O", [tile_ty, scale_ty, np.int32, idx_ty])
    zero = kernels.zero(_B * _B, bfloat16)

    # S and V share one input channel; the reblocked P takes the other.
    of_sv = ObjectFifo(tile_ty, name="sv", depth=2)
    # O is held as the accumulator for the whole round.
    of_o = ObjectFifo(tile_ty, name="o", depth=1)
    of_p = ObjectFifo(tile_ty, name="p", depth=1)
    of_pb = of_p.cons().forward(dims_to_stream=_REBLOCK, depth=1)

    scale_buf = Buffer(scale_ty, name="scale")
    idx_bufs = [
        Buffer(idx_ty, name=f"idx{k}", initial_value=np.array([k, q_block], np.int32))
        for k in range(n_kv)
    ]

    def core(of_sv, of_o, of_p, of_pb, softmax, init, pv, rescale, zero, scale, *idx):
        o = of_o.acquire(1)
        zero(o)
        init(scale, _B)
        for k in range(n_kv):
            softmax(
                of_sv.acquire(1),
                of_p.acquire(1),
                scale,
                idx[k],
                _INV_SCALE,
                _B,
                _B,
                s_q_eff,
                s_kv_eff,
            )
            of_p.release(1)
            of_sv.release(1)
            pv(of_pb.acquire(1), of_sv.acquire(1), o, scale, _B, int(k > 0), idx[k])
            of_pb.release(1)
            of_sv.release(1)
        rescale(o, scale, _B, idx[0])
        of_o.release(1)

    worker = Worker(
        core,
        fn_args=[
            of_sv.cons(),
            of_o.prod(),
            of_p.prod(),
            of_pb.cons(),
            softmax,
            init,
            pv,
            rescale,
            zero,
            scale_buf,
            *idx_bufs,
        ],
        stack_size=kernels.mha_softmax().contract.stack_bytes,
    )

    host = [np.ndarray[(2 * n_kv * _B * _B,), BF], np.ndarray[(_B * _B,), BF]]

    def sequence(sv_h, o_h, svf, of):
        svf.fill(sv_h)
        of.drain(o_h, wait=True)

    rt = Runtime(sequence, [*host, of_sv.prod(), of_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _round_case_data(q_block, n_kv, s_q_eff, s_kv_eff):
    """Inputs for one round, as the mathematical operands and as device buffers."""
    rng = np.random.default_rng(20260923 + 97 * q_block + n_kv + s_q_eff + s_kv_eff)
    q = _dyadic(rng, (_B, _B), 0.25, mag=2)
    k = _dyadic(rng, (n_kv * _B, _B), 0.125, mag=2)
    v = _dyadic(rng, (n_kv * _B, _B), 0.25, mag=2)
    # Aim each query at its diagonal key, in the last block, so the running max
    # rises late and matmul_PV must rescale what earlier blocks accumulated.
    keep = _keep_round(q_block, n_kv, s_q_eff, s_kv_eff)
    for row in range(_B):
        if keep[row].any():
            k[q_block * _B + row] = (q[row].astype(np.float32) * 0.5).astype(bfloat16)

    scores = (q.astype(np.float32) @ k.astype(np.float32).T).astype(bfloat16)
    # Each block's row-major scores, then its blocked V.
    sv_host = np.concatenate(
        [
            arr
            for b in range(n_kv)
            for arr in (
                np.asarray(scores[:, b * _B : (b + 1) * _B]).reshape(-1).copy(),
                _block(v[b * _B : (b + 1) * _B]),
            )
        ]
    )
    return scores, v, keep, sv_host


@pytest.mark.parametrize(
    "q_block,n_kv,s_q_eff,s_kv_eff",
    [
        (0, 1, _B, _B),  # the pure diagonal
        (1, 2, 2 * _B, 2 * _B),  # a real correction carried between blocks
        (1, 2, 100, 100),  # query and key tails both mid-vector
        (0, 1, 100, 40),  # key tail cutting before the diagonal for most rows
        (1, 3, 2 * _B, 3 * _B),  # a block past the diagonal, which must drop
    ],
)
def test_mha_round_matches_masked_attention(q_block, n_kv, s_q_eff, s_kv_eff):
    scores, v, keep, sv_host = _round_case_data(q_block, n_kv, s_q_eff, s_kv_eff)

    o_t = iron.zeros((_B * _B,), dtype=bfloat16)
    mha_round(
        iron.tensor(sv_host, dtype=bfloat16),
        o_t,
        q_block=q_block,
        n_kv=n_kv,
        s_q_eff=s_q_eff,
        s_kv_eff=s_kv_eff,
    )
    got = _unblock(o_t.numpy()).astype(np.float32)

    ref = _attention(scores, v, keep, inv_scale=_INV_SCALE).astype(np.float32)
    live = keep.any(axis=1)
    # Outputs are convex combinations of V rows, so measure in steps of max|v|.
    ulp = 2**-7 * float(np.abs(v.astype(np.float32)).max())
    np.testing.assert_allclose(got[live], ref[live], rtol=0, atol=_ATOL_ULP * ulp)

    # The reference uses the kernel's bf16 scale; check it against the exact one.
    exact = _attention(scores, v, keep, inv_scale=np.log2(np.e) / np.sqrt(_B)).astype(
        np.float32
    )
    np.testing.assert_allclose(ref[live], exact[live], rtol=0, atol=1 * ulp)
