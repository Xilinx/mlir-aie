# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Compose all five flash_attn_prefill.cc entry points into one prefill round.

The device test for ``kernels.prefill_fv`` judges one step, ``y += S*V``, with
S handed to it ready-made. Nothing there reaches the masking, the online
rescale, or the geometry's query-grid mapping -- the parts a wrong index would
silently corrupt. This drives the whole round on one core, over one query
chunk and as many 128-key blocks as the mask needs, and checks the output
against ordinary masked attention.

The reference is deliberately single-pass: writing the online recurrence into
it would only assert the kernel against itself. What that costs is a tolerance,
because the kernel rounds to bf16 at each rescale and inside its polynomial
exp2, and the reference models neither. Measured over the original four cases the
worst deviation is 2.6 bf16 steps of the output's range; the bound allows 4.
AIE2 evaluates exp2 with exp2_bf16.h's cubic instead of aie::exp2, and there
the worst is 0.25 steps and the bound 1.
For scale, the bugs this test exists to catch -- a mask closing a key early, a
query grid mapped to the wrong row, keys permuted inside a chunk, V off by a
row -- move the output by 7 to 112 of those steps.

One blind spot, and it is the reason the sliding-window case earns its place:
admitting *one extra* key past the causal edge is invisible in the long-context
cases, where it is one key in 129 and the kernel's own rounding covers it. Only
the windowed case, with 64 keys visible and both edges live, separates it.
"""

import aie.extras.dialects.arith as arith  # pyright: ignore[reportMissingImports]
import aie.iron as iron
import numpy as np
import pytest
from aie.helpers.util import np_dtype_to_mlir_type
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
from ml_dtypes import bfloat16

BF = np.dtype[bfloat16]
F32 = np.dtype[np.float32]
I32 = np.dtype[np.int32]

# Tolerance, in bf16 steps at the top of the output range; see the assertion
# for why the unit is absolute. Measured, not guessed. The worst of the original four
# cases runs at 2.62 steps and the weakest bug class this test is built to
# catch -- a mask closing one key early -- moves the output by 6.62, so the
# bound sits between them with room on both sides.
_ATOL_ULP = 4

# The same on aie2, where the worst of 31 seeds on npu1 runs at 0.25 steps. It
# catches an exp2 argument 0.2 high in the rescale, 1.66 steps, and scores
# sharpened by half, 1.19; an argument 0.1 high moves the output 0.83.
_ATOL_ULP_AIE2 = 1

# PrefillGeom<DH>: head_dim -> (query chunk, key chunk).
_GEOM = {512: (8, 8), 256: (16, 16)}

# The softmax runs on exp2, and flash_attn_prefill.h folds attention's
# 1/sqrt(head_dim) into its bf16 log2(e) multiplier.
_EXP_SCALE = {
    head_dim: np.float32(bfloat16(np.log2(np.e) / np.sqrt(head_dim)))
    for head_dim in _GEOM
}

_BLOCK_KEYS = 128


def _inner_q(head_dim: int, block_q: int, row: int, col: int) -> int:
    """Absolute position of this core's first query -- PrefillGeom::inner_q."""
    if head_dim == 512:
        return block_q + (col >> 1) * 64 + (col & 1) * 8 + row * 16
    return block_q + col * 16 + row * 32


def _pack_q(mat: np.ndarray) -> np.ndarray:
    """(LQ, DH) -> the MMUL A operand: 8x8 row-major blocks, block-row-major."""
    lq, dh = mat.shape
    return mat.reshape(lq // 8, 8, dh // 8, 8).transpose(0, 2, 1, 3).reshape(-1).copy()


def _pack_kv(mat: np.ndarray) -> np.ndarray:
    """(LK, DH) -> blocks ordered (head-dim, key), each 8x8 row-major (key, dim).

    attn_qk transposes each block back on load and attn_fv takes it as the
    MMUL B operand directly, so K and V share one packing in both geometries.
    """
    lk, dh = mat.shape
    return mat.reshape(lk // 8, 8, dh // 8, 8).transpose(2, 0, 1, 3).reshape(-1).copy()


def _unpack_o(flat: np.ndarray, lq: int, dh: int) -> np.ndarray:
    """Unpack the epilogue's 64-element chunks back to (LQ, DH)."""
    return flat.reshape(lq // 8, dh // 8, 8, 8).transpose(0, 2, 1, 3).reshape(lq, dh)


def _attention(q, k, v, *, q_pos, k_pos, window, exp_scale):
    """Masked attention, rounded where the kernel rounds.

    S and the softmax weights are bf16 stores on the device and float32
    everywhere else. ``exp_scale`` says which exponential is being taken; it is
    a parameter rather than ``log2(e)`` so that the diagnostics below can ask
    what a differently-rounded constant would have cost.
    """
    scores = (q.astype(np.float32) @ k.astype(np.float32).T).astype(bfloat16)
    keep = (k_pos[None, :] <= q_pos[:, None]) & (
        k_pos[None, :] > q_pos[:, None] - window
    )
    assert keep.any(axis=1).all(), "a query with no visible key has no softmax"
    scores = np.where(keep, scores.astype(np.float32), -np.inf)
    shifted = (scores - scores.max(axis=1, keepdims=True)).astype(bfloat16)
    p = np.exp2(shifted.astype(np.float32) * exp_scale).astype(bfloat16)
    y = p.astype(np.float32) @ v.astype(np.float32)
    inv_l = np.float32(1.0) / p.astype(np.float32).sum(axis=1)
    return (
        y.astype(bfloat16).astype(np.float32)
        * inv_l.astype(bfloat16).astype(np.float32)[:, None]
    ).astype(bfloat16)


@jit
def prefill_round(
    q_in: In,
    kv_in: In,
    o_out: Out,
    *,
    head_dim: CompileTime[int] = 512,
    n_blocks: CompileTime[int] = 2,
    block_q: CompileTime[int] = 128,
    window: CompileTime[int] = 1 << 20,
    row: CompileTime[int] = 0,
    col: CompileTime[int] = 0,
):
    """One flash-attention prefill round on one core: all five entry points."""
    lq, lk = _GEOM[head_dim]
    chunks = n_blocks * (_BLOCK_KEYS // lk)
    out_chunks = lq * head_dim // 64
    q_col = (col & 1) if head_dim == 512 else col
    q_slots = q_col + 1

    q_ty = np.ndarray[(q_slots * lq * head_dim,), BF]
    kv_ty = np.ndarray[(lk * head_dim,), BF]
    s_ty = np.ndarray[(_BLOCK_KEYS * lq,), BF]
    m_ty = np.ndarray[(lq * lk,), BF]
    lq_bf = np.ndarray[(lq,), BF]
    lq_f32 = np.ndarray[(lq,), F32]
    y_ty = np.ndarray[(lq * head_dim,), F32]
    o_ty = np.ndarray[(64,), BF]
    scalar_ty = np.ndarray[(1,), I32]

    fv_step = kernels.prefill_fv(head_dim=head_dim)
    obj = fv_step.object_file
    round_begin = obj.bind("prefill_round_begin", [lq_bf, lq_bf, lq_f32, lq_f32, y_ty])
    qk_step = obj.bind(
        "prefill_qk_step",
        [s_ty, q_ty, kv_ty, m_ty, lq_bf, scalar_ty, scalar_ty, *([np.int32] * 5)],
    )
    block_mid = obj.bind(
        "prefill_block_mid", [s_ty, m_ty, lq_bf, lq_bf, lq_f32, lq_f32, y_ty]
    )
    fv = obj.bind("prefill_fv_step", [y_ty, s_ty, kv_ty, np.int32])
    epilogue = obj.bind("prefill_epilogue", [o_ty, lq_f32, y_ty, np.int32])
    setup = fv_step.contract.setup
    assert setup is not None
    setter = setup()

    # Q is acquired once and held for the whole round, so one slot is all it
    # can ever use -- and at head_dim 512 with a second column it is 16 KB.
    of_q = ObjectFifo(q_ty, name="q", depth=1)
    # K and V share one stream: a compute tile has two input DMA channels, and
    # Q holds one for the whole round. Each block sends its key chunks and then
    # its value chunks, which is the order the steps consume them in anyway.
    of_kv = ObjectFifo(kv_ty, name="kv", depth=2)
    of_o = ObjectFifo(o_ty, name="o", depth=2)

    scratch = [
        Buffer(s_ty, name="s"),
        Buffer(m_ty, name="m"),
        Buffer(lq_bf, name="prev_m"),
        Buffer(lq_bf, name="new_m"),
        Buffer(lq_f32, name="c"),
        Buffer(lq_f32, name="l"),
        Buffer(y_ty, name="y"),
        Buffer(scalar_ty, name="l_begin", initial_value=np.array([block_q], np.int32)),
        Buffer(scalar_ty, name="window", initial_value=np.array([window], np.int32)),
    ]

    def core(
        of_q, of_kv, of_o, rb, qk, bm, fv, ep, setter,
        s, m, prev_m, new_m, c, l_sum, y, l_begin, window_size,
    ):  # fmt: skip
        i32 = np_dtype_to_mlir_type(np.int32)
        setter()
        q = of_q.acquire(1)
        rb(prev_m, new_m, c, l_sum, y)
        for block in range_(n_blocks):
            b = arith.index_cast(block, to=i32)
            for chunk in range_(_BLOCK_KEYS // lk):
                j = arith.index_cast(chunk, to=i32)
                qk(
                    s, q, of_kv.acquire(1), m, prev_m, l_begin, window_size,
                    row, col, 0, b, j,
                )  # fmt: skip
                of_kv.release(1)
            bm(s, m, new_m, prev_m, c, l_sum, y)
            for chunk in range_(_BLOCK_KEYS // lk):
                fv(y, s, of_kv.acquire(1), arith.index_cast(chunk, to=i32))
                of_kv.release(1)
        for out_chunk in range_(out_chunks):
            ep(of_o.acquire(1), l_sum, y, arith.index_cast(out_chunk, to=i32))
            of_o.release(1)
        of_q.release(1)

    worker = Worker(
        core,
        fn_args=[
            of_q.cons(),
            of_kv.cons(),
            of_o.prod(),
            round_begin,
            qk_step,
            block_mid,
            fv,
            epilogue,
            setter,
            *scratch,
        ],
        # The contract's number is aiecc's measurement of the fv step alone, and
        # this core runs all five -- but the fv step is the deepest, so it is
        # also the core's number: 960 bytes at head_dim 512, 5824 at 256, both
        # exactly the contract's. No margin, deliberately. aiecc re-measures the
        # linked core and fails the build naming the byte count it wanted, so a
        # future step growing past the fv step's frame is a compile error here
        # rather than something a margin would quietly absorb.
        stack_size=fv_step.contract.stack_bytes,
    )

    host = [
        np.ndarray[(q_slots * lq * head_dim,), BF],
        np.ndarray[(2 * chunks * lk * head_dim,), BF],
        np.ndarray[(lq * head_dim,), BF],
    ]

    def sequence(q_h, kv_h, o_h, qf, kvf, of):
        qf.fill(q_h)
        kvf.fill(kv_h)
        of.drain(o_h, wait=True)

    rt = Runtime(sequence, [*host, of_q.prod(), of_kv.prod(), of_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _dyadic(rng, shape, scale):
    """Small dyadic bf16, so every product is exact in the float32 accumulator."""
    return (rng.integers(-2, 3, size=shape) * scale).astype(bfloat16)


def case_data(head_dim, block_q, n_blocks, window, row, col):
    """Build inputs as mathematical operands and device buffers."""
    lq, lk = _GEOM[head_dim]
    chunks = n_blocks * (_BLOCK_KEYS // lk)
    q_pos = _inner_q(head_dim, block_q, row, col) + np.arange(lq)
    k_start = max(block_q - window, 0)
    k_pos = k_start + np.arange(chunks * lk)
    assert k_pos[-1] >= q_pos[-1], "the key stream must reach the last query"

    rng = np.random.default_rng(20260923 + head_dim + block_q)
    q = _dyadic(rng, (lq, head_dim), 0.25)
    k = _dyadic(rng, (len(k_pos), head_dim), 0.125)
    v = _dyadic(rng, (len(k_pos), head_dim), 0.25)
    # Aim each query at the last key its mask admits, at a quarter the strength
    # that would make the softmax a delta. Two things follow. That key sits in
    # the final block, so the running row max rises there and the online
    # rescale has to correct what the earlier blocks accumulated. And it draws
    # enough of the softmax mass that a mask closing one key early changes the
    # answer outright -- otherwise it is one key in 129, a change the kernel's
    # own bf16 rounding would hide.
    for r, pos in enumerate(q_pos):
        k[pos - k_start] = (q[r].astype(np.float32) * 0.0625).astype(bfloat16)

    q_col = (col & 1) if head_dim == 512 else col
    q_host = np.zeros((q_col + 1, lq * head_dim), bfloat16)
    q_host[q_col] = _pack_q(q)
    # One stream, in consumption order: each block's key chunks, then its value
    # chunks.
    per_block = _BLOCK_KEYS // lk
    kv_host = np.concatenate(
        [
            _pack_kv(mat[(b * per_block + c) * lk :][:lk])
            for b in range(n_blocks)
            for mat in (k, v)
            for c in range(per_block)
        ]
    )
    return q, k, v, q_pos, k_pos, q_host.reshape(-1), kv_host


@pytest.mark.parametrize(
    "head_dim,block_q,n_blocks,window,row,col",
    [
        # Global attention: two blocks, so the second block's row max forces
        # the online rescale, and the causal mask cuts partway through it.
        (512, 128, 2, 1 << 20, 0, 0),
        # The same geometry off the origin of the query grid: a nonzero row
        # shifts the query positions the mask compares against, and a nonzero
        # column additionally shifts which Q tile the core reads.
        (512, 256, 3, 1 << 20, 1, 1),
        (512, 256, 3, 1 << 20, 1, 2),
        (512, 256, 3, 1 << 20, 1, 3),
        # Sliding window: the left edge bites, and one block is all a window
        # narrower than the block ever needs. The only case with few enough
        # keys visible to see the causal edge open one key too far.
        (256, 256, 1, 64, 0, 0),
        (256, 256, 1, 64, 1, 1),
        # The sliding-window geometry with the window open, which is what
        # exercises its 2x2 decomposition against the rescale.
        (256, 128, 2, 1 << 20, 0, 0),
    ],
)
def test_prefill_round_matches_masked_attention(
    head_dim, block_q, n_blocks, window, row, col
):
    lq, _ = _GEOM[head_dim]
    q, k, v, q_pos, k_pos, q_host, kv_host = case_data(
        head_dim, block_q, n_blocks, window, row, col
    )

    o_t = iron.zeros((lq * head_dim,), dtype=bfloat16)
    prefill_round(
        iron.tensor(q_host, dtype=bfloat16),
        iron.tensor(kv_host, dtype=bfloat16),
        o_t,
        head_dim=head_dim,
        n_blocks=n_blocks,
        block_q=block_q,
        window=window,
        row=row,
        col=col,
    )
    got = _unpack_o(o_t.numpy(), lq, head_dim).astype(np.float32)

    ref = _attention(
        q,
        k,
        v,
        q_pos=q_pos,
        k_pos=k_pos,
        window=window,
        exp_scale=_EXP_SCALE[head_dim],
    ).astype(np.float32)
    # Every output is a convex combination of the V rows, so it lives on the
    # scale of max|v| whatever the head dim, and one bf16 step at the top of
    # that range is the natural unit to measure the error in. The bound is
    # purely absolute: where the combination cancels, the output is near zero
    # and its relative error is unbounded -- 180x in the worst element -- while
    # the absolute error stays flat.
    ulp = 2**-7 * float(np.abs(v.astype(np.float32)).max())
    aie2 = resolve_target_arch(iron.get_current_device()) == "aie2"
    atol_ulp = _ATOL_ULP_AIE2 if aie2 else _ATOL_ULP
    np.testing.assert_allclose(got, ref, rtol=0, atol=atol_ulp * ulp)

    # The reference above carries the kernel's bf16 scaled-attention multiplier,
    # so on its own it would pass a systematically sharpened softmax.
    # Against the mathematical exponential the answer barely moves -- that is
    # what says the round computes attention, and not merely what the model
    # says it does. It is also the check that keeps the bf16 constant honest:
    # this is where that argument would fail if it were wrong.
    exact = _attention(
        q,
        k,
        v,
        q_pos=q_pos,
        k_pos=k_pos,
        window=window,
        exp_scale=np.log2(np.e) / np.sqrt(head_dim),
    ).astype(np.float32)
    np.testing.assert_allclose(ref, exact, rtol=0, atol=1 * ulp)
