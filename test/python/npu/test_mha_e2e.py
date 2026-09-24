# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Compose mha.cc's online-softmax entry points into one decode round.

The device cases for ``kernels.mha`` judge the two matmuls with their operands
handed to them ready-made. Nothing there reaches ``partial_softmax``'s causal
and padding masks, the running max and sum it carries in ``scale_buffer``, or
the row rescales ``matmul_PV`` and ``rescale_O`` apply from it -- the parts a
wrong index or a mis-shaped mask would silently corrupt. This drives the whole
round on one core, over one 64-query block and as many 64-key blocks as the
mask needs, and checks the output against ordinary masked attention.

``QK^T`` stays on the host: it is its own entry point with its own device case,
and keeping it there lets this test hand the round a score matrix it chose.
Everything from the mask onwards runs on the core.

The reference is deliberately single-pass. Writing the online recurrence into
it would only assert the kernel against itself, so it takes the whole masked
score matrix, subtracts one max per row and normalizes once. What that costs is
a tolerance, because the kernel rounds to bf16 at every rescale and inside its
polynomial exp2 and the reference models neither. Measured over the five cases
below the worst deviation is 3.25 bf16 steps of the output's range; the bound
allows 4. The bugs this test exists to catch move it by 6 steps and up: a
key-tail mask opening one column too far is the weakest at 6.0, the diagonal
opening one column too far runs 99, dropping the block above the diagonal 15,
and getting either matmul's operands wrong 35 to 330.

Two blind spots, and they are why the single-block cases earn their place.
Admitting *one extra* key is invisible in the two-block cases, where it is one
key in 65 and the kernel's own rounding covers it -- only a case whose whole
key set is one block separates it. And a row rescale that broadcasts the wrong
lane of the correction factor moves those cases by a single element, barely
over the bound, while it moves a one-block case by 270.

One shape the round needs that the kernel does not provide: ``partial_softmax``
writes P row-major and ``matmul_PV`` reads its operand in 8x8 blocks, so P
takes a memtile hop on the way back. That is the transform mha.cc's own comment
records for O, and it happens to be its own inverse.
"""

from collections.abc import Sequence

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

BF = np.dtype[bfloat16]
I32 = np.dtype[np.int32]

# The kernel's micro-tile: 64 queries by 64 keys by a head dim of 64.
# scale_blocked_rows walks exactly eight 8x8 block rows, so 64 is the only
# width it is written for.
_B = 64

# softmax.cc exponentiates with exp2 and mha.cc's caller passes the combined
# constant, so the curve the round takes is 2**(inv_scale * s). Ordinary
# attention wants exp(s / sqrt(head_dim)), which is inv_scale = log2(e) /
# sqrt(head_dim) -- rounded to bf16, because that is the argument type. The
# reference carries the same rounded constant; the second assertion below
# measures what the rounding costs.
_INV_SCALE = float(bfloat16(np.log2(np.e) / np.sqrt(_B)))

# Tolerance in bf16 steps at the top of the output range; see the assertion for
# why the unit is absolute. Measured, not guessed: the worst of the five cases
# runs at 3.25 steps and the weakest bug class this test is built to catch
# moves the output by 6.0, so the bound sits between them. The margin either
# side is thin, and it can be: nothing here is random at run time, so a case
# that measures 3.25 today measures 3.25 every time.
_ATOL_ULP = 4

# P leaves the core row-major and has to come back in the mmul's block order:
# element (row, col) at (row / 8) * 512 + (col / 8) * 64 + (row % 8) * 8 +
# col % 8. Reading a row-major buffer with these dims emits exactly that.
_REBLOCK: list[Sequence[int]] = [(8, 512), (8, 8), (8, 64), (8, 1)]


def _block(mat: np.ndarray) -> np.ndarray:
    """(64, 64) -> 8x8 row-major blocks in block-row-major order.

    mm.cc's operand order, shared by A, B and C: block (i, j) sits at
    (i * 8 + j) * 64. Its own inverse.
    """
    return mat.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(-1).copy()


def _unblock(flat: np.ndarray) -> np.ndarray:
    """The blocked 64x64 tile back to row-major."""
    return flat.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(_B, _B)


def _keep(q_block: int, n_kv: int, s_q_eff: int, s_kv_eff: int) -> np.ndarray:
    """The mask the round applies, as ordinary causal attention states it.

    Rows are this block's queries, columns every key the round streams. A key
    is visible when it is at or before its query and both sit inside the
    sequence. partial_softmax reaches the same set block by block -- it drops
    whole blocks above the diagonal, clamps the padded row and column tails,
    and cuts at the diagonal within the diagonal block -- but only because the
    query and key blocks are the same width, so the block comparison and the
    element comparison agree.
    """
    q_pos = q_block * _B + np.arange(_B)
    k_pos = np.arange(n_kv * _B)
    return (
        (k_pos[None, :] <= q_pos[:, None])
        & (q_pos[:, None] < s_q_eff)
        & (k_pos[None, :] < s_kv_eff)
    )


def _attention(scores, v, keep, *, inv_scale):
    """Masked attention over a given score matrix, rounded where the kernel is.

    P and the running quantities are bf16 stores on the device and float32
    everywhere else. ``inv_scale`` says which exponential is being taken; it is
    a parameter rather than the bf16 constant so the second assertion can ask
    what an unrounded one would have cost.
    """
    scaled = scores.astype(np.float32) * np.float32(inv_scale)
    live = keep.any(axis=1)
    # The row max is a bf16 store on the device and the scaling either side of
    # it is float32; rows the mask leaves empty get no answer at all, so park
    # their max anywhere finite.
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
def mha_round(
    sv_in: In,
    o_out: Out,
    *,
    q_block: CompileTime[int] = 0,
    n_kv: CompileTime[int] = 1,
    s_q_eff: CompileTime[int] = _B,
    s_kv_eff: CompileTime[int] = _B,
):
    """One flash-attention decode round on one core: mha.cc's four steps."""
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

    # S and V share one stream: a compute tile has two input DMA channels and
    # the reblocked P holds the other. Each key block sends its scores and then
    # its values, which is the order the steps consume them in anyway.
    of_sv = ObjectFifo(tile_ty, name="sv", depth=2)
    # O is acquired once and held for the whole round -- it is the accumulator
    # every matmul_PV reads back and rescale_O finishes -- so one slot is all
    # it can use.
    of_o = ObjectFifo(tile_ty, name="o", depth=1)
    of_p = ObjectFifo(tile_ty, name="p", depth=1)
    of_pb = of_p.cons().forward(dims_to_stream=_REBLOCK, depth=1)

    scale_buf = Buffer(scale_ty, name="scale")
    # One index pair per key block. The round is short enough to unroll, which
    # keeps the pair a compile-time constant rather than a buffer the core
    # would have to index at runtime.
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
        stack_size=qkt.contract.stack_bytes,
    )

    host = [np.ndarray[(2 * n_kv * _B * _B,), BF], np.ndarray[(_B * _B,), BF]]

    def sequence(sv_h, o_h, svf, of):
        svf.fill(sv_h)
        of.drain(o_h, wait=True)

    rt = Runtime(sequence, [*host, of_sv.prod(), of_o.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _dyadic(rng, shape, scale):
    """Small dyadic bf16, so every product is exact in the float32 accumulator."""
    return (rng.integers(-2, 3, size=shape) * scale).astype(bfloat16)


def case_data(q_block, n_kv, s_q_eff, s_kv_eff):
    """Inputs for one case, as the mathematical operands and as device buffers."""
    rng = np.random.default_rng(20260923 + 97 * q_block + n_kv + s_q_eff + s_kv_eff)
    q = _dyadic(rng, (_B, _B), 0.25)
    k = _dyadic(rng, (n_kv * _B, _B), 0.125)
    v = _dyadic(rng, (n_kv * _B, _B), 0.25)
    # Aim each query at the diagonal key its mask admits last, at a quarter the
    # strength that would make the softmax a delta. That key is always in the
    # diagonal block, which is the last block the round runs, so the running max
    # rises there and matmul_PV's rescale has to correct what the earlier blocks
    # accumulated -- the code path a flat score matrix would leave at 1.0.
    keep = _keep(q_block, n_kv, s_q_eff, s_kv_eff)
    for row in range(_B):
        if keep[row].any():
            k[q_block * _B + row] = (q[row].astype(np.float32) * 0.5).astype(bfloat16)

    scores = (q.astype(np.float32) @ k.astype(np.float32).T).astype(bfloat16)
    # One stream, in consumption order: each block's scores, then its values.
    # partial_softmax walks its score tile row-major; matmul_PV takes V as the
    # mmul B operand.
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
        # The pure diagonal: one block, no padding, so every row is cut at its
        # own column and nothing else masks.
        (0, 1, _B, _B),
        # Two blocks: the first is fully visible and unmasked, the second is the
        # diagonal. The only shape where the online rescale carries a real
        # correction between blocks.
        (1, 2, 2 * _B, 2 * _B),
        # A query tail and a key tail that both land mid-vector: 36 of 64 rows
        # live, and the diagonal block closes at column 36. The suffix the mask
        # writes starts inside a 64-lane store rather than on its boundary.
        (1, 2, 100, 100),
        # A key tail alone, on the diagonal block, closing before the diagonal
        # does for most rows -- so the column the mask starts at is the padding
        # bound for some rows and the diagonal for others, in the same tile.
        (0, 1, 100, 40),
        # A third key block past the diagonal: partial_softmax must zero its P
        # outright and matmul_PV must decline it, or the keys ahead of these
        # queries leak into the answer.
        (1, 3, 2 * _B, 3 * _B),
    ],
)
def test_mha_round_matches_masked_attention(q_block, n_kv, s_q_eff, s_kv_eff):
    scores, v, keep, sv_host = case_data(q_block, n_kv, s_q_eff, s_kv_eff)

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
    # Only the rows the mask leaves live carry an answer; the padded ones are
    # whatever the zeroed P and the reciprocal of a zero sum leave behind, and
    # the round makes no claim about them.
    live = keep.any(axis=1)
    # Every live output is a convex combination of the V rows, so it lives on
    # the scale of max|v| whatever the block, and one bf16 step at the top of
    # that range is the natural unit to measure the error in. The bound is
    # purely absolute: where the combination cancels the output is near zero
    # and its relative error is unbounded while the absolute error stays flat.
    ulp = 2**-7 * float(np.abs(v.astype(np.float32)).max())
    np.testing.assert_allclose(got[live], ref[live], rtol=0, atol=_ATOL_ULP * ulp)

    # The reference above carries the kernel's bf16 log2(e)/8, so on its own it
    # would pass a round that computed a systematically sharpened softmax.
    # Against the exact constant the answer barely moves, which is what says the
    # round computes attention and not merely what the model says it does.
    exact = _attention(scores, v, keep, inv_scale=np.log2(np.e) / np.sqrt(_B)).astype(
        np.float32
    )
    np.testing.assert_allclose(ref[live], exact[live], rtol=0, atol=1 * ulp)
