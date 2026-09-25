# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Numeric gates for mha.cc: each softmax entry point, and the whole round.

``kernels.mha`` covers the two tile matmuls, and the generic device harness
drives those well: operands in, product out, a numpy matmul to judge it
against. The other half of mha.cc does not fit that shape. ``partial_softmax``
reads and writes a carried ``scale_buffer``, and ``rescale_O`` reads its output
back in and mutates that same buffer -- so neither can be a kernel contract,
whose outputs are always zeroed on core and never filled from the host. Until
this file they had no standing numeric check of any kind.

So each is driven on its own here, and then all four steps are driven together
as one decode round. The two halves answer different questions and neither
subsumes the other. A gate on one entry point sees that kernel's own arithmetic
exactly, because it reads the raw weights and the carried state; the round sees
only what survives normalization, but it is the one thing that says the four
steps agree with each other rather than each being separately defensible.

Every reference here is single-pass, carries none of the kernel's structure,
and exponentiates with true ``np.exp2``: the mask is written as
``key <= query`` over absolute positions, and the softmax as the textbook
online recurrence. Nothing models the device's arithmetic, which matters
because AIE2P's ``exp2`` is not a polynomial but a linear interpolant that
overshoots by up to 6.15% -- see ``_RTOL_EXP2``. Modelling it would buy a
tolerance eleven times tighter, and it is still the wrong trade: it would make
the reference track this device's instruction rather than the mathematics, so a
kernel that moved to ``exp2_poly.h`` the way ``flash_attn_prefill.cc`` already
has would fail a test it had just made more accurate.

What is left is the interpolant's own envelope where the weights are read raw,
and bf16 rounding where they are not. The round never sees the envelope: it
judges ``P*V / sum(P)``, and a smooth multiplicative error divides out of
numerator and denominator alike, leaving it at 3.25 bf16 steps against a bound
of 4. The bugs this file exists to catch clear both bounds by an order of
magnitude: a mask admitting or dropping one key takes that weight between an
exact zero and order one, and the quietest element any of them disturbs still
moves by 31%.

AIE2 has no ``aie::exp2``, and mha.cc evaluates a cubic there instead, so the
envelope does not apply: the weights are held to ``kernels.mha_softmax``'s
aie2 tolerance, 0.4% of ``|a| + |b|``. A weight one bf16 step off always meets
it, and one two steps off only in the top 5% of a binade. That bound is what
sees a wrong coefficient in the cubic, which the envelope passes.

Which configuration catches what is not uniform, and each one below is the only
one that sees something. A key tail that runs one key long shows up only where
the tail falls mid-vector, since a tail on a vector boundary takes the same
path either way. A block past the diagonal that is never skipped shows up only
where there is such a block. A carried correction factor that is dropped shows
up only where a live block precedes another live block. A row rescale that
broadcasts the wrong lane moves a one-block round by 270 steps and a two-block
one barely past the bound. This is the same kind of edge
``test_flash_attn_prefill_e2e`` documents for its own mask.

Not covered, deliberately: a key block that is entirely padding but still
inside the causal region. ``partial_softmax`` returns from that without
updating ``scale_buffer``, which leaves the correction factor from the previous
block in place for the next reader. That is pre-existing kernel behaviour, not
something this file should freeze, so the configurations below stay out of it.
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

# mha.cc is built with -DDIM_M=64 -DDIM_N=64, and its callers pass the same 64
# for B_q and B_kv, so one query block, one key block and one tile are all this
# size. The kernels take the block sizes as runtime arguments, but the P zeroing
# is a compile-time template, so 64 is the only size that is actually wired up.
_B = 64

# The caller's 1/sqrt(d) folded into log2(e), because the softmax runs on exp2.
# The kernel takes it as a bf16 scalar, so the reference carries the same
# rounded constant -- this is a property of the interface, not of the kernel's
# arithmetic, and the caller would round it identically.
_INV_SCALE = float(bfloat16(np.log2(np.e) / np.sqrt(_B)))

# Tolerance, in bf16 steps at the top of each quantity's range; see the
# assertions for why the bound is absolute rather than relative. Measured, not
# guessed: the worst case below runs at 1.26 steps, and the weakest signal any
# of the bugs in the docstring produces is 6.5 on the row sum and 128 on the
# weights themselves. The bound sits three times above the noise and well under
# the quietest bug.
_ATOL_ULP = 4

# bf16's lowest finite value, which is what init_scale_buffer seeds the running
# max with. Not -inf: the kernel's first correction factor is exp2(m_prev -
# m_new), and that has to evaluate to a clean zero rather than a NaN.
_LOWEST = float(ml_dtypes.finfo(bfloat16).min)

# What the unnormalized weights are allowed to differ from true exp2 by, and it
# is a derivation rather than a measurement. ``aie::exp2<bfloat16>`` on AIE2P is
# not a polynomial: it writes the fraction straight into the mantissa field, so
# it evaluates 2**floor(u) * (1 + frac(u)). That interpolant is exact at every
# power of two and, exp2 being convex, overshoots in between -- by at most
# max((1 + f) / 2**f - 1), which is 6.15% at f = 1/ln2 - 1 = 0.443. bf16 rounds
# once on the device and once in the reference, 2**-8 each, so the envelope is
# 1.0615 * 1.0078 - 1 = 6.98%.
#
# Measured against that: the weights run 6.79% and the row sums 6.66%, both
# inside it, and nothing here is random at run time so those hold every run. The
# mutations in this file's history move the same quantities by 31% at their
# quietest and 170% at their loudest, which is what says a bound this wide still
# has teeth. Each was re-run against this bound rather than the tighter one it
# was first proved under, and none of them stopped failing.
_RTOL_EXP2 = 0.07

# P leaves the core row-major and has to come back in the mmul's block order.
# Reading a row-major buffer with these dims emits exactly the layout below.
_REBLOCK: list[Sequence[int]] = [(8, 512), (8, 8), (8, 64), (8, 1)]


def _block(mat: np.ndarray) -> np.ndarray:
    """(64, 64) -> 8x8 row-major blocks in block-row-major order.

    mm.cc's operand order, shared by A, B and C: block (i, j) sits at
    (i * 8 + j) * 64. Its own inverse.
    """
    return mat.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(-1).copy()


def _unblock(flat: np.ndarray) -> np.ndarray:
    """The 8x8-blocked O tile back to (64, 64).

    mha.cc's ``matmul_PV`` records the layout: element (row, col) sits at
    ``(row / 8) * 512 + (col / 8) * 64 + (row % 8) * 8 + col % 8``.
    """
    return flat.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(_B, _B)


def _keep(q_block: int, kv_block: int, s_q_eff: int, s_kv_eff: int) -> np.ndarray:
    """Which (query, key) pairs of this block pair a causal mask admits.

    Written over absolute sequence positions, which is how a mask is ordinarily
    stated. The kernel arrives at the same answer a different way -- it skips
    whole blocks above the diagonal, then masks a suffix of each row -- so the
    two agreeing is the thing worth checking.
    """
    rows = q_block * _B + np.arange(_B)
    cols = kv_block * _B + np.arange(_B)
    return (
        (cols[None, :] <= rows[:, None])
        & (rows[:, None] < s_q_eff)
        & (cols[None, :] < s_kv_eff)
    )


def _keep_round(q_block: int, n_kv: int, s_q_eff: int, s_kv_eff: int) -> np.ndarray:
    """The same mask over every key block a round streams, side by side.

    Rows are the round's queries, columns every key it sees. Stating it as the
    per-block masks laid end to end rather than as its own formula keeps one
    definition of the mask in this file: the round and the gates then agree by
    construction, and a wrong edge cannot hide by being wrong in both.
    """
    return np.concatenate(
        [_keep(q_block, kv, s_q_eff, s_kv_eff) for kv in range(n_kv)], axis=1
    )


def _softmax_step(scores, keep, m_prev, l_prev):
    """One key block of the online softmax recurrence, in float32.

    True ``np.exp2``, not the interpolant the device evaluates; ``_RTOL_EXP2``
    says what that costs and why it is worth paying.

    Returns the unnormalized weights and the updated running max and row sum.
    Rows the mask leaves empty keep the state they came in with, which is what
    the kernel's own early exits amount to.
    """
    scaled = scores.astype(np.float32) * np.float32(_INV_SCALE)
    m_new = np.where(keep, scaled, -np.float32(np.inf)).max(axis=1)
    m_new = np.maximum(m_new, m_prev)
    # Masked entries take the exponent to zero rather than to -inf, so the
    # weight is an exact zero instead of an overflow that np.where discards.
    p = np.exp2(np.where(keep, scaled, m_new[:, None]) - m_new[:, None])
    # The weights are stored as bf16 and the row sum accumulates what was
    # stored, so the reference rounds in the same place.
    p = np.where(keep, p.astype(bfloat16).astype(np.float32), np.float32(0))
    l_new = np.exp2(m_prev - m_new) * l_prev + p.sum(axis=1)
    return p, m_new, l_new


def _attention(scores, v, keep, *, inv_scale):
    """Masked attention over a given score matrix, rounded where the kernel is.

    The normalization below is why this one needs no allowance for the device's
    exp2 at all: the overshoot divides out of numerator and denominator, and
    what is left is bf16 rounding.

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
    """``partial_softmax`` over two key blocks, sharing one ``scale_buffer``.

    Two blocks rather than one because the second block is what reads the state
    the first one wrote: its running max caps the second block's, and its row
    sum is what the correction factor rescales. A single call would exercise
    the recurrence only from its seeded start, where the correction degenerates
    to zero.

    The score tiles arrive on a fifo and are consumed in place -- the kernel
    writes bf16's lowest into the entries it masks -- and the scale buffer is
    an output fifo element for its whole life, seeded on core by
    ``init_scale_buffer`` and drained once both blocks have updated it.
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

    # idx_buffer is (key block, query block); the kernel compares the two to
    # decide whether this pair sits above the diagonal.
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
        stack_size=mha.contract.stack_bytes,
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
    """``rescale_O``: invert the row sums, then divide the O tile by them.

    O is read-modify-write, so the tile is copied into the output fifo element
    first and rescaled there. The scale buffer is consumed in place -- the
    kernel overwrites the row sums with their reciprocals -- which is why it
    needs no path back to the host: the reciprocals are visible in O.
    """
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

    # rescale_O takes an idx_buffer for signature compatibility with the other
    # entry points and never reads it.
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
    """Small dyadic bf16, so the host's arithmetic on them is exact.

    ``mag`` bounds the integer before scaling. The round wants it far smaller
    than the gates do: its operands go through a matmul, and keeping every
    product exact in the float32 accumulator is what lets its tolerance be
    about the kernel's rounding rather than about the reference's.
    """
    return (rng.integers(-mag, mag + 1, size=shape) * scale).astype(bfloat16)


def _softmax_case_data(q_block, kv_first, kv_second, s_q_eff, s_kv_eff):
    """Score tiles for one softmax case, and the masks that go with them."""
    rng = np.random.default_rng(20260923 + q_block * 17 + s_kv_eff)
    keeps = [
        _keep(q_block, kv, s_q_eff, s_kv_eff) for kv in (kv_first, kv_second)
    ]  # fmt: skip
    assert keeps[0].any() or keeps[1].any(), "a case with no live key gates nothing"
    # The second block's scores run hotter than the first's, so its row max
    # rises and the correction factor that rescales the carried row sum is a
    # number well away from 1. With both blocks on the same scale the carry
    # would be near-inert and a broken correction would pass.
    scores = [_dyadic(rng, (_B, _B), 0.25), _dyadic(rng, (_B, _B), 0.75)]
    return scores, keeps


@pytest.mark.parametrize(
    "q_block,kv_first,kv_second,s_q_eff,s_kv_eff",
    [
        # No mask at all on the first block, pure causal diagonal on the
        # second: the two extremes of the mask in one case, with the carry
        # between them.
        (1, 0, 1, 128, 128),
        # The diagonal block alone, then a block past it. The second block is
        # the one the kernel must skip outright, leaving the scale buffer as it
        # found it; the first carries the whole softmax, which is what makes a
        # causal edge one key too wide visible here and nowhere else.
        (0, 0, 1, 64, 128),
        # Both tails mid-vector: 36 live query rows and, on the diagonal block,
        # 36 live keys. The masked suffix of each row starts inside a vector
        # rather than on its boundary, which is the case the lane mask exists
        # for.
        (1, 0, 1, 100, 100),
        # A key tail with no query padding, so the key extent is the only thing
        # cutting the diagonal block short.
        (2, 1, 2, 192, 150),
        # A short key tail -- 6 live keys on the diagonal block -- under padded
        # query rows, where the tail bites before the diagonal does for all but
        # the first six rows.
        (1, 0, 1, 100, 70),
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

    # Relative against the interpolant's envelope, with an absolute floor: a
    # weight near the row max carries the full 6.15% overshoot, while the keys
    # furthest below it land near zero, where relative accuracy means nothing
    # and one bf16 step of the range does. Padded rows are included -- both the
    # kernel and the reference must put an exact zero there, which the floor
    # still holds them to. On aie2 the factory's bound: measured over 31 seeds
    # on npu1, the weights run 0.389% of |a| + |b|, one step.
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

    # The running max and the row sum the kernel carries out for its caller.
    # Only rows some key reached: the kernel runs its epilogue over all 64 rows
    # regardless, so a padded row's state is whatever arithmetic on the seed
    # happens to produce, and the caller discards those rows anyway.
    live = keeps[0].any(axis=1) | keeps[1].any(axis=1)
    # The max is picked out of the scaled scores and never exponentiated, so it
    # gets no relative allowance -- it is a bf16 store and nothing more, and it
    # measures 0.30 steps of 4.
    m_ulp = 2**-7 * float(np.abs(m[live]).max())
    np.testing.assert_allclose(
        got_scale[:_B][live], m[live], rtol=0, atol=_ATOL_ULP * m_ulp
    )
    # The row sum is a sum of weights, so it inherits their envelope rather than
    # accumulating past it: every term overshoots by at most 6.15%, so their
    # sum does too. On aie2 the second block's sum is c * l_prev + sum(P),
    # with c an exp2 as well and each term a bf16 store, so it gets two
    # weights' allowance; it measures 0.54% of |a| + |b| over the same seeds.
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
        # The row sums a 64-key block actually produces: order 1 to order 64,
        # so the reciprocals are all below 1.
        (1.0, 64.0),
        # Sums below 1, where the reciprocal amplifies instead. A row whose
        # keys all sit far below its running max lands here.
        (0.125, 1.0),
        # Three decades in one tile, which is what asks whether the reciprocal
        # holds up across the exponent range rather than near a single scale.
        (0.03125, 512.0),
    ],
)
def test_rescale_o_divides_rows_by_their_sums(l_lo, l_hi):
    rng = np.random.default_rng(20260923 + int(l_hi))
    o_blocked = _dyadic(rng, (_B * _B,), 0.5)
    # Row sums vary within each 8-row block as well as across blocks, so a
    # broadcast that reached the wrong row of a block would change the answer.
    l = np.exp2(rng.uniform(np.log2(l_lo), np.log2(l_hi), _B)).astype(bfloat16)
    scale = np.zeros(4 * _B, bfloat16)
    scale[2 * _B : 3 * _B] = l

    out_t = iron.zeros((_B * _B,), dtype=bfloat16)
    rescale_tile(
        iron.tensor(o_blocked, dtype=bfloat16), iron.tensor(scale, dtype=bfloat16), out_t
    )  # fmt: skip
    got = _unblock(out_t.numpy()).astype(np.float32)

    ref = _unblock(o_blocked).astype(np.float32) / l.astype(np.float32)[:, None]
    # Absolute again, and for the same reason: O entries near zero stay near
    # zero after the division and carry no relative accuracy, while the ones
    # that set the range carry all of it.
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
    """One flash-attention decode round on one core: mha.cc's four steps.

    ``QK^T`` stays on the host -- it is its own entry point with its own device
    case, and keeping it there lets the round be handed a score matrix chosen to
    put the running max where it will exercise the carry. Everything from the
    mask onwards runs on the core.

    One shape the round needs that the kernel does not provide: ``partial_softmax``
    writes P row-major and ``matmul_PV`` reads its operand in 8x8 blocks, so P
    takes a memtile hop on the way back. That is the transform mha.cc's own
    comment records for O, and it happens to be its own inverse.
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


def _round_case_data(q_block, n_kv, s_q_eff, s_kv_eff):
    """Inputs for one round, as the mathematical operands and as device buffers."""
    rng = np.random.default_rng(20260923 + 97 * q_block + n_kv + s_q_eff + s_kv_eff)
    q = _dyadic(rng, (_B, _B), 0.25, mag=2)
    k = _dyadic(rng, (n_kv * _B, _B), 0.125, mag=2)
    v = _dyadic(rng, (n_kv * _B, _B), 0.25, mag=2)
    # Aim each query at the diagonal key its mask admits last, at a quarter the
    # strength that would make the softmax a delta. That key is always in the
    # diagonal block, which is the last block the round runs, so the running max
    # rises there and matmul_PV's rescale has to correct what the earlier blocks
    # accumulated -- the code path a flat score matrix would leave at 1.0.
    keep = _keep_round(q_block, n_kv, s_q_eff, s_kv_eff)
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
