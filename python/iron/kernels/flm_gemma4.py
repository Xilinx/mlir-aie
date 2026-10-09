# flm_gemma4.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Kernels extracted from FastFlowLM's Gemma 4 implementation.

These are not general building blocks. Each source is the kernel code of one
core in one of FastFlowLM's Gemma 4 designs (prefill attention, the LM head
and the decode layer), called from that core's Worker body. Several kernels
acquire and release core locks themselves, by the lock ids the factories
take. The decode kernels build for the whole model's geometry
([`FlmGemma4DecodeGeometry`][iron.kernels.flm_gemma4.FlmGemma4DecodeGeometry])
and for FastFlowLM's array layout. All are AIE2P only; their
sources are in ``aie_kernels/flm_gemma4/``.

A factory returns one ``ExternalFunction``. As
[`cascade_mm`][iron.kernels.linalg.cascade_mm] does with its ``put_only`` and
``put_get``, every entry point of its object, its own included, is an
attribute of it named by its C symbol: ``fn.attn_qk_round``.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import TypeVar, get_args

import numpy as np
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    Trace,
    _arch_traits,
    _include_dirs,
    _kernel_source,
    _make_extern,
    _portable_flags,
    _runtime_lib_include,
    dtypes,
)
from .activation import _bf16_ulp, _vtanh_error
from .linalg import _ZeroInitializedKernel
from .quant import _bf16_floor

_BF16, _F32 = np.dtype[bfloat16], np.dtype[np.float32]
# An RTP buffer: int32 words the host writes.
_RTP_WORD = np.dtype[np.int32]
_K = TypeVar("_K", bound=ExternalFunction)


class _PrefillKernel(ExternalFunction):
    attn_rounds: Kernel
    attn_round_begin: Kernel
    attn_blocks: Kernel
    attn_block_begin: Kernel
    attn_qk_step: Kernel
    attn_block_mid: Kernel
    attn_fv_step: Kernel
    attn_block_end: Kernel
    attn_finalize: Kernel
    attn_epilogue: ExternalFunction


def _prefill_sibling(symbol, head_dim, contract) -> ExternalFunction:
    """``symbol`` of a prefill build as a kernel of its own, judged by ``contract``."""
    if head_dim not in (256, 512):
        raise ValueError(f"head_dim must be 256 or 512, not {head_dim}")
    base = flm_gemma4_swa_prefill() if head_dim == 256 else flm_gemma4_attn_prefill()
    return _on_object(base, symbol, getattr(base, symbol).arg_types(), contract)


def _on_object(base, symbol, arg_types, contract) -> ExternalFunction:
    """``symbol`` of ``base``'s object, judged by ``contract``.

    It names ``base``'s object and compile recipe, so a design that uses both
    compiles and links one copy of the source.
    """
    return ExternalFunction(
        symbol,
        object_file_name=base.object_file_name,
        source_file=base.source_file,
        arg_types=arg_types,
        include_dirs=base.include_dirs,
        compile_flags=base.compile_flags,
        symbol_prefix=base.object_file.symbol_prefix,
        contract=contract,
    )


def _prefill(
    name, dh, lq, chunk, reference, in_prod_lock, in_cons_lock
) -> ExternalFunction:
    """One ``flm_gemma4/prefill.cc`` build, its epilogue bound to ``chunk``.

    A core holds ``lq`` query rows of two cores' q object and folds in ``lq``
    key rows a step, 128 a block.
    """
    if not _arch_traits().bfp16:
        raise NotImplementedError(f"{name}() is only available on aie2p.")
    L = np.ndarray[(8,), _RTP_WORD]
    row_bf16, row_f32 = np.ndarray[(lq,), _BF16], np.ndarray[(lq,), _F32]
    y = np.ndarray[(lq, dh), _F32]
    s = np.ndarray[(lq, 128), _BF16]
    m = np.ndarray[(lq, lq), _BF16]
    kv = np.ndarray[(lq, dh), _BF16]
    fn = _make_extern(
        "attn_epilogue",
        _kernel_source("flm_gemma4/prefill.cc"),
        [np.ndarray[(64,), _BF16], row_bf16, y, np.int32],
        cls=_PrefillKernel,
        compile_flags=[
            # The bf16 mmul lowers onto two bfp16-emulated macs on AIE2P.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            f"-DFLM_GEMMA4_PREFILL_HEAD_DIM={dh}",
            f"-DFLM_GEMMA4_PREFILL_IN_PROD_LOCK={int(in_prod_lock)}",
            f"-DFLM_GEMMA4_PREFILL_IN_CONS_LOCK={int(in_cons_lock)}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, In, In, Param),
            parameter_bindings=((3, chunk),),
            reference=reference,
            tolerance=Tolerance.exact(
                note="two floor bf16 narrowings, as a fresh core rounds"
            ),
            ops_per_call=64,
        ),
    )
    bind = fn.object_file.bind
    q = np.ndarray[(2 * lq, dh), _BF16]
    fn.attn_epilogue = fn
    fn.attn_rounds = bind("attn_rounds", [L, L, L])
    fn.attn_round_begin = bind(
        "attn_round_begin", [row_bf16, row_bf16, row_f32, row_f32, y]
    )
    # The sliding-window build also reads the window size.
    blocks = [L, L, np.int32, L] if dh == 256 else [L, np.int32, L]
    fn.attn_blocks = bind("attn_blocks", blocks)
    fn.attn_block_begin = bind("attn_block_begin", [m, row_bf16])
    fn.attn_qk_step = bind("attn_qk_step", [s, q, kv, kv, m, L, L] + [np.int32] * 5)
    fn.attn_block_mid = bind(
        "attn_block_mid", [s, m, row_bf16, row_bf16, row_f32, row_f32, y]
    )
    fn.attn_fv_step = bind("attn_fv_step", [y, s, kv, kv, np.int32])
    fn.attn_block_end = bind("attn_block_end", [row_bf16, row_bf16])
    fn.attn_finalize = bind("attn_finalize", [row_f32, row_bf16])
    return fn


# The global geometry: query rows per core, head dim.
_ATTN_PREFILL_LQ, _ATTN_PREFILL_DH = 8, 512


def flm_gemma4_attn_prefill(
    *, in_prod_lock: int = 2, in_cons_lock: int = 3
) -> ExternalFunction:
    """Causal flash-attention prefill at a head dim of 512, from ``flm_gemma4/prefill.cc``.

    [`linalg.prefill_fv`][iron.kernels.linalg.prefill_fv]'s algorithm with
    FastFlowLM's numerics and synchronization: the softmax scales by log2(e)
    alone, the epilogue multiplies by a bf16 ``1 / l``, and k and v arrive in
    a ping-pong pair that the core's own locks guard.

    One core holds 8 query rows of a 128-row round and folds the keys in one
    8-row step at a time, with one entry point per step of the caller's
    round, block and step loops. The returned kernel is ``attn_epilogue``,
    which writes one 64-element chunk of ``o = y / l``; its chunk index is
    bound to 0. The entry points, all attributes of the kernel:

    - ``attn_rounds(L_begin, L_end, n_out)``: rounds in the dispatch
    - ``attn_blocks(L_begin, i, n_out)``: key blocks in round ``i``
    - ``attn_round_begin(prev_m, new_m, c, l, y)``
    - ``attn_block_begin(m, prev_m)``
    - ``attn_qk_step(s, q, k_ping, k_pong, m, L_begin, window_size, row, col, i, block, j)``
    - ``attn_block_mid(s, m, new_m, prev_m, c, l, y)``
    - ``attn_fv_step(y, s, v_ping, v_pong, j)``
    - ``attn_block_end(prev_m, new_m)``
    - ``attn_finalize(l, l_bf16)``
    - ``attn_epilogue(o, l_bf16, y, chunk)``

    The design must declare the k/v locks and fill the pair with a tile DMA.

    Args:
        in_prod_lock: Core lock the k/v DMA acquires to fill a buffer.
        in_cons_lock: Core lock the steps acquire to read one.
    """
    return _prefill(
        "flm_gemma4_attn_prefill",
        _ATTN_PREFILL_DH,
        _ATTN_PREFILL_LQ,
        0,
        flm_gemma4_attn_prefill_ref,
        in_prod_lock,
        in_cons_lock,
    )


# The sliding-window geometry, and the output chunk its case writes: the
# second 8-row group, second 64-column slice, so both halves of the chunk
# index reach the reference.
_SWA_PREFILL_LQ, _SWA_PREFILL_DH, _SWA_PREFILL_CHUNK = 16, 256, 33


def flm_gemma4_swa_prefill(
    *, in_prod_lock: int = 2, in_cons_lock: int = 3
) -> ExternalFunction:
    """Sliding-window causal flash-attention prefill at a head dim of 256, from ``flm_gemma4/prefill.cc``.

    The sliding-window build of [`flm_gemma4_attn_prefill`][iron.kernels.flm_gemma4.flm_gemma4_attn_prefill]:
    one core holds 16 query rows of a 128-row round and folds in, 16 rows at
    a time, the keys inside the window its ``window_size`` RTP gives. The
    entry points are those of ``flm_gemma4_attn_prefill``, with
    ``attn_blocks(L_begin, window_size, i, n_out)`` also reading the window.
    The returned kernel is ``attn_epilogue``, bound to chunk 33: rows 8 to 15,
    columns 64 to 127 of the (16, 256) ``y``.

    Args:
        in_prod_lock: Core lock the k/v DMA acquires to fill a buffer.
        in_cons_lock: Core lock the steps acquire to read one.
    """
    return _prefill(
        "flm_gemma4_swa_prefill",
        _SWA_PREFILL_DH,
        _SWA_PREFILL_LQ,
        _SWA_PREFILL_CHUNK,
        flm_gemma4_swa_prefill_ref,
        in_prod_lock,
        in_cons_lock,
    )


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_block_begin(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_block_begin`` of the prefill kernels: row ``j`` of ``m`` is ``prev_m[j]``."""
    return _prefill_sibling(
        "attn_block_begin",
        head_dim,
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, In),
            reference=flm_gemma4_prefill_block_begin_ref,
            tolerance=Tolerance.exact(note="bf16 broadcast"),
            ops_per_call=0,
        ),
    )


def flm_gemma4_prefill_block_begin_ref(prev_m):
    """Numpy reference for [`flm_gemma4_prefill_block_begin`][iron.kernels.flm_gemma4.flm_gemma4_prefill_block_begin].

    ``m`` is ``(LQ, LK)`` with ``LK == LQ`` in both builds.
    """
    prev_m = np.asarray(prev_m)
    return np.repeat(prev_m, prev_m.shape[-1], axis=-1)


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_block_end(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_block_end`` of the prefill kernels: ``prev_m = new_m``."""
    return _prefill_sibling(
        "attn_block_end",
        head_dim,
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, In),
            reference=flm_gemma4_prefill_block_end_ref,
            tolerance=Tolerance.exact(note="bf16 copy"),
            ops_per_call=0,
        ),
    )


def flm_gemma4_prefill_block_end_ref(new_m):
    """Numpy reference for [`flm_gemma4_prefill_block_end`][iron.kernels.flm_gemma4.flm_gemma4_prefill_block_end]."""
    return np.array(new_m, copy=True)


def _rtp_words(rng, word0):
    """Return ``(calls, 8)`` RTP buffers: ``word0`` in word 0, junk the kernel ignores after it."""
    words = rng.integers(-(1 << 31), 1 << 31, size=(len(word0), 8), dtype=np.int64)
    words[:, 0] = word0
    return words.astype(np.int32)


def _word0(buffer):
    return np.asarray(buffer).reshape(-1, 8)[:, 0].astype(np.int32)


def flm_gemma4_prefill_rounds_ref(l_begin, l_end):
    """``(L_end >> 7) - (L_begin >> 7)``: the 128-row rounds a dispatch spans."""
    n = (_word0(l_end) >> 7) - (_word0(l_begin) >> 7)
    return n.reshape(-1, 1)


def _prefill_rounds_sample(rng, calls):
    # Dispatch bounds, some round-aligned, the first an empty one.
    begin = rng.integers(0, 1 << 15, size=calls)
    begin[::2] &= ~127
    end = begin + rng.integers(0, 1 << 13, size=calls)
    end[0] = begin[0]
    return [_rtp_words(rng, begin), _rtp_words(rng, end)]


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_rounds(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_rounds`` of the prefill kernels: the rounds between ``L_begin`` and ``L_end``."""
    return _prefill_sibling(
        "attn_rounds",
        head_dim,
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            sample=_prefill_rounds_sample,
            reference=flm_gemma4_prefill_rounds_ref,
            tolerance=Tolerance.exact(note="integer shifts and a subtraction"),
            out_valid=1,
            ops_per_call=1,
        ),
    )


def flm_gemma4_prefill_blocks_ref(l_begin, i, *, window_size=None):
    """Key blocks round ``i`` folds in: all up to its own, or those in the window.

    ``window_size`` is the sliding-window build's RTP buffer, ``None`` for the
    head-dim-512 build.
    """
    begin = _word0(l_begin)
    if window_size is None:
        n = (begin >> 7) + np.int32(i) + 1
    else:
        q = begin + np.int32(i) * 128
        k = np.maximum(q - _word0(window_size), 0)
        n = ((q - k) >> 7) + 1
    return n.astype(np.int32).reshape(-1, 1)


def _prefill_swa_blocks_ref(l_begin, window_size, i):
    return flm_gemma4_prefill_blocks_ref(l_begin, i, window_size=window_size)


def _prefill_blocks_sample(rng, calls, *, window):
    # Positions below the window clamp the first key block to 0: the first
    # call starts at 0, the second past any window.
    begin = rng.integers(0, 1 << 12, size=calls)
    begin[0] = 0
    if calls > 1:
        begin[1] = rng.integers(1 << 12, 1 << 15)
    begin[::2] &= ~127
    buffers = [_rtp_words(rng, begin)]
    if window:
        buffers.append(_rtp_words(rng, np.resize([512, 1024], calls)))
    return buffers


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_blocks(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_blocks`` of the prefill kernels: the key blocks round ``i`` folds in.

    At a head dim of 256, the sliding-window build, it also reads the window
    size: ``attn_blocks(L_begin, window_size, i, n_out)``.
    """
    window = head_dim == 256
    return _prefill_sibling(
        "attn_blocks",
        head_dim,
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Param, Out) if window else (In, Param, Out),
            sample=partial(_prefill_blocks_sample, window=window),
            reference=(
                _prefill_swa_blocks_ref if window else flm_gemma4_prefill_blocks_ref
            ),
            tolerance=Tolerance.exact(note="integer shifts, adds and a clamp"),
            out_valid=1,
            ops_per_call=1,
        ),
    )


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_finalize(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_finalize`` of the prefill kernels: ``inv_l = bf16(1 / l)`` per query row.

    Args:
        head_dim: 512 builds the global kernel's 8 rows; 256 builds the
            sliding-window kernel's 16 rows.
    """
    return _prefill_sibling(
        "attn_finalize",
        head_dim,
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=flm_gemma4_prefill_finalize_ref,
            sample=partial(
                _prefill_finalize_sample,
                lq=_ATTN_PREFILL_LQ if head_dim == 512 else _SWA_PREFILL_LQ,
            ),
            # Floor narrowing lands at most one bf16 step from the nearest-even
            # cast of 1 / l if aie::inv errs by less than half a step (2**-9
            # relative).
            tolerance=Tolerance.bf16_ulps(
                1,
                note="floor bf16 narrowing of AIE2P's scalar inv instruction, "
                "whose error the AIE-ML v2 intrinsics guide does not document",
            ),
        ),
    )


def flm_gemma4_prefill_finalize_ref(row_sums):
    """Numpy reference for [`flm_gemma4_prefill_finalize`][iron.kernels.flm_gemma4.flm_gemma4_prefill_finalize]: ``1 / l``.

    The tolerance covers the kernel's hardware reciprocal and its floor
    narrowing to bf16.
    """
    return 1.0 / np.asarray(row_sums, dtype=np.float32)


def _prefill_finalize_sample(rng, calls, *, lq):
    # Softmax row sums: at least 1, since the row maximum adds exp2(0), and at
    # most a few hundred keys' worth.
    return [rng.uniform(1.0, 300.0, (calls, lq)).astype(np.float32)]


# The prefill arithmetic behind attn_qk_step's and attn_fv_step's locks and
# attn_block_mid's state, built from flm_gemma4/prefill_core.cc.

# FastFlowLM's softmax scale: log2(e) in bf16, with no 1 / sqrt(head_dim).
_LOG2E_BF16 = float(bfloat16(np.log2(np.e)))
# prefill.cc's NEG_INF: the most negative finite bf16.
_PREFILL_NEG_INF = float.fromhex("-0x1.FEp127")
# aie::exp2<bfloat16> overshoots 2**x by up to 6.98% with its bf16 roundings
# (linalg.mha_softmax, test_mha_e2e.py). The reference's own cast to bf16
# adds half a step, 2**-9.
_PREFILL_EXP2_RTOL = 0.075
# AIE2P has no float32 multiplier. aie::mul on floats runs Peano's
# accuracy_safe product of bf16 limbs; this bound leaves a wide margin over it.
_PREFILL_F32_MUL_RTOL = 2.0**-16


def _prefill_lq(head_dim):
    if head_dim not in (256, 512):
        raise ValueError(f"head_dim must be 256 or 512, not {head_dim}")
    return _ATTN_PREFILL_LQ if head_dim == 512 else _SWA_PREFILL_LQ


def _prefill_core(symbol, head_dim, arg_types, contract) -> ExternalFunction:
    """``symbol`` of ``flm_gemma4/prefill_core.cc``, built with ``head_dim``'s prefill flags."""
    _prefill_lq(head_dim)
    base = flm_gemma4_swa_prefill() if head_dim == 256 else flm_gemma4_attn_prefill()
    return _wrapper(base, "prefill_core.cc", symbol, arg_types, contract)


def _tiles(x, rows, cols, *, col_major=False):
    """Return ``(calls, rows, cols)`` from 8x8 row-major tiles, in row- or column-major tile order."""
    t = np.asarray(x, np.float64).reshape(-1, rows // 8, cols // 8, 8, 8)
    if col_major:
        t = t.reshape(-1, cols // 8, rows // 8, 8, 8).transpose(0, 2, 1, 3, 4)
    return t.transpose(0, 1, 3, 2, 4).reshape(-1, rows, cols)


def _to_tiles(x, *, col_major=False):
    """Invert ``_tiles``, flattened per call."""
    calls, rows, cols = x.shape
    t = x.reshape(calls, rows // 8, 8, cols // 8, 8).transpose(0, 1, 3, 2, 4)
    if col_major:
        t = t.transpose(0, 2, 1, 3, 4)
    return t.reshape(calls, -1)


def _bf16_rounding_error(x):
    """0 where ``x`` is a bf16 value, else one bf16 step at ``x``."""
    x = np.asarray(x, np.float64)
    exact = x.astype(np.float32).astype(bfloat16).astype(np.float64) == x
    return np.where(exact, 0.0, np.ldexp(1.0, np.frexp(np.abs(x))[1] - 8))


def _bfp16_error(x):
    """Bound on the bf16 -> bfp16ebs8 conversion error, blocks of 8 along the last axis.

    0 on the block's grid of 2**(E - 6), E the exponent of its largest value;
    elsewhere two grid steps, which also covers a rounding carry that raises E.
    """
    x = np.asarray(x, np.float64)
    blocks = x.reshape(*x.shape[:-1], -1, 8)
    top = np.abs(blocks).max(axis=-1, keepdims=True)
    step = np.ldexp(1.0, np.frexp(top)[1] - 7)
    on_grid = np.mod(blocks, step) == 0
    return np.where(on_grid | (top == 0), 0.0, 2 * step).reshape(x.shape)


def _bfp16_product_error(a, b):
    """Bound on ``|a @ b - Q(a) @ Q(b)|`` for the bfp16-emulated mmul, ``Q`` along the inner axis."""
    ea, eb = _bfp16_error(a), _bfp16_error(np.swapaxes(b, -1, -2))
    eb = np.swapaxes(eb, -1, -2)
    return np.abs(a) @ eb + ea @ np.abs(b) + ea @ eb


def _grid_bf16(rng, shape, step):
    """Integers -127..127 times ``step``: bf16 values on every bfp16ebs8 block's grid."""
    return (rng.integers(-127, 128, size=shape) * step).astype(bfloat16)


def flm_gemma4_prefill_round_begin_ref(*, head_dim=512):
    """Numpy reference for [`flm_gemma4_prefill_round_begin`][iron.kernels.flm_gemma4.flm_gemma4_prefill_round_begin].

    The bytes of ``y | c | l | prev_m | new_m``: zeros, ones, zeros and two
    rows of NEG_INF.
    """
    lq = _prefill_lq(head_dim)
    return np.concatenate(
        [
            np.zeros(lq * head_dim, np.float32).view(np.uint8),
            np.ones(lq, np.float32).view(np.uint8),
            np.zeros(lq, np.float32).view(np.uint8),
            np.full(2 * lq, _PREFILL_NEG_INF, bfloat16).view(np.uint8),
        ]
    )


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_round_begin(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_round_begin`` of the prefill kernels, its five outputs packed as bytes.

    ``out`` is ``y | c | l | prev_m | new_m``.
    """
    lq = _prefill_lq(head_dim)
    n = 4 * lq * head_dim + 12 * lq
    return _prefill_core(
        "attn_round_begin_core",
        head_dim,
        [np.ndarray[(n,), np.dtype[np.uint8]]],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            reference=partial(flm_gemma4_prefill_round_begin_ref, head_dim=head_dim),
            tolerance=Tolerance.exact(note="constant stores"),
            ops_per_call=0,
        ),
    )


def _prefill_keep(lq, lk, inner_k, inner_q, inner_k_current):
    """apply_mask_and_get_max's mask: key ``inner_k_current + lane`` against query ``inner_q + row``."""
    row, lane = np.indices((lq, lk))
    key = inner_k_current + lane
    return (key <= inner_q + row) & (key > inner_k + row)


def flm_gemma4_prefill_qk_core_ref(
    q, k, m, inner_k, inner_q, inner_k_current, *, head_dim=512
):
    """Numpy reference for [`flm_gemma4_prefill_qk_core`][iron.kernels.flm_gemma4.flm_gemma4_prefill_qk_core]: ``s | m``.

    ``s = q @ k.T``, NEG_INF where masked, and ``m`` folds in ``s``
    lanewise. The tolerance covers the bfp16 operands and the bf16 narrowing.
    """
    lq = lk = _prefill_lq(head_dim)
    q, k = _tiles(q, lq, head_dim), _tiles(k, lk, head_dim, col_major=True)
    s = q @ k.transpose(0, 2, 1)
    keep = _prefill_keep(lq, lk, inner_k, inner_q, inner_k_current)
    s = np.where(keep, s, _PREFILL_NEG_INF)
    m = np.maximum(np.asarray(m, np.float64).reshape(-1, lq, lk), s)
    return np.concatenate([s.reshape(len(s), -1), m.reshape(len(m), -1)], axis=1)


def _prefill_qk_bound(q, k, m, inner_k, inner_q, inner_k_current, *, head_dim):
    lq = lk = _prefill_lq(head_dim)
    q, k = _tiles(q, lq, head_dim), _tiles(k, lk, head_dim, col_major=True)
    kt = k.transpose(0, 2, 1)
    s = np.abs(q @ kt)
    # bfp16 operands; float32 accumulation over head_dim products.
    err = _bfp16_product_error(q, kt) + head_dim * 2.0**-24 * (np.abs(q) @ np.abs(kt))
    # The device's narrowing (floor or nearest) and the reference's cast.
    bound = err + 2.0**-7 * (s + err) + 2.0**-8 * s
    bound = np.where(_prefill_keep(lq, lk, inner_k, inner_q, inner_k_current), bound, 0)
    bound = bound.reshape(len(bound), -1)
    return np.concatenate([bound, bound], axis=1)


def _prefill_qk_sample(rng, calls, *, head_dim):
    # Operands on the bfp16 grid, so the mmul converts them exactly. A running
    # max near the scores' spread, so either side wins some lanes.
    lq = lk = _prefill_lq(head_dim)
    q = _grid_bf16(rng, (calls, lq, head_dim), 2.0**-6)
    k = _grid_bf16(rng, (calls, lk, head_dim), 2.0**-6)
    m = rng.normal(0.0, 20.0, (calls, lq * lk)).astype(bfloat16)
    return [
        _to_tiles(q.astype(np.float32)).astype(bfloat16),
        _to_tiles(k.astype(np.float32), col_major=True).astype(bfloat16),
        m,
    ]


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_qk_core(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_qk_step``'s arithmetic for one key chunk, without its k lock.

    ``attn_qk_core(q, k, m, inner_k, inner_q, inner_k_current, out)``:
    ``s = q @ k.T`` through the bfp16-emulated mmul, masked, and ``m``
    folded with it, into ``out = s | m``. The three positions are
    ``apply_mask_and_get_max``'s, which attn_qk_step derives from its RTPs.
    q and k arrive in the mmul's 8x8 tiles, k's tiles in column-major order.
    """
    lq = lk = _prefill_lq(head_dim)
    kv = np.ndarray[(lq * head_dim,), _BF16]
    m = np.ndarray[(lq * lk,), _BF16]
    return _prefill_core(
        "attn_qk_core",
        head_dim,
        [kv, kv, m, np.int32, np.int32, np.int32, np.ndarray[(2 * lq * lk,), _BF16]],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, In, Param, Param, Param, Out),
            sample=partial(_prefill_qk_sample, head_dim=head_dim),
            reference=partial(flm_gemma4_prefill_qk_core_ref, head_dim=head_dim),
            tolerance=Tolerance.bounded(
                partial(_prefill_qk_bound, head_dim=head_dim),
                note="bfp16ebs8 operand rounding (zero on the sample's grid), "
                "float32 accumulation, and the bf16 narrowing of s",
            ),
            ops_per_call=2 * lq * lk * head_dim,
        ),
    )


def flm_gemma4_prefill_fv_core_ref(y, sv, *, head_dim=512):
    """Numpy reference for [`flm_gemma4_prefill_fv_core`][iron.kernels.flm_gemma4.flm_gemma4_prefill_fv_core]: ``y + s @ v``.

    ``y`` and ``s`` are in the mmul's 8x8 tiles, ``v`` in column-major tile
    order; the tolerance covers the bfp16 operands.
    """
    lq = lk = _prefill_lq(head_dim)
    s, v = _prefill_sv(sv, lq, lk, head_dim)
    return _to_tiles(_tiles(y, lq, head_dim) + s @ v)


def _prefill_sv(sv, lq, lk, head_dim):
    sv = np.asarray(sv).reshape(-1, lq * lk + lk * head_dim)
    s = _tiles(sv[:, : lq * lk], lq, lk)
    return s, _tiles(sv[:, lq * lk :], lk, head_dim, col_major=True)


def _prefill_fv_core_bound(y, sv, *, head_dim):
    lq = lk = _prefill_lq(head_dim)
    s, v = _prefill_sv(sv, lq, lk, head_dim)
    mag = np.abs(_tiles(y, lq, head_dim)) + np.abs(s) @ np.abs(v)
    # bfp16 operands; float32 accumulation of lk / 8 mmul steps into y and
    # the reference's float32 cast, far inside 2**-20.
    return _to_tiles(_bfp16_product_error(s, v) + 2.0**-20 * mag)


def _prefill_fv_sample(rng, calls, *, head_dim):
    # Softmax weights in [0, 1) and v on the bfp16 grid, so the mmul converts
    # them exactly.
    lq = lk = _prefill_lq(head_dim)
    y = rng.normal(0.0, 4.0, (calls, lq * head_dim)).astype(np.float32)
    s = rng.integers(0, 128, (calls, lq, lk)) * 2.0**-7
    v = _grid_bf16(rng, (calls, lk, head_dim), 2.0**-6).astype(np.float64)
    sv = np.concatenate([_to_tiles(s), _to_tiles(v, col_major=True)], axis=1)
    return [y, sv.astype(bfloat16)]


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_fv_core(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_fv_step``'s arithmetic for one key chunk, without its v lock.

    ``attn_fv_core(y, sv, y_out)``: ``y_out = y + s @ v`` through
    ``flm_attn_fv``'s bfp16-emulated mmul, with ``sv = s | v``. y and s are in
    the mmul's 8x8 tiles, v's tiles in column-major order.
    """
    lq = lk = _prefill_lq(head_dim)
    y = np.ndarray[(lq * head_dim,), _F32]
    return _prefill_core(
        "attn_fv_core",
        head_dim,
        [y, np.ndarray[(lq * lk + lk * head_dim,), _BF16], y],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            sample=partial(_prefill_fv_sample, head_dim=head_dim),
            reference=partial(flm_gemma4_prefill_fv_core_ref, head_dim=head_dim),
            tolerance=Tolerance.bounded(
                partial(_prefill_fv_core_bound, head_dim=head_dim),
                note="bfp16ebs8 operand rounding (zero on the sample's grid) "
                "and float32 accumulation",
            ),
            ops_per_call=2 * lq * lk * head_dim,
        ),
    )


def _prefill_block_mid_math(in_bf16, in_f32, head_dim):
    """attn_block_mid in float64: ``(p, new_m, y, l, c, d, dc)``, ``p`` as ``(calls, lq, 128)``.

    ``d`` and ``dc`` are the differences the kernel narrows to bf16 before
    its exp2: ``s - new_m`` and ``prev_m - new_m``.
    """
    lq = lk = _prefill_lq(head_dim)
    b = np.asarray(in_bf16, np.float64).reshape(-1, lq * 128 + lq * lk + lq)
    f = np.asarray(in_f32, np.float64).reshape(-1, lq * head_dim + lq)
    # s arrives as attn_qk_step stores it: (chunk, query row, lane).
    s = b[:, : lq * 128].reshape(-1, 128 // lk, lq, lk).transpose(0, 2, 1, 3)
    s = s.reshape(-1, lq, 128)
    new_m = b[:, lq * 128 : lq * 128 + lq * lk].reshape(-1, lq, lk).max(axis=2)
    prev_m = b[:, -lq:]
    d = s - new_m[..., None]
    dc = prev_m - new_m
    with np.errstate(over="ignore", under="ignore"):
        p = np.exp2(d * _LOG2E_BF16)
    c = np.exp2(dc * _LOG2E_BF16)
    y = f[:, : lq * head_dim].reshape(-1, lq // 8, head_dim // 8, 8, 8)
    y = y * c.reshape(-1, lq // 8, 1, 8, 1)
    row_sums = p.sum(axis=2) + c * f[:, lq * head_dim :]
    return p, new_m, y.reshape(len(y), -1), row_sums, c, d, dc


def _prefill_block_mid_s(p, lq):
    """``p`` as attn_block_mid leaves s: per chunk, row-major or, at lq 16, 8x8 tiles."""
    t = p.reshape(-1, lq, 128 // lq, lq).transpose(0, 2, 1, 3)
    if lq == 16:
        t = t.reshape(-1, 8, 2, 8, 2, 8).transpose(0, 1, 2, 4, 3, 5)
    return t.reshape(len(p), -1)


def flm_gemma4_prefill_block_mid_core_ref(in_bf16, in_f32, *, head_dim=512):
    """Numpy reference for [`flm_gemma4_prefill_block_mid_core`][iron.kernels.flm_gemma4.flm_gemma4_prefill_block_mid_core].

    ``(s | new_m, y | l | c)``: ``new_m`` the row max of ``m``,
    ``p = 2**((s - new_m) * log2e)`` and ``c = 2**((prev_m - new_m) * log2e)``
    with FastFlowLM's bf16 log2(e), ``l = sum(p) + c * l`` and ``y = c * y``.
    """
    lq = _prefill_lq(head_dim)
    p, new_m, y, row_sums, c, _, _ = _prefill_block_mid_math(in_bf16, in_f32, head_dim)
    return (
        np.concatenate([_prefill_block_mid_s(p, lq), new_m], axis=1),
        np.concatenate([y, row_sums, c], axis=1),
    )


def _prefill_exp2_bound(value, arg_error):
    """Bound on an exp2 output: the interpolant's error and an argument off by ``arg_error``."""
    with np.errstate(over="ignore", invalid="ignore"):
        rel = (1 + _PREFILL_EXP2_RTOL) * np.exp2(arg_error) - 1
        return np.where(value > 0, value * rel, 0.0)


def _prefill_block_mid_bound(in_bf16, in_f32, *, head_dim):
    lq = _prefill_lq(head_dim)
    p, new_m, _, _, c, d, dc = _prefill_block_mid_math(in_bf16, in_f32, head_dim)
    f = np.asarray(in_f32, np.float64).reshape(len(p), -1)
    y, l_in = np.abs(f[:, : lq * head_dim]), f[:, lq * head_dim :]
    # The kernel narrows s - new_m to bf16; prev_m - new_m to bf16, and its
    # product with log2(e) to bf16 again. aie::exp2 also narrows its float
    # argument to bf16: on npu2 its error grows with one bf16 step of the
    # argument, 2**0.25 at arguments in [-64, -32).
    with np.errstate(over="ignore"):
        arg_error = _bf16_rounding_error(d * _LOG2E_BF16)
    bp = _prefill_exp2_bound(p, _bf16_rounding_error(d) * _LOG2E_BF16 + arg_error)
    e1 = _bf16_rounding_error(dc)
    e2 = np.where(e1 == 0, _bf16_rounding_error(dc * _LOG2E_BF16), 2.0**-6 * np.abs(dc))
    bc = _prefill_exp2_bound(c, e1 * _LOG2E_BF16 + e2)
    # calculate_l sums p through the bfp16-emulated mmul: blocks of 8 keys,
    # at most two grid steps each from the largest p the device can return.
    top = (p + bp).reshape(len(p), lq, 16, 8).max(axis=3)
    quant = (16 * np.ldexp(1.0, np.frexp(top)[1] - 7) * (top > 0)).sum(axis=2)
    cl = (c + bc) * np.abs(l_in)
    bl = bp.sum(axis=2) + quant + bc * np.abs(l_in) + _PREFILL_F32_MUL_RTOL * cl
    bl += 2.0**-20 * (p.sum(axis=2) + cl)
    rows = (bc + _PREFILL_F32_MUL_RTOL * (c + bc)).reshape(-1, lq // 8, 1, 8, 1)
    by = (y.reshape(-1, lq // 8, head_dim // 8, 8, 8) * rows).reshape(len(p), -1)
    return (
        np.concatenate([_prefill_block_mid_s(bp, lq), np.zeros_like(new_m)], axis=1),
        np.concatenate([by, bl, bc], axis=1),
    )


def _prefill_block_mid_sample(rng, calls, *, head_dim):
    """State as attn_qk_step leaves it, on grids that make the kernel's bf16 differences exact.

    Scores are multiples of 1/8 in [-12, 12], so every ``s - new_m`` is a
    bf16. The last chunk masks the keys past each row's own, as the causal
    diagonal does. ``prev_m`` sits 0.5 to 4 below the scores' row max (a
    power of two, whose product with the bf16 log2(e) is a bf16) or 1 above
    it (c = 1).
    """
    lq = lk = _prefill_lq(head_dim)
    chunks = 128 // lk
    s = rng.integers(-96, 97, (calls, chunks, lq, lk)) / 8.0
    s[:, -1] = np.where(np.triu(np.ones((lq, lk), bool), 1), _PREFILL_NEG_INF, s[:, -1])
    row_max = s.max(axis=(1, 3))
    prev_m = row_max - rng.choice([-1.0, 0.5, 1.0, 2.0, 4.0], (calls, lq))
    m = np.maximum(s.max(axis=1), prev_m[..., None])
    in_bf16 = np.concatenate(
        [s.reshape(calls, -1), m.reshape(calls, -1), prev_m], axis=1
    )
    y = rng.normal(0.0, 4.0, (calls, lq * head_dim))
    l_in = rng.uniform(1.0, 300.0, (calls, lq))
    return [
        in_bf16.astype(bfloat16),
        np.concatenate([y, l_in], axis=1).astype(np.float32),
    ]


@dtypes([{"head_dim": 256}])
def flm_gemma4_prefill_block_mid_core(*, head_dim: int = 512) -> ExternalFunction:
    """``attn_block_mid`` of the prefill kernels, its state passed in and out.

    ``attn_block_mid_core(in_bf16, in_f32, out_bf16, out_f32)`` with
    ``in_bf16 = s | m | prev_m``, ``in_f32 = y | l``, ``out_bf16 = s | new_m``
    and ``out_f32 = y | l | c``: the row max, FastFlowLM's softmax, the
    correction ``c`` and the rescaled ``l`` and ``y``.
    """
    lq = lk = _prefill_lq(head_dim)
    return _prefill_core(
        "attn_block_mid_core",
        head_dim,
        [
            np.ndarray[(lq * 128 + lq * lk + lq,), _BF16],
            np.ndarray[(lq * head_dim + lq,), _F32],
            np.ndarray[(lq * 128 + lq,), _BF16],
            np.ndarray[(lq * head_dim + 2 * lq,), _F32],
        ],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Out),
            sample=partial(_prefill_block_mid_sample, head_dim=head_dim),
            reference=partial(flm_gemma4_prefill_block_mid_core_ref, head_dim=head_dim),
            tolerance=Tolerance.bounded(
                partial(_prefill_block_mid_bound, head_dim=head_dim),
                note="aie::exp2's 6.98% (linalg.mha_softmax), bf16 rounding of "
                "the exp2 arguments, in the kernel (zero on the sample's grid) "
                "and inside aie::exp2 (measured on npu2), bfp16ebs8 "
                "rounding of p in the row sums, the emulated float32 multiply",
            ),
            # Per score: subtract, scale, exp2, sum; per y element: one multiply.
            ops_per_call=4 * lq * 128 + lq * head_dim,
        ),
    )


@dataclass(frozen=True)
class FlmGemma4DecodeGeometry:
    """The model geometry the ``flm_gemma4/decode_*.cc`` kernels build for.

    The kernels are Gemma 4's fused decode layer, one core per stage, as
    FastFlowLM builds it. Every field reaches the kernels as a ``-DFLM_GEMMA4_DECODE_*``
    flag. ``FLM_GEMMA4_E2B_DECODE`` and ``FLM_GEMMA4_E4B_DECODE`` are the two
    variants' values, and the factories build only those: the kernels keep
    assumptions other values break, such as eight query heads, the GELU
    table, the q/k norm and projections that divide into whole rounds.
    """

    model_dim: int
    num_attn_heads: int
    num_kv_heads: int
    intermediate_size: int
    glu_slice: int
    pli_d: int
    dh: int
    swa_dh: int
    attn_scale: float
    swa_attn_scale: float
    pli_projection_scale: float
    pli_input_scale: float
    gelu: bool = True
    qk_norm: bool = True
    double_wide_mlp: bool = False
    name: str = field(default="custom", compare=False)

    def __str__(self) -> str:
        return self.name

    def flags(self) -> list[str]:
        """Return the ``-DFLM_GEMMA4_DECODE_*`` flags, floats as shortest round-trip literals."""
        values = {
            "MODEL_DIM": self.model_dim,
            "NUM_ATTN_HEADS": self.num_attn_heads,
            "NUM_KV_HEADS": self.num_kv_heads,
            "INTERMEDIATE_SIZE": self.intermediate_size,
            "GLU_SLICE": self.glu_slice,
            "PLI_D": self.pli_d,
            "DH": self.dh,
            "SWA_DH": self.swa_dh,
            "ATTN_SCALE": f"{float(self.attn_scale)!r}f",
            "SWA_ATTN_SCALE": f"{float(self.swa_attn_scale)!r}f",
            "PLI_PROJECTION_SCALE": f"{float(self.pli_projection_scale)!r}f",
            "PLI_INPUT_SCALE": f"{float(self.pli_input_scale)!r}f",
            "GELU": int(self.gelu),
            "QK_NORM": int(self.qk_norm),
            "DOUBLE_WIDE_MLP": int(self.double_wide_mlp),
        }
        return [f"-DFLM_GEMMA4_DECODE_{k}={v}" for k, v in values.items()]


FLM_GEMMA4_E2B_DECODE = FlmGemma4DecodeGeometry(
    model_dim=1536,
    num_attn_heads=8,
    num_kv_heads=1,
    intermediate_size=6144,
    glu_slice=1024,
    pli_d=256,
    dh=512,
    swa_dh=256,
    attn_scale=1.0,
    swa_attn_scale=1.0,
    pli_projection_scale=0.7071067811865476,
    pli_input_scale=0.02551551815399144,
    double_wide_mlp=True,
    name="gemma4_e2b",
)
FLM_GEMMA4_E4B_DECODE = FlmGemma4DecodeGeometry(
    model_dim=2560,
    num_attn_heads=8,
    num_kv_heads=2,
    intermediate_size=10240,
    glu_slice=1024,
    pli_d=256,
    dh=512,
    swa_dh=256,
    attn_scale=0.04419417382415922,
    swa_attn_scale=0.0625,
    pli_projection_scale=0.7071067811865476,
    pli_input_scale=0.01976423537605237,
    name="gemma4_e4b",
)

_DECODE_UNTIMED = (
    "blocks on core locks its design releases, so the generic harness can "
    "neither run nor time it"
)


def _decode_kernel(
    stem: str,
    symbol: str,
    arg_types: list,
    roles: tuple,
    lock_defaults: dict[str, int],
    locks: dict[str, int],
    geometry: FlmGemma4DecodeGeometry,
    trace: Trace | None = None,
    flags: tuple[str, ...] = (),
    lut: bool = False,
    cls: type[_K] = ExternalFunction,
) -> _K:
    """One ``flm_gemma4/decode_<stem>.cc`` build, returning ``symbol`` as a ``cls``.

    ``lock_defaults`` names the kernel's core locks, each set by a
    ``-DFLM_GEMMA4_DECODE_<STEM>_<NAME>`` flag; ``locks`` overrides any of
    them. ``trace`` defaults to ``Trace.none``; ``flags`` are further compile
    flags. A ``lut`` kernel reads aie_runtime_lib's tables, so it builds
    inside ``common/lut_kernel.cc``, which defines them.
    """
    unknown = set(locks) - set(lock_defaults)
    if unknown:
        raise TypeError(
            f"flm_gemma4_decode_{stem}: unknown lock(s) {sorted(unknown)}; "
            f"its locks are {sorted(lock_defaults)}"
        )
    if not _arch_traits().bfp16:
        raise NotImplementedError(
            f"flm_gemma4_decode_{stem}() is only available on aie2p."
        )
    if geometry not in (FLM_GEMMA4_E2B_DECODE, FLM_GEMMA4_E4B_DECODE):
        raise ValueError(
            f"flm_gemma4_decode_{stem}: builds only FLM_GEMMA4_E2B_DECODE and "
            "FLM_GEMMA4_E4B_DECODE, not a custom geometry"
        )
    lock_flags = [
        f"-DFLM_GEMMA4_DECODE_{stem.upper()}_{name.upper()}={int(value)}"
        for name, value in {**lock_defaults, **locks}.items()
    ]
    source = _kernel_source(f"flm_gemma4/decode_{stem}.cc")
    include_dirs = None
    if lut:
        flags = (*flags, f'-DAIE_LUT_KERNEL_SOURCE="{source}"')
        include_dirs = [*_include_dirs(), str(source.parent), _runtime_lib_include()]
        source = _kernel_source("common/lut_kernel.cc")
    fn = _make_extern(
        symbol,
        source,
        arg_types,
        cls=cls,
        include_dirs=include_dirs,
        compile_flags=[
            *geometry.flags(),
            # The bf16 mmul lowers onto two bfp16-emulated macs on AIE2P.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            *lock_flags,
            *flags,
        ],
        contract=KernelContract(
            trace=trace or Trace.none(_DECODE_UNTIMED),
            roles=roles,
            unsupported=_DECODE_UNTIMED,
        ),
    )
    # IRON's designs look up every entry point by its symbol, this one too.
    setattr(fn, symbol, fn)
    return fn


def _wrapper(base: ExternalFunction, source: str, symbol, arg_types, contract):
    """``symbol`` of ``flm_gemma4/<source>``, a lock-free wrapper that includes ``base``'s source.

    It builds with ``base``'s flags and include path, so the wrapper sees the
    same geometry and lock ids, and is judged by ``contract``.
    """
    path = str(_kernel_source(f"flm_gemma4/{source}"))
    flags = [f for f in base.compile_flags if f not in _portable_flags()]
    lut = [i for i, f in enumerate(flags) if f.startswith("-DAIE_LUT_KERNEL_SOURCE=")]
    if lut:
        flags[lut[0]] = f'-DAIE_LUT_KERNEL_SOURCE="{path}"'
        path = str(_kernel_source("common/lut_kernel.cc"))
    # Only the wrapper is built: base was constructed for its recipe.
    ExternalFunction._instances.discard(base)
    return _make_extern(
        symbol,
        path,
        arg_types,
        include_dirs=base.include_dirs,
        compile_flags=flags,
        contract=contract,
    )


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_glu(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the decode layer's gated linear unit, from ``flm_gemma4/decode_glu.cc``.

    ``glu(y, x_ping, x_pong, y_ping, y_pong, skip)``: waits on the RTP lock,
    then, unless ``skip[0]`` is set, turns ``2 * intermediate_size`` gate and
    up values, arriving ``glu_slice`` at a time in the ``x`` ping-pong pair,
    into ``intermediate_size`` activations in the core-local ``y`` (twice that
    with ``double_wide_mlp``), and sends them out ``glu_slice // 2`` at a time
    through the ``y`` pair once per down-projection repeat. ``skip`` is an
    int32 RTP buffer of 16 words.

    Args:
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults ``x_prod_lock=0``,
            ``x_cons_lock=1``, ``y_prod_lock=2``, ``y_cons_lock=3`` and
            ``y_cons_lock=3``.
    """
    bf16 = np.dtype[bfloat16]
    hid = geometry.intermediate_size * (2 if geometry.double_wide_mlp else 1)
    x_ty = np.ndarray[(geometry.glu_slice,), bf16]
    y_ty = np.ndarray[(geometry.glu_slice // 2,), bf16]
    return _decode_kernel(
        "glu",
        "glu",
        [
            np.ndarray[(hid,), bf16],
            x_ty,
            x_ty,
            y_ty,
            y_ty,
            np.ndarray[(16,), np.dtype[np.int32]],
        ],
        (Out, In, In, Out, Out, Param),
        dict(x_prod_lock=0, x_cons_lock=1, y_prod_lock=2, y_cons_lock=3),
        locks,
        geometry,
        lut=True,
    )


# getGeluBf16 reads one line, slope and offset, per 1/8-wide segment of
# [-4, 4) from aie_runtime_lib's table and clamps to the end segments, 0 and x,
# outside. Over every bf16 input those lines stay within 1.04e-3 of tanh-GELU
# (1.45e-3 if an input within 2**-7 of a segment boundary indexes its
# neighbour), measured on the host from the table; rounded up.
_GELU_LUT_ERROR = 2e-3
# The table's largest slope.
_GELU_LUT_MAX_SLOPE = 1.13


def _gelu_tanh(x):
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _gelu_lut_bound(x):
    """Bound on getGeluBf16's bf16 result against tanh-GELU at ``x``.

    The table's lines, plus one bf16 ulp (2**-7 relative) of the slope, which
    the multiplier takes as bf16, times ``|x|`` up to the clamp at 4, plus one
    ulp for the floor store of the result.
    """
    x = np.asarray(x, np.float64)
    e = _GELU_LUT_ERROR + 2.0**-7 * _GELU_LUT_MAX_SLOPE * np.minimum(np.abs(x), 4.0)
    return e + 2.0**-7 * (np.abs(_gelu_tanh(x)) + e)


def _glu_split(x):
    x = np.asarray(x, np.float64)
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]


def flm_gemma4_glu_core_ref(x):
    """Numpy reference for [`flm_gemma4_glu_core`][iron.kernels.flm_gemma4.flm_gemma4_glu_core]: ``gelu(gate) * up``.

    ``x`` holds, per call, the up half then the gate half. GELU is the tanh
    form Gemma 4 uses; the tolerance covers the kernel's table.
    """
    up, gate = _glu_split(x)
    return (_gelu_tanh(gate) * up).astype(np.float32)


def _glu_core_bound(x):
    up, gate = _glu_split(x)
    g, d = np.abs(_gelu_tanh(gate)), _gelu_lut_bound(gate)
    # The product of two bf16 values is exact in float32; its floor store adds
    # one ulp, and the reference's float32 result 2**-24 relative.
    return np.abs(up) * (d + 2.0**-7 * (g + d)) + 2.0**-23 * np.abs(up) * g


def _glu_core_sample(rng, calls, *, n):
    # Gate values reach both clamps of the table; up values are signed.
    up = rng.uniform(-4.0, 4.0, (calls, n // 2))
    gate = rng.uniform(-6.0, 6.0, (calls, n // 2))
    return [np.concatenate([up, gate], axis=1).astype(bfloat16)]


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_glu_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE
) -> ExternalFunction:
    """``pseduo_glu`` of [`flm_gemma4_decode_glu`][iron.kernels.flm_gemma4.flm_gemma4_decode_glu]: ``gelu(gate) * up`` for one slice.

    ``glu_core(x, y)``: ``x`` is ``glu_slice`` bf16, the up half then the
    gate half; ``y`` is ``glu_slice // 2`` bf16. GELU goes through
    aie_runtime_lib's table (``getGeluBf16``).

    Args:
        geometry: The model the kernel builds for.
    """
    n = geometry.glu_slice
    return _wrapper(
        flm_gemma4_decode_glu(geometry=geometry),
        "decode_glu_core.cc",
        "glu_core",
        [np.ndarray[(n,), _BF16], np.ndarray[(n // 2,), _BF16]],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=flm_gemma4_glu_core_ref,
            sample=partial(_glu_core_sample, n=n),
            tolerance=Tolerance.bounded(
                _glu_core_bound,
                note="the GELU table's error against tanh-GELU, measured on the "
                "host from aie_runtime_lib's table, plus a bf16 ulp of its slope "
                "and the floor bf16 stores of the GELU and the product",
            ),
            uses_lut=True,
            # Per output: the table line's multiply and add, then the product.
            ops_per_call=3 * (n // 2),
        ),
    )


# Keys the attention kernels fold in per round (FastFlowLM's lq and lk).
_DECODE_KEYS_PER_ROUND = 16
# One q4nx weight block: 32 rows by 256 columns at 5 bits a weight.
_DECODE_Q4NX_ROWS, _DECODE_Q4NX_COLS = 32, 256
_DECODE_Q4NX_BLOCK_BYTES = _DECODE_Q4NX_ROWS * _DECODE_Q4NX_COLS * 5 // 8
# One bf16 weight block of the per-layer-input projections: 32 rows by 256.
_DECODE_BF16_BLOCK = 32 * 256
_DECODE_RTP = np.ndarray[(16,), np.dtype[np.int32]]
_DECODE_KV_LOCKS = {}
_DECODE_ROPE_LOCKS = dict(
    qkv_prod_lock=0,
    qkv_cons_lock=1,
    k_prod_lock=4,
    k_cons_lock=5,
    v_prod_lock=6,
    v_cons_lock=7,
)


def _decode_kv_heads(
    name: str, geometry: FlmGemma4DecodeGeometry, kv_heads: int
) -> None:
    # The source compiles to an object without entry points for another count.
    if geometry.num_kv_heads != kv_heads:
        raise ValueError(
            f"{name}: builds for {kv_heads} KV head(s), the geometry has "
            f"{geometry.num_kv_heads}"
        )


def _decode_q_heads_padded(geometry: FlmGemma4DecodeGeometry) -> int:
    """Query heads per attention core, each KV head's group padded as decode_geometry.h does."""
    group = geometry.num_attn_heads // geometry.num_kv_heads
    segment = 8 if geometry.num_kv_heads == 1 else 4
    return geometry.num_kv_heads * (-(-group // segment) * segment)


def _decode_attn_types(geometry, dh, kv_width):
    """Return an attention core pair's buffer types.

    The head dim is ``dh``, and a k or v row is ``kv_width`` wide.
    """
    q_heads = _decode_q_heads_padded(geometry)
    return dict(
        q=np.ndarray[(q_heads * dh,), _BF16],
        kv=np.ndarray[(_DECODE_KEYS_PER_ROUND, kv_width), _BF16],
        # A round's scores, then its eight float32 corrections.
        s=np.ndarray[(_DECODE_KEYS_PER_ROUND * q_heads + 32,), _BF16],
        m=np.ndarray[(16,), _BF16],
        c=np.ndarray[(8,), _F32],
        y=np.ndarray[(geometry.num_attn_heads * dh,), _F32],
        o=np.ndarray[(geometry.num_attn_heads * dh,), _BF16],
        l=np.ndarray[(8,), _F32],
    )


class _AttnKvKernel(ExternalFunction):
    attn_kv_round: Kernel
    attn_kv_finish: Kernel


class _AttnKvKvh2Kernel(ExternalFunction):
    attn_kv_s_begin: Kernel
    attn_kv_v_half: Kernel
    attn_kv_finish: Kernel


class _SwaAttnKvKernel(ExternalFunction):
    swa_attn_kv_round: Kernel
    swa_attn_kv_finish: Kernel


class _AttnQkKernel(ExternalFunction):
    attn_qk_round: Kernel


class _AttnQkKvh2Kernel(ExternalFunction):
    attn_qk_half: Kernel
    attn_qk_store_c: Kernel


def flm_gemma4_decode_attn_kv(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the global-attention score-times-value core for one KV head, from ``flm_gemma4/decode_attn_kv.cc``.

    The decode layer's global attention runs on two cores. This one folds
    each round's scores and running-max corrections from the qk core into
    the softmax denominator ``l`` and the float32 accumulator ``y``, and at
    the end writes ``o = y / l``. The returned kernel is ``attn_kv_begin``,
    which zeroes ``y`` and ``l`` and waits for the RTPs. Per round the
    Worker calls ``attn_kv_round``, then once ``attn_kv_finish``. ``y`` and
    ``o`` need 64-byte-aligned base pointers.

    Args:
        geometry: The model the kernel builds for; ``num_kv_heads`` must be 1.
        **locks: The kernel takes no core lock.
    """
    _decode_kv_heads("flm_gemma4_decode_attn_kv", geometry, 1)
    t = _decode_attn_types(geometry, geometry.dh, geometry.dh)
    fn = _decode_kernel(
        "attn_kv",
        "attn_kv_begin",
        [t["y"], t["l"]],
        (Out, Out),
        _DECODE_KV_LOCKS,
        locks,
        geometry,
        lut=True,
        cls=_AttnKvKernel,
    )
    bind = fn.object_file.bind
    fn.attn_kv_round = bind("attn_kv_round", [t["s"], t["kv"], t["kv"], t["y"], t["l"]])
    fn.attn_kv_finish = bind("attn_kv_finish", [t["y"], t["o"], t["l"]])
    return fn


# The kv round cores' state buffer: y, then l in a 16-float slot that keeps
# the buffer a multiple of 64 bytes. The kernel copies the slot's last 8
# floats through. Their first input is s then v, because a core has two
# input DMA channels.
_ATTN_KV_L_SLOT = 16
_ATTN_KV_S = _DECODE_KEYS_PER_ROUND * 8 + 32
_ATTN_KV_ROUND_TOLERANCE = (
    "The bf16 mmul runs as a bfp16 mac: each operand rounds to an 8-bit "
    "mantissa within 2**-6 of the maximum of its 8-key block, so s * v errs "
    "by up to 2**-6 (|s|max(v) + max(s)|v|) + 2**-12 max(s)max(v). l's row sum "
    "narrows to bf16, within 2**-7 of it. 2**-20 relative covers the float32 "
    "multiplies and adds"
)


def _attn_kv_ly(dh):
    return np.ndarray[(8 * dh + _ATTN_KV_L_SLOT,), _F32]


def _attn_kv_round_terms(sv, ly, kv_head, *, dh, impl):
    """Return a round's scores, c, y, l and the v rows each query head reads.

    The scores are ``(calls, 8 heads, 16 keys)``, y ``(calls, 8, dh)`` and the
    v rows ``(calls, 8, 16, dh)``. ``impl`` names the ``attn_fv`` whose v
    layout to read: ``"1x8x1"`` one KV head, ``"2x4x1"`` two KV heads in one
    buffer, ``"kvh2"`` KV head ``kv_head`` of two, the other heads reading
    zeros.
    """
    sv, ly = np.asarray(sv), np.asarray(ly, dtype=np.float32)
    s, v = sv[:, :_ATTN_KV_S], sv[:, _ATTN_KV_S:].astype(np.float64)
    n, col_q = len(s), dh // 8
    scores = s[:, :128].astype(np.float64).reshape(n, 8, 16)
    c = np.ascontiguousarray(s[:, 128:144]).view(np.float32).astype(np.float64)
    y = ly[:, : 8 * dh].reshape(n, col_q, 8, 8).transpose(0, 2, 1, 3)
    denom = ly[:, 8 * dh : 8 * dh + 8].astype(np.float64)
    if impl == "1x8x1":
        # v is (key half, column block, key, column).
        one = v.reshape(n, 2, col_q, 8, 8).transpose(0, 1, 3, 2, 4)
        rows = np.broadcast_to(one.reshape(n, 1, 16, dh), (n, 8, 16, dh))
    elif impl == "2x4x1":
        # v is (key half, KV head, column block, key, column).
        two = v.reshape(n, 2, 2, col_q, 8, 8).transpose(0, 2, 1, 4, 3, 5)
        rows = np.repeat(two.reshape(n, 2, 16, dh), 4, axis=1)
    else:
        # v is (column block, key half, key, column).
        one = v.reshape(n, col_q, 2, 8, 8).transpose(0, 2, 3, 1, 4)
        rows = np.zeros((n, 8, 16, dh))
        rows[:, 4 * kv_head : 4 * kv_head + 4] = one.reshape(n, 1, 16, dh)
    return scores, c, y.reshape(n, 8, dh).astype(np.float64), denom, rows


def _attn_kv_pack(y, denom, tail):
    n, _, dh = y.shape
    y = y.reshape(n, 8, dh // 8, 8).transpose(0, 2, 1, 3).reshape(n, 8 * dh)
    return np.concatenate([y, denom, tail], axis=1).astype(np.float32)


def _attn_kv_round_ref(sv, ly, kv_head=0, *, dh, impl):
    scores, c, y, denom, rows = _attn_kv_round_terms(sv, ly, kv_head, dh=dh, impl=impl)
    y = c[..., None] * y + np.einsum("nhk,nhkd->nhd", scores, rows)
    denom = c * denom + scores.sum(-1)
    return _attn_kv_pack(y, denom, np.asarray(ly)[:, 8 * dh + 8 :])


def _attn_kv_round_bound(sv, ly, kv_head=0, *, dh, impl):
    scores, c, y, denom, rows = _attn_kv_round_terms(sv, ly, kv_head, dh=dh, impl=impl)
    n = len(scores)
    a, b = np.abs(scores), np.abs(rows)
    a_max = np.repeat(a.reshape(n, 8, 2, 8).max(-1), 8, axis=2)
    b_max = np.repeat(b.reshape(n, 8, 2, 8, dh).max(3), 8, axis=2)
    bfp = 2**-6 * (
        np.einsum("nhk,nhkd->nhd", a_max, b) + np.einsum("nhk,nhkd->nhd", a, b_max)
    ) + 2**-12 * np.einsum("nhk,nhkd->nhd", a_max, b_max)
    y_bound = bfp + 2**-20 * (
        np.abs(c[..., None] * y) + np.einsum("nhk,nhkd->nhd", a, b)
    )
    sums = a.sum(-1)
    l_bound = 2**-7 * sums + 2**-20 * (np.abs(c * denom) + sums)
    return _attn_kv_pack(y_bound, l_bound, np.zeros((n, _ATTN_KV_L_SLOT - 8)))


def _attn_kv_round_sample(rng, calls, *, dh, kv_width):
    # Scores are exponentials of non-positive numbers, and each correction c
    # is one too, stored as float32 in the scores' tail.
    v = rng.standard_normal((calls, _DECODE_KEYS_PER_ROUND * kv_width))
    sv = np.concatenate([np.zeros((calls, _ATTN_KV_S)), v], axis=1).astype(bfloat16)
    sv[:, :128] = rng.uniform(0.0, 1.0, (calls, 128)).astype(bfloat16)
    c = rng.uniform(0.05, 1.0, (calls, 8)).astype(np.float32)
    sv[:, 128:144] = c.view(bfloat16)
    ly = 4 * rng.standard_normal((calls, 8 * dh + _ATTN_KV_L_SLOT))
    ly[:, 8 * dh : 8 * dh + 8] = rng.uniform(1.0, 100.0, (calls, 8))
    return [sv, ly.astype(np.float32)]


def _attn_kv_round_core(base, source, symbol, *, dh, kv_width, impl, rows, ref):
    ly = _attn_kv_ly(dh)
    sv = np.ndarray[(_ATTN_KV_S + _DECODE_KEYS_PER_ROUND * kv_width,), _BF16]
    head = (np.int32,) if impl == "kvh2" else ()
    return _wrapper(
        base,
        source,
        symbol,
        [sv, ly, ly, *head],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, *(Param for _ in head)),
            reference=ref,
            sample=partial(_attn_kv_round_sample, dh=dh, kv_width=kv_width),
            tolerance=Tolerance.bounded(
                partial(_attn_kv_round_bound, dh=dh, impl=impl),
                note=_ATTN_KV_ROUND_TOLERANCE,
            ),
            # Two per multiply-add of the rows' s @ v, one per rescaled y.
            ops_per_call=2 * rows * _DECODE_KEYS_PER_ROUND * dh + 8 * dh,
        ),
    )


def flm_gemma4_attn_kv_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE
) -> ExternalFunction:
    """One ``attn_kv_round`` of [`flm_gemma4_decode_attn_kv`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_kv] without its v lock.

    ``attn_kv_round_core(sv, ly_in, ly_out)``: ``ly_out`` is ``ly_in`` with
    ``l = c * l + rowsum(s)`` and ``y = c * y + s @ v`` per query head. ``sv``
    holds ``s``, then ``v``. ``ly`` holds ``y``, then ``l`` in a 16-float slot.
    """
    base = flm_gemma4_decode_attn_kv(geometry=geometry)
    dh = geometry.dh
    return _attn_kv_round_core(
        base,
        "decode_attn_kv_core.cc",
        "attn_kv_round_core",
        dh=dh,
        kv_width=dh,
        impl="1x8x1",
        rows=8,
        ref=partial(flm_gemma4_attn_kv_core_ref, dh=dh),
    )


def flm_gemma4_attn_kv_core_ref(sv, ly, *, dh=512):
    """Numpy reference for [`flm_gemma4_attn_kv_core`][iron.kernels.flm_gemma4.flm_gemma4_attn_kv_core], in float64."""
    return _attn_kv_round_ref(sv, ly, dh=dh, impl="1x8x1")


def flm_gemma4_decode_attn_kv_kvh2(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E4B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the global-attention score-times-value core for two KV heads, from ``flm_gemma4/decode_attn_kv_kvh2.cc``.

    The two-KV-head (E4B) sibling of
    [`flm_gemma4_decode_attn_kv`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_kv]:
    the Worker runs a round as one ``attn_kv_s_begin``, which folds the
    scores into ``l`` and rescales ``y``, then one ``attn_kv_v_half`` per
    KV head.

    Args:
        geometry: The model the kernel builds for; ``num_kv_heads`` must be 2.
        **locks: The kernel takes no core lock.
    """
    _decode_kv_heads("flm_gemma4_decode_attn_kv_kvh2", geometry, 2)
    t = _decode_attn_types(geometry, geometry.dh, geometry.dh)
    fn = _decode_kernel(
        "attn_kv_kvh2",
        "attn_kv_begin",
        [t["y"], t["l"]],
        (Out, Out),
        _DECODE_KV_LOCKS,
        locks,
        geometry,
        lut=True,
        cls=_AttnKvKvh2Kernel,
    )
    bind = fn.object_file.bind
    fn.attn_kv_s_begin = bind("attn_kv_s_begin", [t["s"], t["y"], t["l"]])
    fn.attn_kv_v_half = bind(
        "attn_kv_v_half", [t["s"], t["kv"], t["kv"], t["y"], np.int32]
    )
    fn.attn_kv_finish = bind("attn_kv_finish", [t["y"], t["o"], t["l"]])
    return fn


def flm_gemma4_attn_kv_kvh2_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E4B_DECODE
) -> ExternalFunction:
    """``attn_kv_s_begin``, then one KV head's ``attn_kv_v_half``, without the v lock.

    The entry points are those of
    [`flm_gemma4_decode_attn_kv_kvh2`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_kv_kvh2].
    ``attn_kv_kvh2_round_core(sv, ly_in, ly_out, kv_head)``: ``ly_out`` is
    ``ly_in`` with ``l = c * l + rowsum(s)`` and ``y = c * y``, plus ``s @ v``
    for query heads ``4 * kv_head`` to ``4 * kv_head + 3``. ``sv`` holds
    ``s``, then ``v``. ``ly`` holds ``y``, then ``l`` in a 16-float slot.
    Both KV heads' v buffers do not fit in core memory beside ``ly``, so a
    call reads one.
    """
    base = flm_gemma4_decode_attn_kv_kvh2(geometry=geometry)
    dh = geometry.dh
    return _attn_kv_round_core(
        base,
        "decode_attn_kv_kvh2_core.cc",
        "attn_kv_kvh2_round_core",
        dh=dh,
        kv_width=dh,
        impl="kvh2",
        rows=4,
        ref=partial(flm_gemma4_attn_kv_kvh2_core_ref, dh=dh),
    )


def flm_gemma4_attn_kv_kvh2_core_ref(sv, ly, kv_head, *, dh=512):
    """Numpy reference for [`flm_gemma4_attn_kv_kvh2_core`][iron.kernels.flm_gemma4.flm_gemma4_attn_kv_kvh2_core], in float64."""
    return _attn_kv_round_ref(sv, ly, int(kv_head), dh=dh, impl="kvh2")


@dtypes(
    [
        {"sliding_window": True},
        {"sliding_window": True, "geometry": FLM_GEMMA4_E4B_DECODE},
    ]
)
def flm_gemma4_decode_attn_qk(
    *,
    sliding_window: bool = False,
    geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE,
    **locks: int,
) -> ExternalFunction:
    """Build an attention query-times-key core, from ``flm_gemma4/decode_attn_qk.cc``.

    The other half of [`flm_gemma4_decode_attn_kv`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_kv]:
    it multiplies the query heads into each round's 16 key rows, masks the
    keys past the sequence length, keeps the running row maximum ``m`` and
    hands the kv core the exponentiated scores with their corrections. The
    returned kernel is ``attn_qk_begin``, which sets ``m`` to -inf and
    releases the kv core; per round the Worker calls ``attn_qk_round``.
    The global build serves one KV head (E2B), the sliding-window build both.

    Args:
        sliding_window: Build for the sliding-window layers' head dim.
        geometry: The model the kernel builds for; without ``sliding_window``
            ``num_kv_heads`` must be 1.
        **locks: The kernel takes no core lock.
    """
    if not sliding_window:
        _decode_kv_heads("flm_gemma4_decode_attn_qk", geometry, 1)
    dh = geometry.swa_dh if sliding_window else geometry.dh
    t = _decode_attn_types(geometry, dh, geometry.num_kv_heads * dh)
    fn = _decode_kernel(
        "attn_qk",
        "attn_qk_begin",
        [t["m"]],
        (Out,),
        {},
        locks,
        geometry,
        flags=(f"-DFLM_GEMMA4_DECODE_ATTN_QK_SWA={int(sliding_window)}",),
        cls=_AttnQkKernel,
    )
    fn.attn_qk_round = fn.object_file.bind(
        "attn_qk_round",
        [t["q"], t["kv"], t["kv"], t["s"], t["m"], t["c"], np.int32, np.int32],
    )
    return fn


def flm_gemma4_decode_attn_qk_kvh2(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E4B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the global-attention query-times-key core for two KV heads, from ``flm_gemma4/decode_attn_qk_kvh2.cc``.

    The two-KV-head (E4B) sibling of
    [`flm_gemma4_decode_attn_qk`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_qk]:
    the Worker runs a round as one ``attn_qk_half`` per KV head, then
    ``attn_qk_store_c``, which copies the corrections into the scores.

    Args:
        geometry: The model the kernel builds for; ``num_kv_heads`` must be 2.
        **locks: The kernel takes no core lock.
    """
    _decode_kv_heads("flm_gemma4_decode_attn_qk_kvh2", geometry, 2)
    t = _decode_attn_types(geometry, geometry.dh, geometry.dh)
    fn = _decode_kernel(
        "attn_qk_kvh2",
        "attn_qk_begin",
        [t["m"]],
        (Out,),
        {},
        locks,
        geometry,
        cls=_AttnQkKvh2Kernel,
    )
    bind = fn.object_file.bind
    fn.attn_qk_half = bind(
        "attn_qk_half",
        [
            t["q"],
            t["kv"],
            t["kv"],
            t["s"],
            t["m"],
            t["c"],
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    fn.attn_qk_store_c = bind("attn_qk_store_c", [t["s"], t["c"]])
    return fn


# The running max attn_qk_begin starts from, bf16's lowest finite value.
_ATTN_QK_M_START = -float.fromhex("0x1.FEp127")
_ATTN_QK_EXP_FLOOR = -87.0
# Table entries are within 2**-8 of exp (measured over decode_lut_exp.h), and
# exp_ilut multiplies by exp_flut.
_ATTN_QK_TABLE_REL = (1 + 2.0**-8) ** 2 - 1
# Absolute slack for exponents near the clamp, where table entries are
# subnormal and may flush to zero.
_ATTN_QK_ATOL = 2.0**-120


def _attn_qk_parts(qm, k, iter, L0):
    """Return the scores, mask, old and new running max, ``exp`` scores and corrections.

    ``qm`` holds 8 query rows as 8x8 tiles, ``[dh / 8][row][8 dims]``, then
    the 16 running maxima. Each KV head's k object holds 16 keys as 8x8
    tiles, ``[dh / 8][key half][key][8 dims]``, the heads back to back. Query
    row ``r`` meets KV head ``r // (8 / kv_heads)``.
    """
    qm, k = np.asarray(qm, np.float64), np.asarray(k, np.float64)
    q, m_in = qm[:, :-16], qm[:, -16:]
    calls, dh = q.shape[0], q.shape[1] // 8
    kv = k.shape[1] // (_DECODE_KEYS_PER_ROUND * dh)
    qr = q.reshape(calls, dh // 8, 8, 8).transpose(0, 2, 1, 3).reshape(calls, 8, dh)
    kr = k.reshape(calls, kv, dh // 8, 2, 8, 8).transpose(0, 1, 3, 4, 2, 5)
    kr = np.repeat(kr.reshape(calls, kv, 16, dh), 8 // kv, axis=1)
    s = np.einsum("crd,crkd->crk", qr, kr)
    valid = np.arange(16) < np.int64(L0) - 16 * np.int64(iter)
    m_old = m_in[:, :8]
    m_new = np.maximum(m_old, np.where(valid, s, -np.inf).max(axis=-1))
    d = np.where(valid, s - m_new[..., None], 0.0)
    e = np.where(valid, np.exp(np.maximum(d, _ATTN_QK_EXP_FLOOR)), 0.0)
    c = np.exp(np.maximum(m_old - m_new, _ATTN_QK_EXP_FLOOR))
    return s, valid, m_in, m_new, d, e, c


def flm_gemma4_attn_qk_core_ref(qm, k, iter, L0):
    """Numpy reference for [`flm_gemma4_attn_qk_core`][iron.kernels.flm_gemma4.flm_gemma4_attn_qk_core].

    Keys ``j`` with ``j < L0 - 16 * iter`` are valid. Per query row the new
    running max is ``max(m_in, valid scores)``; ``s`` holds ``exp(score -
    max)`` for valid keys and 0 for the others, then the 16 running maxima
    (rows 8 to 15 copied from ``m_in``). The second output is ``c = exp(m_in
    - max)``. Both exponents are clamped at -87, as the kernel does.
    """
    _, _, m_in, m_new, _, e, c = _attn_qk_parts(qm, k, iter, L0)
    m_out = m_in.copy()
    m_out[:, :8] = m_new
    return np.concatenate([e.reshape(len(e), -1), m_out], axis=1), c


def _attn_qk_exp_bound(ref, d, delta, rel):
    """Bound ``|kernel - ref|`` for an exponent ``d <= 0`` off by up to ``delta``.

    The kernel clamps its exponent at -87, and its exponent is at most 0.
    """
    lo = np.exp(np.clip(d - delta, _ATTN_QK_EXP_FLOOR, 0.0)) * (1 - rel)
    hi = np.exp(np.clip(d + delta, _ATTN_QK_EXP_FLOOR, 0.0)) * (1 + rel)
    return np.maximum(hi - ref, ref - lo) + _ATTN_QK_ATOL


def _attn_qk_core_bound(qm, k, iter, L0):
    """Per-element bound for the kernel's bf16 narrowings and exp table.

    The sample data makes the bfp16-emulated mmul exact, so a score's error
    is its floor narrowing to bf16, under one ulp. A new max taken from the
    scores has the same error. The bf16 subtraction narrows once more, and
    the table's fixed-point input truncates the exponent to 2**-8. The table
    entries add ``_ATTN_QK_TABLE_REL``. The exponentiated scores narrow to
    bf16 once more; the float32 corrections do not.
    """
    s, valid, m_in, m_new, d, e, c = _attn_qk_parts(qm, k, iter, L0)
    m_old = m_in[:, :8]
    # A max kept from m_in is exact.
    u_m = np.where(m_new > m_old, _bf16_ulp(m_new), 0.0)
    u_s, u_m = _bf16_ulp(s), u_m[..., None]
    delta = u_s + u_m + _bf16_ulp(np.abs(d) + u_s + u_m) + 2.0**-8
    rel = (1 + _ATTN_QK_TABLE_REL) * (1 + 2.0**-7) - 1
    b_e = np.where(valid, _attn_qk_exp_bound(e, d, delta, rel), 0.0)
    dc, u_m = m_old - m_new, u_m[..., 0]
    delta_c = u_m + _bf16_ulp(np.abs(dc) + u_m) + 2.0**-8
    b_c = _attn_qk_exp_bound(c, dc, delta_c, _ATTN_QK_TABLE_REL)
    b_m = np.zeros((len(s), 16))
    b_m[:, :8] = u_m
    return np.concatenate([b_e.reshape(len(s), -1), b_m], axis=1), b_c


def _attn_qk_core_sample(rng, calls, *, dh, kv_heads):
    # Multiples of 2**-4 and 2**-6 with 4-bit magnitudes: every bfp16ebs8
    # block holds them exactly, and their dot products are exact in float32.
    # Scores spread over a few units, so the exponents cover the table.
    q = rng.integers(-15, 16, (calls, 8 * dh)) * 2.0**-4
    k = rng.integers(-15, 16, (calls, kv_heads * _DECODE_KEYS_PER_ROUND * dh))
    # A first round's running max, a max under the scores and one above them.
    m = rng.choice([_ATTN_QK_M_START, -1.0, 0.5, 8.0], (calls, 16))
    m += rng.integers(0, 8, (calls, 16)) * 2.0**-3
    qm = np.concatenate([q, m], axis=1).astype(bfloat16)
    return [qm, (k * 2.0**-6).astype(bfloat16)]


def _attn_qk_core_contract(dh, kv_heads, reference):
    return KernelContract(
        trace=Trace.whole_call(),
        roles=(In, In, Out, Out, Param, Param),
        sample=partial(_attn_qk_core_sample, dh=dh, kv_heads=kv_heads),
        reference=reference,
        tolerance=Tolerance.bounded(
            _attn_qk_core_bound,
            note="floor bf16 narrowings of score, max and difference, the "
            "2**-8 fixed-point exponent and the exp tables' measured 2**-8 "
            "entries; the sample keeps the bfp16-emulated mmul exact",
        ),
        ops_per_call=2 * 8 * _DECODE_KEYS_PER_ROUND * dh,
    )


def _attn_qk_core_types(geometry, dh, kv_heads):
    t = _decode_attn_types(geometry, dh, dh)
    q_heads = _decode_q_heads_padded(geometry)
    # q, then the running max.
    qm = np.ndarray[(q_heads * dh + 16,), _BF16]
    k = np.ndarray[(kv_heads * _DECODE_KEYS_PER_ROUND, dh), _BF16]
    # The scores, then the running max.
    s = np.ndarray[(_DECODE_KEYS_PER_ROUND * q_heads + 16,), _BF16]
    return [qm, k, s, t["c"], np.int32, np.int32]


@dtypes(
    [
        {"sliding_window": True},
        {"sliding_window": True, "geometry": FLM_GEMMA4_E4B_DECODE},
    ]
)
def flm_gemma4_attn_qk_core(
    *,
    sliding_window: bool = False,
    geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE,
) -> ExternalFunction:
    """One ``attn_qk_round`` of [`flm_gemma4_decode_attn_qk`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_qk], without its k lock.

    ``attn_qk_round_core(qm, k, s, c, iter, L0)`` takes the running max
    after q in ``qm`` and writes it after the scores in ``s``. ``c`` holds the
    corrections the round stores in the tail of its s object.
    """
    base = flm_gemma4_decode_attn_qk(sliding_window=sliding_window, geometry=geometry)
    dh = geometry.swa_dh if sliding_window else geometry.dh
    kv_heads = geometry.num_kv_heads
    return _wrapper(
        base,
        "decode_attn_qk_core.cc",
        "attn_qk_round_core",
        _attn_qk_core_types(geometry, dh, kv_heads),
        _attn_qk_core_contract(dh, kv_heads, flm_gemma4_attn_qk_core_ref),
    )


def flm_gemma4_attn_qk_kvh2_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E4B_DECODE
) -> ExternalFunction:
    """One round of [`flm_gemma4_decode_attn_qk_kvh2`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_qk_kvh2], without its k lock.

    ``attn_qk_kvh2_round_core(qm, k, s, c, iter, L0)`` runs
    ``attn_qk_half`` for both KV heads, whose k objects ``k`` holds back to
    back. Otherwise as
    [`flm_gemma4_attn_qk_core`][iron.kernels.flm_gemma4.flm_gemma4_attn_qk_core].
    """
    base = flm_gemma4_decode_attn_qk_kvh2(geometry=geometry)
    return _wrapper(
        base,
        "decode_attn_qk_kvh2_core.cc",
        "attn_qk_kvh2_round_core",
        _attn_qk_core_types(geometry, geometry.dh, 2),
        _attn_qk_core_contract(geometry.dh, 2, flm_gemma4_attn_qk_kvh2_core_ref),
    )


def flm_gemma4_attn_qk_kvh2_core_ref(qm, k, iter, L0):
    """Numpy reference for [`flm_gemma4_attn_qk_kvh2_core`][iron.kernels.flm_gemma4.flm_gemma4_attn_qk_kvh2_core]; see ``flm_gemma4_attn_qk_core_ref``."""
    return flm_gemma4_attn_qk_core_ref(qm, k, iter, L0)


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_swa_attn_kv(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the sliding-window-attention score-times-value core, from ``flm_gemma4/decode_swa_attn_kv.cc``.

    [`flm_gemma4_decode_attn_kv`][iron.kernels.flm_gemma4.flm_gemma4_decode_attn_kv]
    at head dim ``swa_dh``, with entry points ``swa_attn_kv_begin``,
    ``swa_attn_kv_round`` and ``swa_attn_kv_finish``. One round covers every
    KV head, so it builds for one and for two.

    Args:
        geometry: The model the kernel builds for.
        **locks: The kernel takes no core lock.
    """
    t = _decode_attn_types(
        geometry, geometry.swa_dh, geometry.num_kv_heads * geometry.swa_dh
    )
    fn = _decode_kernel(
        "swa_attn_kv",
        "swa_attn_kv_begin",
        [t["y"], t["l"]],
        (Out, Out),
        _DECODE_KV_LOCKS,
        locks,
        geometry,
        lut=True,
        cls=_SwaAttnKvKernel,
    )
    bind = fn.object_file.bind
    fn.swa_attn_kv_round = bind(
        "swa_attn_kv_round", [t["s"], t["kv"], t["kv"], t["y"], t["l"]]
    )
    fn.swa_attn_kv_finish = bind("swa_attn_kv_finish", [t["y"], t["o"], t["l"]])
    return fn


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_swa_attn_kv_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE
) -> ExternalFunction:
    """One ``swa_attn_kv_round`` of [`flm_gemma4_decode_swa_attn_kv`][iron.kernels.flm_gemma4.flm_gemma4_decode_swa_attn_kv] without its v lock.

    ``swa_attn_kv_round_core(sv, ly_in, ly_out)``: ``ly_out`` is ``ly_in``
    with ``l = c * l + rowsum(s)`` and ``y = c * y + s @ v`` per query head,
    each head reading its KV head's half of ``v`` when there are two. ``sv``
    holds ``s``, then ``v``. ``ly`` holds ``y``, then ``l`` in a 16-float
    slot.
    """
    base = flm_gemma4_decode_swa_attn_kv(geometry=geometry)
    dh, kv_heads = geometry.swa_dh, geometry.num_kv_heads
    return _attn_kv_round_core(
        base,
        "decode_swa_attn_kv_core.cc",
        "swa_attn_kv_round_core",
        dh=dh,
        kv_width=kv_heads * dh,
        impl="1x8x1" if kv_heads == 1 else "2x4x1",
        rows=8,
        ref=partial(flm_gemma4_swa_attn_kv_core_ref, dh=dh, kv_heads=kv_heads),
    )


def flm_gemma4_swa_attn_kv_core_ref(sv, ly, *, dh=256, kv_heads=1):
    """Numpy reference for [`flm_gemma4_swa_attn_kv_core`][iron.kernels.flm_gemma4.flm_gemma4_swa_attn_kv_core], in float64."""
    impl = "1x8x1" if kv_heads == 1 else "2x4x1"
    return _attn_kv_round_ref(sv, ly, dh=dh, impl=impl)


@dtypes(
    [
        {"geometry": FLM_GEMMA4_E4B_DECODE},
        {"sliding_window": True},
        {"sliding_window": True, "geometry": FLM_GEMMA4_E4B_DECODE},
    ]
)
def flm_gemma4_decode_rope(
    *,
    sliding_window: bool = False,
    geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE,
    **locks: int,
) -> ExternalFunction:
    """Build a layer's q/k norm and rotary embedding, from ``flm_gemma4/decode_rope.cc``.

    ``rope(q, k, v, qkv_ping, qkv_pong, rope_w, skip_kv)``: takes the q, k
    and v projections one head (``dh`` bf16) at a time through the
    ``qkv`` ping-pong pair, RMS-normalizes each q and k head in place against
    its weight and rotates it into ``q`` or ``k``, and RMS-normalizes each v
    head into ``v``. Unless ``skip_kv[0]`` is set, when the layer reuses
    another layer's KV cache, it hands ``k`` and ``v`` on. ``q`` is
    ``q_heads_padded * dh`` bf16, ``k`` and ``v`` ``num_kv_heads * dh``,
    ``rope_w`` the cos and sin halves then, with ``qk_norm``, the q and k
    norm weights (``3 * dh``), and ``skip_kv`` an int32 RTP buffer of 16
    words. ``dh`` is the geometry's ``swa_dh`` with ``sliding_window``, else
    its ``dh``.

    Args:
        sliding_window: Build for the sliding-window layers' head dim.
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults ``qkv_prod_lock=0``,
            ``qkv_cons_lock=1``, ``k_prod_lock=4``, ``k_cons_lock=5``,
            ``v_prod_lock=6`` and ``v_cons_lock=7``.
    """
    dh = geometry.swa_dh if sliding_window else geometry.dh
    bf16 = np.dtype[bfloat16]
    kv = geometry.num_kv_heads * dh
    qkv_ty = np.ndarray[(dh,), bf16]
    return _decode_kernel(
        "rope",
        "rope",
        [
            np.ndarray[(_decode_q_heads_padded(geometry) * dh,), bf16],
            np.ndarray[(kv,), bf16],
            np.ndarray[(kv,), bf16],
            qkv_ty,
            qkv_ty,
            np.ndarray[(dh * (3 if geometry.qk_norm else 1),), bf16],
            _DECODE_RTP,
        ],
        (Out, Out, Out, InOut, InOut, In, Param),
        _DECODE_ROPE_LOCKS,
        locks,
        geometry,
        flags=(f"-DFLM_GEMMA4_DECODE_ROPE_SWA={int(sliding_window)}",),
    )


_ROPE_CORE_VARIANTS = [
    {"geometry": FLM_GEMMA4_E4B_DECODE},
    {"sliding_window": True},
    {"sliding_window": True, "geometry": FLM_GEMMA4_E4B_DECODE},
]
# rms_scale's error: two Newton steps of the fast inverse square root, and
# a float32 sum of squares of up to 512 terms, both far below this.
_RMS_SCALE_REL = 2.0**-12
_BF16_TINY = 2.0**-126


def _rms(x):
    """Return ``1 / sqrt(mean(x^2) + 1e-6)`` per row, in float64."""
    return 1 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1e-6)


def _rope_core_terms(x, rope_w):
    """Return the rotation's two products per output, float64, each (calls, 2, dh)."""
    rope_w = np.asarray(rope_w, np.float64)
    dh = rope_w.shape[-1] // 3
    x = np.asarray(x, np.float64).reshape(len(rope_w), 2, dh)
    cos, sin = rope_w[:, None, : dh // 2], rope_w[:, None, dh // 2 : dh]
    w = rope_w[:, dh:].reshape(-1, 2, dh)
    n = x * _rms(x) * w
    n1, n2 = n[..., : dh // 2], n[..., dh // 2 :]
    a = np.concatenate([n1 * cos, n1 * sin], axis=-1)
    b = np.concatenate([-n2 * sin, n2 * cos], axis=-1)
    return a, b


def flm_gemma4_rope_core_ref(x, rope_w):
    """Numpy reference for [`flm_gemma4_rope_core`][iron.kernels.flm_gemma4.flm_gemma4_rope_core].

    Per call, ``x`` is a q head then a k head and ``rope_w`` is ``[cos |
    sin | q weight | k weight]``. Each head ``n = x * w / sqrt(mean(x^2) +
    1e-6)`` is rotated by halves: ``y1 = n1 cos - n2 sin``, ``y2 = n1 sin +
    n2 cos``, with ``n1``, ``n2`` its first and second half.
    """
    a, b = _rope_core_terms(x, rope_w)
    y = (a + b).reshape(len(a), -1)
    return y.astype(np.float32).astype(bfloat16)


def _rope_core_bound(x, rope_w):
    a, b = _rope_core_terms(x, rope_w)
    terms, y = np.abs(a) + np.abs(b), np.abs(a + b)
    eps = 2.0**-7 + _RMS_SCALE_REL
    bound = eps * (1 + 2.0**-7) * terms + 1.5 * 2.0**-7 * y + _BF16_TINY
    return bound.reshape(len(a), -1)


def _rope_core_sample(rng, calls, *, dh):
    x = rng.normal(0, rng.uniform(0.1, 10, (calls, 1)), (calls, 2 * dh))
    theta = rng.uniform(-np.pi, np.pi, (calls, dh // 2))
    w = rng.uniform(0.25, 2, (calls, 2 * dh))
    rope_w = np.concatenate([np.cos(theta), np.sin(theta), w], axis=1)
    return [x.astype(bfloat16), rope_w.astype(bfloat16)]


@dtypes(_ROPE_CORE_VARIANTS)
def flm_gemma4_rope_core(
    *,
    sliding_window: bool = False,
    geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE,
) -> ExternalFunction:
    """``_rotate_t`` of [`flm_gemma4_decode_rope`][iron.kernels.flm_gemma4.flm_gemma4_decode_rope] on one q and one k head.

    ``rope_head_core(x, rope_w, y)``: ``x`` is a q head then a k head,
    ``rope_w`` the base kernel's ``[cos | sin | q weight | k weight]``.
    The kernel normalizes ``x`` in place; the write lands in the core's input
    element, which the next DMA fill overwrites.
    """
    base = flm_gemma4_decode_rope(sliding_window=sliding_window, geometry=geometry)
    dh = geometry.swa_dh if sliding_window else geometry.dh
    heads = np.ndarray[(2 * dh,), _BF16]
    return _wrapper(
        base,
        "decode_rope_core.cc",
        "rope_head_core",
        [heads, np.ndarray[(3 * dh,), _BF16], heads],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            reference=flm_gemma4_rope_core_ref,
            sample=partial(_rope_core_sample, dh=dh),
            tolerance=Tolerance.bounded(
                _rope_core_bound,
                note="bf16 ulp <= 2**-7 |v|. The kernel floors each normalized "
                "value to bf16 (1 ulp, plus the rms scale's error, under "
                "2**-12), so each rotation product is off by that much; it "
                "floors the rotated sum to bf16 (1 ulp) and the reference "
                "rounds it to nearest (1/2 ulp)",
            ),
            # Per element: square-add and two multiplies for the norm, then
            # a multiply and a multiply-add for the rotation.
            ops_per_call=2 * 7 * dh,
        ),
    )


def flm_gemma4_v_norm_core_ref(x):
    """Numpy reference for [`flm_gemma4_v_norm_core`][iron.kernels.flm_gemma4.flm_gemma4_v_norm_core]: ``x / sqrt(mean(x^2) + 1e-6)``."""
    x = np.asarray(x, np.float64)
    return (x * _rms(x)).astype(np.float32).astype(bfloat16)


def _v_norm_core_bound(x):
    x = np.asarray(x, np.float64)
    y = np.abs(x * _rms(x))
    return (1.5 * 2.0**-7 + 2 * _RMS_SCALE_REL) * y + _BF16_TINY


def _v_norm_core_sample(rng, calls, *, dh):
    x = rng.normal(0, rng.uniform(0.1, 10, (calls, 1)), (calls, dh))
    return [x.astype(bfloat16)]


@dtypes(_ROPE_CORE_VARIANTS)
def flm_gemma4_v_norm_core(
    *,
    sliding_window: bool = False,
    geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE,
) -> ExternalFunction:
    """``rms_norm_unweighted`` of [`flm_gemma4_decode_rope`][iron.kernels.flm_gemma4.flm_gemma4_decode_rope] on one v head."""
    base = flm_gemma4_decode_rope(sliding_window=sliding_window, geometry=geometry)
    dh = geometry.swa_dh if sliding_window else geometry.dh
    head = np.ndarray[(dh,), _BF16]
    return _wrapper(
        base,
        "decode_rope_core.cc",
        "v_norm_core",
        [head, head],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=flm_gemma4_v_norm_core_ref,
            sample=partial(_v_norm_core_sample, dh=dh),
            tolerance=Tolerance.bounded(
                _v_norm_core_bound,
                note="bf16 ulp <= 2**-7 |v|: the kernel floors to bf16 (1 ulp), "
                "the reference rounds to nearest (1/2 ulp), and the rms scale "
                "is within 2**-12",
            ),
            # Per element: square-add, then one multiply.
            ops_per_call=3 * dh,
        ),
    )


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_rms_residual(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the decode layer's four RMS norms and two residual adds, from ``flm_gemma4/decode_rms_residual.cc``.

    ``rms_residual(y, x_ping, x_pong, y_final, w, x_temp_buf, is_swa, skip_kv)``:
    one layer's worth of norms. It normalizes the layer input into ``y`` for
    the QKV projection, then normalizes the attention output, adds the
    residual and normalizes the sum into ``y`` for the up/gate projection,
    and finally normalizes the MLP output and adds the second residual into
    ``y_final``. The attention and MLP outputs arrive in the ``x`` ping-pong
    pair (``model_dim`` bf16 each). ``y`` is ``model_dim + 16`` bf16, a
    16-element packet header then the row; ``w`` holds the four norm weights
    (``4 * model_dim``); ``x_temp_buf`` is ``2 * model_dim`` of scratch.
    ``is_swa`` and ``skip_kv`` are int32 RTP buffers of 16 words that select
    how many projection repeats ``y`` is released for.

    Args:
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults ``y_prod_lock=2``, ``y_cons_lock=3``,
            ``x_prod_lock=4`` and ``x_cons_lock=5``.
    """
    bf16 = np.dtype[bfloat16]
    d = geometry.model_dim
    x_ty = np.ndarray[(d,), bf16]
    return _decode_kernel(
        "rms_residual",
        "rms_residual",
        [
            np.ndarray[(d + 16,), bf16],
            x_ty,
            x_ty,
            x_ty,
            np.ndarray[(4, d), bf16],
            np.ndarray[(2, d), bf16],
            _DECODE_RTP,
            _DECODE_RTP,
        ],
        (Out, In, In, Out, In, Out, Param, Param),
        dict(
            y_prod_lock=2,
            y_cons_lock=3,
            x_prod_lock=4,
            x_cons_lock=5,
        ),
        locks,
        geometry,
    )


def _rms_residual_core_terms(x, w, residual):
    """Return the normalized ``x`` and its sum with ``residual`` in float64."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    rms = 1 / np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + 1e-6)
    n = x * w * rms
    return n, n + np.asarray(residual, np.float64)


def flm_gemma4_rms_residual_core_ref(x, w, residual):
    """Numpy reference for [`flm_gemma4_rms_residual_core`][iron.kernels.flm_gemma4.flm_gemma4_rms_residual_core].

    ``residual + x * w / sqrt(mean(x^2) + 1e-6)``, rounded to bf16.
    """
    return (
        _rms_residual_core_terms(x, w, residual)[1].astype(np.float32).astype(bfloat16)
    )


def _rms_residual_core_bound(x, w, residual):
    n, s = map(np.abs, _rms_residual_core_terms(x, w, residual))
    norm = (2.0**-7 + 2.0**-12) * (1 + 2.0**-7) * n
    return norm + (1.5 * 2.0**-7 + 2.0**-12) * s + 2 * 2.0**-126


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_rms_residual_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE
) -> ExternalFunction:
    """``rms_norm`` then ``residual_add`` of [`flm_gemma4_decode_rms_residual`][iron.kernels.flm_gemma4.flm_gemma4_decode_rms_residual].

    The post-attention norm and residual add. ``rms_residual_core(x, w,
    residual, y)``, all ``model_dim`` bf16: ``y = residual + rms_norm(x) * w``.
    """
    base = flm_gemma4_decode_rms_residual(geometry=geometry)
    row = np.ndarray[(geometry.model_dim,), _BF16]
    return _wrapper(
        base,
        "decode_rms_residual_core.cc",
        "rms_residual_core",
        [row, row, row, row],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, In, Out),
            reference=flm_gemma4_rms_residual_core_ref,
            tolerance=Tolerance.bounded(
                _rms_residual_core_bound,
                note="bf16 ulp <= 2**-7 |v|. The kernel floors the normalized "
                "value to bf16 (1 ulp, plus the fast inverse sqrt's and the "
                "float32 sum of squares' error, under 2**-12), floors the sum "
                "to bf16 (1 ulp, plus its float32 add's error, under 2**-12), "
                "and the reference rounds the sum to nearest (1/2 ulp). Each "
                "flush of a subnormal adds the smallest normal",
            ),
            # Per element: square-add and two multiplies, then the add.
            ops_per_call=5 * geometry.model_dim,
        ),
    )


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_proj_main(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """One of the decode layer's 16 q4nx projection cores, from ``flm_gemma4/decode_proj_main.cc``.

    ``proj_main(y_ping, w_ping, x_ping, y_pong, w_pong, x_pong, is_swa, skip_kv, send_x_output)``:
    runs this core's share of every projection in a layer (QKV, output,
    up/gate and down), multiplying q4nx weight blocks from the ``w``
    ping-pong pair into 256-element input slices from the ``x`` pair and
    writing each block's 32 outputs into the ``y`` pair. ``y_ping`` and
    ``y_pong`` are ``2 * 32 + 16`` bf16: a 16-element packet header, then
    two 32-output slots. With ``send_x_output`` set the core fills the first
    slot of its own pair and sends it; with it 0 the pair is the tile
    below's and the core fills the second slot. ``w_ping`` and ``w_pong`` are one q4nx
    block each (5120 bytes: 32 by 256 weights, their bf16 scales and mins,
    then the 4-bit codes). ``is_swa`` and ``skip_kv`` are int32 RTP buffers
    of 16 words that select the projection shapes.

    Args:
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults ``x_prod_lock=0``,
            ``x_cons_lock=1``, ``w_prod_lock=2``, ``w_cons_lock=3``,
            ``y_prod_ping_lock=4``, ``y_prod_pong_lock=5``,
            ``y_cons_ping_lock=7`` and
            ``y_cons_pong_lock=8``. When ``send_x_output`` is 0 the four
            ``y`` locks are the tile below's.
    """
    bf16 = np.dtype[bfloat16]
    y_ty = np.ndarray[(2 * _DECODE_Q4NX_ROWS + 16,), bf16]
    # One q4nx block, as the bf16 words the weight streams move.
    w_ty = np.ndarray[(_DECODE_Q4NX_BLOCK_BYTES // 2,), bf16]
    x_ty = np.ndarray[(_DECODE_Q4NX_COLS,), bf16]
    return _decode_kernel(
        "proj_main",
        "proj_main",
        [
            y_ty,
            w_ty,
            x_ty,
            y_ty,
            w_ty,
            x_ty,
            _DECODE_RTP,
            _DECODE_RTP,
            np.int32,
        ],
        (Out, In, In, Out, In, In, Param, Param, Param),
        dict(
            x_prod_lock=0,
            x_cons_lock=1,
            w_prod_lock=2,
            w_cons_lock=3,
            y_prod_ping_lock=4,
            y_prod_pong_lock=5,
            y_cons_ping_lock=7,
            y_cons_pong_lock=8,
        ),
        locks,
        geometry,
    )


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_per_layer_up(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the per-layer-input up projection, from ``flm_gemma4/decode_per_layer_up.cc``.

    ``per_layer_up(x, proj_w_ping, proj_w_pong, y)``: gates the layer's
    per-layer input, projects it from ``pli_d`` up to ``model_dim`` with bf16
    weight blocks (32 by 256 each) streamed through the ``proj_w`` ping-pong
    pair, RMS-normalizes the result, adds the residual and scales it by the
    layer scale into ``y`` (``model_dim`` bf16). ``x`` is
    ``2 * (pli_d + model_dim) + 32`` bf16: the norm weight, the layer scale
    padded to 32, the per-layer input, the residual and the gate. The kernel
    gates the per-layer input in place.

    ``y`` needs a 64-byte-aligned base pointer.

    Args:
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults ``x_prod_lock=0``,
            ``x_cons_lock=1``, ``proj_w_prod_lock=2``, ``proj_w_cons_lock=3``,
            ``y_prod_lock=4`` and ``y_cons_lock=5``.
    """
    bf16 = np.dtype[bfloat16]
    d, pli = geometry.model_dim, geometry.pli_d
    w_ty = np.ndarray[(_DECODE_BF16_BLOCK,), bf16]
    return _decode_kernel(
        "per_layer_up",
        "per_layer_up",
        [
            np.ndarray[(2 * (pli + d) + 32,), bf16],
            w_ty,
            w_ty,
            np.ndarray[(d,), bf16],
        ],
        (InOut, In, In, Out),
        dict(
            proj_w_prod_lock=2,
            proj_w_cons_lock=3,
        ),
        locks,
        geometry,
    )


_BF16_PROJ_M, _BF16_PROJ_K = 32, 256


def flm_gemma4_bf16_proj_core_ref(w, x):
    """Numpy reference for [`flm_gemma4_bf16_proj_core`][iron.kernels.flm_gemma4.flm_gemma4_bf16_proj_core]: one weight block times ``x``.

    The block is stored column by column: ``w[k * 32 + m]`` multiplies
    ``x[k]`` into out-feature ``m``. Sums in float64.
    """
    w = np.asarray(w, np.float64).reshape(-1, _BF16_PROJ_K, _BF16_PROJ_M)
    x = np.asarray(x, np.float64).reshape(-1, _BF16_PROJ_K)
    return np.einsum("ckm,ck->cm", w, x).astype(np.float32)


def _bf16_proj_core_sample(rng, calls):
    # Integers in [-8, 8]: every product and partial sum (below 2**14) is
    # exact in float32 in any order, and each survives a bfp16 block
    # conversion unchanged.
    w = rng.integers(-8, 9, (calls, _DECODE_BF16_BLOCK)).astype(bfloat16)
    x = rng.integers(-8, 9, (calls, _BF16_PROJ_K)).astype(bfloat16)
    return [w, x]


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_bf16_proj_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE
) -> ExternalFunction:
    """``_mvm_bf16_bf16`` of the per-layer-input projections, one block.

    ``bf16_proj_block_core(w, x, y)``: zeroes a float32 accumulator in its own
    frame, accumulates one 32 by 256 bf16 weight block times 256 inputs into
    it, as ``linear_proj`` does, and copies it to ``y``. The block does not
    depend on ``geometry``.
    """
    base = flm_gemma4_decode_per_layer_up(geometry=geometry)
    return _wrapper(
        base,
        "decode_per_layer_up_core.cc",
        "bf16_proj_block_core",
        [
            np.ndarray[(_DECODE_BF16_BLOCK,), _BF16],
            np.ndarray[(_BF16_PROJ_K,), _BF16],
            np.ndarray[(_BF16_PROJ_M,), _F32],
        ],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            reference=flm_gemma4_bf16_proj_core_ref,
            sample=_bf16_proj_core_sample,
            tolerance=Tolerance.exact(
                note="the sample keeps every product and sum exact in float32"
            ),
            ops_per_call=2 * _DECODE_BF16_BLOCK,
            acc_dtype=np.float32,
            reduction=_BF16_PROJ_K,
        ),
    )


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_proj_layer_embedding(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the per-layer-input embedding projection, from ``flm_gemma4/decode_proj_layer_embedding.cc``.

    ``proj_layer_embedding(norm_w, x0_per_layer, x0, x_proj, y, proj_w_ping, proj_w_pong)``:
    projects the layer input ``x0`` (``model_dim`` bf16) down to ``pli_d``
    with bf16 weight blocks (32 by 256 each) streamed through the ``proj_w``
    ping-pong pair, scales, RMS-normalizes and adds the token's per-layer
    embedding ``x0_per_layer`` (``pli_d``), and scales again, in the scratch
    ``x_proj`` (``pli_d``). ``norm_w`` (``pli_d + model_dim + 32`` bf16)
    holds the norm weight, then ``model_dim + 32`` values the kernel copies
    to the start of ``y`` (same size) for the up projection; the result
    follows them.

    ``x_proj`` needs a 64-byte-aligned base pointer.

    Args:
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults
            ``x0_per_layer_prod_lock=1``,
            ``xw_cons_lock=3``, ``proj_w_prod_lock=4``,
            ``proj_w_cons_lock=5``, ``y_prod_lock=6`` and ``y_cons_lock=7``.
    """
    bf16 = np.dtype[bfloat16]
    d, pli = geometry.model_dim, geometry.pli_d
    pli_ty = np.ndarray[(pli,), bf16]
    y_ty = np.ndarray[(pli + d + 32,), bf16]
    w_ty = np.ndarray[(_DECODE_BF16_BLOCK,), bf16]
    return _decode_kernel(
        "proj_layer_embedding",
        "proj_layer_embedding",
        [
            y_ty,
            pli_ty,
            np.ndarray[(d,), bf16],
            pli_ty,
            y_ty,
            w_ty,
            w_ty,
        ],
        (In, In, In, Out, Out, In, In),
        dict(
            proj_w_prod_lock=4,
            proj_w_cons_lock=5,
        ),
        locks,
        geometry,
    )


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_decode_gate_layer_embedding(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE, **locks: int
) -> ExternalFunction:
    """Build the per-layer-input gate, from ``flm_gemma4/decode_gate_layer_embedding.cc``.

    ``gate_layer_embedding(x, proj_w_ping, proj_w_pong, y)``: copies the
    layer output ``x`` (``model_dim`` bf16, a buffer of the tile to the left)
    into ``y``, projects it to ``pli_d`` with bf16 weight blocks (32 by 256
    each) streamed through the ``proj_w`` ping-pong pair, and applies the
    activation into ``y + model_dim``. ``y`` is ``model_dim + pli_d`` bf16.

    ``y`` needs a 64-byte-aligned base pointer; the kernel also stores
    512 bits at a time at ``y + model_dim``.

    Args:
        geometry: The model the kernel builds for.
        **locks: Core lock ids overriding the defaults ``proj_w_prod_lock=2``
            and ``proj_w_cons_lock=3``.
    """
    bf16 = np.dtype[bfloat16]
    d = geometry.model_dim
    w_ty = np.ndarray[(_DECODE_BF16_BLOCK,), bf16]
    return _decode_kernel(
        "gate_layer_embedding",
        "gate_layer_embedding",
        [
            np.ndarray[(d,), bf16],
            w_ty,
            w_ty,
            np.ndarray[(d + geometry.pli_d,), bf16],
        ],
        (In, In, In, Out),
        dict(
            proj_w_prod_lock=2,
            proj_w_cons_lock=3,
        ),
        locks,
        geometry,
        lut=True,
    )


def flm_gemma4_pli_gelu_core_ref(x):
    """Numpy reference for [`flm_gemma4_pli_gelu_core`][iron.kernels.flm_gemma4.flm_gemma4_pli_gelu_core]: tanh-GELU.

    The tolerance covers the kernel's table.
    """
    return _gelu_tanh(x).astype(np.float32)


def _pli_gelu_core_bound(x):
    # Plus the reference's float32 result, 2**-24 relative.
    return _gelu_lut_bound(x) + 2.0**-23 * np.abs(_gelu_tanh(x))


def _pli_gelu_core_sample(rng, calls, *, n):
    # Reaches both clamps of the table.
    return [rng.uniform(-6.0, 6.0, (calls, n)).astype(bfloat16)]


@dtypes([{"geometry": FLM_GEMMA4_E4B_DECODE}])
def flm_gemma4_pli_gelu_core(
    *, geometry: FlmGemma4DecodeGeometry = FLM_GEMMA4_E2B_DECODE
) -> ExternalFunction:
    """``_activate`` of [`flm_gemma4_decode_gate_layer_embedding`][iron.kernels.flm_gemma4.flm_gemma4_decode_gate_layer_embedding]: GELU over ``pli_d`` values.

    ``pli_gelu_core(x, y)``: copies ``x`` into ``y`` and applies GELU there,
    through aie_runtime_lib's table (``getGeluBf16``). Both are ``pli_d``
    bf16.

    Args:
        geometry: The model the kernel builds for.
    """
    n = geometry.pli_d
    return _wrapper(
        flm_gemma4_decode_gate_layer_embedding(geometry=geometry),
        "decode_gate_layer_embedding_core.cc",
        "pli_gelu_core",
        [np.ndarray[(n,), _BF16], np.ndarray[(n,), _BF16]],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=flm_gemma4_pli_gelu_core_ref,
            sample=partial(_pli_gelu_core_sample, n=n),
            tolerance=Tolerance.bounded(
                _pli_gelu_core_bound,
                note="the GELU table's error against tanh-GELU, measured on the "
                "host from aie_runtime_lib's table, plus a bf16 ulp of its slope "
                "and the floor bf16 store",
            ),
            uses_lut=True,
            # Per value: the table line's multiply and add.
            ops_per_call=2 * n,
        ),
    )


# The LM head's RTP buffer, in int32 words: the softcap's float32 bits in word
# 0, padded as the IRON operator's RTP writes need.
_LM_HEAD_RTP_WORDS = 32


class _LmHeadKernel(_ZeroInitializedKernel):
    q4nx_lm_head_rms: Kernel
    q4nx_lm_head_zero: Kernel
    q4nx_lm_head_block: ExternalFunction
    q4nx_lm_head_epilogue: Kernel


def _lm_head_geometry(dim, m_tile, k_tile, group):
    for name, value in dict(dim=dim, m_tile=m_tile, k_tile=k_tile, group=group).items():
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(
                f"flm_gemma4_q4nx_lm_head: {name} must be a positive integer"
            )
    if group != 32:
        raise ValueError(
            "flm_gemma4_q4nx_lm_head: group must be 32, the span of one column sum"
        )
    if k_tile % group:
        raise ValueError("flm_gemma4_q4nx_lm_head: k_tile must be a multiple of group")
    if m_tile % 16:
        raise ValueError("flm_gemma4_q4nx_lm_head: m_tile must be a multiple of 16")
    if dim % k_tile:
        # The RMS entry point and the block loop both cover whole k_tile
        # blocks, so a partial block would read past the token and write
        # past its column sums.
        raise ValueError("flm_gemma4_q4nx_lm_head: dim must be a multiple of k_tile")


def _f32(x):
    """Round float64 to float32 once, as an AIE float accumulator does."""
    return np.asarray(x, np.float64).astype(np.float32)


def flm_gemma4_q4nx_lm_head_ref(w, x, sums, *, m_tile=32, k_tile=256, group=32):
    """Numpy reference for [`flm_gemma4_q4nx_lm_head`][iron.kernels.flm_gemma4.flm_gemma4_q4nx_lm_head]: one ``q4nx_lm_head_block`` call.

    ``w`` is one q4nx block per call, as bytes or bf16 words: bf16 scales,
    then bf16 minima, both indexed ``[k // group, m]``, then 4-bit codes, low
    nibble first, indexed ``[m // 16, k // 32, (k % 32) // 8, k % 8, m % 16]``.
    ``x`` is the token then its RMS weight, and ``sums`` the token's per-32
    column sums, both shared by every call; slice 0 of the token is the one
    the call reads. Returns the float32
    accumulator, zero before the call, holding ``sum_k (min + scale * code) *
    x`` with the minima folded in through ``sums``.

    Per 16 rows and 32 columns the kernel sums the codes times ``x`` in
    float32, narrows that dot product to bf16 with the floor rounding a fresh
    core uses, then accumulates it times the scale, and the minimum times the
    column sum, into float32. Every product is exact, so each
    multiply-accumulate rounds once.
    """
    w = np.ascontiguousarray(w).view(np.uint8).reshape(-1, m_tile * k_tile * 5 // 8)
    calls, n_groups = len(w), k_tile // group
    params = np.ascontiguousarray(w[:, : 4 * m_tile * n_groups]).view("<u2")
    params = (params.astype(np.uint32) << 16).view(np.float32)
    scales, mins = params.reshape(calls, 2, n_groups, m_tile).transpose(1, 0, 2, 3)
    packed = w[:, 4 * m_tile * n_groups :]
    codes = np.empty((calls, m_tile * k_tile), np.float64)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    codes = codes.reshape(calls, m_tile // 16, n_groups, 32, 16)
    x = np.asarray(x, np.float32).reshape(-1)[:k_tile]
    x = x.astype(np.float64).reshape(1, n_groups, 32)
    sums = np.asarray(sums, np.float32).reshape(-1)[:n_groups]
    sums = sums.astype(np.float64).reshape(1, n_groups)
    acc = np.zeros((calls, m_tile // 16, 16), np.float32)
    for g in range(n_groups):
        dot = np.zeros((calls, m_tile // 16, 16), np.float32)
        for c in range(32):
            dot = _f32(dot + codes[:, :, g, c, :] * x[:, g, c, None, None])
        dot = _bf16_floor(dot).astype(np.float64)
        scale = scales[:, g].reshape(calls, m_tile // 16, 16).astype(np.float64)
        low = mins[:, g].reshape(calls, m_tile // 16, 16).astype(np.float64)
        acc = _f32(acc + dot * scale)
        acc = _f32(acc + low * sums[:, g, None, None])
    return acc.reshape(calls, m_tile)


def _q4nx_lm_head_sample(rng, calls, *, dim, m_tile, k_tile, group):
    # Every value is exact in float32, whatever order the kernel sums in: each
    # 32-column dot product is an integer below 2048, and the accumulator
    # holds multiples of 2**-4 below 2**15. The floor narrowing of a dot
    # product above 256 to bf16 still drops bits.
    n_groups = k_tile // group
    params = rng.choice([-1.0, 1.0], (calls, 2 * n_groups * m_tile)) * 2.0 ** (
        -rng.integers(0, 5, (calls, 2 * n_groups * m_tile))
    )
    params = params.astype(bfloat16).view(np.uint16).astype("<u2").view(np.uint8)
    packed = rng.integers(0, 256, (calls, m_tile * k_tile // 2), dtype=np.uint8)
    w = np.concatenate([params, packed], axis=1).view(bfloat16)
    token = rng.integers(-4, 5, dim).astype(bfloat16)
    x = np.concatenate([token, np.ones(dim, bfloat16)])
    sums = token.astype(np.float32).reshape(dim // group, group).sum(axis=1)
    return [w, x, sums.astype(bfloat16)]


def _lm_head_sibling(symbol, contract, **geometry) -> ExternalFunction:
    """``symbol`` of an LM-head build as a kernel of its own, judged by ``contract``."""
    base = flm_gemma4_q4nx_lm_head(**geometry)
    return _on_object(base, symbol, getattr(base, symbol).arg_types(), contract)


def _lm_head_zero(fn) -> ExternalFunction:
    """``fn``'s ``q4nx_lm_head_zero``, the initializer of its accumulator.

    It shares ``fn``'s object, so the design links one copy of the source.
    """
    y_acc = fn.arg_types()[2]
    return _on_object(
        fn,
        "q4nx_lm_head_zero",
        [y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            reference=lambda: np.zeros((1, *get_args(y_acc)[0]), np.float32),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )


def flm_gemma4_q4nx_lm_head(
    *, dim: int = 1536, m_tile: int = 32, k_tile: int = 256, group: int = 32
) -> ExternalFunction:
    """AIE2P logits from a q4nx vocabulary, from ``flm_gemma4/q4nx_lm_head.cc``.

    One core owns ``m_tile`` out-features at a time and streams their q4nx
    blocks past a token of ``dim`` values. The returned kernel is
    ``q4nx_lm_head_block``, which accumulates one ``m_tile`` by ``k_tile``
    block times one slice of the token into float32; its slice index is bound
    to 0. See ``flm_gemma4_q4nx_lm_head_ref`` for the block layout.

    The other entry points, attributes of the kernel:

    - ``q4nx_lm_head_rms(x, sums)``: once per token; normalizes the token in
      ``x`` (the token, then its RMS weight) in place and writes its per-32
      column sums.
    - ``q4nx_lm_head_zero(y_acc)``: before a tile's k loop.
    - ``q4nx_lm_head_epilogue(y, y_acc, softcap)``: after the k loop,
      ``y = c * tanh(y_acc / c)``, with the float32 ``c`` in RTP word 0.

    Args:
        dim: The token's length, a multiple of ``k_tile``; sets the RMS
            norm's span.
        m_tile: Out-features per block, a multiple of 16.
        k_tile: In-features per block, a multiple of ``group``.
        group: In-features per scale and minimum; must be 32.
    """
    _lm_head_geometry(dim, m_tile, k_tile, group)
    if not _arch_traits().bfp16:
        raise NotImplementedError(
            "flm_gemma4_q4nx_lm_head() is only available on aie2p."
        )
    geometry = dict(m_tile=m_tile, k_tile=k_tile, group=group)
    x = np.ndarray[(2, dim), _BF16]
    sums = np.ndarray[(dim // group,), _BF16]
    y_acc = np.ndarray[(m_tile,), _F32]
    fn = _make_extern(
        "q4nx_lm_head_block",
        _kernel_source("flm_gemma4/q4nx_lm_head.cc"),
        [np.ndarray[(m_tile * k_tile * 5 // 8 // 2,), _BF16], x, y_acc, sums, np.int32],
        compile_flags=[
            f"-DQ4NX_M_TILE={m_tile}",
            f"-DQ4NX_K_TILE={k_tile}",
            f"-DQ4NX_GROUP={group}",
            f"-DFLM_GEMMA4_LM_HEAD_DIM={dim}",
            # Without it the bf16 mmul emulation runs about 8x slower.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
        cls=_LmHeadKernel,
        contract=KernelContract(
            trace=Trace.whole_call(),
            # The token and its sums are fixed while the blocks stream past.
            roles=(In, Param, InOut, Param, Param),
            parameter_bindings=((4, 0),),
            initializers=((2, _lm_head_zero),),
            reference=partial(flm_gemma4_q4nx_lm_head_ref, **geometry),
            sample=partial(_q4nx_lm_head_sample, dim=dim, **geometry),
            tolerance=Tolerance.exact(
                note="the sample keeps every sum exact in float32; the floor "
                "bf16 narrowing of each dot product is modeled"
            ),
            ops_per_call=2 * m_tile * k_tile,
            acc_dtype=np.float32,
            reduction=k_tile,
        ),
    )
    bind = fn.object_file.bind
    fn.q4nx_lm_head_block = fn
    fn.q4nx_lm_head_rms = bind("q4nx_lm_head_rms", [x, sums])
    fn.q4nx_lm_head_zero = bind("q4nx_lm_head_zero", [y_acc])
    rtp = np.ndarray[(_LM_HEAD_RTP_WORDS,), _RTP_WORD]
    fn.q4nx_lm_head_epilogue = bind(
        "q4nx_lm_head_epilogue", [np.ndarray[(m_tile,), _BF16], y_acc, rtp]
    )
    return fn


def _lm_head_softcap(rtp):
    """Return the float32 softcap in word 0 of the LM head's RTP buffer."""
    word = np.ascontiguousarray(rtp, np.int32).reshape(-1)[:1]
    return float(word.view(np.float32)[0])


def flm_gemma4_q4nx_lm_head_epilogue_ref(y_acc, rtp):
    """Numpy reference for [`flm_gemma4_q4nx_lm_head_epilogue`][iron.kernels.flm_gemma4.flm_gemma4_q4nx_lm_head_epilogue]: ``c * tanh(y_acc / c)``.

    ``rtp`` holds the float32 bits of ``c`` in word 0. The reference computes
    in float64 and rounds to bf16 once; the tolerance covers the kernel's
    narrowings and its tanh.
    """
    c = _lm_head_softcap(rtp)
    y = np.asarray(y_acc, np.float64)
    return (c * np.tanh(y / c)).astype(bfloat16)


def _lm_head_epilogue_bound(y_acc, rtp):
    """Bound on the epilogue's error against its reference.

    The kernel floors ``y_acc`` and ``1 / c`` to bf16, each within ``2**-7``
    relative, so its tanh argument lies within ``2**-6 * |u|`` of ``u``. Over
    that interval tanh moves by at most ``tanh(hi) - tanh(lo)``, and vtanh
    adds ``_vtanh_error`` at its worst endpoint plus one ulp of ``t``: that
    bound was measured in ``conv_even``, and the kernel runs in floor mode.
    ``c`` scales both. The floor store of ``c * t`` adds one ulp, and the
    reference's nearest-even store half of one. ``c`` must be a bf16 value.
    """
    c = _lm_head_softcap(rtp)
    a = np.abs(np.asarray(y_acc, np.float64) / c)
    lo, hi = a * (1 - 2.0**-6), a * (1 + 2.0**-6)
    t = np.tanh(a)
    vtanh = np.maximum.reduce([_vtanh_error(lo), _vtanh_error(a), _vtanh_error(hi)])
    err = np.tanh(hi) - np.tanh(lo) + vtanh + _bf16_ulp(t)
    out = c * np.minimum(t + err, 1.0)
    return c * err + 1.5 * _bf16_ulp(out)


def _lm_head_epilogue_sample(rng, calls, *, m_tile, softcap=30.0):
    # u = y / c spans [-4, 4]: vtanh's identity band, its piecewise middle
    # and its saturation past |u| = 3.
    y_acc = rng.uniform(-4 * softcap, 4 * softcap, (calls, m_tile))
    rtp = np.zeros(_LM_HEAD_RTP_WORDS, np.int32)
    rtp[0] = np.float32(softcap).view(np.int32)
    return [y_acc.astype(np.float32), rtp]


def flm_gemma4_q4nx_lm_head_epilogue(
    *, dim: int = 1536, m_tile: int = 32, k_tile: int = 256, group: int = 32
) -> ExternalFunction:
    """``q4nx_lm_head_epilogue`` of the LM head: ``y = c * tanh(y_acc / c)``.

    The softcap ``c`` is float32 bits in word 0 of a 32-word RTP buffer. The
    tolerance assumes ``c`` is a bf16 value, so the kernel's narrowing of it
    is exact; Gemma 4's 30 is.
    """
    return _lm_head_sibling(
        "q4nx_lm_head_epilogue",
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, In, Param),
            reference=flm_gemma4_q4nx_lm_head_epilogue_ref,
            sample=partial(_lm_head_epilogue_sample, m_tile=m_tile),
            tolerance=Tolerance.bounded(
                _lm_head_epilogue_bound,
                note="c times vtanh's error, measured on npu2 (see "
                "activation._vtanh_error), plus the floor bf16 roundings",
            ),
        ),
        dim=dim,
        m_tile=m_tile,
        k_tile=k_tile,
        group=group,
    )


def _lm_head_rms_y(x, dim):
    """Per call, ``x * w / sqrt(mean(x^2) + 1e-6)`` in float64, shape (calls, dim)."""
    x = np.asarray(x, np.float64).reshape(-1, 2, dim)
    token, w = x[:, 0], x[:, 1]
    return token * w / np.sqrt(np.mean(token**2, axis=1, keepdims=True) + 1e-6)


def flm_gemma4_q4nx_lm_head_rms_ref(x, *, dim=1536):
    """Numpy reference for [`flm_gemma4_q4nx_lm_head_rms`][iron.kernels.flm_gemma4.flm_gemma4_q4nx_lm_head_rms]: column sums of the normalized token.

    ``x`` holds, per call, the token then its RMS weight. The reference
    narrows each normalized value to bf16 and sums each 32 of them in float64.
    """
    y = _lm_head_rms_y(x, dim).astype(np.float32).astype(bfloat16)
    sums = y.astype(np.float64).reshape(len(y), dim // 32, 32).sum(axis=2)
    return sums.astype(np.float32)


def _lm_head_rms_bound(x, *, dim):
    y = np.abs(_lm_head_rms_y(x, dim)).reshape(-1, dim // 32, 32)
    return 3 * 2.0**-7 * y.sum(axis=2)


def flm_gemma4_q4nx_lm_head_rms(
    *, dim: int = 1536, m_tile: int = 32, k_tile: int = 256, group: int = 32
) -> ExternalFunction:
    """``q4nx_lm_head_rms`` of [`flm_gemma4_q4nx_lm_head`][iron.kernels.flm_gemma4.flm_gemma4_q4nx_lm_head]: the token's RMS norm and per-32 column sums.

    The kernel normalizes the token in place. The contract declares the token
    ``In``: the write lands in the core's input element, and the next DMA fill
    overwrites it. The column sums depend on every normalized value.
    """
    return _lm_head_sibling(
        "q4nx_lm_head_rms",
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=partial(flm_gemma4_q4nx_lm_head_rms_ref, dim=dim),
            tolerance=Tolerance.bounded(
                partial(_lm_head_rms_bound, dim=dim),
                note="bf16 ulp <= 2**-7 |v|. Per term: the kernel floors to bf16 "
                "(1 ulp), the reference rounds to nearest (1/2 ulp), and the fast "
                "inverse sqrt after two Newton steps adds 5e-6 relative. The "
                "kernel floors the sum to bf16 (1 ulp of the sum). 2.5 ulps of "
                "the sum of |y| plus that 5e-6 stays under 3 * 2**-7 of it",
            ),
            # Per element: square and add, two multiplies, one column-sum add.
            ops_per_call=5 * dim,
        ),
        dim=dim,
        m_tile=m_tile,
        k_tile=k_tile,
        group=group,
    )


def flm_gemma4_attn_prefill_ref(l_bf16, y):
    """Numpy reference for [`flm_gemma4_attn_prefill`][iron.kernels.flm_gemma4.flm_gemma4_attn_prefill]: ``attn_epilogue`` at chunk 0.

    ``y`` is the round's (8, 512) float32 accumulator, flat; chunk 0 is its
    first 64 values, eight per row of ``l_bf16``. The kernel narrows ``y`` to
    bf16, multiplies by the row's ``1 / l`` exactly and narrows again, both
    times with the floor rounding a fresh core uses.
    """
    return _prefill_epilogue(
        l_bf16, y, lq=_ATTN_PREFILL_LQ, dh=_ATTN_PREFILL_DH, chunk=0
    )


def flm_gemma4_swa_prefill_ref(l_bf16, y):
    """Numpy reference for [`flm_gemma4_swa_prefill`][iron.kernels.flm_gemma4.flm_gemma4_swa_prefill]: ``attn_epilogue`` at chunk 33.

    As [`flm_gemma4_attn_prefill_ref`][iron.kernels.flm_gemma4.flm_gemma4_attn_prefill_ref], over a
    (16, 256) ``y``: rows 8 to 15, columns 64 to 127, each row scaled by its
    own ``1 / l``.
    """
    return _prefill_epilogue(
        l_bf16, y, lq=_SWA_PREFILL_LQ, dh=_SWA_PREFILL_DH, chunk=_SWA_PREFILL_CHUNK
    )


def _prefill_epilogue(l_bf16, y, *, lq, dh, chunk):
    """One prefill ``attn_epilogue`` chunk: 8 rows by 8 columns, floor-rounded twice."""
    row = 8 * (chunk // (dh // 8))
    col = 64 * (chunk % (dh // 8))
    l_bf16 = np.asarray(l_bf16, dtype=np.float32).reshape(-1, lq)
    y = np.asarray(y, dtype=np.float32).reshape(len(l_bf16), lq * dh)
    y = y[:, row * dh + col : row * dh + col + 64]
    scale = np.repeat(l_bf16[:, row : row + 8], 8, axis=1)
    return _bf16_floor(_bf16_floor(y) * scale).astype(bfloat16)
