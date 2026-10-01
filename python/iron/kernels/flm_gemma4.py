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


def _prefill_flags(dh, in_prod_lock=2, in_cons_lock=3) -> list[str]:
    return [
        # The bf16 mmul lowers onto two bfp16-emulated macs on AIE2P.
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DFLM_GEMMA4_PREFILL_HEAD_DIM={dh}",
        f"-DFLM_GEMMA4_PREFILL_IN_PROD_LOCK={int(in_prod_lock)}",
        f"-DFLM_GEMMA4_PREFILL_IN_CONS_LOCK={int(in_cons_lock)}",
    ]


def _prefill_sibling(symbol, head_dim, contract) -> ExternalFunction:
    """``symbol`` of a prefill build as a kernel of its own, judged by ``contract``."""
    if head_dim not in (256, 512):
        raise ValueError(f"head_dim must be 256 or 512, not {head_dim}")
    base = flm_gemma4_swa_prefill() if head_dim == 256 else flm_gemma4_attn_prefill()
    return _make_extern(
        symbol,
        _kernel_source("flm_gemma4/prefill.cc"),
        getattr(base, symbol).arg_types(),
        compile_flags=_prefill_flags(head_dim),
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
        compile_flags=_prefill_flags(dh, in_prod_lock, in_cons_lock),
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


def _decode_core(base: ExternalFunction, source: str, symbol, arg_types, contract):
    """``symbol`` of ``flm_gemma4/<source>``, a lock-free wrapper that includes ``base``'s source.

    It builds with ``base``'s flags and include path, so the wrapper sees the
    same geometry and lock ids, and is judged by ``contract``.
    """
    path = str(_kernel_source(f"flm_gemma4/{source}"))
    flags = [f for f in base.compile_flags if f not in _portable_flags()]
    lut = [i for i, f in enumerate(flags) if f.startswith("-DAIE_LUT_KERNEL_SOURCE=")]
    if lut:
        flags[lut[0]] = f'-DAIE_LUT_KERNEL_SOURCE="{path}"'
        path = base.source_file
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
            ``rtp_lock=6``.
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
        dict(x_prod_lock=0, x_cons_lock=1, y_prod_lock=2, y_cons_lock=3, rtp_lock=6),
        locks,
        geometry,
        lut=True,
    )


# Keys the attention kernels fold in per round (FastFlowLM's lq and lk).
_DECODE_KEYS_PER_ROUND = 16
# One q4nx weight block: 32 rows by 256 columns at 5 bits a weight.
_DECODE_Q4NX_ROWS, _DECODE_Q4NX_COLS = 32, 256
_DECODE_Q4NX_BLOCK_BYTES = _DECODE_Q4NX_ROWS * _DECODE_Q4NX_COLS * 5 // 8
# One bf16 weight block of the per-layer-input projections: 32 rows by 256.
_DECODE_BF16_BLOCK = 32 * 256
_DECODE_RTP = np.ndarray[(16,), np.dtype[np.int32]]
_DECODE_KV_LOCKS = dict(
    v_prod_lock=2, v_cons_lock=3, o_prod_lock=0, o_cons_lock=1, l_cons_lock=8
)
_DECODE_ROPE_LOCKS = dict(
    qkv_prod_lock=0,
    qkv_cons_lock=1,
    k_prod_lock=4,
    k_cons_lock=5,
    v_prod_lock=6,
    v_cons_lock=7,
    rope_prod_lock=8,
    rope_cons_lock=9,
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
        **locks: Core lock ids overriding the defaults ``v_prod_lock=2``,
            ``v_cons_lock=3``, ``o_prod_lock=0``, ``o_cons_lock=1`` and
            ``l_cons_lock=8`` (a lock of the qk core, on the tile below).
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
        **locks: Core lock ids overriding the defaults ``v_prod_lock=2``,
            ``v_cons_lock=3``, ``o_prod_lock=0``, ``o_cons_lock=1`` and
            ``l_cons_lock=8`` (a lock of the qk core, on the tile below).
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
        **locks: Core lock ids overriding the defaults ``k_prod_lock=2``,
            ``k_cons_lock=3`` and ``l_cons_lock=8``.
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
        dict(k_prod_lock=2, k_cons_lock=3, l_cons_lock=8),
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
        **locks: Core lock ids overriding the defaults ``k_prod_lock=2``,
            ``k_cons_lock=3`` and ``l_prod_lock=8``.
    """
    _decode_kv_heads("flm_gemma4_decode_attn_qk_kvh2", geometry, 2)
    t = _decode_attn_types(geometry, geometry.dh, geometry.dh)
    fn = _decode_kernel(
        "attn_qk_kvh2",
        "attn_qk_begin",
        [t["m"]],
        (Out,),
        dict(k_prod_lock=2, k_cons_lock=3, l_prod_lock=8),
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
        **locks: Core lock ids overriding the defaults ``v_prod_lock=2``,
            ``v_cons_lock=3``, ``o_prod_lock=0``, ``o_cons_lock=1`` and
            ``l_cons_lock=8`` (a lock of the qk core, on the tile below).
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
            ``v_prod_lock=6``, ``v_cons_lock=7``, ``rope_prod_lock=8`` and
            ``rope_cons_lock=9``.
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
        **locks: Core lock ids overriding the defaults ``w_prod_lock=0``,
            ``w_cons_lock=1``, ``y_prod_lock=2``, ``y_cons_lock=3``,
            ``x_prod_lock=4``, ``x_cons_lock=5``, ``rtp_available_lock=6``,
            ``lm_head_out_prod_lock=7`` and ``lm_head_out_cons_lock=8``.
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
            w_prod_lock=0,
            w_cons_lock=1,
            y_prod_lock=2,
            y_cons_lock=3,
            x_prod_lock=4,
            x_cons_lock=5,
            rtp_available_lock=6,
            lm_head_out_prod_lock=7,
            lm_head_out_cons_lock=8,
        ),
        locks,
        geometry,
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
            ``rtp_available_lock=6``, ``y_cons_ping_lock=7`` and
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
            rtp_available_lock=6,
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
            x_prod_lock=0,
            x_cons_lock=1,
            proj_w_prod_lock=2,
            proj_w_cons_lock=3,
            y_prod_lock=4,
            y_cons_lock=5,
        ),
        locks,
        geometry,
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
        **locks: Core lock ids overriding the defaults ``norm_w_prod_lock=0``,
            ``x0_per_layer_prod_lock=1``, ``x0_prod_lock=2``,
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
            norm_w_prod_lock=0,
            x0_per_layer_prod_lock=1,
            x0_prod_lock=2,
            xw_cons_lock=3,
            proj_w_prod_lock=4,
            proj_w_cons_lock=5,
            y_prod_lock=6,
            y_cons_lock=7,
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
        **locks: Core lock ids overriding the defaults ``x_prod_lock=0``,
            ``x_cons_lock=1``, ``proj_w_prod_lock=2``, ``proj_w_cons_lock=3``,
            ``y_prod_lock=4``, ``y_cons_lock=5``, ``final_x_prod_lock=7`` and
            ``final_x_cons_lock=8`` (the last two locks of the tile to the
            left).
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
            x_prod_lock=0,
            x_cons_lock=1,
            proj_w_prod_lock=2,
            proj_w_cons_lock=3,
            y_prod_lock=4,
            y_cons_lock=5,
            final_x_prod_lock=7,
            final_x_cons_lock=8,
        ),
        locks,
        geometry,
        lut=True,
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


def _lm_head_flags(dim, m_tile, k_tile, group) -> list[str]:
    return [
        f"-DQ4NX_M_TILE={m_tile}",
        f"-DQ4NX_K_TILE={k_tile}",
        f"-DQ4NX_GROUP={group}",
        f"-DFLM_GEMMA4_LM_HEAD_DIM={dim}",
        # Without it the bf16 mmul emulation runs about 8x slower.
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]


def _lm_head_sibling(symbol, contract, **geometry) -> ExternalFunction:
    """``symbol`` of an LM-head build as a kernel of its own, judged by ``contract``."""
    base = flm_gemma4_q4nx_lm_head(**geometry)
    return _make_extern(
        symbol,
        _kernel_source("flm_gemma4/q4nx_lm_head.cc"),
        getattr(base, symbol).arg_types(),
        compile_flags=_lm_head_flags(**geometry),
        contract=contract,
    )


def _lm_head_zero(fn) -> ExternalFunction:
    """``fn``'s ``q4nx_lm_head_zero``, the initializer of its accumulator.

    It names ``fn``'s object and compile recipe, so the design compiles and
    links one copy of ``q4nx_lm_head.cc``.
    """
    y_acc = fn.arg_types()[2]
    return ExternalFunction(
        "q4nx_lm_head_zero",
        object_file_name=fn.object_file_name,
        source_file=fn.source_file,
        arg_types=[y_acc],
        include_dirs=fn.include_dirs,
        compile_flags=fn.compile_flags,
        symbol_prefix=fn.object_file.symbol_prefix,
        contract=KernelContract(
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
        compile_flags=_lm_head_flags(dim, m_tile, k_tile, group),
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
