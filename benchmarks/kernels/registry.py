# benchmarks/kernels/registry.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Which kernels the nightly measures, at which shapes, on which data.

What a kernel *computes* -- argument roles, reference, tolerance, ops -- lives
with its factory as a ``KernelContract`` and is not repeated here. This file
is policy only: the cases worth timing, the edge-case data each must survive,
and the canary that decides whether a runner's numbers can be trusted.

Adding a kernel to the nightly is one ``Case`` line, provided its factory
carries a contract (``test/python/test_kernel_contracts.py`` enforces that).
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import get_current_device
from aie.utils import kernel_harness as kh
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16

_DEVICE_FOR_ARCH = {"aie2": NPU1Col1, "aie2p": NPU2Col1}


@contextmanager
def _device_for(arch: str | None):
    """Bind the iron device an arch-restricted factory needs, then restore.

    Factories pick their source through the current device; an aie2p-only one
    (``exp2f_vec``) refuses to build under aie2. Naming and describing such a
    case on a host with no device set must still work, so the case binds its
    own arch around the factory call.
    """
    if arch is None:
        yield
        return
    try:
        previous = get_current_device(probe_runtime=False)
    except Exception:  # noqa: BLE001 - no device bound
        previous = None
    set_current_device(_DEVICE_FOR_ARCH[arch]())
    try:
        yield
    finally:
        set_current_device(previous)


# --------------------------------------------------------------------------
# Case
# --------------------------------------------------------------------------


@dataclass
class Case:
    """One (kernel, shape) the suite builds, checks and times.

    ``calls`` iterations of a streaming kernel, or ``shape`` = (M, K, N) /
    (M, K) host operands for ``mm`` / ``mv``. ``params`` overrides the value
    of ``param`` arguments (``scale``'s factor); ``scalars`` supplies
    ``scalar`` arguments in order. ``arch`` restricts a case to one target
    when its source exists only there.
    """

    factory: str
    kwargs: dict = field(default_factory=dict)
    calls: int = 1
    shape: tuple | None = None
    scalars: tuple = ()
    params: tuple = ()
    tag: str = ""
    perf: bool = True
    correctness: bool = True
    data_cases: tuple[str, ...] | None = None  # None: derived from the contract
    arch: str | None = None

    def fn(self):
        with _device_for(self.arch):
            return getattr(kernels, self.factory)(**self.kwargs)

    def harness_opts(self) -> dict:
        return dict(calls=self.calls, shape=self.shape, scalars=self.scalars)

    # Factory kwargs that the dims / dtype segments of the name already encode.
    # ``skip_dtype`` is not among them: it types a residual that is neither
    # the first input nor the output, so it must be spelled out.
    _NAMED_KWARGS = frozenset(
        {
            "tile_size",
            "dtype",
            "act_dtype",
            "input_dtype",
            "output_dtype",
            "dim_m",
            "dim_k",
            "dim_n",
        }
    )

    @functools.cached_property
    def name(self) -> str:
        """Stable graph key: ``factory/<dims>/<in dtype>[_<out dtype>][/k=v...][/tag]``.

        The dtype segment names the input dtype, and the output dtype too when
        it differs (``mm/256x256x256/int8_int32``). Factory kwargs that the
        dims and dtypes do not encode (``subtile=8``) are appended, so two
        cases of one factory never share a series. Renaming a case renames
        every series recorded under it. Cached: building the kernel is the
        expensive part, and the LUT factories are not memoized.
        """
        fn = self.fn()
        types = kh._arg_types(fn)
        in_dt = kh.dtype_name(kh._shape_dtype(types[fn.contract.roles.index("in")])[1])
        out_dt = kh.dtype_name(kh._shape_dtype(types[fn.contract.out_index])[1])
        dtypes = in_dt if in_dt == out_dt else f"{in_dt}_{out_dt}"
        if self.shape:
            dims = "x".join(str(d) for d in self.shape)
        else:
            dims = f"{kh._elems(types[fn.contract.roles.index('in')])}x{self.calls}"
        extra = [
            f"{k}={kh.dtype_name(v) if isinstance(v, type) else v}"
            for k, v in sorted(self.kwargs.items())
            if k not in self._NAMED_KWARGS
        ]
        parts = [self.factory, dims, dtypes, *extra] + ([self.tag] if self.tag else [])
        return "/".join(parts)

    def data_policy(self) -> tuple[str, ...]:
        """The edge-data cases this case runs: ``data_cases`` or :func:`data_policy`."""
        return (
            self.data_cases if self.data_cases is not None else data_policy(self.fn())
        )

    def kernel_calls(self) -> int:
        """How many times the core invokes the kernel in one run."""
        if self.shape is None:
            return self.calls
        m, k = self.kwargs["dim_m"], self.kwargs["dim_k"]
        if len(self.shape) == 3:
            M, K, N = self.shape
            return (M // m) * (N // self.kwargs["dim_n"]) * (K // k)
        M, K = self.shape
        return (M // m) * (K // k)

    def work(self) -> int:
        """Arithmetic operations per run, from the contract's ``ops_per_call``."""
        fn = self.fn()
        per_call = fn.contract.ops_per_call
        if per_call is None:
            per_call = kh._elems(kh._arg_types(fn)[fn.contract.out_index])
        return per_call * self.kernel_calls()


# --------------------------------------------------------------------------
# Edge-case data
# --------------------------------------------------------------------------

# Data cases every kernel of a kind should survive. Random data finds nothing
# a vectorised tail, a saturating add or a NaN path gets wrong. The policy
# is derived from each contract by `data_policy`: integer kernels get the
# extremes (inside `input_limit`); float kernels get subnormal inputs only
# when the contract says what the core does with them (`subnormals`) and
# NaN / inf only when it declares they propagate (`nonfinite`); the LUT
# activations, the norms and the reductions leave both unspecified until a
# device run pins them, and widening is then a contract change, not a table
# edit here. Matmul operands skip NaN (the reference accumulates them into
# every output) and bfp16ebs8 operands skip "max" too (3e38 products
# overflow the float reference; what the core's bfp16 store makes of an inf
# is unknown).
INT_DATA = ("random", "zeros", "max", "min", "alternating")
FLOAT_BASE = ("random", "zeros", "ones", "large", "alternating")
MATRIX_DATA = ("random", "zeros", "ones", "alternating", "max")


def data_policy(fn) -> tuple[str, ...]:
    """The edge-data cases a kernel's contract admits (see the note above)."""
    c = fn.contract
    if c.sample is not None:
        return ("random",)  # structured inputs have no edge variants
    types = kh._arg_types(fn)
    in_dt = kh._shape_dtype(types[c.roles.index("in")])[1]
    if kh._is_bfp(in_dt):
        return tuple(d for d in MATRIX_DATA if d != "max")
    if kh.is_matmul(fn) or kh.is_matvec(fn):
        return MATRIX_DATA
    if np.issubdtype(np.dtype(in_dt), np.integer):
        return INT_DATA
    cases = list(FLOAT_BASE)
    if c.subnormals != "unspecified":
        cases.append("subnormal")
    if c.nonfinite == "propagate":
        cases.append("nan_inf")
    return tuple(cases)


def _edge(shape, dtype, rng, case: str, limit: int | None = None) -> np.ndarray:
    """One edge-case array. ``limit`` bounds the integer extremes ("max",
    "min") to what the kernel's accumulator admits (``kernel_harness.input_limit``)."""
    dt = np.dtype(dtype)
    is_int = np.issubdtype(dt, np.integer)
    hi = np.iinfo(dt).max if is_int else None
    lo = np.iinfo(dt).min if is_int else None
    if is_int and limit is not None:
        hi = min(hi, limit)
        lo = max(lo, -limit) if dt.kind != "u" else 0
    if case == "zeros":
        a = np.zeros(shape)
    elif case == "ones":
        a = np.ones(shape)
    elif case == "max":
        a = np.full(shape, hi if is_int else 3.0e38)
    elif case == "min":
        a = np.full(shape, lo if is_int else -3.0e38)
    elif case == "alternating":
        a = (np.indices(shape).sum(0) % 2) * 2 - 1
    elif case == "subnormal":
        a = rng.uniform(-1e-39, 1e-39, shape)
    elif case == "nan_inf":
        a = rng.standard_normal(shape)
        a.flat[0], a.flat[-1], a.flat[a.size // 2] = np.nan, np.inf, -np.inf
    elif case == "large":
        a = rng.standard_normal(shape) * 1e4
    else:
        raise KeyError(case)
    if is_int:
        return np.clip(a, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
    return a.astype(np.float32).astype(dt)


def inputs_for(case: Case, data_case: str, rng) -> list[np.ndarray]:
    """Host inputs for ``case`` under one data case, in contract order."""
    fn = case.fn()
    c = fn.contract
    inputs = kh.sample_inputs(fn, calls=case.calls, shape=case.shape, rng=rng)
    tensor_pos = [i for i, r in enumerate(c.roles) if r in ("in", "param")]
    if data_case != "random":
        if c.sample is not None:
            raise ValueError(
                f"{case.factory}: structured inputs have no '{data_case}' variant"
            )
        # Edge data is about the streamed inputs; a `param` (scale's factor,
        # filter2d's kernel) keeps its value, so one design serves every case.
        # Integer extremes stay inside what the kernel's accumulator admits,
        # so "max" tests the datapath, not an overflow the source leaves open.
        k_total = case.shape[1] if case.shape else None
        inputs = [
            (
                a
                if c.roles[i] == "param"
                else _edge(
                    a.shape,
                    a.dtype,
                    rng,
                    data_case,
                    kh.input_limit(fn, a.dtype, reduction=k_total),
                )
            )
            for a, i in zip(inputs, tensor_pos)
        ]
    if case.params:
        params = iter(case.params)
        inputs = [
            (
                np.full(a.shape, next(params), dtype=a.dtype)
                if c.roles[i] == "param"
                else a
            )
            for a, i in zip(inputs, tensor_pos)
        ]
    return inputs


# --------------------------------------------------------------------------
# The cases
# --------------------------------------------------------------------------

_bf16 = dict(dtype=bfloat16)
# Tile sizes are chosen so two sets of tiles (ping-pong) plus the stack fit a
# core's 64 KB: a 64x32x64 matmul tile set is 8 KB + 8 KB + 16 KB of C.
# aie.utils.kernel_harness drops to depth 1 when a set does not fit, which
# still checks the kernel but is not the buffering anyone benchmarks.
_mm = dict(dim_m=64, dim_k=32, dim_n=64)
_mm_bf16 = dict(**_mm, input_dtype=bfloat16, output_dtype=np.float32)
_mm_bfp = dict(dim_m=64, dim_k=64, dim_n=64)  # the block_datatypes examples' tile

CASES: list[Case] = [
    # eltwise
    Case("passthrough", dict(tile_size=2048), calls=16),
    Case("passthrough", dict(tile_size=2048), calls=256),
    Case("passthrough", dict(dtype=np.int16), calls=16),
    Case("passthrough", dict(dtype=np.uint8), calls=16),
    Case("passthrough", dict(tile_size=64), calls=4, tag="edge-tiny", perf=False),
    Case("scale", dict(dtype=np.int16), calls=16),
    Case("scale", dict(dtype=np.int16), calls=256),
    Case("scale", dict(dtype=np.int32), calls=16),
    # No int16 overflow case: scale.cc stores acc32 with to_vector(0) and no
    # set_sat, so whether a product beyond int16 wraps or saturates is a core
    # setting the source leaves open (overflow="undefined"); the judge refuses
    # to grade such a reference until the kernel declares it.
    Case(
        "scale",
        dict(dtype=np.int32),
        calls=16,
        params=(-7,),
        tag="edge-negfactor",
        perf=False,
    ),
    Case("add", calls=16),
    Case("add", calls=256),
    Case("mul", calls=16),
    Case("mul", calls=256),
    Case("relu", calls=16),
    Case("relu", calls=256),
    # reduce
    Case("reduce_add", calls=16),
    Case("reduce_add", calls=256),
    Case("reduce_add", calls=1, tag="edge-single", perf=False),
    Case("reduce_min", calls=16),
    Case("reduce_min", calls=256),
    Case("reduce_max", calls=16),
    Case("reduce_max", calls=256),
    Case("reduce_max", _bf16, calls=16),
    Case("reduce_max", _bf16, calls=256),
    # activation
    Case("gelu", calls=16),
    Case("gelu", calls=256),
    Case("silu", calls=16),
    Case("silu", calls=256),
    Case("bf16_exp", calls=16),
    Case("bf16_exp", calls=256),
    Case("tanh", calls=16),
    Case("tanh", calls=256),
    Case("sigmoid", calls=16),
    Case("sigmoid", calls=256),
    Case("softmax", calls=16),
    Case("softmax", calls=256),
    Case("leaky_relu", calls=16, scalars=(0.5,)),
    Case("leaky_relu", calls=256, scalars=(0.5,)),
    Case("exp2f_vec", calls=16, arch="aie2p"),
    Case("exp2f_vec", calls=256, arch="aie2p"),
    # datamovement
    Case("axpy", calls=16, scalars=(2.5,)),
    Case("axpy", calls=256, scalars=(2.5,)),
    Case("convert_copy", calls=16, arch="aie2p"),
    Case("convert_copy", calls=256, arch="aie2p"),
    Case("expand", calls=16),
    Case("expand", calls=256),
    Case("transpose", dict(subtile=4), calls=16),
    Case("transpose", dict(subtile=8), calls=16),
    Case("transpose", dict(subtile=4, dtype=np.uint8), calls=16),
    Case("transpose", dict(subtile=8, dtype=np.uint32), calls=16),
    # linalg
    Case("mm", _mm_bf16, shape=(256, 256, 256)),
    Case("mm", _mm_bf16, shape=(512, 512, 512)),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int16, output_dtype=np.int32),
        shape=(256, 256, 256),
    ),
    Case(
        "mm",
        dict(**_mm, input_dtype=np.int8, output_dtype=np.int32),
        shape=(256, 256, 256),
    ),
    Case("mm", _mm_bf16, shape=(64, 64, 64), tag="edge-single-tile", perf=False),
    # block floating point (aie2p): bfp16ebs8 A, B and C, and the mixed
    # kernel with bf16 A and C; host encode/shuffle via aie.utils.bfp.
    Case("mm_bfp", _mm_bfp, shape=(256, 256, 256), arch="aie2p"),
    Case(
        "mm_bfp",
        dict(**_mm_bfp, mixed=True),
        shape=(256, 256, 256),
        arch="aie2p",
    ),
    Case(
        "mm",
        dict(
            dim_m=32, dim_k=32, dim_n=32, input_dtype=bfloat16, output_dtype=np.float32
        ),
        shape=(256, 256, 256),
        tag="edge-small-tile",
        perf=False,
    ),
    Case("mv", dict(dim_m=32, dim_k=32), shape=(256, 256)),
    # reduce companion, gated activation
    Case("compute_max", calls=16),
    Case("compute_max", _bf16, calls=16),
    Case("swiglu", calls=16),
    Case("swiglu", calls=256),
    # vision: uint8 lines of 1920 pixels
    Case("gray2rgba", calls=16),
    Case("rgba2gray", calls=16),
    Case("threshold", calls=16, scalars=(100, 255, 0)),
    Case("threshold", calls=16, scalars=(100, 255, 2), tag="trunc", perf=False),
    Case("threshold", calls=16, scalars=(100, 255, 4), tag="tozero-inv", perf=False),
    Case("bitwise_or", calls=16),
    Case("bitwise_and", calls=16),
    # alpha = beta = 0.5 in Q2.14; gamma = 0, where the kernel's two paths agree.
    Case("add_weighted", calls=16, scalars=(8192, 8192, 0)),
    Case("filter2d", calls=16),
    Case("rgba2hue", calls=16),
    # conv: full-range int8 data (the kernels saturate, so `input_limit` only
    # keeps the int32 accumulator safe); the shift puts random sums around
    # uint8's range (64 channels x 127^2 ~ 2**20 >> 12 for k1; 9x that >> 15
    # for k3) so saturation is exercised without being the whole picture.
    Case("conv2dk1", calls=8, scalars=(32, 64, 64, 12)),
    Case("conv2dk1_i8", calls=8, scalars=(32, 64, 64, 12)),
    # conv2dk1_skip streams three tensors; the harness packs them into one
    # fifo only when they share a type, i.e. input_channels == 2 *
    # output_channels with a uint8 residual (the int8 residual build shares
    # the contract and is covered by the host reference tests).
    Case(
        "conv2dk1_skip",
        dict(input_channels=128, output_channels=64, act_dtype=np.uint8),
        calls=8,
        scalars=(32, 128, 64, 12, 1),
    ),
    # conv2dk1_skip_init: the residual is a 1x1 conv of its own; packable with
    # a uint8 residual of half the input channels (one type across the fifo).
    Case(
        "conv2dk1_skip_init",
        dict(input_channels=64, skip_input_channels=32, act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 32, 12, 1, 11),
    ),
    # conv2dk14 (aie2p): 16 patches of 14x14 RGBA pixels per call, 784 taps.
    Case(
        "conv2dk14",
        calls=4,
        scalars=(224, 4, 16, 14, 17),
        arch="aie2p",
    ),
    # bottleneck (bn_*) single-core kernels: scalar sources, round-half-even.
    Case("bn_conv2dk1_relu", calls=8, scalars=(32, 64, 64, 12)),
    Case("bn_conv2dk1_i8", calls=8, scalars=(32, 64, 64, 13)),
    Case("bn_conv2dk1_skip", calls=8, scalars=(32, 64, 64, 13, 1)),
    Case(
        "bn_conv2dk1_skip",
        dict(skip_dtype=np.int8),
        calls=8,
        scalars=(32, 64, 64, 13, 1),
    ),
    Case(
        "bn_conv2dk3_dw",
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 11, 0),
    ),
    Case(
        "bn_conv2dk3_dw",
        dict(stride=2),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 11, 0),
    ),
    Case(
        "bn_conv2dk3",
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 15, 0),
    ),
    # MobileNet's classifier FC: one (1, 1, 1280) uint16 vector in, 16 uint16
    # logits per call; weights [16/8][1280/8][8][8] unpadded (pad == IC).
    Case(
        "bn_fc_relu_ui16_pad",
        dict(input_channels=1280, output_channels=16),
        calls=8,
        scalars=(1, 1280, 1280, 16, 13),
    ),
    Case(
        "conv2dk1",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 12),
    ),
    Case("conv2dk3", calls=8, scalars=(32, 64, 64, 3, 3, 1, 15, 0)),
    Case(
        "conv2dk3",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 1, 15, 0),
    ),
    Case(
        "conv2dk3",
        dict(act_dtype=np.uint8),
        calls=8,
        scalars=(32, 64, 64, 3, 3, 0, 15, 0),
        tag="top-row",
        perf=False,
    ),
    # eltwise mul/add selected per call (programming_examples/ml/scale_shift)
    Case("mul_add", calls=16, scalars=(1,)),
    Case("mul_add", calls=16, scalars=(0,), tag="add"),
    # transformer blocks (aie2p): one row per call
    Case("rms_norm", dict(cols=1024), calls=16, arch="aie2p"),
    Case("layer_norm", dict(cols=1024), calls=16, arch="aie2p"),
    Case(
        "layer_norm_f32",
        dict(cols=1024),
        calls=16,
        arch="aie2p",
    ),
    Case(
        "layer_norm_affine_cast",
        dict(cols=1024),
        calls=16,
        arch="aie2p",
    ),
    Case("rope", dict(cols=1024), calls=16, arch="aie2p"),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(0,),
        tag="identity",
        arch="aie2p",
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(1,),
        tag="silu",
        arch="aie2p",
    ),
    Case(
        "mm_activation_epilogue",
        calls=16,
        scalars=(2,),
        tag="gelu",
        arch="aie2p",
    ),
    # depthwise 1-D conv (aie2p): 1024 outputs per call from a padded row
    Case(
        "dwconv1d",
        dict(seq_len=1024, kernel_size=9),
        calls=16,
        scalars=(1024,),
        arch="aie2p",
    ),
]

# The canary runs first: a trivially correct kernel that must be bit-exact and
# inside a wide cycle band, otherwise the machine is declared bad and nothing
# is recorded. Tighten the band after the first few nights.
CANARY = Case("passthrough", dict(tile_size=2048), calls=16)
CANARY_CYCLE_BAND = (1_000, 2_000_000)


def perf_cases(only: str | None = None):
    import re

    for c in CASES:
        if c.perf and (only is None or re.search(only, c.name)):
            yield c
