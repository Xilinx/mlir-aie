# test_kernels.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Unit tests for aie.iron.kernels factory functions.

Each kernel factory is described by a single row in KERNEL_SPECS. Generic
parametrized tests exercise the common surface (returns ExternalFunction,
source is locatable, _arg_types length, default _name, invalid-kwargs raise).
Per-kernel name and shape variants are listed alongside the spec.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction
from aie.iron import kernels
from aie.iron.kernels import _common as _kernels_common


@pytest.fixture(autouse=True)
def _isolate_extern_state():
    """Reset the process-wide ExternalFunction registry + _make_extern cache
    between tests so cross-test pollution doesn't trip the new auto-prefix-
    on-collision logic.  Without this, a kernels.X() call in test N+1 sees
    test N's ExternalFunction still in ``_instances`` with the same
    ``_original_name`` and gets spuriously prefixed, breaking ``_name``
    assertions in name-variant tests."""
    ExternalFunction._instances.clear()
    _kernels_common._EXTERN_CACHE.clear()
    yield
    ExternalFunction._instances.clear()
    _kernels_common._EXTERN_CACHE.clear()


# ---------------------------------------------------------------------------
# Spec table
# ---------------------------------------------------------------------------


@dataclass
class KernelSpec:
    """Declarative description of a kernel factory's expected surface."""

    name: str  # spec id (used as pytest parameter id)
    factory: Callable
    kwargs: dict  # baseline kwargs that should produce a valid kernel
    arg_count: int
    expected_name: str  # expected ef._name with baseline kwargs
    # Source is either a real .cc file (source_substring=None checks _source_file)
    # or an embedded source_string containing a particular #include.
    source_kind: str = "file"  # "file" | "string_or_file"
    source_substring: str | None = None  # for "string_or_file": substring to find
    # Additional (kwargs_overrides, expected_name) pairs
    name_variants: list[tuple[dict, str]] = field(default_factory=list)
    # (kwargs_overrides, error_pattern) pairs
    invalid_kwargs: list[tuple[dict, str]] = field(default_factory=list)
    # (kwargs_overrides, arg_index, expected_shape_tuple) — for shape sanity
    shape_checks: list[tuple[dict, int, tuple]] = field(default_factory=list)
    # (kwargs_overrides, expected_tile_size_at_arg_0)
    tile_size_checks: list[tuple[dict, int]] = field(default_factory=list)


KERNEL_SPECS: list[KernelSpec] = [
    # ----- eltwise -----
    KernelSpec(
        name="passthrough",
        factory=kernels.passthrough,
        kwargs=dict(tile_size=4096, dtype=np.int32),
        arg_count=3,
        expected_name="passThroughLine",
        shape_checks=[
            (dict(tile_size=64, dtype=np.int16), 0, (64,)),
        ],
        tile_size_checks=[
            (dict(tile_size=256, dtype=np.uint8), 256),
        ],
    ),
    KernelSpec(
        name="scale",
        factory=kernels.scale,
        kwargs=dict(tile_size=1024, dtype=np.int32),
        arg_count=4,
        expected_name="vector_scalar_mul_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=True),
                "vector_scalar_mul_vector",
            ),
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=False),
                "vector_scalar_mul_scalar",
            ),
        ],
        invalid_kwargs=[
            (
                dict(tile_size=1024, dtype=np.float32),
                "dtype must be np.int16 or np.int32",
            ),
        ],
    ),
    KernelSpec(
        name="add",
        factory=kernels.add,
        kwargs=dict(tile_size=1024, dtype=bfloat16),
        arg_count=3,
        expected_name="eltwise_add_bf16_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=True),
                "eltwise_add_bf16_vector",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "eltwise_add_bf16_scalar",
            ),
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=np.float32), "dtype must be bfloat16"),
        ],
    ),
    KernelSpec(
        name="mul",
        factory=kernels.mul,
        kwargs=dict(tile_size=1024, dtype=bfloat16),
        arg_count=3,
        expected_name="eltwise_mul_bf16_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=True),
                "eltwise_mul_bf16_vector",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "eltwise_mul_bf16_scalar",
            ),
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=np.float32), "dtype must be bfloat16"),
            (dict(tile_size=512), "tile_size must be 1024"),
        ],
    ),
    # ----- reduce -----
    KernelSpec(
        name="reduce_add",
        factory=kernels.reduce_add,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="reduce_add_vector",
        name_variants=[
            (dict(tile_size=1024, vectorized=True), "reduce_add_vector"),
            (dict(tile_size=1024, vectorized=False), "reduce_add_scalar"),
            (dict(tile_size=512, dtype=np.int32), "reduce_add_vector"),
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=bfloat16), "dtype must be np.int32"),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            # int32 output: 1 element = 4 bytes → already DMA-aligned.
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
        ],
        tile_size_checks=[(dict(tile_size=2048, dtype=np.int32), 2048)],
    ),
    KernelSpec(
        name="reduce_min",
        factory=kernels.reduce_min,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="reduce_min_vector",
        name_variants=[
            (dict(tile_size=1024, vectorized=True), "reduce_min_vector"),
            (dict(tile_size=1024, vectorized=False), "reduce_min_scalar"),
            (dict(tile_size=512, dtype=np.int32), "reduce_min_vector"),
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=bfloat16), "dtype must be np.int32"),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
        ],
        tile_size_checks=[(dict(tile_size=2048, dtype=np.int32), 2048)],
    ),
    KernelSpec(
        name="reduce_max",
        factory=kernels.reduce_max,
        kwargs=dict(tile_size=1024, dtype=np.int32),
        arg_count=3,
        expected_name="reduce_max_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=True),
                "reduce_max_vector",
            ),
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=False),
                "reduce_max_scalar",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=True),
                "reduce_max_vector_bfloat16",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "reduce_max_scalar_bfloat16",
            ),
            (dict(tile_size=1024, dtype=bfloat16), "reduce_max_vector_bfloat16"),
        ],
        invalid_kwargs=[
            (
                dict(tile_size=1024, dtype=np.float32),
                "dtype must be np.int32 or bfloat16",
            ),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            # int32: out is 4-byte aligned with a single element.
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
            # bfloat16: out must be padded to 2 elements (4 bytes) for DMA alignment.
            (dict(tile_size=1024, dtype=bfloat16), 1, (2,)),
        ],
    ),
    # ----- activation -----
    KernelSpec(
        name="relu",
        factory=kernels.relu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="bf16_relu",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="softmax",
        factory=kernels.softmax,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="softmax_bf16",
        source_kind="string_or_file",
        source_substring="softmax.cc",
        invalid_kwargs=[(dict(tile_size=2048), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="gelu",
        factory=kernels.gelu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="gelu_bf16",
        source_kind="string_or_file",
        source_substring="gelu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="silu",
        factory=kernels.silu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="silu_bf16",
        source_kind="string_or_file",
        source_substring="silu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="swiglu",
        factory=kernels.swiglu,
        kwargs=dict(tile_size=1024),
        arg_count=4,
        expected_name="swiglu_bf16",
        source_kind="string_or_file",
        source_substring="swiglu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="bf16_exp",
        factory=kernels.bf16_exp,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="exp_bf16_1024",
        source_kind="string_or_file",
        source_substring="bf16_exp.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    # ----- vision -----
    KernelSpec(
        name="rgba2hue",
        factory=kernels.rgba2hue,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="rgba2hueLine",
        shape_checks=[
            (dict(line_width=640), 0, (640 * 4,)),
            (dict(line_width=640), 1, (640,)),
        ],
    ),
    KernelSpec(
        name="rgba2gray",
        factory=kernels.rgba2gray,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="rgba2grayLine",
        shape_checks=[
            (dict(line_width=640), 0, (640 * 4,)),
            (dict(line_width=640), 1, (640,)),
        ],
    ),
    KernelSpec(
        name="gray2rgba",
        factory=kernels.gray2rgba,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="gray2rgbaLine",
        shape_checks=[
            (dict(line_width=640), 0, (640,)),
            (dict(line_width=640), 1, (640 * 4,)),
        ],
    ),
    KernelSpec(
        name="threshold",
        factory=kernels.threshold,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=6,
        expected_name="thresholdLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "thresholdLine"),
            (dict(line_width=1920, dtype=np.int32), "thresholdLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    KernelSpec(
        name="bitwise_or",
        factory=kernels.bitwise_or,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=4,
        expected_name="bitwiseORLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "bitwiseORLine"),
            (dict(line_width=1920, dtype=np.int32), "bitwiseORLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    KernelSpec(
        name="bitwise_and",
        factory=kernels.bitwise_and,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=4,
        expected_name="bitwiseANDLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "bitwiseANDLine"),
            (dict(line_width=1920, dtype=np.int32), "bitwiseANDLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    KernelSpec(
        name="filter2d",
        factory=kernels.filter2d,
        kwargs=dict(line_width=1920),
        arg_count=6,
        expected_name="filter2dLine",
        shape_checks=[(dict(line_width=640), 0, (640,))],
    ),
    KernelSpec(
        name="add_weighted",
        factory=kernels.add_weighted,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=7,
        expected_name="addWeightedLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "addWeightedLine"),
            (dict(line_width=1920, dtype=np.int32), "addWeightedLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    # ----- linalg -----
    KernelSpec(
        name="mm",
        factory=kernels.mm,
        kwargs=dict(),
        arg_count=3,
        expected_name="matmul_i16_i16",
        name_variants=[
            (
                dict(input_dtype=np.int16, output_dtype=np.int16, vectorized=True),
                "matmul_i16_i16",
            ),
            (
                dict(input_dtype=np.int16, output_dtype=np.int16, vectorized=False),
                "matmul_scalar_i16_i16",
            ),
            (dict(input_dtype=bfloat16, output_dtype=bfloat16), "matmul_bf16_bf16"),
            (dict(input_dtype=np.int8, output_dtype=np.int8), "matmul_i8_i8"),
            (dict(input_dtype=bfloat16, output_dtype=np.float32), "matmul_bf16_f32"),
        ],
        invalid_kwargs=[
            (dict(input_dtype=np.float64, output_dtype=np.float64), "unsupported"),
        ],
        shape_checks=[
            (dict(dim_m=32, dim_k=16, dim_n=48), 2, (32 * 48,)),
        ],
    ),
    KernelSpec(
        name="mm_zero",
        factory=kernels.mm_zero,
        kwargs=dict(),
        arg_count=1,
        expected_name="zero_i16",
        name_variants=[
            (dict(output_dtype=np.int16, vectorized=True), "zero_i16"),
            (dict(output_dtype=np.int16, vectorized=False), "zero_scalar_i16"),
        ],
        invalid_kwargs=[(dict(output_dtype=np.float64), "unsupported")],
    ),
    KernelSpec(
        name="mv",
        factory=kernels.mv,
        kwargs=dict(),
        arg_count=3,
        expected_name="matvec_vectorized_i16_i32",
        name_variants=[
            (dict(vectorized=True), "matvec_vectorized_i16_i32"),
            (dict(vectorized=False), "matvec_scalar_i16_i32"),
        ],
        invalid_kwargs=[
            (dict(input_dtype=np.int8, output_dtype=np.int8), "only.*supported"),
        ],
        shape_checks=[
            (dict(dim_m=16, dim_k=64), 1, (64,)),
            (dict(dim_m=16, dim_k=64), 2, (16,)),
        ],
    ),
    KernelSpec(
        name="cascade_mm",
        factory=kernels.cascade_mm,
        kwargs=dict(),
        arg_count=3,
        expected_name="matmul_scalar_cascade_get_only_i16_i16",
        name_variants=[
            (dict(cascade_mode="get_only"), "matmul_scalar_cascade_get_only_i16_i16"),
            (dict(cascade_mode="put_only"), "matmul_scalar_cascade_put_only_i16_i16"),
            (dict(cascade_mode="put_get"), "matmul_scalar_cascade_put_get_i16_i16"),
            (
                dict(
                    input_dtype=bfloat16, output_dtype=bfloat16, cascade_mode="get_only"
                ),
                "matmul_scalar_cascade_get_only_bf16_bf16",
            ),
        ],
        invalid_kwargs=[
            (dict(cascade_mode="invalid"), "cascade_mode"),
            (dict(input_dtype=np.int8, output_dtype=np.int8), "unsupported"),
        ],
    ),
    # ----- conv -----
    KernelSpec(
        name="conv2dk1",
        factory=kernels.conv2dk1,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk1_i8"),
            (dict(act_dtype=np.uint8), "conv2dk1_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="conv2dk3",
        factory=kernels.conv2dk3,
        kwargs=dict(),
        arg_count=13,
        expected_name="conv2dk3_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk3_i8"),
            (dict(act_dtype=np.uint8), "conv2dk3_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="conv2dk1_skip",
        factory=kernels.conv2dk1_skip,
        kwargs=dict(),
        arg_count=10,
        expected_name="conv2dk1_skip_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk1_skip_i8"),
            (dict(act_dtype=np.uint8), "conv2dk1_skip_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="conv2dk1_i8",
        factory=kernels.conv2dk1_i8,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_i8",
    ),
    KernelSpec(
        name="conv2dk14",
        factory=kernels.conv2dk14,
        kwargs=dict(),
        arg_count=8,
        expected_name="conv2dk14_i8",
    ),
    KernelSpec(
        name="conv2dk1_skip_init",
        factory=kernels.conv2dk1_skip_init,
        kwargs=dict(),
        arg_count=12,
        expected_name="conv2dk1_skip_init_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk1_skip_init_i8"),
            (dict(act_dtype=np.uint8), "conv2dk1_skip_init_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="bn_conv2dk1_relu",
        factory=kernels.bn_conv2dk1_relu,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_relu_i8_ui8",
    ),
    KernelSpec(
        name="bn_conv2dk3",
        factory=kernels.bn_conv2dk3,
        kwargs=dict(),
        arg_count=13,
        expected_name="conv2dk3_stride2_i8",
    ),
    KernelSpec(
        name="bn_conv2dk1_i8",
        factory=kernels.bn_conv2dk1_i8,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_ui8_i8",
    ),
    KernelSpec(
        name="bn_conv2dk1_skip",
        factory=kernels.bn_conv2dk1_skip,
        kwargs=dict(),
        arg_count=9,
        expected_name="conv2dk1_skip_ui8_ui8_i8",
        name_variants=[
            (dict(skip_dtype=np.uint8), "conv2dk1_skip_ui8_ui8_i8"),
            (dict(skip_dtype=np.int8), "conv2dk1_skip_ui8_i8_i8"),
        ],
        invalid_kwargs=[(dict(skip_dtype=np.float32), "skip_dtype")],
    ),
    KernelSpec(
        name="bn_conv2dk3_dw",
        factory=kernels.bn_conv2dk3_dw,
        kwargs=dict(stride=2),
        arg_count=13,
        expected_name="conv2dk3_dw_stride2_relu_ui8_ui8",
        name_variants=[
            (dict(stride=1), "conv2dk3_dw_stride1_relu_ui8_ui8"),
            (dict(stride=2), "conv2dk3_dw_stride2_relu_ui8_ui8"),
        ],
        invalid_kwargs=[(dict(stride=3), "stride")],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(seq):
    return [s.name for s in seq]


def _flat(specs, attr):
    """Flatten (spec, *tuple_items) for parametrize tables."""
    out = []
    for s in specs:
        for entry in getattr(s, attr):
            out.append((s, *entry))
    return out


def _flat_ids(rows, label):
    return [f"{r[0].name}-{label}{i}" for i, r in enumerate(rows)]


# Special case: bn_conv2dk3_dw arg_count differs by `stride`.
# The base spec uses stride=2 (arg_count=13); add stride=1 (arg_count=14).
ARG_COUNT_OVERRIDES: list[tuple[KernelSpec, dict, int]] = [
    (
        next(s for s in KERNEL_SPECS if s.name == "bn_conv2dk3_dw"),
        dict(stride=1),
        14,
    ),
]


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_returns_external_function(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    assert isinstance(ef, ExternalFunction)


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_source_locatable(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    if spec.source_kind == "file":
        src = ef._source_file
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"
    else:
        # source_string OR source_file must be set; if string, must reference the .cc
        assert ef._source_string is not None or ef._source_file is not None
        if ef._source_string is not None and spec.source_substring is not None:
            assert spec.source_substring in ef._source_string


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_arg_types_length(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    assert len(ef._arg_types) == spec.arg_count


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_default_function_name(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    assert ef._name == spec.expected_name


_NAME_VARIANTS = _flat(KERNEL_SPECS, "name_variants")


@pytest.mark.parametrize(
    "spec,kwargs,expected_name",
    _NAME_VARIANTS,
    ids=_flat_ids(_NAME_VARIANTS, "v"),
)
def test_name_variant(spec: KernelSpec, kwargs: dict, expected_name: str):
    ef = spec.factory(**kwargs)
    assert ef._name == expected_name


_INVALID = _flat(KERNEL_SPECS, "invalid_kwargs")


@pytest.mark.parametrize(
    "spec,kwargs,pattern",
    _INVALID,
    ids=_flat_ids(_INVALID, "bad"),
)
def test_invalid_kwargs_raise(spec: KernelSpec, kwargs: dict, pattern: str):
    with pytest.raises(ValueError, match=pattern):
        spec.factory(**kwargs)


_SHAPES = _flat(KERNEL_SPECS, "shape_checks")


@pytest.mark.parametrize(
    "spec,kwargs,arg_idx,expected_shape",
    _SHAPES,
    ids=_flat_ids(_SHAPES, "shape"),
)
def test_arg_shape(spec: KernelSpec, kwargs: dict, arg_idx: int, expected_shape: tuple):
    ef = spec.factory(**kwargs)
    arg = ef._arg_types[arg_idx]
    assert arg.__args__[0] == expected_shape


_TILE_SIZES = _flat(KERNEL_SPECS, "tile_size_checks")


@pytest.mark.parametrize(
    "spec,kwargs,expected_tile_size",
    _TILE_SIZES,
    ids=_flat_ids(_TILE_SIZES, "ts"),
)
def test_tile_size_at_arg_0(spec: KernelSpec, kwargs: dict, expected_tile_size: int):
    ef = spec.factory(**kwargs)
    assert ef.tile_size(0) == expected_tile_size


@pytest.mark.parametrize(
    "spec,kwargs,expected_arg_count",
    ARG_COUNT_OVERRIDES,
    ids=[f"{r[0].name}-argc{i}" for i, r in enumerate(ARG_COUNT_OVERRIDES)],
)
def test_arg_count_override(spec: KernelSpec, kwargs: dict, expected_arg_count: int):
    """Variant arg_counts (e.g. bn_conv2dk3_dw stride=1 has an extra arg)."""
    ef = spec.factory(**kwargs)
    assert len(ef._arg_types) == expected_arg_count


# ---------------------------------------------------------------------------
# Memoization + collision protection (kernels.X helpers and ExternalFunction)
#
# Background: a real bug caught while porting whole_array — calling
# kernels.mm() twice with different parameters produced two ExternalFunctions
# with the same default `<name>.o` filename.  Both were registered with the
# JIT, both got compiled, and the .o files silently overwrote each other.
# The wrong-flag kernel won the race, producing wrong hardware output.
#
# These tests pin down the two defenses:
#   (A) _make_extern memoizes on the full input parameter set, so identical
#       helper calls return the SAME instance.  Different parameterizations
#       get distinct instances AND distinct object_file_names (auto-suffixed
#       with a digest of compile_flags).
#   (B) ExternalFunction.__init__ refuses to register two instances with
#       the same (name, object_file_name) but a different content digest —
#       a backstop for code that bypasses the helper (constructs
#       ExternalFunction directly).
# ---------------------------------------------------------------------------


def test_kernels_mm_memoized_same_params_returns_same_instance():
    """Defense A: identical kernels.mm() calls return the exact same instance."""
    ef1 = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    ef2 = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert ef1 is ef2


def test_kernels_mm_different_params_returns_different_instances():
    """Defense A: different params get distinct instances (no spurious sharing)."""
    ef_plain = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)
    ef_ccm = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef_plain is not ef_ccm


def test_kernels_mm_different_params_have_distinct_object_files():
    """Defense A: distinct parameterizations get distinct object_file_names so
    JIT compilation outputs cannot overwrite each other on disk.
    This is the precise bug that broke whole_array c_col_maj support."""
    ef_plain = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)
    ef_ccm = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef_plain.object_file_name != ef_ccm.object_file_name


def test_kernels_mm_b_and_c_col_maj_independent():
    """Defense A: (b_col_maj, c_col_maj) tuples produce four distinct instances."""
    instances = {
        (b, c): kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=b, c_col_maj=c)
        for b in (False, True)
        for c in (False, True)
    }
    # All four are distinct ExternalFunction instances
    assert len({id(v) for v in instances.values()}) == 4
    # And all four have distinct .o filenames
    assert len({v.object_file_name for v in instances.values()}) == 4


def test_external_function_collision_check_fires():
    """Defense B: directly constructing two ExternalFunctions with the same
    (name, object_file_name) but different compile_flags is rejected."""
    # Use a unique source_string so we don't collide with real kernels.X
    # registrations from other tests in this session.
    src = "/* dummy */ int sentinel_for_collision_test() { return 0; }"
    a_ty = np.ndarray[(16,), np.dtype[np.int32]]
    name = "sentinel_for_collision_test"
    obj = "sentinel_for_collision_test.o"
    ExternalFunction(
        name=name,
        object_file_name=obj,
        source_string=src,
        arg_types=[],
        compile_flags=["-DA"],
    )
    with pytest.raises(ValueError, match="would collide"):
        ExternalFunction(
            name=name,
            object_file_name=obj,
            source_string=src,
            arg_types=[],
            compile_flags=["-DB"],  # different flags → different content digest
        )


def test_external_function_collision_check_allows_identical_redeclaration():
    """Defense B: re-registering an EXACT duplicate is fine (set semantics).

    Only differing-content collisions are rejected; identical re-instantiation
    might happen if a helper happens to be called twice without the memoization
    layer (e.g. by tests).  It must not raise.
    """
    src = "/* dummy */ int sentinel_redeclare_ok() { return 0; }"
    name = "sentinel_redeclare_ok"
    obj = "sentinel_redeclare_ok.o"
    ExternalFunction(
        name=name,
        object_file_name=obj,
        source_string=src,
        arg_types=[],
        compile_flags=["-DA"],
    )
    # Same content → same digest → no collision.
    ExternalFunction(
        name=name,
        object_file_name=obj,
        source_string=src,
        arg_types=[],
        compile_flags=["-DA"],
    )


# ---------------------------------------------------------------------------
# kernels.mm + kernels.mm_zero: disjoint symbol sets via -DMATMUL_ONLY /
# -DZERO_ONLY (added so both helpers can be used in the same design without
# duplicate-symbol link errors).  The matching gating is in
# aie_kernels/aie2{,p}/mm.cc.
# ---------------------------------------------------------------------------


def test_mm_carries_matmul_only_flag():
    """kernels.mm() must compile with -DMATMUL_ONLY so its .o exports only
    matmul_* (no zero_*).  Otherwise pairing with kernels.mm_zero() collides."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert "-DMATMUL_ONLY" in ef._compile_flags


def test_mm_zero_carries_zero_only_flag():
    """kernels.mm_zero() must compile with -DZERO_ONLY so its .o exports
    only zero_* (no matmul_*).  Otherwise pairing with kernels.mm() collides."""
    ef = kernels.mm_zero(dim_m=64, dim_k=64, dim_n=32, output_dtype=np.int16)
    assert "-DZERO_ONLY" in ef._compile_flags


def test_mm_and_mm_zero_have_disjoint_gating_flags():
    """The two flags are exclusive: mm carries MATMUL_ONLY but not ZERO_ONLY;
    mm_zero carries ZERO_ONLY but not MATMUL_ONLY.  Together their .o files
    have non-overlapping symbol sets.  This is the exact test that would have
    caught the duplicate-symbol footgun the matmul whole_array port hit."""
    mm_ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    mz_ef = kernels.mm_zero(dim_m=64, dim_k=64, dim_n=32, output_dtype=np.int16)
    assert "-DMATMUL_ONLY" in mm_ef._compile_flags
    assert "-DZERO_ONLY" not in mm_ef._compile_flags
    assert "-DZERO_ONLY" in mz_ef._compile_flags
    assert "-DMATMUL_ONLY" not in mz_ef._compile_flags


# ---------------------------------------------------------------------------
# Auto-prefix on symbol collision (Defense C — see _make_extern in
# python/iron/kernels/_common.py).
#
# Two kernels.X() calls with different parameterizations would otherwise
# produce two ExternalFunctions with the same C symbol name and trip
# duplicate-symbol errors at MLIR-verify and link time.  _make_extern now
# auto-prefixes the second-and-later instance's symbol with the same digest
# already used for object_file_name disambiguation.  The first call keeps
# the unprefixed canonical name (preserves byte-identity for existing
# single-version designs).
# ---------------------------------------------------------------------------


def test_kernels_mm_first_version_keeps_unprefixed_name():
    """First call (no existing same-named ExternalFunction): no auto-prefix."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int16, output_dtype=np.int16
    )
    assert ef._name == ef._original_name == "matmul_i16_i16"
    assert ef._symbol_prefix is None


def test_kernels_mm_two_versions_get_distinct_symbol_names():
    """Second call (existing same-named ExternalFunction) gets auto-prefixed
    so MLIR + linker don't see two `func.func @matmul_i16_i16` declarations."""
    ef1 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)
    ef2 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef1._name != ef2._name
    # The first one keeps the canonical symbol, the second one is prefixed.
    assert ef1._name == "matmul_i16_i16"
    assert ef2._name.endswith("_matmul_i16_i16")
    assert ef2._symbol_prefix is not None and len(ef2._symbol_prefix) == 8


def test_kernels_mm_three_versions_all_distinct_names():
    """Three distinct parameterizations get three distinct effective names."""
    efs = [
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=False, c_col_maj=False),
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=True, c_col_maj=False),
        kernels.mm(dim_m=64, dim_k=64, dim_n=32, b_col_maj=False, c_col_maj=True),
    ]
    names = [e._name for e in efs]
    assert len(set(names)) == 3
    # First instance still canonical, others prefixed.
    assert efs[0]._symbol_prefix is None
    assert all(e._symbol_prefix is not None for e in efs[1:])


def test_kernels_mm_prefixed_object_file_matches_prefixed_name():
    """The auto-prefixed instance's object_file_name aligns with the prefix
    so the .o filename advertises the same disambiguation as the symbol."""
    kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=False)  # prime the registry
    ef2 = kernels.mm(dim_m=64, dim_k=64, dim_n=32, c_col_maj=True)
    assert ef2._object_file_name.startswith(ef2._symbol_prefix)
    assert "matmul_i16_i16" in ef2._object_file_name
