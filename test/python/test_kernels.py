# test_kernels.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Unit tests for aie.iron.kernels factory functions."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Path setup: make python/iron/kernels/ discoverable when running against the
# installed package (which doesn't yet include the kernels subpackage).
# We extend aie.iron.__path__ so that Python can find the source-tree kernels/
# directory even when aie.iron itself is loaded from the install tree.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).parents[2]
_src_iron = _repo_root / "python" / "iron"
if _src_iron.is_dir():
    import aie.iron as _iron_pkg

    _src_iron_str = str(_src_iron)
    if _src_iron_str not in _iron_pkg.__path__:
        _iron_pkg.__path__.append(_src_iron_str)

from aie.iron.kernel import ExternalFunction
from aie.iron import kernels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_file(ef: ExternalFunction) -> str:
    """Return the source_file attribute of an ExternalFunction."""
    return ef._source_file


# ---------------------------------------------------------------------------
# passthrough
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_returns_external_function(self):
        ef = kernels.passthrough(tile_size=4096, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.passthrough(tile_size=4096, dtype=np.int32)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        ef = kernels.passthrough(tile_size=4096, dtype=np.int32)
        # (in_tile, out_tile, lineWidth: np.int32)
        assert len(ef._arg_types) == 3

    def test_function_name(self):
        ef = kernels.passthrough(tile_size=4096, dtype=np.int32)
        assert ef._name == "passThroughLine"

    def test_custom_tile_size(self):
        ef = kernels.passthrough(tile_size=256, dtype=np.uint8)
        assert ef.tile_size(0) == 256

    def test_dtype_reflected_in_arg_types(self):
        ef = kernels.passthrough(tile_size=64, dtype=np.int16)
        # first arg_type shape should be (64,)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (64,)


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


class TestScale:
    def test_returns_external_function(self):
        ef = kernels.scale(tile_size=1024, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.scale(tile_size=1024, dtype=np.int32)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_tile, out_tile, factor_tile, N: np.int32)
        ef = kernels.scale(tile_size=1024, dtype=np.int32)
        assert len(ef._arg_types) == 4

    def test_vectorized_function_name(self):
        ef = kernels.scale(tile_size=1024, dtype=np.int32, vectorized=True)
        assert ef._name == "vector_scalar_mul_vector"

    def test_scalar_function_name(self):
        ef = kernels.scale(tile_size=1024, dtype=np.int32, vectorized=False)
        assert ef._name == "vector_scalar_mul_scalar"

    def test_int16_dtype(self):
        ef = kernels.scale(tile_size=1024, dtype=np.int16)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 4

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be np.int16 or np.int32"):
            kernels.scale(tile_size=1024, dtype=np.float32)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_returns_external_function(self):
        ef = kernels.add(tile_size=1024, dtype=bfloat16)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.add(tile_size=1024, dtype=bfloat16)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (a_tile, b_tile, c_tile)
        ef = kernels.add(tile_size=1024, dtype=bfloat16)
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name(self):
        ef = kernels.add(tile_size=1024, dtype=bfloat16, vectorized=True)
        assert ef._name == "eltwise_add_bf16_vector"

    def test_scalar_function_name(self):
        ef = kernels.add(tile_size=1024, dtype=bfloat16, vectorized=False)
        assert ef._name == "eltwise_add_bf16_scalar"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be bfloat16"):
            kernels.add(tile_size=1024, dtype=np.float32)


# ---------------------------------------------------------------------------
# mul
# ---------------------------------------------------------------------------


class TestMul:
    def test_returns_external_function(self):
        ef = kernels.mul(tile_size=1024, dtype=bfloat16)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.mul(tile_size=1024, dtype=bfloat16)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (a_tile, b_tile, c_tile)
        ef = kernels.mul(tile_size=1024, dtype=bfloat16)
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name(self):
        ef = kernels.mul(tile_size=1024, dtype=bfloat16, vectorized=True)
        assert ef._name == "eltwise_mul_bf16_vector"

    def test_scalar_function_name(self):
        ef = kernels.mul(tile_size=1024, dtype=bfloat16, vectorized=False)
        assert ef._name == "eltwise_mul_bf16_scalar"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be bfloat16"):
            kernels.mul(tile_size=1024, dtype=np.float32)

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.mul(tile_size=512, dtype=bfloat16)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (512,)


# ---------------------------------------------------------------------------
# reduce_add
# ---------------------------------------------------------------------------


class TestReduceAdd:
    def test_returns_external_function(self):
        ef = kernels.reduce_add(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.reduce_add(tile_size=1024)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_tile, out_scalar, input_size: np.int32)
        ef = kernels.reduce_add(tile_size=1024)
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name(self):
        ef = kernels.reduce_add(tile_size=1024, vectorized=True)
        assert ef._name == "reduce_add_vector"

    def test_scalar_function_name(self):
        ef = kernels.reduce_add(tile_size=1024, vectorized=False)
        assert ef._name == "reduce_add_scalar"

    def test_explicit_int32_dtype(self):
        ef = kernels.reduce_add(tile_size=512, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)
        assert ef._name == "reduce_add_vector"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be np.int32"):
            kernels.reduce_add(tile_size=1024, dtype=bfloat16)

    def test_custom_tile_size(self):
        ef = kernels.reduce_add(tile_size=2048, dtype=np.int32)
        assert ef.tile_size(0) == 2048

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.reduce_add(tile_size=2048, dtype=np.int32)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (2048,)


# ---------------------------------------------------------------------------
# reduce_min
# ---------------------------------------------------------------------------


class TestReduceMin:
    def test_returns_external_function(self):
        ef = kernels.reduce_min(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.reduce_min(tile_size=1024)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_tile, out_scalar, input_size: np.int32)
        ef = kernels.reduce_min(tile_size=1024)
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name(self):
        ef = kernels.reduce_min(tile_size=1024, vectorized=True)
        assert ef._name == "reduce_min_vector"

    def test_scalar_function_name(self):
        ef = kernels.reduce_min(tile_size=1024, vectorized=False)
        assert ef._name == "reduce_min_scalar"

    def test_explicit_int32_dtype(self):
        ef = kernels.reduce_min(tile_size=512, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)
        assert ef._name == "reduce_min_vector"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be np.int32"):
            kernels.reduce_min(tile_size=1024, dtype=bfloat16)

    def test_custom_tile_size(self):
        ef = kernels.reduce_min(tile_size=2048, dtype=np.int32)
        assert ef.tile_size(0) == 2048

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.reduce_min(tile_size=2048, dtype=np.int32)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (2048,)


# ---------------------------------------------------------------------------
# reduce_max
# ---------------------------------------------------------------------------


class TestReduceMax:
    def test_returns_external_function_int32(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)

    def test_returns_external_function_bf16(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=bfloat16)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=np.int32)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_tile, out_scalar, input_size: np.int32)
        ef = kernels.reduce_max(tile_size=1024, dtype=np.int32)
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name_int32(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=np.int32, vectorized=True)
        assert ef._name == "reduce_max_vector"

    def test_vectorized_function_name_bf16(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=bfloat16, vectorized=True)
        assert ef._name == "reduce_max_vector_bfloat16"

    def test_scalar_function_name_int32(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=np.int32, vectorized=False)
        assert ef._name == "reduce_max_scalar"

    def test_scalar_function_name_bf16(self):
        ef = kernels.reduce_max(tile_size=1024, dtype=bfloat16, vectorized=False)
        assert ef._name == "reduce_max_scalar_bfloat16"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be np.int32 or bfloat16"):
            kernels.reduce_max(tile_size=1024, dtype=np.float32)

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.reduce_max(tile_size=2048, dtype=np.int32)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (2048,)


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


class TestRelu:
    def test_returns_external_function(self):
        ef = kernels.relu(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.relu(tile_size=1024)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_tile, out_tile)
        ef = kernels.relu(tile_size=1024)
        assert len(ef._arg_types) == 2

    def test_function_name(self):
        ef = kernels.relu(tile_size=1024)
        assert ef._name == "bf16_relu"

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.relu(tile_size=512)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (512,)


# ---------------------------------------------------------------------------
# rgba2hue
# ---------------------------------------------------------------------------


class TestRgba2Hue:
    def test_returns_external_function(self):
        ef = kernels.rgba2hue(line_width=1920)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.rgba2hue(line_width=1920)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_rgba, out_hue, lineWidth: np.int32)
        ef = kernels.rgba2hue(line_width=1920)
        assert len(ef._arg_types) == 3

    def test_function_name(self):
        ef = kernels.rgba2hue(line_width=1920)
        assert ef._name == "rgba2hueLine"

    def test_input_shape_is_4x_line_width(self):
        ef = kernels.rgba2hue(line_width=640)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640 * 4,)

    def test_output_shape_is_line_width(self):
        ef = kernels.rgba2hue(line_width=640)
        arg1 = ef._arg_types[1]
        assert arg1.__args__[0] == (640,)


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_returns_external_function(self):
        ef = kernels.threshold(line_width=1920, dtype=np.uint8)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.threshold(line_width=1920, dtype=np.uint8)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_line, out_line, lineWidth, thresholdValue, maxValue, thresholdType)
        ef = kernels.threshold(line_width=1920, dtype=np.uint8)
        assert len(ef._arg_types) == 6

    def test_function_name(self):
        ef = kernels.threshold(line_width=1920, dtype=np.uint8)
        assert ef._name == "thresholdLine"

    def test_int16_dtype(self):
        ef = kernels.threshold(line_width=1920, dtype=np.int16)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 6

    def test_int32_dtype(self):
        ef = kernels.threshold(line_width=1920, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 6

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            kernels.threshold(line_width=1920, dtype=np.float32)

    def test_custom_line_width_reflected_in_arg_types(self):
        ef = kernels.threshold(line_width=640, dtype=np.uint8)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640,)


# ---------------------------------------------------------------------------
# bitwiseOR
# ---------------------------------------------------------------------------


class TestBitwiseOR:
    def test_returns_external_function(self):
        ef = kernels.bitwiseOR(line_width=1920, dtype=np.uint8)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bitwiseOR(line_width=1920, dtype=np.uint8)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in1, in2, out, lineWidth: np.int32)
        ef = kernels.bitwiseOR(line_width=1920, dtype=np.uint8)
        assert len(ef._arg_types) == 4

    def test_function_name(self):
        ef = kernels.bitwiseOR(line_width=1920, dtype=np.uint8)
        assert ef._name == "bitwiseORLine"

    def test_int16_dtype(self):
        ef = kernels.bitwiseOR(line_width=1920, dtype=np.int16)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 4

    def test_int32_dtype(self):
        ef = kernels.bitwiseOR(line_width=1920, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 4

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            kernels.bitwiseOR(line_width=1920, dtype=np.float32)

    def test_custom_line_width_reflected_in_arg_types(self):
        ef = kernels.bitwiseOR(line_width=640, dtype=np.uint8)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640,)


# ---------------------------------------------------------------------------
# bitwiseAND
# ---------------------------------------------------------------------------


class TestBitwiseAND:
    def test_returns_external_function(self):
        ef = kernels.bitwiseAND(line_width=1920, dtype=np.uint8)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bitwiseAND(line_width=1920, dtype=np.uint8)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in1, in2, out, lineWidth: np.int32)
        ef = kernels.bitwiseAND(line_width=1920, dtype=np.uint8)
        assert len(ef._arg_types) == 4

    def test_function_name(self):
        ef = kernels.bitwiseAND(line_width=1920, dtype=np.uint8)
        assert ef._name == "bitwiseANDLine"

    def test_int16_dtype(self):
        ef = kernels.bitwiseAND(line_width=1920, dtype=np.int16)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 4

    def test_int32_dtype(self):
        ef = kernels.bitwiseAND(line_width=1920, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 4

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            kernels.bitwiseAND(line_width=1920, dtype=np.float32)

    def test_custom_line_width_reflected_in_arg_types(self):
        ef = kernels.bitwiseAND(line_width=640, dtype=np.uint8)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640,)


# ---------------------------------------------------------------------------
# gray2rgba
# ---------------------------------------------------------------------------


class TestGray2Rgba:
    def test_returns_external_function(self):
        ef = kernels.gray2rgba(line_width=1920)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.gray2rgba(line_width=1920)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_gray, out_rgba, lineWidth: np.int32)
        ef = kernels.gray2rgba(line_width=1920)
        assert len(ef._arg_types) == 3

    def test_function_name(self):
        ef = kernels.gray2rgba(line_width=1920)
        assert ef._name == "gray2rgbaLine"

    def test_input_shape_is_line_width(self):
        ef = kernels.gray2rgba(line_width=640)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640,)

    def test_output_shape_is_4x_line_width(self):
        ef = kernels.gray2rgba(line_width=640)
        arg1 = ef._arg_types[1]
        assert arg1.__args__[0] == (640 * 4,)


# ---------------------------------------------------------------------------
# rgba2gray
# ---------------------------------------------------------------------------


class TestRgba2Gray:
    def test_returns_external_function(self):
        ef = kernels.rgba2gray(line_width=1920)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.rgba2gray(line_width=1920)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in_rgba, out_gray, lineWidth: np.int32)
        ef = kernels.rgba2gray(line_width=1920)
        assert len(ef._arg_types) == 3

    def test_function_name(self):
        ef = kernels.rgba2gray(line_width=1920)
        assert ef._name == "rgba2grayLine"

    def test_input_shape_is_4x_line_width(self):
        ef = kernels.rgba2gray(line_width=640)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640 * 4,)

    def test_output_shape_is_line_width(self):
        ef = kernels.rgba2gray(line_width=640)
        arg1 = ef._arg_types[1]
        assert arg1.__args__[0] == (640,)


# ---------------------------------------------------------------------------
# filter2d
# ---------------------------------------------------------------------------


class TestFilter2d:
    def test_returns_external_function(self):
        ef = kernels.filter2d(line_width=1920)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.filter2d(line_width=1920)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (lineIn0, lineIn1, lineIn2, out, lineWidth: np.int32, filterKernel)
        ef = kernels.filter2d(line_width=1920)
        assert len(ef._arg_types) == 6

    def test_function_name(self):
        ef = kernels.filter2d(line_width=1920)
        assert ef._name == "filter2dLine"

    def test_custom_line_width_reflected_in_arg_types(self):
        ef = kernels.filter2d(line_width=640)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640,)


# ---------------------------------------------------------------------------
# addWeighted
# ---------------------------------------------------------------------------


class TestAddWeighted:
    def test_returns_external_function(self):
        ef = kernels.addWeighted(line_width=1920, dtype=np.uint8)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.addWeighted(line_width=1920, dtype=np.uint8)
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (in1, in2, out, lineWidth, alpha, beta, gamma)
        ef = kernels.addWeighted(line_width=1920, dtype=np.uint8)
        assert len(ef._arg_types) == 7

    def test_function_name(self):
        ef = kernels.addWeighted(line_width=1920, dtype=np.uint8)
        assert ef._name == "addWeightedLine"

    def test_int16_dtype(self):
        ef = kernels.addWeighted(line_width=1920, dtype=np.int16)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 7

    def test_int32_dtype(self):
        ef = kernels.addWeighted(line_width=1920, dtype=np.int32)
        assert isinstance(ef, ExternalFunction)
        assert len(ef._arg_types) == 7

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            kernels.addWeighted(line_width=1920, dtype=np.float32)

    def test_custom_line_width_reflected_in_arg_types(self):
        ef = kernels.addWeighted(line_width=640, dtype=np.uint8)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (640,)


# saxpy tests are intentionally omitted: the saxpy factory was removed because
# saxpy.cc lives only in programming_examples/, not in the installed aie_kernels/ tree.


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_returns_external_function(self):
        ef = kernels.softmax(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_has_source(self):
        ef = kernels.softmax(tile_size=1024)
        # On aie2 (default), uses source_string with #include directives
        assert ef._source_string is not None or ef._source_file is not None

    def test_source_string_includes_kernel(self):
        ef = kernels.softmax(tile_size=1024)
        if ef._source_string is not None:
            assert "softmax.cc" in ef._source_string

    def test_arg_types_length(self):
        # (input_tile, output_tile, input_size: np.int32)
        ef = kernels.softmax(tile_size=1024)
        assert len(ef._arg_types) == 3

    def test_function_name(self):
        ef = kernels.softmax(tile_size=1024)
        assert ef._name == "softmax_bf16"

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.softmax(tile_size=2048)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (2048,)


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------


class TestGelu:
    def test_returns_external_function(self):
        ef = kernels.gelu(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_has_source(self):
        ef = kernels.gelu(tile_size=1024)
        assert ef._source_string is not None or ef._source_file is not None

    def test_source_string_includes_kernel(self):
        ef = kernels.gelu(tile_size=1024)
        if ef._source_string is not None:
            assert "gelu.cc" in ef._source_string

    def test_arg_types_length(self):
        # (input_tile, output_tile)
        ef = kernels.gelu(tile_size=1024)
        assert len(ef._arg_types) == 2

    def test_function_name(self):
        ef = kernels.gelu(tile_size=1024)
        assert ef._name == "gelu_bf16"

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.gelu(tile_size=512)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (512,)


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------


class TestSilu:
    def test_returns_external_function(self):
        ef = kernels.silu(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_has_source(self):
        ef = kernels.silu(tile_size=1024)
        assert ef._source_string is not None or ef._source_file is not None

    def test_source_string_includes_kernel(self):
        ef = kernels.silu(tile_size=1024)
        if ef._source_string is not None:
            assert "silu.cc" in ef._source_string

    def test_arg_types_length(self):
        # (input_tile, output_tile)
        ef = kernels.silu(tile_size=1024)
        assert len(ef._arg_types) == 2

    def test_function_name(self):
        ef = kernels.silu(tile_size=1024)
        assert ef._name == "silu_bf16"

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.silu(tile_size=512)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (512,)


# ---------------------------------------------------------------------------
# swiglu
# ---------------------------------------------------------------------------


class TestSwiglu:
    def test_returns_external_function(self):
        ef = kernels.swiglu(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_has_source(self):
        ef = kernels.swiglu(tile_size=1024)
        assert ef._source_string is not None or ef._source_file is not None

    def test_source_string_includes_kernel(self):
        ef = kernels.swiglu(tile_size=1024)
        if ef._source_string is not None:
            assert "swiglu.cc" in ef._source_string

    def test_arg_types_length(self):
        # (input, weights_1, weights_2, output)
        ef = kernels.swiglu(tile_size=1024)
        assert len(ef._arg_types) == 4

    def test_function_name(self):
        ef = kernels.swiglu(tile_size=1024)
        assert ef._name == "swiglu_bf16"

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.swiglu(tile_size=512)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (512,)


# ---------------------------------------------------------------------------
# bf16_exp
# ---------------------------------------------------------------------------


class TestBf16Exp:
    def test_returns_external_function(self):
        ef = kernels.bf16_exp(tile_size=1024)
        assert isinstance(ef, ExternalFunction)

    def test_has_source(self):
        ef = kernels.bf16_exp(tile_size=1024)
        assert ef._source_string is not None or ef._source_file is not None

    def test_source_string_includes_kernel(self):
        ef = kernels.bf16_exp(tile_size=1024)
        if ef._source_string is not None:
            assert "bf16_exp.cc" in ef._source_string

    def test_arg_types_length(self):
        # (input_tile, output_tile)
        ef = kernels.bf16_exp(tile_size=1024)
        assert len(ef._arg_types) == 2

    def test_function_name(self):
        ef = kernels.bf16_exp(tile_size=1024)
        assert ef._name == "exp_bf16_1024"

    def test_custom_tile_size_reflected_in_arg_types(self):
        ef = kernels.bf16_exp(tile_size=512)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (512,)


# ---------------------------------------------------------------------------
# mm (matrix multiply)
# ---------------------------------------------------------------------------


class TestMM:
    def test_returns_external_function(self):
        ef = kernels.mm(dim_m=64, dim_k=64, dim_n=64)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.mm()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (a_in, b_in, c_out)
        ef = kernels.mm()
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name(self):
        ef = kernels.mm(input_dtype=np.int16, output_dtype=np.int16, vectorized=True)
        assert ef._name == "matmul_i16_i16"

    def test_scalar_function_name(self):
        ef = kernels.mm(input_dtype=np.int16, output_dtype=np.int16, vectorized=False)
        assert ef._name == "matmul_scalar_i16_i16"

    def test_bf16_combo(self):
        ef = kernels.mm(input_dtype=bfloat16, output_dtype=bfloat16)
        assert ef._name == "matmul_bf16_bf16"

    def test_i8_i8_combo(self):
        ef = kernels.mm(input_dtype=np.int8, output_dtype=np.int8)
        assert ef._name == "matmul_i8_i8"

    def test_bf16_f32_combo(self):
        ef = kernels.mm(input_dtype=bfloat16, output_dtype=np.float32)
        assert ef._name == "matmul_bf16_f32"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            kernels.mm(input_dtype=np.float64, output_dtype=np.float64)

    def test_custom_dims_reflected_in_arg_types(self):
        ef = kernels.mm(dim_m=32, dim_k=16, dim_n=48)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (32 * 16,)
        arg2 = ef._arg_types[2]
        assert arg2.__args__[0] == (32 * 48,)


# ---------------------------------------------------------------------------
# mm_zero
# ---------------------------------------------------------------------------


class TestMMZero:
    def test_returns_external_function(self):
        ef = kernels.mm_zero(dim_m=64, dim_n=64, output_dtype=np.int16)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.mm_zero()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (c_out,)
        ef = kernels.mm_zero()
        assert len(ef._arg_types) == 1

    def test_vectorized_function_name(self):
        ef = kernels.mm_zero(output_dtype=np.int16, vectorized=True)
        assert ef._name == "zero_i16"

    def test_scalar_function_name(self):
        ef = kernels.mm_zero(output_dtype=np.int16, vectorized=False)
        assert ef._name == "zero_scalar_i16"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            kernels.mm_zero(output_dtype=np.float64)


# ---------------------------------------------------------------------------
# mv (matrix-vector multiply)
# ---------------------------------------------------------------------------


class TestMV:
    def test_returns_external_function(self):
        ef = kernels.mv(dim_m=32, dim_k=32)
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.mv()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (a_in, b_in, c_out)
        ef = kernels.mv()
        assert len(ef._arg_types) == 3

    def test_vectorized_function_name(self):
        ef = kernels.mv(vectorized=True)
        assert ef._name == "matvec_vectorized_i16_i32"

    def test_scalar_function_name(self):
        ef = kernels.mv(vectorized=False)
        assert ef._name == "matvec_scalar_i16_i32"

    def test_custom_dims_reflected_in_arg_types(self):
        ef = kernels.mv(dim_m=16, dim_k=64)
        arg0 = ef._arg_types[0]
        assert arg0.__args__[0] == (16 * 64,)
        arg1 = ef._arg_types[1]
        assert arg1.__args__[0] == (64,)
        arg2 = ef._arg_types[2]
        assert arg2.__args__[0] == (16,)

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="only.*supported"):
            kernels.mv(input_dtype=np.int8, output_dtype=np.int8)


# ---------------------------------------------------------------------------
# cascade_mm
# ---------------------------------------------------------------------------


class TestCascadeMM:
    def test_returns_external_function(self):
        ef = kernels.cascade_mm()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.cascade_mm()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (a_in, b_in, c_out)
        ef = kernels.cascade_mm()
        assert len(ef._arg_types) == 3

    def test_get_only_function_name(self):
        ef = kernels.cascade_mm(cascade_mode="get_only")
        assert ef._name == "matmul_scalar_cascade_get_only_i16_i16"

    def test_put_only_function_name(self):
        ef = kernels.cascade_mm(cascade_mode="put_only")
        assert ef._name == "matmul_scalar_cascade_put_only_i16_i16"

    def test_put_get_function_name(self):
        ef = kernels.cascade_mm(cascade_mode="put_get")
        assert ef._name == "matmul_scalar_cascade_put_get_i16_i16"

    def test_bf16_combo(self):
        ef = kernels.cascade_mm(
            input_dtype=bfloat16, output_dtype=bfloat16, cascade_mode="get_only"
        )
        assert ef._name == "matmul_scalar_cascade_get_only_bf16_bf16"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="cascade_mode"):
            kernels.cascade_mm(cascade_mode="invalid")

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            kernels.cascade_mm(input_dtype=np.int8, output_dtype=np.int8)


# ---------------------------------------------------------------------------
# conv2dk1
# ---------------------------------------------------------------------------


class TestConv2dk1:
    def test_returns_external_function(self):
        ef = kernels.conv2dk1()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.conv2dk1()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input, kernels, output, input_width, input_channels,
        #  output_channels, scale)
        ef = kernels.conv2dk1()
        assert len(ef._arg_types) == 7

    def test_i8_function_name(self):
        ef = kernels.conv2dk1(act_dtype=np.int8)
        assert ef._name == "conv2dk1_i8"

    def test_ui8_function_name(self):
        ef = kernels.conv2dk1(act_dtype=np.uint8)
        assert ef._name == "conv2dk1_ui8"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="act_dtype"):
            kernels.conv2dk1(act_dtype=np.float32)


# ---------------------------------------------------------------------------
# conv2dk3
# ---------------------------------------------------------------------------


class TestConv2dk3:
    def test_returns_external_function(self):
        ef = kernels.conv2dk3()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.conv2dk3()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (line0, line1, line2, wts, output,
        #  input_width, input_channels, output_channels, kernel_width,
        #  kernel_height, check, scale, channel_offset)
        ef = kernels.conv2dk3()
        assert len(ef._arg_types) == 13

    def test_i8_function_name(self):
        ef = kernels.conv2dk3(act_dtype=np.int8)
        assert ef._name == "conv2dk3_i8"

    def test_ui8_function_name(self):
        ef = kernels.conv2dk3(act_dtype=np.uint8)
        assert ef._name == "conv2dk3_ui8"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="act_dtype"):
            kernels.conv2dk3(act_dtype=np.float32)


# ---------------------------------------------------------------------------
# conv2dk1_skip
# ---------------------------------------------------------------------------


class TestConv2dk1Skip:
    def test_returns_external_function(self):
        ef = kernels.conv2dk1_skip()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.conv2dk1_skip()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input0, input1, kernels, output, skip,
        #  input_width, input_channels, output_channels, scale, skip_scale)
        ef = kernels.conv2dk1_skip()
        assert len(ef._arg_types) == 10

    def test_i8_function_name(self):
        ef = kernels.conv2dk1_skip(act_dtype=np.int8)
        assert ef._name == "conv2dk1_skip_i8"

    def test_ui8_function_name(self):
        ef = kernels.conv2dk1_skip(act_dtype=np.uint8)
        assert ef._name == "conv2dk1_skip_ui8"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="act_dtype"):
            kernels.conv2dk1_skip(act_dtype=np.float32)


# ---------------------------------------------------------------------------
# conv2dk1_i8
# ---------------------------------------------------------------------------


class TestConv2dk1I8:
    def test_returns_external_function(self):
        ef = kernels.conv2dk1_i8()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.conv2dk1_i8()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input, kernels, output, input_width, input_channels,
        #  output_channels, scale)
        ef = kernels.conv2dk1_i8()
        assert len(ef._arg_types) == 7

    def test_function_name(self):
        ef = kernels.conv2dk1_i8()
        assert ef._name == "conv2dk1_i8"


# ---------------------------------------------------------------------------
# conv2dk14
# ---------------------------------------------------------------------------


class TestConv2dk14:
    def test_returns_external_function(self):
        ef = kernels.conv2dk14()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.conv2dk14()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input, kernels, output, input_width, input_channels,
        #  output_channels, kernel_width, scale)
        ef = kernels.conv2dk14()
        assert len(ef._arg_types) == 8

    def test_function_name(self):
        ef = kernels.conv2dk14()
        assert ef._name == "conv2dk14_i8"


# ---------------------------------------------------------------------------
# conv2dk1_skip_init
# ---------------------------------------------------------------------------


class TestConv2dk1SkipInit:
    def test_returns_external_function(self):
        ef = kernels.conv2dk1_skip_init()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.conv2dk1_skip_init()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input0, input1, kernels, output, skip,
        #  input_width, input_channels, output_channels,
        #  input_channels_skip, scale, skip_scale, scale_skip_conv)
        ef = kernels.conv2dk1_skip_init()
        assert len(ef._arg_types) == 12

    def test_i8_function_name(self):
        ef = kernels.conv2dk1_skip_init(act_dtype=np.int8)
        assert ef._name == "conv2dk1_skip_init_i8"

    def test_ui8_function_name(self):
        ef = kernels.conv2dk1_skip_init(act_dtype=np.uint8)
        assert ef._name == "conv2dk1_skip_init_ui8"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="act_dtype"):
            kernels.conv2dk1_skip_init(act_dtype=np.float32)


# ---------------------------------------------------------------------------
# bn_conv2dk1_relu
# ---------------------------------------------------------------------------


class TestBnConv2dk1Relu:
    def test_returns_external_function(self):
        ef = kernels.bn_conv2dk1_relu()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bn_conv2dk1_relu()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input, kernels, output, input_width, input_channels,
        #  output_channels, scale)
        ef = kernels.bn_conv2dk1_relu()
        assert len(ef._arg_types) == 7

    def test_function_name(self):
        ef = kernels.bn_conv2dk1_relu()
        assert ef._name == "conv2dk1_relu_i8_ui8"


# ---------------------------------------------------------------------------
# bn_conv2dk3
# ---------------------------------------------------------------------------


class TestBnConv2dk3:
    def test_returns_external_function(self):
        ef = kernels.bn_conv2dk3()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bn_conv2dk3()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (line0, line1, line2, wts, output,
        #  input_width, input_channels, output_channels, kernel_width,
        #  kernel_height, check, scale, channel_offset)
        ef = kernels.bn_conv2dk3()
        assert len(ef._arg_types) == 13

    def test_function_name(self):
        ef = kernels.bn_conv2dk3()
        assert ef._name == "conv2dk3_stride2_i8"


# ---------------------------------------------------------------------------
# bn_conv2dk1_i8
# ---------------------------------------------------------------------------


class TestBnConv2dk1I8:
    def test_returns_external_function(self):
        ef = kernels.bn_conv2dk1_i8()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bn_conv2dk1_i8()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input, kernels, output, input_width, input_channels,
        #  output_channels, scale)
        ef = kernels.bn_conv2dk1_i8()
        assert len(ef._arg_types) == 7

    def test_function_name(self):
        ef = kernels.bn_conv2dk1_i8()
        assert ef._name == "conv2dk1_ui8_i8"


# ---------------------------------------------------------------------------
# bn_conv2dk1_skip
# ---------------------------------------------------------------------------


class TestBnConv2dk1Skip:
    def test_returns_external_function(self):
        ef = kernels.bn_conv2dk1_skip()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bn_conv2dk1_skip()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_arg_types_length(self):
        # (input, kernels, output, skip,
        #  input_width, input_channels, output_channels, scale, skip_scale)
        ef = kernels.bn_conv2dk1_skip()
        assert len(ef._arg_types) == 9

    def test_ui8_skip_function_name(self):
        ef = kernels.bn_conv2dk1_skip(skip_dtype=np.uint8)
        assert ef._name == "conv2dk1_skip_ui8_ui8_i8"

    def test_i8_skip_function_name(self):
        ef = kernels.bn_conv2dk1_skip(skip_dtype=np.int8)
        assert ef._name == "conv2dk1_skip_ui8_i8_i8"

    def test_invalid_skip_dtype_raises(self):
        with pytest.raises(ValueError, match="skip_dtype"):
            kernels.bn_conv2dk1_skip(skip_dtype=np.float32)


# ---------------------------------------------------------------------------
# bn_conv2dk3_dw
# ---------------------------------------------------------------------------


class TestBnConv2dk3Dw:
    def test_returns_external_function(self):
        ef = kernels.bn_conv2dk3_dw()
        assert isinstance(ef, ExternalFunction)

    def test_source_file_exists(self):
        ef = kernels.bn_conv2dk3_dw()
        src = _source_file(ef)
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"

    def test_stride1_function_name(self):
        ef = kernels.bn_conv2dk3_dw(stride=1)
        assert ef._name == "conv2dk3_dw_stride1_relu_ui8_ui8"

    def test_stride2_function_name(self):
        ef = kernels.bn_conv2dk3_dw(stride=2)
        assert ef._name == "conv2dk3_dw_stride2_relu_ui8_ui8"

    def test_stride2_arg_types_length(self):
        # (line0, line1, line2, wts, output,
        #  input_width, input_channels, output_channels, kernel_width,
        #  kernel_height, check, scale, channel_offset)
        ef = kernels.bn_conv2dk3_dw(stride=2)
        assert len(ef._arg_types) == 13

    def test_stride1_arg_types_length(self):
        # stride1 has an extra output split arg
        ef = kernels.bn_conv2dk3_dw(stride=1)
        assert len(ef._arg_types) == 14

    def test_invalid_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            kernels.bn_conv2dk3_dw(stride=3)
