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


# saxpy tests are intentionally omitted: the saxpy factory was removed because
# saxpy.cc lives only in programming_examples/, not in the installed aie_kernels/ tree.
