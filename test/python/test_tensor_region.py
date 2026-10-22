# test_tensor_region.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""``NpuTensor.region``: a kernel argument that may start or end mid cache line.

What a runtime does with a region is tested on hardware in
``npu-xrt/test_xrt_tensor_region.py``. These tests cover the region itself.
"""

import gc

import numpy as np
import pytest

from aie.utils.compile.cache.utils import _create_function_cache_key
from aie.utils.hostruntime.tensor_class import CPUOnlyTensor, NpuTensor, TensorRegion

GRANULE = 64


def test_region_may_start_mid_granule():
    tensor = CPUOnlyTensor((1024,), dtype=np.int16)
    region = tensor.region(2000, (24,))
    assert isinstance(region, TensorRegion)
    assert region.tensor is tensor
    assert region.storage_offset == 2000
    assert region.shape == (24,)
    assert region.dtype == np.dtype(np.int16)
    assert region.nbytes == 48


def test_subview_refuses_the_same_region():
    tensor = CPUOnlyTensor((1024,), dtype=np.int16)
    with pytest.raises(ValueError, match="coherence granule"):
        tensor.subview(2000, (24,))


def test_region_dtype_reinterprets_the_bytes():
    tensor = CPUOnlyTensor((256,), dtype=np.uint8)
    region = tensor.region(8, (2, 3), dtype=np.int32)
    assert region.shape == (2, 3)
    assert region.dtype == np.dtype(np.int32)
    assert region.nbytes == 24


def test_region_of_a_subview_is_measured_from_the_allocation():
    tensor = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    view = tensor.subview(GRANULE, (2 * GRANULE,))
    region = view.region(12, (8,))
    assert region.tensor is view
    assert region.storage_offset == GRANULE + 12


def test_region_must_start_on_a_word():
    tensor = CPUOnlyTensor((256,), dtype=np.uint8)
    with pytest.raises(ValueError, match="not a multiple of 4"):
        tensor.region(2, (8,))


def test_region_word_rule_applies_to_the_allocation_offset():
    tensor = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    view = tensor.subview(GRANULE, (2 * GRANULE,))
    with pytest.raises(ValueError, match=f"starts at byte {GRANULE + 6}"):
        view.region(6, (8,))


def test_region_must_fit_in_its_tensor():
    tensor = CPUOnlyTensor((16,), dtype=np.int32)
    with pytest.raises(ValueError, match=r"region\(.*out of bounds"):
        tensor.region(60, (2,))
    with pytest.raises(ValueError, match="out of bounds"):
        tensor.region(-4, (1,))


def test_region_rejects_a_non_integer_offset():
    tensor = CPUOnlyTensor((16,), dtype=np.int32)
    with pytest.raises(TypeError, match=r"region\(\) offset must be an integer"):
        tensor.region(4.0, (1,))


def test_region_keeps_its_tensor_alive():
    tensor = CPUOnlyTensor((64,), dtype=np.uint8)
    tensor.data[:] = 0x22
    region = tensor.region(4, (8,))
    del tensor
    gc.collect()
    np.testing.assert_array_equal(region.tensor.data, np.full(64, 0x22, dtype=np.uint8))


def test_region_is_not_a_tensor():
    region = CPUOnlyTensor((64,), dtype=np.uint8).region(4, (8,))
    assert not isinstance(region, NpuTensor)


def test_regions_of_one_shape_share_a_compile_cache_key():
    def design():
        pass

    tensor = CPUOnlyTensor((1024,), dtype=np.int32)
    first = _create_function_cache_key(design, (tensor.region(0, (16,)),), {})
    second = _create_function_cache_key(design, (tensor.region(68, (16,)),), {})
    other = _create_function_cache_key(design, (tensor.region(0, (32,)),), {})
    assert first == second
    assert first != other
