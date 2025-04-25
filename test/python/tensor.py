# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu %pytest %s

import pytest
import numpy as np
import aie.iron as iron

@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_tensor_creation(dtype):
    t = iron.tensor((2, 2), dtype=dtype, device="npu")
    expected = np.zeros((2, 2), dtype=dtype)
    assert np.allclose(t, expected)
    assert t.shape == (2, 2)
    assert str(t.device) == "npu"


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_to_device(dtype):
    t = iron.ones((2, 2), dtype=dtype, device="cpu")
    t.to("npu")
    t.to("cpu")


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_zeros(dtype):
    assert np.allclose(iron.zeros(2, 3, dtype=dtype), np.zeros((2, 3), dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_ones(dtype):
    assert np.allclose(iron.ones((2, 2), dtype=dtype), np.ones((2, 2), dtype=dtype))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_random_with_bounds(dtype):
    t = iron.randint(0, 32, (2, 4), dtype=dtype, device="npu")
    assert t.shape == (2, 4)
    arr = t.numpy()
    assert np.all((arr >= 0) & (arr < 32))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rand(dtype):
    t = iron.rand(2, 2, dtype=dtype, device="npu")
    arr = t.numpy()
    assert np.all((arr >= 0) & (arr < 1.0))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_arange_integer(dtype):
    assert np.array_equal(iron.arange(3, 9, dtype=dtype), np.arange(3, 9, dtype=dtype))


def test_arange_floats():
    assert np.allclose(iron.arange(1.0, 5.0, 1.5), np.arange(1.0, 5.0, 1.5))


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_zeros_like(dtype):
    t = iron.tensor([[1, 2], [3, 4]], dtype=dtype)
    z = iron.zeros_like(t)
    expected = np.zeros_like(t)
    assert np.array_equal(z, expected)
