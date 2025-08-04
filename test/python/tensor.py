# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

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


def test_tensor_repr():
    """Test that __repr__ properly syncs from device and shows correct data."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="npu")
    # Modify data on device
    t.to("npu")
    # Get string representation (should sync from device)
    repr_str = repr(t)
    print(repr_str)
    assert "tensor(" in repr_str
    assert "device='npu'" in repr_str
    # Check that the data values are present
    assert "1" in repr_str and "2" in repr_str and "3" in repr_str and "4" in repr_str


def test_tensor_getitem():
    """Test that __getitem__ properly syncs from device."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="npu")
    # Modify data on device
    t.to("npu")
    # Get item (should sync from device)
    value = t[0, 1]
    assert value == 2


def test_tensor_setitem():
    """Test that __setitem__ properly syncs to and from device."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="npu")
    t[0, 1] = 42
    # Verify the change is reflected
    assert t[0, 1] == 42
    # Verify other elements are unchanged
    assert t[0, 0] == 1
    assert t[1, 0] == 3
    assert t[1, 1] == 4


def test_tensor_getitem_setitem_consistency():
    """Test that getitem and setitem work consistently with device sync."""
    t = iron.zeros((2, 2), dtype=np.int32, device="npu")
    # Set values
    t[0, 0] = 10
    t[0, 1] = 20
    t[1, 0] = 30
    t[1, 1] = 40
    # Get values back
    assert t[0, 0] == 10
    assert t[0, 1] == 20
    assert t[1, 0] == 30
    assert t[1, 1] == 40
    # Verify the entire tensor
    expected = np.array([[10, 20], [30, 40]], dtype=np.int32)
    assert np.array_equal(t.numpy(), expected)
