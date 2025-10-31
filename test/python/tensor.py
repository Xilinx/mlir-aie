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
def test_fill(dtype):
    """Test the fill_ method for in-place tensor filling."""
    t = iron.zeros((2, 3), dtype=dtype, device="npu")

    # Fill with a specific value
    fill_value = 42 if dtype == np.int32 else 42.5
    t.fill_(fill_value)

    # Verify the tensor is filled with the correct value
    expected = np.full((2, 3), fill_value, dtype=dtype)
    assert np.allclose(t.numpy(), expected)

    # Test with different value
    new_fill_value = 99 if dtype == np.int32 else 99.9
    t.fill_(new_fill_value)
    expected = np.full((2, 3), new_fill_value, dtype=dtype)
    assert np.allclose(t.numpy(), expected)


def test_fill_cpu_tensor():
    """Test fill_ method on CPU tensors."""
    t = iron.zeros((2, 2), dtype=np.int32, device="cpu")
    t.fill_(123)
    expected = np.full((2, 2), 123, dtype=np.int32)
    assert np.array_equal(t.numpy(), expected)


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


def test_cpu_tensor_no_sync():
    """Test that CPU tensors operations."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="cpu")
    assert t[0, 1] == 2
    t[0, 1] = 42
    assert t[0, 1] == 42
    assert "device='cpu'" in repr(t)
    arr = t.numpy()
    assert np.array_equal(arr, np.array([[1, 42], [3, 4]], dtype=np.int32))


def test_device_attribute_update():
    """Test that to() method properly updates the device attribute."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="cpu")
    assert t.device == "cpu"

    # Move to NPU
    t.to("npu")
    assert t.device == "npu"
    assert "device='npu'" in repr(t)

    # Move back to CPU
    t.to("cpu")
    assert t.device == "cpu"
    assert "device='cpu'" in repr(t)


def test_npu_tensor_sync_behavior():
    """Test that NPU tensors when implicit sync is required."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="npu")
    assert t.device == "npu"

    # Test that accessing data works correctly
    assert t[0, 1] == 2
    t[0, 1] = 42
    assert t[0, 1] == 42

    # Test that numpy() returns correct data
    arr = t.numpy()
    expected = np.array([[1, 42], [3, 4]], dtype=np.int32)
    assert np.array_equal(arr, expected)

    # Test that __array__ protocol works
    np_arr = np.array(t)
    assert np.array_equal(np_arr, expected)


def test_mixed_device_operations():
    """Test operations between CPU and NPU tensors."""
    # Create tensors on different devices
    cpu_tensor = iron.tensor([[1, 2], [3, 4]], dtype=np.int32, device="cpu")
    npu_tensor = iron.tensor([[5, 6], [7, 8]], dtype=np.int32, device="npu")

    # Test device attributes
    assert cpu_tensor.device == "cpu"
    assert npu_tensor.device == "npu"

    # Test that both can be accessed without issues
    assert cpu_tensor[0, 0] == 1
    assert npu_tensor[0, 0] == 5

    # Test moving between devices
    cpu_tensor = cpu_tensor.to("npu")
    assert cpu_tensor.device == "npu"

    npu_tensor = npu_tensor.to("cpu")
    assert npu_tensor.device == "cpu"
