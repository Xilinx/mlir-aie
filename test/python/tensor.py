# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np
import aie.iron as iron
from aie.iron.hostruntime.tensor import CPUOnlyTensor, Tensor
from aie.iron.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16

TENSOR_CLASSES = [CPUOnlyTensor, XRTTensor]
TEST_DTYPES = [np.float32, np.int32, bfloat16]


def bfloat16_safe_allclose(dtype, arr1, arr2):
    if dtype == bfloat16:
        if isinstance(arr1, Tensor):
            arr1 = np.array(arr1, dtype=np.float16)
        else:
            arr1 = arr1.astype(np.float16)
        if isinstance(arr2, Tensor):
            arr2 = np.array(arr2, dtype=np.float16)
        else:
            arr2 = arr2.astype(np.float16)
    return np.allclose(arr1, arr2)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_tensor_creation(dtype, tensorclass):
    for d in tensorclass.DEVICES:
        t = tensorclass((2, 2), dtype=dtype, device=d)
        assert t.dtype == dtype
        assert isinstance(t, iron.hostruntime.Tensor)
        assert isinstance(t, tensorclass)
        expected = np.zeros((2, 2), dtype=dtype)
        assert bfloat16_safe_allclose(dtype, t, expected)
        assert t.shape == (2, 2)
        assert str(t.device) == d


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_to_device(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.ones((2, 2), dtype=dtype, device=d)
        assert isinstance(t, iron.hostruntime.Tensor)
        assert isinstance(t, tensorclass)
        assert t.dtype == dtype
        for d2 in tensorclass.DEVICES:
            t.to(d2)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_zeros(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    t = iron.zeros(2, 3, dtype=dtype)
    assert isinstance(t, tensorclass)
    assert bfloat16_safe_allclose(dtype, t, np.zeros((2, 3), dtype=dtype))


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_ones(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    t = iron.ones((2, 2), dtype=dtype)
    assert isinstance(t, tensorclass)
    assert bfloat16_safe_allclose(dtype, t, np.ones((2, 2), dtype=dtype))


@pytest.mark.parametrize(
    "dtype", [d for d in TEST_DTYPES if np.issubdtype(d, np.integer)]
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_random_with_bounds(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.randint(0, 32, (2, 4), dtype=dtype, device=d)
        assert t.shape == (2, 4)
        arr = t.numpy()
        assert np.all((arr >= 0) & (arr < 32))


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_rand(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.rand(2, 2, dtype=dtype, device=d)
        arr = t.numpy()
        assert np.all((arr >= 0) & (arr < 1.0))


@pytest.mark.parametrize(
    "dtype", [d for d in TEST_DTYPES if np.issubdtype(d, np.integer)]
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_arange_integer(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    assert np.array_equal(iron.arange(3, 9, dtype=dtype), np.arange(3, 9, dtype=dtype))


@pytest.mark.parametrize(
    "dtype", [d for d in TEST_DTYPES if np.issubdtype(d, np.floating)]
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_arange_floats(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    assert bfloat16_safe_allclose(
        dtype,
        iron.arange(1.0, 5.0, 1.5, dtype=dtype),
        np.arange(1.0, 5.0, 1.5, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_fill(dtype, tensorclass):
    """Test the fill method for in-place tensor filling."""
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.zeros((2, 3), dtype=dtype, device=d)

        # Fill with a specific value
        fill_value = 42 if np.issubdtype(dtype, np.integer) else 42.5
        t.fill_(fill_value)

        # Verify the tensor is filled with the correct value
        expected = np.full((2, 3), fill_value, dtype=dtype)
        assert bfloat16_safe_allclose(dtype, t.numpy(), expected)

        # Test with different value
        new_fill_value = 99 if np.issubdtype(dtype, np.integer) else 99.9
        t.fill_(new_fill_value)
        expected = np.full((2, 3), new_fill_value, dtype=dtype)
        assert bfloat16_safe_allclose(dtype, t.numpy(), expected)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_zeros_like(dtype, tensorclass):
    iron.set_iron_tensor_class(tensorclass)
    t = iron.tensor([[1, 2], [3, 4]], dtype=dtype)
    z = iron.zeros_like(t)
    expected = np.zeros_like(t)
    assert np.array_equal(z, expected)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_tensor_repr(dtype, tensorclass):
    """Test that __repr__ properly syncs from device and shows correct data."""
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device=d)
        # Modify data on device
        t.to(d)
        # Get string representation (should sync from device)
        repr_str = repr(t)
        print(repr_str)
        assert f"{t.__class__.__name__}(" in repr_str
        assert f"device='{d}'" in repr_str
        # Check that the data values are present
        assert (
            "1" in repr_str and "2" in repr_str and "3" in repr_str and "4" in repr_str
        )


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_tensor_getitem(dtype, tensorclass):
    """Test that __getitem__ properly syncs from device."""
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device=d)
        # Modify data on device
        t.to(d)
        # Get item (should sync from device)
        value = t[0, 1]
        assert value == 2


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_tensor_setitem(dtype, tensorclass):
    """Test that __setitem__ properly syncs to and from device."""
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device=d)
        t[0, 1] = 42
        # Verify the change is reflected
        assert t[0, 1] == 42
        # Verify other elements are unchanged
        assert t[0, 0] == 1
        assert t[1, 0] == 3
        assert t[1, 1] == 4


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_tensor_getitem_setitem_consistency(dtype, tensorclass):
    """Test that getitem and setitem work consistently with device sync."""
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.zeros((2, 2), dtype=dtype, device=d)
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
        expected = np.array([[10, 20], [30, 40]], dtype=dtype)
        assert np.array_equal(t.numpy(), expected)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize(
    "tensorclass", [t for t in TENSOR_CLASSES if "cpu" in t.DEVICES]
)
def test_cpu_tensor_no_sync(dtype, tensorclass):
    """Test that CPU tensors operations."""
    iron.set_iron_tensor_class(tensorclass)
    t = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device="cpu")
    assert t[0, 1] == 2
    t[0, 1] = 42
    assert t[0, 1] == 42
    assert f"device='cpu'" in repr(t)
    arr = t.numpy()
    assert np.array_equal(arr, np.array([[1, 42], [3, 4]], dtype=dtype))


@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_device_attribute_update(dtype):
    """Test that to() method properly updates the device attribute."""
    t = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device="cpu")
    assert isinstance(t, XRTTensor)
    assert t.device == "cpu"

    # Move to NPU
    t.to("npu")
    assert t.device == "npu"
    assert f"device='npu'" in repr(t)

    # Move back to CPU
    t.to("cpu")
    assert t.device == "cpu"
    assert f"device='cpu'" in repr(t)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_npu_tensor_sync_behavior(dtype, tensorclass):
    """Test that NPU tensors when implicit sync is required."""
    iron.set_iron_tensor_class(tensorclass)
    for d in tensorclass.DEVICES:
        t = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device=d)
        assert t.device == d

        # Test that accessing data works correctly
        assert t[0, 1] == 2
        t[0, 1] = 42
        assert t[0, 1] == 42

        # Test that numpy() returns correct data
        arr = t.numpy()
        expected = np.array([[1, 42], [3, 4]], dtype=dtype)
        assert np.array_equal(arr, expected)

        # Test that __array__ protocol works
        np_arr = np.array(t)
        assert np.array_equal(np_arr, expected)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_mixed_device_operations(dtype):
    """Test operations between CPU and NPU tensors."""
    # Create tensors on different devices
    cpu_tensor = iron.tensor([[1, 2], [3, 4]], dtype=dtype, device="cpu")
    npu_tensor = iron.tensor([[5, 6], [7, 8]], dtype=dtype, device="npu")

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
