# test_for_each.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import pytest
import numpy as np
from aie.iron.tensor import Tensor
from aie.iron.algorithms import for_each
from aie.iron.functional import plus, negate, multiplies, equal_to, greater


def test_for_each_basic():
    """Test basic for_each functionality with different sizes."""
    # Test with size 32
    M = 32
    dtype = np.int32
    tensor = Tensor((M,), dtype=dtype)
    tensor.data[:] = np.arange(1, M + 1, dtype=dtype)  # [1, 2, 3, ..., 32]

    print(f"Original tensor (size {M}): {tensor.data}")

    # Test 1: Negate all elements
    for_each(tensor, lambda x: negate(x))
    expected_negate = -np.arange(1, M + 1, dtype=dtype)
    print(f"After negate: {tensor.data}")
    np.testing.assert_array_equal(tensor.data, expected_negate)

    # Test 2: Add 10 to all elements
    for_each(tensor, lambda x: plus(x, 10))
    expected_plus = expected_negate + 10
    print(f"After plus 10: {tensor.data}")
    np.testing.assert_array_equal(tensor.data, expected_plus)

    # Test 3: Multiply all elements by 2
    for_each(tensor, lambda x: multiplies(x, 2))
    expected_mult = expected_plus * 2
    print(f"After multiply 2: {tensor.data}")
    np.testing.assert_array_equal(tensor.data, expected_mult)


def test_for_each_functional_operations():
    """Test for_each with various functional operations."""
    # Create test tensor with size 16
    M = 16
    dtype = np.int32
    tensor = Tensor((M,), dtype=dtype)
    tensor.data[:] = np.arange(1, M + 1, dtype=dtype)  # [1, 2, 3, ..., 16]

    print(f"Original tensor (size {M}): {tensor.data}")

    # Test unary operations
    for_each(tensor, lambda x: negate(x))
    expected = -np.arange(1, M + 1, dtype=dtype)
    np.testing.assert_array_equal(tensor.data, expected)
    print(f"After negate: {tensor.data}")

    # Test binary operations with constants
    for_each(tensor, lambda x: plus(x, 5))
    expected = expected + 5
    np.testing.assert_array_equal(tensor.data, expected)
    print(f"After plus 5: {tensor.data}")

    for_each(tensor, lambda x: multiplies(x, 3))
    expected = expected * 3
    np.testing.assert_array_equal(tensor.data, expected)
    print(f"After multiply 3: {tensor.data}")


def test_for_each_with_lambda_functions():
    """Test for_each with custom lambda functions using functional API."""
    # Create test tensor with size 64
    M = 64
    dtype = np.int32
    tensor = Tensor((M,), dtype=dtype)
    tensor.data[:] = np.arange(2, 2 * M + 2, 2, dtype=dtype)  # [2, 4, 6, 8, ..., 128]

    print(f"Original tensor (size {M}): {tensor.data[:8]}...")  # Show first 8 elements

    # Test complex lambda with multiple operations
    for_each(tensor, lambda x: plus(multiplies(x, 2), 1))
    expected = np.arange(2, 2 * M + 2, 2, dtype=dtype) * 2 + 1  # (x * 2) + 1
    np.testing.assert_array_equal(tensor.data, expected)
    print(f"After (x * 2) + 1: {tensor.data[:8]}...")

    # Test another complex operation
    for_each(tensor, lambda x: multiplies(plus(x, 3), 2))
    expected = (expected + 3) * 2  # (x + 3) * 2
    np.testing.assert_array_equal(tensor.data, expected)
    print(f"After (x + 3) * 2: {tensor.data[:8]}...")


def test_for_each_in_place():
    """Test that for_each modifies tensor in-place."""
    # Create test tensor with size 16
    M = 16
    dtype = np.int32
    tensor = Tensor((M,), dtype=dtype)
    tensor.data[:] = np.arange(1, M + 1, dtype=dtype)  # [1, 2, 3, ..., 16]


    original_id = id(tensor)
    original_data = tensor.data.copy()

    # Apply operation
    result = for_each(tensor, lambda x: plus(x, 10))

    # Check that the same tensor object is returned
    assert result is tensor, "for_each should return the same tensor object"
    assert id(result) == original_id, "Object ID should be the same"

    # Check that data was modified
    expected = original_data + 10
    print(f"Original: {original_data}")
    print(f"Modified: {tensor.data}")

    np.testing.assert_array_equal(tensor.data, expected)



def test_for_each_different_dtypes():
    """Test for_each with different data types."""
    # Test with float32 (size 16)
    tensor_f32 = Tensor((16,), dtype=np.float32)
    tensor_f32.data[:] = np.arange(1.5, 17.5, dtype=np.float32)  # [1.5, 2.5, ..., 16.5]

    for_each(tensor_f32, lambda x: plus(x, 0.5))
    expected_f32 = np.arange(1.5, 17.5, dtype=np.float32) + 0.5
    np.testing.assert_array_almost_equal(tensor_f32.data, expected_f32)
    print(f"Float32 result: {tensor_f32.data}")

    # Test with int64 (size 32)
    tensor_i64 = Tensor((32,), dtype=np.int64)
    tensor_i64.data[:] = np.arange(100, 132, dtype=np.int64)  # [100, 101, ..., 131]

    for_each(tensor_i64, lambda x: multiplies(x, 2))
    expected_i64 = np.arange(100, 132, dtype=np.int64) * 2
    np.testing.assert_array_equal(tensor_i64.data, expected_i64)
    print(f"Int64 result: {tensor_i64.data}")


if __name__ == "__main__":
    #pytest.main([__file__])
    #test_for_each_basic()
    #test_for_each_functional_operations()
    #test_for_each_with_lambda_functions()
    #test_for_each_in_place()
    test_for_each_different_dtypes()
