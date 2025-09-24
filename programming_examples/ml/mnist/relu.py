# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from ml_dtypes import bfloat16
import numpy as np
import sys

import aie.iron as iron
from aie.iron.algorithms import for_each
from aie.iron.functional import relu


def test_relu_with_dtype(dtype, dtype_name):
    """Test ReLU with a specific data type."""
    print(f"\nReLU Test with {dtype_name}")
    print("=" * (20 + len(dtype_name)))
    
    # Define tensor shapes and data types
    data_size = 1024
    
    # Construct input tensor
    input0 = iron.arange(data_size, dtype=dtype, device="npu")
    
    # Store original values for comparison
    original_values = input0.numpy().copy()
    
    # Apply ReLU using functional.relu with specific dtype
    relu_func = relu(dtype)
    for_each(input0, relu_func)
    
    # Show results
    print(f"Input shape: {input0.shape}")
    print(f"Sample input (first 5): {original_values[:5]}")
    print(f"Sample output (first 5): {input0.numpy()[:5]}")
    
    # Check correctness
    ref_vec = [max(0, val) for val in original_values]
    actual_values = input0.numpy()
    errors = 0
    
    for index, (actual, ref) in enumerate(zip(actual_values, ref_vec)):
        if actual != ref:
            print(f"Error at {index}: {actual} != {ref}")
            errors += 1
    
    if not errors:
        print(f"PASS! ({dtype_name})")
    else:
        print(f"FAILED! ({dtype_name}) - Error count: {errors}")
    
    return errors == 0


def main():
    """Test ReLU using functional.relu with different data types."""
    
    print("ReLU Functional Test - Multiple Data Types")
    print("=" * 45)
    
    # Test different data types
    test_results = []
    
    # Test bfloat16 (default)
    test_results.append(test_relu_with_dtype(bfloat16, "bfloat16"))
    
    # Test float32
    import numpy as np
    test_results.append(test_relu_with_dtype(np.float32, "float32"))
    
    # Test int32
    test_results.append(test_relu_with_dtype(np.int32, "int32"))
    
    # Summary
    print(f"\nSummary:")
    print(f"Passed: {sum(test_results)}/{len(test_results)} tests")
    
    if all(test_results):
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")


if __name__ == "__main__":
    main()
