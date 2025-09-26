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
from aie.iron.graph import capture_graph


class ReLU:
    """ReLU activation layer implementation using Iron operations."""
    
    def __init__(self, dtype=bfloat16, device="npu"):
        """
        Initialize ReLU layer.
        
        Args:
            dtype: Data type for computations
            device (str): Device to run on
        """
        self.dtype = dtype
        self.device = device
        self.relu_func = relu(dtype)
    
    def forward(self, x):
        """
        Forward pass through ReLU layer.
        
        Args:
            x (iron.Tensor): Input tensor
            
        Returns:
            iron.Tensor: Output tensor with ReLU applied
        """
        # Apply ReLU activation to each element
        for_each(x.view(-1), self.relu_func, out=x.view(-1))
        return x
    
    def parameters(self):
        """Return layer parameters (ReLU has no parameters)."""
        return []


def test_relu_with_dtype(dtype, dtype_name):
    """Test ReLU with graph capture - multiple executions."""
    print(f"\nReLU Graph Capture Test with {dtype_name}")
    print("=" * (35 + len(dtype_name)))
    
    # Define tensor shapes and data types
    data_size = 1024
    
    # Create ReLU layer
    relu_layer = ReLU(dtype=dtype, device="npu")
    
    # First execution - capture the graph
    input_data0 = np.arange(data_size, dtype=dtype) - data_size // 2  # [-512, -511, ..., 511]
    input = iron.tensor(input_data0, dtype=dtype, device="npu")
    
    with capture_graph() as graph:
        _ = relu_layer.forward(input)
    
        # Compile the graph
        graph.compile()
    
        # Execute first time
        result0 = graph.replay()
        
        
        # Check correctness
        ref1 = [max(0, val) for val in input_data0]
        actual0 = result0.numpy()
        errors0 = sum(1 for a, r in zip(actual0, ref1) if a != r)
        
        # Reuse compiled graph with different input
        input_data1 = np.arange(data_size, dtype=dtype) - data_size // 4  # Different input: [-256, -255, ..., 767]
        input[:] = input_data1
        
        # Execute with new data
        result1 = graph.replay()
        
        # Check correctness
        ref2 = [max(0, val) for val in input_data1]
        actual1 = result1.numpy()
        errors1 = sum(1 for a, r in zip(actual1, ref2) if a != r)
        
        verbose = True
        if verbose:
            print(f"   Input[0]: {input_data0}")
            print(f"   Output[0]: {result0.numpy()}")
        
            print(f"   Input[1]: {input_data1}")
            print(f"   Output[1]: {result1.numpy()}")
            
            
        if errors0 == 0 and errors1 == 0:
            print(f"PASS! ({dtype_name}) - Graph capture working correctly")
        else:
            print(f"FAILED! ({dtype_name}) - Total errors: {errors0 + errors1} ({errors0} + {errors1})")
        
        return errors0 == 0 and errors1 == 0


def main():
    """Test ReLU using ReLU class with different data types."""
    
    print("ReLU Class Test - Multiple Data Types")
    print("=" * 40)
    
    # Test different data types
    test_results = []
    
    # Test bfloat16 (default)
    test_results.append(test_relu_with_dtype(bfloat16, "bfloat16"))
    
    # Summary
    print(f"\nSummary:")
    print(f"Passed: {sum(test_results)}/{len(test_results)} tests")
    
    if all(test_results):
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")


if __name__ == "__main__":
    main()
