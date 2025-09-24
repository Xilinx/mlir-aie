# test_graph_capture.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import pytest
import numpy as np
from aie.iron.tensor import Tensor
from aie.iron.algorithms import matmul
from aie.iron.graph import capture_graph
from ml_dtypes import bfloat16


def test_graph_capture_basic():
    """Test basic graph capture functionality."""
    # Create some test tensors
    M, K, N = 256, 256, 256
    dtype = np.int16
    a = Tensor((M, K), dtype=dtype)
    b = Tensor((K, N), dtype=dtype)

    # Fill with some data
    a.data[:] = 1.0
    b.data[:] = 2.0

    # Capture operations in a graph
    with capture_graph() as graph:
        result1 = matmul(a, b)
        # Create a compatible tensor for the second operation
        c = Tensor((N, M), dtype=dtype)
        c.data[:] = 1.0
        result2 = matmul(result1, c)  # Chain operations

    # Check graph structure
    assert len(graph) == 2, f"Expected 2 operations, got {len(graph)}"

    # Check first operation
    node1 = list(graph)[0]
    assert node1.operation == "matmul"
    assert node1.func.__name__ == "matmul_impl"
    assert len(node1.inputs) == 2
    assert node1.inputs[0] is a
    assert node1.inputs[1] is b

    # Check second operation
    node2 = list(graph)[1]
    assert node2.operation == "matmul"
    assert node2.func.__name__ == "matmul_impl"
    assert len(node2.inputs) == 2
    assert node2.inputs[0] is result1  # Uses output from first operation
    assert node2.inputs[1] is c


def test_graph_replay():
    """Test graph replay functionality."""
    # Create test tensors
    M, K, N = 256, 256, 256
    dtype = np.int16
    a = Tensor((M, K), dtype=dtype)
    b = Tensor((K, N), dtype=dtype)

    # Fill with initial data
    a.data[:] = 1.0
    b.data[:] = 2.0

    # Capture graph
    with capture_graph() as graph:
        result = matmul(a, b)

    # First replay
    results1 = graph.replay()
    assert len(results1) == 1
    expected1 = np.matmul(a.numpy(), b.numpy())
    np.testing.assert_array_almost_equal(results1[0].numpy(), expected1)

    # Update input data and replay again
    a.data[:] = 3.0
    b.data[:] = 4.0

    results2 = graph.replay()
    assert len(results2) == 1
    expected2 = np.matmul(a.numpy(), b.numpy())
    np.testing.assert_array_almost_equal(results2[0].numpy(), expected2)

    # Verify different results
    assert not np.array_equal(results1[0].numpy(), results2[0].numpy())


def test_graph_replay_multiple_operations():
    """Test graph replay with multiple chained operations."""
    # Create test tensors
    M, K, N = 256, 256, 256
    dtype = np.int16
    a = Tensor((M, K), dtype=dtype)
    b = Tensor((K, N), dtype=dtype)
    c = Tensor((N, M), dtype=dtype)

    # Fill with initial data
    a.data[:] = 1.0
    b.data[:] = 2.0
    c.data[:] = 3.0

    # Capture graph with multiple operations
    with capture_graph() as graph:
        result1 = matmul(a, b)
        result2 = matmul(result1, c)

    # Check graph has 2 operations
    assert len(graph) == 2

    # Replay and verify results
    results = graph.replay()
    assert len(results) == 2

    # Verify first result
    expected1 = np.matmul(a.numpy(), b.numpy())
    np.testing.assert_array_almost_equal(results[0].numpy(), expected1)

    # Verify second result
    expected2 = np.matmul(expected1, c.numpy())
    np.testing.assert_array_almost_equal(results[1].numpy(), expected2)


def test_graph_capture_without_context():
    """Test that operations outside capture context are not captured."""
    M, K, N = 256, 256, 256
    dtype = np.int16
    a = Tensor((M, K), dtype=dtype)
    b = Tensor((K, N), dtype=dtype)

    # Operations outside capture context
    result1 = matmul(a, b)

    # Capture graph
    with capture_graph() as graph:
        result2 = matmul(a, b)

    # Should only capture one operation
    assert len(graph) == 1

    # Verify the captured operation
    node = list(graph)[0]
    assert node.operation == "matmul"
    assert node.inputs[0] is a
    assert node.inputs[1] is b


def test_graph_metadata():
    """Test that graph nodes contain proper metadata."""
    M, K, N = 256, 256, 256
    dtype = np.int16
    a = Tensor((M, K), dtype=dtype)
    b = Tensor((K, N), dtype=dtype)
    a.data[:] = 1.0
    b.data[:] = 2.0

    with capture_graph() as graph:
        result = matmul(a, b)

    node = list(graph)[0]

    # Check metadata
    assert "input_shapes" in node.metadata
    assert "output_shape" in node.metadata
    assert "input_dtypes" in node.metadata
    assert "output_dtype" in node.metadata
    assert "has_out_param" in node.metadata

    # Verify metadata values
    assert node.metadata["input_shapes"] == (a.shape, b.shape)
    assert node.metadata["output_shape"] == result.shape
    assert node.metadata["input_dtypes"] == (a.dtype, b.dtype)
    assert node.metadata["output_dtype"] == result.dtype
    assert node.metadata["has_out_param"] == False


def test_graph_capture_matmul_negate_chain():
    """Test graph capture with matmul followed by negate operation."""
    from aie.iron.algorithms import matmul, for_each
    from aie.iron.functional import negate

    # Create test matrices
    M, N, K = 128, 128, 128
    dtype = np.int16

    A = Tensor((M, K), dtype=dtype)
    B = Tensor((K, N), dtype=dtype)
    C = Tensor((M, N), dtype=dtype)

    # Initialize with test data
    A.data[:] = np.random.randn(M, K).astype(dtype)
    B.data[:] = np.random.randn(K, N).astype(dtype)
    C.data[:] = np.zeros((M, N), dtype=dtype)

    print(f"Matrix A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
    print(f"A sample: {A.data[0, :4]}")
    print(f"B sample: {B.data[:4, 0]}")

    # Enable graph capture
    with capture_graph() as graph:
        # Perform matmul operation - returns Promise object
        matmul(A, B, C)

        # Apply negate to the result - returns Promise object
        for_each(C, lambda x: negate(x))  # Use C as input since matmul is in-place

    # Verify graph was captured
    assert len(graph.nodes) == 2, f"Expected 2 nodes, got {len(graph.nodes)}"

    # Check first node (matmul)
    matmul_node = graph.nodes[0]
    assert matmul_node.operation == "matmul"
    assert matmul_node.output is C  # matmul writes to C

    # Check second node (for_each with negate)
    negate_node = graph.nodes[1]
    assert negate_node.operation == "for_each"
    assert negate_node.output is C  # for_each is in-place, modifies C

    print(f"Graph captured {len(graph.nodes)} operations")
    print(f"Node 0: {matmul_node.operation}")
    print(f"Node 1: {negate_node.operation}")

    # Replay the graph
    print("Replaying graph...")
    graph.replay()

    # Wait for completion of all Promise objects
    matmul_promise.done()
    negate_promise.done()

    # Verify the final result
    expected_matmul = np.dot(A.data, B.data)
    expected_negated = -expected_matmul

    print(f"Expected negated sample: {expected_negated[0, :4]}")
    print(f"Actual negated sample: {C.data[0, :4]}")

    np.testing.assert_array_almost_equal(C.data, expected_negated, decimal=5)
    print("✓ Matmul + negate chain test passed!")


def test_for_each_direct_operations():
    """Test for_each operations directly without graph capture."""
    from aie.iron.algorithms import matmul, for_each
    from aie.iron.functional import identity

    # Create test matrices
    M, N, K = 128, 128, 128
    dtype = bfloat16

    A = Tensor((M, K), dtype=dtype)
    B = Tensor((K, N), dtype=dtype)
    C = Tensor((M, N), dtype=dtype)

    # Initialize with test data
    A.data[:] = np.random.randn(M, K).astype(dtype)
    B.data[:] = np.random.randn(K, N).astype(dtype)
    C.data[:] = np.zeros((M, N), dtype=dtype)

    print(f"Matrix A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
    print(f"A sample: {A.data[0, :4]}")
    print(f"B sample: {B.data[:4, 0]}")

    # Perform matmul operation
    matmul(A, B, C)
    print(f"Matmul result sample: {C.data[0, :4]}")

    # Store the matmul result before identity operation
    matmul_result = C.data.copy()

    # Apply identity to the result (flatten to 1D for for_each)
    for_each(C.view(-1), lambda x: identity(x))
    print(f"Identity result sample: {C.data[0, :4]}")

    # Verify the final result
    # For identity operation, the result should be the same as the matmul result
    # Since identity(x) = x, we expect C.data to be unchanged after the identity operation
    print(f"Expected identity sample: {matmul_result[0, :4]}")
    print(f"Actual identity sample: {C.data[0, :4]}")

    # Compare the result after identity with the original matmul result
    # Convert both to float32 to avoid dtype promotion issues
    np.testing.assert_array_almost_equal(
        C.data.astype(np.float32), matmul_result.astype(np.float32), decimal=5
    )
    print("✓ Direct matmul + identity operations test passed!")


def test_plus():
    from aie.iron.functional import identity
    from aie.iron.algorithms import for_each

    num_elemtns = 128 * 128
    dtype = bfloat16
    tensor = Tensor((num_elemtns,), dtype=dtype)
    tensor.data[:] = np.arange(1, num_elemtns + 1, dtype=dtype)

    print(f"Original tensor: {tensor.data}")

    for_each(tensor, lambda x: identity(x))
    print(f"Identity tensor: {tensor.data}")

    expected = np.arange(1, num_elemtns + 1, dtype=dtype)
    np.testing.assert_array_equal(tensor.data, expected)


if __name__ == "__main__":
    # pytest.main([__file__])
    # test_graph_capture_basic()
    # test_graph_replay()
    # test_graph_replay_multiple_operations()
    # test_graph_capture_without_context()
    # test_graph_metadata()
    test_graph_capture_matmul_negate_chain()
    #test_for_each_direct_operations()
    # test_plus()
