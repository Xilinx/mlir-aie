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


@pytest.mark.parametrize(
    "M,K,N,dtype",
    [
        (256, 784, 128, np.int16),
        (256, 128, 32, np.int16),
        (256, 32, 32, np.int16),
    ],
)
def test_matmul_dimensions(M, K, N, dtype):
    """Test matrix multiplication with various dimensions that are multiples of required values."""
    a = iron.tensor(np.random.randn(M, K), dtype=dtype)
    b = iron.tensor(np.random.randn(K, N), dtype=dtype)

    result = iron.matmul(a, b)
    expected = np.matmul(a.numpy(), b.numpy())

    assert result.shape == expected.shape
    assert np.allclose(result.numpy(), expected)


def test_matmul_device_consistency():
    """Test that matmul preserves device of first input."""
    M, K, N = 256, 256, 256
    a_npu = iron.tensor(np.random.randn(M, K), dtype=np.int16, device="npu")
    b_cpu = iron.tensor(np.random.randn(K, N), dtype=np.int16, device="cpu")

    result = iron.matmul(a_npu, b_cpu)
    assert result.device == "npu"

    result = iron.matmul(b_cpu, a_npu)
    assert result.device == "cpu"


if __name__ == "__main__":
    # Run parameterized tests with different dimensions
    test_matmul_dimensions(256, 784, 128, np.int16)
    test_matmul_dimensions(256, 128, 32, np.int16)
    test_matmul_dimensions(256, 32, 32, np.int16)
    test_matmul_device_consistency()
