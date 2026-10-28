# test_whole_array_dispatch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: peano
# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s

# The whole-array GEMM example with DispatchTime M, K and N, swept over
# n_aie_cols 1, 2 and 4: one compiled artifact per column count serves every
# shape, and a static specialization of the same generator agrees with it.

import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3]
        / "programming_examples"
        / "basic"
        / "matrix_multiplication"
        / "whole_array"
    ),
)
from whole_array import whole_array  # noqa: E402

# M must be a multiple of m * 4 rows and N of n * n_aie_cols, for every column
# count in the sweep. The host buffers fit the largest shape.
SHAPES = [(512, 512, 512), (256, 512, 1024), (768, 256, 768)]
TILE = dict(m=64, k=64, n=32)
A_LEN = max(M * K for M, K, _ in SHAPES)
B_LEN = max(K * N for _, K, N in SHAPES)
C_LEN = max(M * N for M, _, N in SHAPES)
BUFFERS = dict(
    A=np.ndarray[(A_LEN,), np.dtype[np.int16]],
    B=np.ndarray[(B_LEN,), np.dtype[np.int16]],
    C=np.ndarray[(C_LEN,), np.dtype[np.int32]],
)


def _run_shape(design, M, K, N, *, static=False):
    """Dispatch one shape and return (actual, expected) as M x N."""
    # The design reads and writes packed at the dispatched shape, so the live
    # data is a dense prefix of each buffer.
    rng = np.random.default_rng(1726250518)
    a = np.zeros(A_LEN, dtype=np.int16)
    b = np.zeros(B_LEN, dtype=np.int16)
    a[: M * K] = rng.integers(-8, 9, size=M * K, dtype=np.int16)
    b[: K * N] = rng.integers(-8, 9, size=K * N, dtype=np.int16)

    A = iron.tensor(a, dtype=np.int16, device="npu")
    B = iron.tensor(b, dtype=np.int16, device="npu")
    C = iron.zeros((C_LEN,), dtype=np.int32, device="npu")
    if static:
        design.specialize(M=M, K=K, N=N)(A, B, C)
    else:
        design(A, B, C, M=M, K=K, N=N)

    a_mat = a[: M * K].reshape(M, K).astype(np.int32)
    b_mat = b[: K * N].reshape(K, N).astype(np.int32)
    # C.numpy() views the device buffer, which is released with C.
    return C.numpy()[: M * N].reshape(M, N).copy(), a_mat @ b_mat


@pytest.mark.parametrize("n_aie_cols", [1, 2, 4])
def test_shapes_share_one_compiled_artifact(n_aie_cols):
    """Every shape is correct, and all of them come from a single compile.

    Recompiling per shape would also be correct, so the kernel cache is what
    shows the shape reached the descriptors at dispatch time.
    """
    design = whole_array.specialize(**BUFFERS, **TILE, n_aie_cols=n_aie_cols)
    artifacts = None
    for M, K, N in SHAPES:
        actual, expected = _run_shape(design, M, K, N)
        np.testing.assert_array_equal(actual, expected, err_msg=f"M={M} K={K} N={N}")
        assert len(design._kernel_cache) == 1
        kernel = next(iter(design._kernel_cache.values()))
        if artifacts is None:
            artifacts = (kernel.xclbin_path, kernel.dispatch_lib_path)
        assert (kernel.xclbin_path, kernel.dispatch_lib_path) == artifacts


@pytest.mark.parametrize("n_aie_cols", [1, 2, 4])
@pytest.mark.parametrize("shape", SHAPES)
def test_same_generator_static_specialization(n_aie_cols, shape):
    design = whole_array.specialize(**BUFFERS, **TILE, n_aie_cols=n_aie_cols)
    actual, expected = _run_shape(design, *shape, static=True)
    np.testing.assert_array_equal(actual, expected)
