# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Boundary contracts for the row-major bf16 matvec kernel."""

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from ml_dtypes import bfloat16


@pytest.mark.parametrize("vec_size", [16, 32, 64])
@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("vectorized", [False, True])
def test_bf16_matvec_accepts_complete_chunks(npu2_device, vec_size, chunks, vectorized):
    dim_k = vec_size * chunks
    fn = kernels.mv(
        dim_m=4,
        dim_k=dim_k,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        vec_size=vec_size,
        vectorized=vectorized,
    )
    assert f"-DDIM_K={dim_k}" in fn.compile_flags
    assert f"-DVEC_SIZE={vec_size}" in fn.compile_flags
    assert [kd.shape_dtype(t)[0] for t in fn.arg_types()[2:]] == [
        (4 * dim_k,),
        (dim_k,),
        (4,),
    ]
    a = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
    a = np.broadcast_to(a, (2, 4, dim_k)).astype(bfloat16)
    b = np.ones((2, dim_k), dtype=bfloat16)
    np.testing.assert_array_equal(
        fn.contract.reference(a, b),
        np.broadcast_to(np.arange(4) * dim_k, (2, 4)),
    )


@pytest.mark.parametrize(
    "dim_k, vec_size",
    [(0, 64), (-64, 64), (32, 64), (65, 64), (64, 0), (64, -64)],
)
def test_bf16_matvec_rejects_invalid_chunks(npu2_device, dim_k, vec_size):
    with pytest.raises(ValueError, match="positive multiple of vec_size"):
        kernels.mv(
            dim_m=4,
            dim_k=dim_k,
            input_dtype=bfloat16,
            output_dtype=bfloat16,
            vec_size=vec_size,
        )
