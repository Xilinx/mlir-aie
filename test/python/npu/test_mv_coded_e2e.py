# test_mv_coded_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

"""The coded bf16 matvec returns plain mv's bits on the decoded matrix.

``mv_bf16.cc`` built with ``-DA_CODED`` decodes its coded rows into a
scratch tile and runs the row-major kernel on it, so it is lossless only if
the decode is exact: every case runs the coded kernel on coded rows and the
plain kernel on ``coded.decode_rows`` of them, at the same ``dim_m``,
``vec_size`` and ``output_rows``, and compares the bits. The comparison is
device against device, since no host model reproduces either kernel's
bits. The rows are the factory's sample, which
fills each call's first row to capacity with exceptions at both of its ends.
"""

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils import coded
from ml_dtypes import bfloat16

# (dim_m, dim_k, output_rows). The first two are llama 3.2 1B's decode
# GEMVs as its npu2 profile runs them: four rows of 2048 a call, and the ffn
# down projection's one row of 8192. The rest cover a call per C tile, the
# shortest row, a row whose code plane ends off a 64-byte boundary, and three
# rows, which runs matvec's tail past its groups of four.
_SHAPES = [
    (4, 2048, 32),
    (1, 8192, 4),
    (4, 2048, None),
    (4, 64, 16),
    (2, 192, 4),
    (3, 256, 6),
]


@pytest.mark.parametrize(
    "dim_m, dim_k, output_rows",
    _SHAPES,
    ids=[f"m{m}-k{k}-rows{r}" for m, k, r in _SHAPES],
)
def test_coded_matvec_returns_the_plain_bits(dim_m, dim_k, output_rows):
    shape = dict(
        dim_m=dim_m,
        dim_k=dim_k,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        output_rows=output_rows,
    )
    calls = output_rows // dim_m if output_rows else 4
    fn = kernels.mv(**shape, a_coded=True)
    a, b = fn.contract.sample(np.random.default_rng(dim_m * dim_k), calls)
    harness = kd.KernelDesign(fn, calls=calls)
    got = harness.run([a, b])
    verdict = harness.check(got, [a, b])
    assert verdict, verdict.detail
    plain = kd.KernelDesign(kernels.mv(**shape), calls=calls)
    expected = plain.run([coded.decode_rows(a, dim_k), b])
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
