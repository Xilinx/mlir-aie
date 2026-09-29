# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu1_xrt% %pytest -m "not extensive" %s
# RUN: %run_on_npu2_xrt% %pytest -m "not extensive" %s
# RUN: %run_on_npu2_hrx% %pytest -m "not extensive" %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Compile, link and execute all three mm_fused.h entry points together."""

import numpy as np
import pytest
from aie.iron.algorithms import kernel_design as kd
from aie.iron.kernels.fused import fused_mm
from aie.utils import ensure_current_device
from ml_dtypes import bfloat16


@pytest.mark.parametrize(
    "epilogue,clamp",
    [
        ("none", None),
        pytest.param("gelu", None, marks=pytest.mark.extensive),
        pytest.param("silu", None, marks=pytest.mark.extensive),
        pytest.param("sigmoid", None, marks=pytest.mark.extensive),
        pytest.param("none", (-0.125, 0.75), marks=pytest.mark.extensive),
        pytest.param("silu", (-0.125, 0.75), marks=pytest.mark.extensive),
    ],
)
def test_fused_init_k_bands_and_epilogue(epilogue, clamp):
    # Bind before constructing the host codecs: JIT generation binds the runtime
    # device too, but by then an unbound factory has already chosen AIE2 layouts.
    ensure_current_device()
    # Small dyadic operands make the unactivated GEMM exact in both f32 and
    # bf16. Different rows/columns, two A bands and three K chunks expose a
    # missing accumulation or an incorrect offset. Three consecutive calls
    # include a zero result after a nonzero one to detect stale accumulator
    # state. Eight output chunks exercise nonzero outer AND half indices.
    rng = np.random.default_rng(3740)
    a = (rng.integers(-2, 3, size=(3, 32, 48)) / 8).astype(bfloat16)
    b = (rng.integers(-2, 3, size=(3, 48, 16)) / 8).astype(bfloat16)
    a[0, 0, :] = 0.25
    b[0, :, 0] = 0.25
    b[0, :, 1] = -0.25
    a[1] = 0
    fn = fused_mm(dim_k=48, epilogue=epilogue, clamp=clamp)
    design = kd.design(fused_mm, calls=3, dim_k=48, epilogue=epilogue, clamp=clamp)
    inputs, output = kd.upload(
        [a, b], 3 * 32 * 16, np.dtype(bfloat16), fn=fn, poison=True
    )
    design(*inputs, output)
    got = (
        fn.contract.layouts[2].decode(output.numpy().copy(), calls=3).astype(np.float32)
    )
    product = a.astype(np.float32) @ b.astype(np.float32)
    if epilogue == "none":
        expected = product
    else:
        scale = 1.702 if epilogue == "gelu" else 1
        sigmoid = 1 / (1 + np.exp(-scale * product.astype(np.float64)))
        expected = sigmoid if epilogue == "sigmoid" else product * sigmoid
    if clamp is not None:
        expected = np.clip(expected, *clamp)
    expected = expected.astype(bfloat16).astype(np.float32)
    if epilogue == "none":
        np.testing.assert_array_equal(got, expected)
    else:
        # tanh is narrowed to bf16 on both arches; AIE2 additionally uses
        # the tanh LUT. No mismatch budget: every output must satisfy the bound.
        np.testing.assert_allclose(got, expected, rtol=0.04, atol=0.02)
    np.testing.assert_array_equal(
        got[1], np.full((32, 16), 0.5 if epilogue == "sigmoid" else 0)
    )
    if clamp is not None:
        assert np.min(got) == clamp[0]
        assert np.max(got) == clamp[1]
