# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""The fused tile contract's codecs and reference, without an NPU."""

import numpy as np
import pytest
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels.fused import fused_mm
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16


@pytest.fixture(params=[NPU1Col1, NPU2Col1])
def device(request):
    set_current_device(request.param())
    yield request.param
    set_current_device(None)


def test_fused_variants_have_distinct_symbols_and_reuse_bindings(device):
    plain = fused_mm()
    activated = fused_mm(epilogue="silu")
    assert fused_mm() is plain
    assert fused_mm(epilogue="silu") is activated
    assert plain.name != activated.name
    assert plain.object_file_name != activated.object_file_name
    assert plain._symbol_prefix and activated._symbol_prefix
    for fn in (plain, activated):
        fn.siblings(
            init=("mm_fused_acc_init", [np.ndarray[(512,), np.dtype[np.float32]]])
        )
        assert fn.also.init.name == f"{fn._symbol_prefix}_mm_fused_acc_init"
        assert fn.also.init.object_file is fn.object_file


def test_fused_layouts_match_microblock_addressing(device):
    fn = fused_mm()
    assert fn.contract.stack_bytes == (
        np.dtype(np.float32).itemsize * 32 * 16 + device().default_core_stack_bytes
    )
    r, s, t = (4, 8, 4) if device is NPU1Col1 else (4, 8, 8)
    a = np.arange(32 * 32).reshape(1, 32, 32)
    b = np.arange(32 * 16).reshape(1, 32, 16)
    c = np.arange(32 * 16).reshape(1, 32, 16)
    layouts = fn.contract.layouts
    pa, pb, pc = [layout.encode(x)[0] for layout, x in zip(layouts, (a, b, c))]
    # Check the actual pointer arithmetic, not only a round trip: mutually
    # inverse but incorrectly ordered codecs would otherwise pass.
    for k in range(2):
        for band in range(2):
            for row in range(16 // r):
                for z in range(16 // s):
                    offset = (k * 32 + band * 16) * 16 + (row * 16 // s + z) * r * s
                    np.testing.assert_array_equal(
                        pa[offset : offset + r * s],
                        a[
                            0,
                            band * 16 + row * r : band * 16 + (row + 1) * r,
                            k * 16 + z * s : k * 16 + (z + 1) * s,
                        ].ravel(),
                    )
        for col in range(16 // t):
            for z in range(16 // s):
                offset = k * 16 * 16 + (col * 16 // s + z) * s * t
                np.testing.assert_array_equal(
                    pb[offset : offset + s * t],
                    b[
                        0,
                        k * 16 + z * s : k * 16 + (z + 1) * s,
                        col * t : (col + 1) * t,
                    ].ravel(),
                )
    expected_c = np.concatenate(
        [
            c[0, row : row + r, col : col + t].ravel()
            for row in range(0, 32, r)
            for col in range(0, 16, t)
        ]
    )
    np.testing.assert_array_equal(pc, expected_c)
    for layout, values in zip(layouts, (a, b, c)):
        batch = np.concatenate([values, values + 10000])
        np.testing.assert_array_equal(
            layout.decode(layout.encode(batch), calls=2), batch
        )


@pytest.mark.parametrize("epilogue", ["none", "gelu", "silu", "sigmoid"])
def test_fused_reference_and_generic_lowering(device, epilogue):
    fn = fused_mm(epilogue=epilogue, clamp=(-0.125, 0.75))
    rng = np.random.default_rng(24)
    a = rng.normal(size=(2, 32, 32)).astype(bfloat16)
    b = rng.normal(size=(2, 32, 16)).astype(bfloat16)
    product = a.astype(np.float32) @ b.astype(np.float32)
    if epilogue == "none":
        expected = product
    else:
        scale = 1.702 if epilogue == "gelu" else 1
        sigmoid = 1 / (1 + np.exp(-scale * product.astype(np.float64)))
        expected = sigmoid if epilogue == "sigmoid" else product * sigmoid
    expected = np.clip(expected, -0.125, 0.75)
    np.testing.assert_allclose(
        fn.contract.reference(a, b).reshape(expected.shape),
        expected,
        atol=2e-6,
        rtol=2e-6,
    )
    design = kd.design(fused_mm, calls=2, epilogue=epilogue, clamp=(-0.125, 0.75))
    assert "fused_mm_tile" in str(design.as_mlir())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim_m": 0},
        {"band_m": 7},
        {"dim_k": 24},
        {"chunk_k": 12},
        {"dim_n": 9},
        {"out_chunk": 48},
        {"out_chunk": 8},
        {"epilogue": "relu"},
        {"clamp": (2, 1)},
        {"clamp": (0, np.inf)},
    ],
)
def test_fused_rejects_invalid_geometry_and_epilogue(device, kwargs):
    with pytest.raises(ValueError):
        fused_mm(**kwargs)
