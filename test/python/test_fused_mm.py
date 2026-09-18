# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""The fused tile contract's codecs and reference, without an NPU."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels.fused import fused_mm
from aie.utils import ensure_current_device, get_current_device
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


def test_fused_accumulator_uses_target_vector_alignment(device):
    assert "alignas(aie::vector_decl_align) float acc[" in fused_mm().source_string


def test_fused_e2e_binds_runtime_before_constructing_host_layouts(monkeypatch):
    monkeypatch.setattr("aie.utils.hostruntime._CURRENT_DEVICE", None)
    monkeypatch.setattr(
        "aie.utils._get_default_npu_runtime",
        lambda: SimpleNamespace(device=NPU2Col1),
    )
    case = runpy.run_path(str(Path(__file__).parent / "npu" / "test_fused_mm_e2e.py"))[
        "test_fused_init_k_bands_and_epilogue"
    ]

    class ContractChecked(Exception):
        pass

    def check_contract(**kwargs):
        assert isinstance(get_current_device(probe_runtime=False), NPU2Col1)
        assert "-DMM_FUSED_T=8" in fused_mm(**kwargs).compile_flags
        raise ContractChecked

    monkeypatch.setitem(case.__globals__, "fused_mm", check_contract)
    with pytest.raises(ContractChecked):
        case("none", None)


def test_fused_unbound_layout_reproduces_npu2_failure(monkeypatch):
    monkeypatch.setattr("aie.utils.hostruntime._CURRENT_DEVICE", None)
    monkeypatch.setattr(
        "aie.utils._get_default_npu_runtime",
        lambda: SimpleNamespace(device=NPU2Col1),
    )
    unbound = fused_mm(dim_k=48)
    ensure_current_device()
    bound = fused_mm(dim_k=48)
    rng = np.random.default_rng(3740)
    a = (rng.integers(-2, 3, size=(3, 32, 48)) / 8).astype(bfloat16)
    b = (rng.integers(-2, 3, size=(3, 48, 16)) / 8).astype(bfloat16)
    a[0, 0, :] = 0.25
    b[0, :, 0] = 0.25
    b[0, :, 1] = -0.25
    a[1] = 0
    expected = a.astype(np.float32) @ b.astype(np.float32)

    def execute(host):
        # mm_fused_mmul_2x2 consumes row-major bf16 microblocks on both
        # targets; the kernel's target, not the uploader, fixes their size.
        operands = [
            native.decode(upload.encode(x), calls=3).astype(np.float32)
            for native, upload, x in zip(
                bound.contract.layouts, host.contract.layouts, (a, b)
            )
        ]
        output = bound.contract.layouts[2].encode(operands[0] @ operands[1])
        return host.contract.layouts[2].decode(output, calls=3)

    wrong = execute(unbound)
    assert np.count_nonzero(wrong != expected) == 998
    np.testing.assert_array_equal(wrong[0, 0, :2], [1.28125, -1.8125])
    np.testing.assert_array_equal(execute(bound), expected)


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
