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
    assert fused_mm() == plain
    assert fused_mm(epilogue="silu") == activated
    assert plain.name != activated.name
    assert plain.object_file_name != activated.object_file_name
    assert plain._symbol_prefix and activated._symbol_prefix
    for fn in (plain, activated):
        init = fn.object_file.bind(
            "mm_fused_acc_init", [np.ndarray[(512,), np.dtype[np.float32]]]
        )
        assert init.name == f"{fn._symbol_prefix}_mm_fused_acc_init"
        assert init.object_file is fn.object_file


def test_fused_uses_native_source(device):
    fn = fused_mm()
    assert fn.source_string is None
    assert Path(fn.source_file).is_file()
    assert Path(fn.source_file).name == "fused_mm_tile.cc"


def test_fused_architectures_have_distinct_symbols():
    try:
        set_current_device(NPU1Col1())
        aie2 = fused_mm()
        set_current_device(NPU2Col1())
        aie2p = fused_mm()
        assert aie2.name != aie2p.name
        assert aie2.object_file_name != aie2p.object_file_name
        assert fused_mm() == aie2p
        set_current_device(NPU1Col1())
        assert fused_mm() == aie2
    finally:
        set_current_device(None)


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


def test_fused_binds_every_entry_point_of_its_object(device):
    fn = fused_mm(dim_m=64, band_m=32, dim_k=128, dim_n=64, chunk_k=128)
    assert fn.fused_mm_tile is fn
    for name in ("mm_fused_acc_init", "mm_fused_k_step", "mm_fused_epilogue_chunk"):
        entry = getattr(fn, name)
        assert entry.name == f"{fn._symbol_prefix}_{name}"
        assert entry.object_file is fn.object_file
    shapes = [kd.shape_dtype(t)[0] for t in fn.mm_fused_k_step.arg_types()[:3]]
    assert shapes == [(32 * 128,), (128 * 64,), (64 * 64,)]
    assert kd.shape_dtype(fn.mm_fused_epilogue_chunk.arg_types()[0])[0] == (64,)


def test_fused_options_reach_the_compile_flags(device):
    assert "-DROUND_CONV_EVEN" in fused_mm().compile_flags
    assert "-DROUND_CONV_EVEN" not in fused_mm(rounding="floor").compile_flags
    steps = fused_mm(epilogue="gelu", gelu="bf16_steps").compile_flags
    assert "-DMM_FUSED_GELU_BF16_STEPS" in steps
    marked = fused_mm(step_markers=True)
    assert "-DMM_FUSED_STEP_MARKERS" in marked.compile_flags
    assert marked.contract.trace.shape == "partial"
    every = fused_mm(epilogue_modes=("none", "gelu", "silu", "sigmoid"))
    assert "-DMM_FUSED_EPILOGUE_MODE_MASK=15" in every.compile_flags
    assert "-DMM_FUSED_C_DEPTH=4" in fused_mm(c_depth=4).compile_flags
    rst = fused_mm(dim_n=32, mmul_shape=(8, 8, 8)).compile_flags
    assert {"-DMM_FUSED_R=8", "-DMM_FUSED_S=8", "-DMM_FUSED_T=8"} <= set(rst)


def test_fused_bound_mode_distinguishes_kernels_with_one_mask(device):
    modes = ("none", "gelu")
    a = fused_mm(epilogue="none", epilogue_modes=modes)
    b = fused_mm(epilogue="gelu", epilogue_modes=modes)
    assert a.compile_flags == b.compile_flags
    assert a.name != b.name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epilogue": "gelu", "epilogue_modes": ("none",)},
        {"epilogue_modes": ("none", "relu")},
        {"rounding": "nearest"},
        {"gelu": "fp64"},
        {"c_depth": 3},
        {"mmul_shape": (4, 8, 3)},
    ],
)
def test_fused_rejects_invalid_options(device, kwargs):
    with pytest.raises(ValueError):
        fused_mm(**kwargs)


def _bf16_floor_ref(x):
    """Round float64 toward minus infinity at bf16 precision."""
    x = np.asarray(x, np.float64)
    ulp = np.exp2(np.floor(np.log2(np.where(x == 0, 1.0, np.abs(x)))) - 7)
    return np.where(x == 0, 0.0, np.floor(x / ulp) * ulp)


def test_fused_gelu_bf16_steps_reference_rounds_after_each_step(device):
    """The reference follows gelu_bf16_steps_vec: x * sigmoid(1.703125x), each
    of five results rounded toward minus infinity."""
    fn = fused_mm(epilogue="gelu", gelu="bf16_steps", rounding="floor")
    rng = np.random.default_rng(7)
    a = (rng.normal(size=(1, 32, 32)) / 4).astype(bfloat16)
    b = (rng.normal(size=(1, 32, 16)) / 4).astype(bfloat16)
    c = a.astype(np.float64) @ b.astype(np.float64)
    x = _bf16_floor_ref(c)
    y = _bf16_floor_ref(x * 1.703125)
    sig = _bf16_floor_ref(np.tanh(y / 2) + 1) / 2
    expected = _bf16_floor_ref(x * sig)
    got = fn.contract.reference(a, b).reshape(expected.shape)
    np.testing.assert_array_equal(got, expected)
    fp32 = fused_mm(epilogue="gelu", rounding="floor").contract.reference(a, b)
    assert np.count_nonzero(fp32.reshape(expected.shape) != expected) > 100
