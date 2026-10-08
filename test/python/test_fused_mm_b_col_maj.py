# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""fused_mm's transposed-B form: its codec and where it is refused, without an NPU."""

import numpy as np
import pytest
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels.fused import fused_mm
from aie.utils.hostruntime import set_current_device


@pytest.fixture
def npu1():
    set_current_device(NPU1Col1())
    yield
    set_current_device(None)


@pytest.fixture
def npu2():
    set_current_device(NPU2Col1())
    yield
    set_current_device(None)


def test_b_col_maj_packs_transposed_panels(npu1):
    dim_k, dim_n, chunk_k, s, t = 64, 32, 32, 8, 8
    fn = fused_mm(
        dim_k=dim_k, dim_n=dim_n, chunk_k=chunk_k, mmul_shape=(4, 8, 8), b_col_maj=True
    )
    assert "-DMM_FUSED_B_COL_MAJ" in fn.compile_flags
    layout = fn.contract.layouts[1]
    b = np.arange(dim_k * dim_n).reshape(1, dim_k, dim_n)
    packed = layout.encode(b)[0]
    # The kernel's addressing: block (i, j) of chunk k at (i * colB + j), and
    # within it row n holds s consecutive k of column j * t + n.
    col_b = dim_n // t
    for k in range(dim_k // chunk_k):
        for i in range(chunk_k // s):
            for j in range(col_b):
                offset = k * chunk_k * dim_n + (i * col_b + j) * s * t
                rows = slice(k * chunk_k + i * s, k * chunk_k + (i + 1) * s)
                np.testing.assert_array_equal(
                    packed[offset : offset + s * t],
                    b[0, rows, j * t : (j + 1) * t].T.ravel(),
                )
    batch = np.concatenate([b, b + 10000])
    np.testing.assert_array_equal(layout.decode(layout.encode(batch), calls=2), batch)


def test_b_col_maj_is_distinct_from_row_major_b(npu1):
    plain = fused_mm(mmul_shape=(4, 8, 8))
    transposed = fused_mm(mmul_shape=(4, 8, 8), b_col_maj=True)
    assert plain.name != transposed.name


@pytest.mark.parametrize("kwargs", [{}, {"mmul_shape": (4, 8, 4)}])
def test_b_col_maj_needs_an_8_by_8_b_block(npu1, kwargs):
    with pytest.raises(ValueError, match="b_col_maj"):
        fused_mm(b_col_maj=True, **kwargs)


@pytest.mark.parametrize("emulate", [False, True])
def test_b_col_maj_builds_on_aie2p(npu2, emulate):
    fn = fused_mm(
        b_col_maj=True, mmul_shape=(8, 8, 8), emulate_bf16_mmul_with_bfp16=emulate
    )
    assert "-DMM_FUSED_B_COL_MAJ" in fn.compile_flags
    assert (
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" in fn.compile_flags
    ) == emulate


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bfp16_b": True},
        {"emulate_bf16_mmul_with_bfp16": True, "mmul_shape": (4, 8, 8)},
        {"mmul_shape": (4, 8, 8)},
    ],
)
def test_b_col_maj_refuses_packed_b_and_short_aie2p_mmuls(npu2, kwargs):
    with pytest.raises(ValueError, match="b_col_maj"):
        fused_mm(b_col_maj=True, **kwargs)


def test_bfp16_macs_are_aie2p_only(npu1):
    with pytest.raises(ValueError, match="aie2p"):
        fused_mm(emulate_bf16_mmul_with_bfp16=True)


@pytest.mark.parametrize("b_col_maj", [False, True])
def test_bfp16_macs_see_the_packed_blocks(npu2, b_col_maj):
    # In floor mode the core's conversion and the host's packer agree, so a
    # B the core converts is the B the packed form ships.
    dims = dict(dim_m=16, band_m=16, dim_k=64, dim_n=32, chunk_k=32, rounding="floor")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 64)).astype(np.float32)
    b = rng.standard_normal((64, 32)) * np.exp2(rng.integers(-12, 12, (1, 32)))
    b = b.astype(np.float32)
    converted = fused_mm(
        **dims, b_col_maj=b_col_maj, emulate_bf16_mmul_with_bfp16=True
    ).contract.reference(a, b)
    packed = fused_mm(**dims, bfp16_b=True).contract.reference(a, b)
    np.testing.assert_array_equal(converted, packed)
    assert not np.array_equal(converted, a @ b)
