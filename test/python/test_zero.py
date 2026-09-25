# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

from pathlib import Path

import numpy as np
import pytest
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron import Out, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils import bfp
from ml_dtypes import bfloat16


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32, bfloat16],
)
@pytest.mark.parametrize("vectorized", [False, True])
def test_zero_is_an_independent_output_only_kernel(dtype, vectorized):
    fn = kernels.zero((4, 16), dtype, vectorized=vectorized)
    assert Path(fn.source_file).name == "zero.cc"
    assert Path(fn.source_file).parent.name == "zero"
    assert fn.source_string is None
    assert fn.contract.roles == (Out,)
    assert fn.arg_types() == [np.ndarray[(4, 16), np.dtype[dtype]]]
    assert fn.contract.reference_indices() == []
    np.testing.assert_array_equal(fn.expected([]), np.zeros((1, 4, 16), dtype=dtype))
    assert not hasattr(fn, "also")
    assert not hasattr(fn, "siblings")


@pytest.mark.parametrize("tile_size", [0, -1, 1.5, None, (), (8, 0), (3, 1.5)])
def test_zero_rejects_invalid_shapes(tile_size):
    with pytest.raises(ValueError, match="positive integer or shape"):
        kernels.zero(tile_size)


def test_zero_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        kernels.zero(64, np.float64)


def test_zero_bfp_storage_and_reference(npu2_device):
    fn = kernels.zero((4, 8), v8bfp16ebs8)
    assert "-DTILE_SIZE=288" in fn.compile_flags
    assert "-DZERO_TYPE=uint8_t" in fn.compile_flags
    reference = fn.expected([])
    assert reference.shape == (1, 32 * 8)
    assert np.count_nonzero(reference) == 0
    layout = fn.contract.layouts[0]
    encoded = layout.encode(reference)
    assert encoded.dtype == np.uint8 and encoded.shape == (1, 32 * 9)
    assert np.count_nonzero(encoded) == 0
    np.testing.assert_array_equal(layout.decode(encoded), reference)
    np.testing.assert_array_equal(bfp.decode(encoded), reference)


def test_zero_harness_builds_without_input_fifos(npu2_device):
    fn = kernels.zero(64, np.int32)
    assert kd.sample_inputs(fn, calls=3) == []
    args = kd.host_args(fn, calls=3)
    assert len(args) == 1 and args[0].direction is Out
    assert args[0].n_elements == 192
    mlir = str(kd.design(kernels.zero, tile_size=64, calls=3).as_mlir())
    assert fn.name in mlir
    assert "func.call" in mlir


def test_zero_initializer_matches_bfp_accumulator(npu2_device):
    for mixed in (False, True):
        fn = kernels.mm_bfp(dim_m=32, dim_k=64, dim_n=32, mixed=mixed)
        initialize = fn.contract.initializers[0][1](fn)
        assert initialize.object_file is not fn.object_file
        assert initialize.arg_types() == [fn.arg_types()[2]]
        assert initialize.contract.roles == (Out,)


@pytest.mark.parametrize("dtype", [np.int32, bfloat16, v8bfp16ebs8])
def test_zero_judges_every_repeated_output_tile(npu2_device, dtype):
    fn = kernels.zero(64, dtype)
    reference = fn.expected([])
    logical = np.zeros((3, *fn.contract.layouts[0].shape), reference.dtype)
    actual = fn.contract.layouts[0].encode(logical).ravel()
    verdict = fn.judge(actual, reference, calls=3)
    assert verdict and verdict.n_checked == logical.size
    logical[-1, -1] = 1
    corrupted = fn.contract.layouts[0].encode(logical).ravel()
    verdict = fn.judge(corrupted, reference, calls=3)
    assert not verdict and verdict.n_mismatch == 1
    assert verdict.first_bad_index == logical.size - 1


def test_zero_device_cases_cover_smoke_and_static_sweeps(monkeypatch, npu2_device):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "npu"))
    from cases import inputs_for
    from kernel_cases import CASES

    cases = [case for case in CASES if case.factory == "zero"]
    assert len([case for case in cases if case.smoke]) == 3
    names = [case.name for case in cases]
    assert len(names) == len(set(names))
    for case in cases:
        assert case.calls > 1
        assert case.data_policy() == ("random",)
        assert inputs_for(case, "random", np.random.default_rng(0)) == []
        fn = case.fn()
        assert kd.elems(fn.arg_types()[0]) * bfp.itemsize(fn.arg_dtype(0)) % 4 == 0
        assert kd.design(kernels.zero, **case.kwargs, **case.harness_opts())
