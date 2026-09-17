# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Kernel host results follow argument declarations, not algorithm names."""

import numpy as np
import pytest
from aie.helpers.util import v8bfp16ebs8
from aie.iron.kernel import DesignShape, ExternalFunction
from aie.iron.kernels._common import KernelContract, TensorLayout
from aie.utils.compile.jit.markers import In, Out, Scalar
from aie.utils.verify import Tolerance


def _kernel(types, contract):
    fn = ExternalFunction(
        "result_test", source_string="void result_test() {}", arg_types=types
    )
    fn.contract = contract
    return fn


def _tile(dtype=np.int32, size=4):
    return np.ndarray[(size,), np.dtype[dtype]]


def test_expected_omits_bound_scalars_but_passes_free_scalars():
    fn = _kernel(
        [_tile(), np.int32, np.float32, _tile()],
        KernelContract(
            roles=(In, Scalar, Scalar, Out),
            scalar_bindings=((1, 4),),
            reference=lambda x, scale: x * scale,
        ),
    )
    result = fn.expected([np.array([1, 2, 3, 4])], scalars=(2.5,))
    np.testing.assert_array_equal(result, [2, 5, 7, 10])
    assert result.dtype == np.int32


@pytest.mark.parametrize("shape", list(DesignShape))
def test_judge_uses_declared_layout_for_every_design_shape(shape):
    layout = TensorLayout(
        (2, 2),
        pack=lambda x: x.transpose(0, 2, 1).reshape(-1, 4),
        unpack=lambda x: x.reshape(-1, 2, 2).transpose(0, 2, 1),
    )
    fn = _kernel(
        [_tile(), _tile()],
        KernelContract(roles=(In, Out), layouts=(None, layout)),
    )
    fn.design_shape = shape
    ref = np.arange(8).reshape(2, 2, 2)
    assert fn.judge(layout.encode(ref).ravel(), ref, calls=2)
    assert not fn.judge(ref.ravel(), ref, calls=2)


def test_judge_trims_padding_per_call_after_decoding():
    layout = TensorLayout((4,), unpack=lambda x: x[:, ::-1])
    fn = _kernel(
        [_tile(), _tile()],
        KernelContract(roles=(In, Out), layouts=(None, layout), out_valid=1),
    )
    assert fn.judge([99, 99, 99, 1, 88, 88, 88, 2], [[1], [2]], calls=2)
    assert not fn.judge([99, 99, 99, 1, 88, 88, 88, 3], [[1], [2]], calls=2)


def test_multiple_outputs_cast_and_judge_independently():
    reverse = TensorLayout((4,), unpack=lambda x: x[:, ::-1])
    fn = _kernel(
        [_tile(), _tile(np.int16), _tile(np.float32)],
        KernelContract(
            roles=(In, Out, Out),
            reference=lambda x: (x + 1, x / 2),
            layouts=(None, None, reverse),
        ),
    )
    refs = fn.expected([np.arange(8).reshape(2, 4)])
    assert tuple(r.dtype for r in refs) == (np.dtype(np.int16), np.dtype(np.float32))
    assert fn.output_dtype(tuple(r.dtype for r in refs)) == (np.int16, np.float32)
    actual = (refs[0].ravel(), refs[1][:, ::-1].ravel())
    assert all(fn.judge(actual, refs, calls=2))
    actual[1][0] += 1
    verdicts = fn.judge(actual, refs, calls=2)
    assert verdicts[0] and not verdicts[1]


def test_bfp_dtype_is_per_output_not_per_kernel():
    fn = _kernel(
        [_tile(), _tile(v8bfp16ebs8), _tile(np.int16)],
        KernelContract(roles=(In, Out, Out), reference=lambda x: (x / 2, x)),
    )
    refs = fn.expected([np.arange(4)])
    assert tuple(r.dtype for r in refs) == (np.dtype(np.float32), np.dtype(np.int16))
    assert fn.output_dtype(tuple(r.dtype for r in refs)) == (np.uint8, np.int16)


def test_multioutput_arity_errors_are_explicit():
    fn = _kernel(
        [_tile(), _tile(), _tile()],
        KernelContract(roles=(In, Out, Out), reference=lambda x: x),
    )
    with pytest.raises(ValueError, match="reference must return"):
        fn.expected([np.arange(4)])
    with pytest.raises(ValueError, match="one reference dtype"):
        fn.output_dtype(np.dtype(np.int32))
    with pytest.raises(ValueError, match="one actual and reference"):
        fn.judge(np.arange(8).reshape(2, 4), (np.arange(4), np.arange(4)))


def test_judge_honors_explicit_tolerance_override():
    fn = _kernel(
        [_tile(np.float32), _tile(np.float32)],
        KernelContract(roles=(In, Out), tolerance=Tolerance.exact()),
    )
    ref = np.ones(4, dtype=np.float32)
    got = ref + 0.001
    assert not fn.judge(got, ref)
    assert fn.judge(got, ref, tolerance=Tolerance.relative(0.01))
