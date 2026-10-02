# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Kernel host results follow argument declarations, not algorithm names."""

import numpy as np
import pytest
from aie.helpers.npdtypes import v8bfp16ebs8
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels import Param
from aie.iron.kernels._common import KernelContract, TensorLayout
from aie.utils.compile.jit.markers import In, Out
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
            roles=(In, Param, Param, Out),
            parameter_bindings=((1, 4),),
            reference=lambda x, scale: x * scale,
        ),
    )
    result = fn.expected([np.array([1, 2, 3, 4])], scalars=(2.5,))
    np.testing.assert_array_equal(result, [2, 5, 7, 10])
    assert result.dtype == np.int32


@pytest.mark.parametrize("binding", [(-1, 4), (3, 4), (0, 4), ("1", 4), (1.0, 4)])
def test_parameter_bindings_require_param_indices(binding):
    with pytest.raises(ValueError, match="distinct Param"):
        KernelContract(roles=(In, Param, Out), parameter_bindings=(binding,))


@pytest.mark.parametrize(
    "arg_type,value,error",
    [
        (np.int32, np.arange(4), "expected scalar parameter"),
        (_tile(), 4, "tensor parameter"),
        (_tile(), np.ones(3), "tensor parameter"),
    ],
)
def test_parameter_bindings_validate_against_abi(arg_type, value, error):
    fn = _kernel(
        [_tile(), arg_type, _tile()],
        KernelContract(
            roles=(In, Param, Out),
            parameter_bindings=((1, value),),
            reference=lambda x: x,
        ),
    )
    with pytest.raises(ValueError, match=error):
        fn.expected([np.arange(4)])


def test_input_limit_counts_tensor_params_not_scalar_params():
    fn = _kernel(
        [_tile(), np.int32, np.int32, _tile()],
        KernelContract(
            roles=(In, Param, Param, Out),
            parameter_bindings=((1, 4),),
            acc_dtype=np.int32,
            reduction=4,
        ),
    )
    assert fn.input_limit(np.int32) == np.iinfo(np.int32).max // 4 // 4
    fn = _kernel(
        [_tile(), _tile(), np.int32, _tile()],
        KernelContract(
            roles=(In, Param, Param, Out),
            parameter_bindings=((1, np.ones(4)),),
            acc_dtype=np.int32,
            reduction=4,
        ),
    )
    assert fn.input_limit(np.int32) == int(np.sqrt(np.iinfo(np.int32).max // 4 // 4))


def test_reference_checks_tensor_and_scalar_arity_separately():
    fn = _kernel(
        [_tile(), np.int32, _tile(), _tile()],
        KernelContract(roles=(In, Param, Param, Out), reference=lambda x, n, w: x),
    )
    with pytest.raises(ValueError, match="expected 2 input arrays"):
        fn.expected([np.arange(4)], scalars=(1,))
    with pytest.raises(ValueError, match="expected 1 scalar"):
        fn.expected([np.arange(4), np.arange(4)])
    with pytest.raises(ValueError, match="expected 2 input arrays"):
        fn.param_values([np.arange(4), np.arange(4), np.arange(4)])


def test_scalar_abi_requires_param_and_cannot_have_layout():
    with pytest.raises(ValueError, match="scalar arguments require Param"):
        KernelContract(roles=(In, Out)).validate_types([np.int32, _tile()])
    with pytest.raises(ValueError, match="layouts require tensor"):
        KernelContract(
            roles=(Param, Out), layouts=(TensorLayout((4,)), None)
        ).validate_types([np.int32, _tile()])
    with pytest.raises(ValueError, match="one entry"):
        KernelContract(roles=(In, Out)).validate_types([_tile()])


@pytest.mark.parametrize(
    "arg_type", [tuple[int, float], tuple[(4,), np.dtype[np.int32]], str]
)
def test_contract_rejects_non_numpy_abi_types(arg_type):
    from aie.iron.kernels._common import _is_tensor_type

    assert not _is_tensor_type(arg_type)
    with pytest.raises(ValueError, match="require NumPy"):
        KernelContract(roles=(Param, Out)).validate_types([arg_type, _tile()])


def test_contract_rejects_malformed_tensor_alias():
    with pytest.raises(ValueError, match="expected np.ndarray"):
        KernelContract(roles=(In, Out)).validate_types(
            [np.ndarray[(4,), np.int32], _tile()]
        )


def test_contract_rejects_raw_mlir_types_with_host_diagnostic():
    from aie.ir import Context, IntegerType, Location, MemRefType

    with Context(), Location.unknown():
        scalar = IntegerType.get_signless(32)
        for arg_type in (scalar, MemRefType.get((4,), scalar)):
            with pytest.raises(ValueError, match="require NumPy"):
                KernelContract(roles=(Param, Out)).validate_types([arg_type, _tile()])


def test_judge_uses_declared_layout():
    layout = TensorLayout(
        (2, 2),
        pack=lambda x: x.transpose(0, 2, 1).reshape(-1, 4),
        unpack=lambda x: x.reshape(-1, 2, 2).transpose(0, 2, 1),
    )
    fn = _kernel(
        [_tile(), _tile()],
        KernelContract(roles=(In, Out), layouts=(None, layout)),
    )
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
    assert fn.output_dtype() == (np.int16, np.float32)
    actual = (refs[0].ravel(), refs[1][:, ::-1].ravel())
    assert fn.judge(actual, refs, calls=2)
    actual[1][0] += 1
    verdict = fn.judge(actual, refs, calls=2)
    assert not verdict
    assert verdict.n_checked == 16 and verdict.n_mismatch == 1
    assert verdict.first_bad_index == 11
    assert "output 1 (argument 2)" in verdict.detail


def test_bfp_dtype_is_per_output_not_per_kernel():
    fn = _kernel(
        [_tile(), _tile(v8bfp16ebs8), _tile(np.int16)],
        KernelContract(roles=(In, Out, Out), reference=lambda x: (x / 2, x)),
    )
    refs = fn.expected([np.arange(4)])
    assert tuple(r.dtype for r in refs) == (np.dtype(np.float32), np.dtype(np.int16))
    assert fn.output_dtype(tuple(r.dtype for r in refs)) == (np.uint8, np.int16)
    assert fn.output_dtype() == (np.uint8, np.int16)


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


def test_judge_evaluates_a_bound_tolerance_on_the_inputs_and_scalars():
    fn = _kernel(
        [_tile(np.float32), np.float32, _tile(np.float32)],
        KernelContract(
            roles=(In, Param, Out),
            reference=lambda x, scale: x * scale,
            tolerance=Tolerance.bounded(lambda x, scale: np.abs(x) * scale / 100),
        ),
    )
    x = np.array([[1, 2, 3, 4], [0, 10, 20, 30]], dtype=np.float32)
    ref = fn.expected([x], scalars=(2.0,))
    got = ref.ravel() + np.float32(0.015) * np.abs(x.ravel())
    assert fn.judge(got, ref, calls=2, inputs=[x], scalars=(2.0,))
    # The bound follows the scalar the reference was given.
    assert not fn.judge(got, ref, calls=2, inputs=[x], scalars=(1.0,))
    with pytest.raises(ValueError, match="function of the inputs"):
        fn.judge(got, ref, calls=2)


def test_output_only_reference_repeats_per_call_after_layout_decode():
    layout = TensorLayout((2, 2), unpack=lambda x: x.reshape(-1, 2, 2))
    fn = _kernel(
        [_tile()],
        KernelContract(
            roles=(Out,),
            layouts=(layout,),
            reference=lambda: np.zeros((1, 2, 2), np.int32),
        ),
    )
    actual = np.zeros(12, np.int32)
    reference = fn.expected([])
    verdict = fn.judge(actual, reference, calls=3)
    assert verdict and verdict.n_checked == 12
    actual[9] = 1
    verdict = fn.judge(actual, reference, calls=3)
    assert not verdict and verdict.first_bad_index == 9
    assert fn.judge(actual, actual.reshape(3, 2, 2), calls=3)


def test_parameter_only_multioutput_reference_repeats_valid_elements():
    fn = _kernel(
        [np.int32, _tile(), _tile()],
        KernelContract(
            roles=(Param, Out, Out),
            out_valid=1,
            reference=lambda value: (np.array([value]), np.array([-value])),
        ),
    )
    refs = fn.expected([], scalars=(3,))
    first = np.array([3, 99, 99, 99] * 3)
    second = np.array([-3, 99, 99, 99] * 3)
    verdict = fn.judge((first, second), refs, calls=3)
    assert verdict and verdict.n_checked == 6
    second[8] = 0
    verdict = fn.judge((first, second), refs, calls=3)
    assert not verdict and verdict.first_bad_index == 5


def test_tensor_parameter_only_reference_repeats_one_complete_tile():
    fn = _kernel(
        [_tile(), _tile()],
        KernelContract(roles=(Param, Out), reference=lambda weights: weights),
    )
    weights = np.arange(4, dtype=np.int32)
    reference = fn.expected([weights])
    actual = np.tile(weights, 3)
    verdict = fn.judge(actual, reference, calls=3)
    assert verdict and verdict.n_checked == 12
    actual[-1] += 1
    verdict = fn.judge(actual, reference, calls=3)
    assert not verdict and verdict.first_bad_index == 11
    with pytest.raises(ValueError):
        fn.judge(actual, reference[:2], calls=3)


def test_streamed_inputs_do_not_broadcast_a_one_tile_reference():
    fn = _kernel(
        [_tile(), _tile()],
        KernelContract(roles=(In, Out), reference=lambda x: x),
    )
    with pytest.raises(ValueError):
        fn.judge(np.zeros(12), np.zeros(4), calls=3)


def test_output_only_broadcast_requires_a_complete_reference_tile():
    fn = _kernel([_tile()], KernelContract(roles=(Out,)))
    with pytest.raises(ValueError):
        fn.judge(np.zeros(12), np.zeros(2), calls=3)
