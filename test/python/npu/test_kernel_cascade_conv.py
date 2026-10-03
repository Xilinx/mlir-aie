# test_kernel_cascade_conv.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu1_xrt% %pytest -m "not extensive" %s
# RUN: %run_on_npu2_xrt% %pytest -m "not extensive" %s
# RUN: %run_on_npu2_hrx% %pytest -m "not extensive" %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

"""Check MobileNet's cascade convolutions against independent integer algebra."""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects import memref
from aie.extras.dialects.arith import constant
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron import (
    CascadeFlow,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

# These scalar cascade entry points process exactly seven pixels per call.
_WIDTH, _IC, _OC = 7, 32, 32
_INPUT_SPLIT = _OUTPUT_SPLIT = 2
_HALF_IC = _IC // _INPUT_SPLIT
_OC8 = _OC // (8 * _OUTPUT_SPLIT)


def _cascade_program(with_skip, block_index):
    channels = _HALF_IC if with_skip else _IC
    weight_count = _HALF_IC * (_OC // _OUTPUT_SPLIT)
    opts = dict(
        input_width=_WIDTH,
        input_channels=channels,
        weight_count=weight_count,
        block_index=block_index,
    )
    if with_skip:
        put = kernels.bn_conv2dk1_input_split_partial_put_ui8(**opts)
        get = kernels.bn_conv2dk1_input_split_partial_skip_get(
            **opts, output_channels=_OC
        )
    else:
        put = kernels.bn_conv2dk1_partial_put_i8(**opts)
        get = kernels.bn_conv2dk1_partial_get_relu_i8(**opts, output_channels=_OC)
    in_ty, wt_ty, out_ty = get.arg_types()[:3]
    full_wt_ty = np.ndarray[(_HALF_IC * _OC,), np.dtype[np.int8]]
    get_input_ty = in_ty
    if with_skip:
        # One byte buffer carries activation and residual: GET has only two
        # incoming DMA channels, one for this row and one for its weights.
        get_input_ty = np.ndarray[(_WIDTH * (_HALF_IC + _OC),), np.dtype[np.int8]]
    input_types = [in_ty, full_wt_ty, get_input_ty, full_wt_ty]
    inputs = [
        ObjectFifo(ty, name=name)
        for name, ty in (
            ("input_put", in_ty),
            ("weights_put", wt_ty),
            ("input_get", get_input_ty),
            ("weights_get", wt_ty),
        )
    ]
    output = ObjectFifo(out_ty, name="output")

    def put_core(act, weights, kernel):
        row = act.acquire(1)
        for weight_index in range_(_OUTPUT_SPLIT):
            chunk = weights.acquire(1)
            for oc in range_(_OC8):
                # The split-input ABI still takes the full channel count.
                kernel(row, chunk, _WIDTH, _IC, _OC, _INPUT_SPLIT, weight_index, 0, oc)
            weights.release(1)
        act.release(1)

    def get_core(act, weights, out, kernel):
        raw, result = act.acquire(1), out.acquire(1)
        if with_skip:
            # Typed byte views preserve unsigned activations and signed skips,
            # following kernel_design's guarded-buffer memref.view pattern.
            row = memref.view(
                np_ndarray_type_to_memref_type(in_ty),
                raw,
                constant(0, index=True),
                [],
            )
            skip_row = memref.view(
                np_ndarray_type_to_memref_type(kernel.arg_types()[3]),
                raw,
                constant(_WIDTH * _HALF_IC, index=True),
                [],
            )
        else:
            row = raw
        for weight_index in range_(_OUTPUT_SPLIT):
            chunk = weights.acquire(1)
            for oc in range_(_OC8):
                if with_skip:
                    kernel(
                        row,
                        chunk,
                        result,
                        skip_row,
                        _WIDTH,
                        _IC,
                        _OC,
                        3,
                        1,
                        _INPUT_SPLIT,
                        _OUTPUT_SPLIT,
                        weight_index,
                        0,
                        oc,
                    )
                else:
                    kernel(
                        row,
                        chunk,
                        result,
                        _WIDTH,
                        _IC,
                        _OC,
                        2,
                        _INPUT_SPLIT,
                        _OUTPUT_SPLIT,
                        weight_index,
                        0,
                        oc,
                    )
            weights.release(1)
        act.release(1)
        out.release(1)

    w_put = Worker(put_core, [inputs[0].cons(), inputs[1].cons(), put], tile=Tile(0, 3))
    get_args = [inputs[2].cons(), inputs[3].cons(), output.prod(), get]
    w_get = Worker(get_core, get_args, tile=Tile(0, 2))
    CascadeFlow(w_put, w_get)

    def seq(*args):
        hosts = args[: len(input_types)]
        result = args[len(input_types)]
        handles = args[len(input_types) + 1 :]
        for host, handle in zip(hosts, handles[:-1]):
            handle.fill(host)
        handles[-1].drain(result, wait=True)

    rt = Runtime(
        seq,
        input_types + [out_ty] + [f.prod() for f in inputs] + [output.cons()],
    )
    return Program(
        iron.get_current_device(), rt, workers=[w_put, w_get]
    ).resolve_program()


@iron.jit
def _relu_design(
    x_put: In,
    w_put: In,
    x_get: In,
    w_get: In,
    out: Out,
    *,
    block_index: CompileTime[int]
):
    return _cascade_program(False, block_index)


@iron.jit
def _skip_design(
    x_put: In,
    w_put: In,
    x_get_skip: In,
    w_get: In,
    out: Out,
    *,
    block_index: CompileTime[int]
):
    return _cascade_program(True, block_index)


def _pack_row(row):
    """[W,C] -> the kernels' [C/8,W,8] activation/output layout."""
    return row.reshape(_WIDTH, -1, 8).transpose(1, 0, 2).copy().ravel()


def _pack_weights(weights):
    """[IC/2,OC] -> consecutive [OC/8,IC/16,ic8,oc8] weight chunks."""
    return (
        weights.reshape(_HALF_IC // 8, 8, _OC // 8, 8)
        .transpose(2, 0, 1, 3)
        .copy()
        .ravel()
    )


def _round_even(value, shift):
    quotient, remainder = np.divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + (
        (remainder > halfway) | ((remainder == halfway) & ((quotient & 1) != 0))
    )


def _reference(put_sum, get_sum, skip=None):
    acc = put_sum + get_sum
    if skip is None:
        return np.clip(_round_even(acc, 2), 0, 255).astype(np.uint8)
    conv = np.clip(_round_even(acc, 3), -128, 127)
    return np.clip(_round_even(conv + skip.astype(np.int64), 1), -128, 127).astype(
        np.int8
    )


def _data(with_skip):
    rng = np.random.default_rng(42)
    dtype = np.uint8 if with_skip else np.int8
    lo, hi = (0, 256) if with_skip else (-32, 33)
    x = rng.integers(lo, hi, size=(_WIDTH, _IC)).astype(dtype)
    weights = [
        rng.integers(-8, 9, size=(_HALF_IC, _OC)).astype(np.int8) for _ in range(2)
    ]
    # Exact halfway values with even/odd quotients, negative values, and
    # saturating lanes coexist with dense random multi-channel dot products.
    x[:, 0] = x[:, _HALF_IC] = 2 if with_skip else 1
    x[:, 1] = x[:, _HALF_IC + 1] = 127
    for w in weights:
        w[:, :8] = 0
        w[0, :8] = [1, 3, 5, 7, -1, -3, 127, -128]
        w[1, 6:8] = [127, -128]
    skip = None
    if with_skip:
        skip = rng.integers(-128, 128, size=(_WIDTH, _OC)).astype(np.int8)
        skip[:, :8] = [1, 1, 3, -1, -3, -1, 127, -128]
    put_sum = x[:, :_HALF_IC].astype(np.int64) @ weights[0].astype(np.int64)
    get_sum = x[:, _HALF_IC:].astype(np.int64) @ weights[1].astype(np.int64)
    expected = _reference(put_sum, get_sum, skip)
    # A zeroed/dropped cascade, GET product, or residual must not pass.
    assert np.any(expected != _reference(0, get_sum, skip))
    assert np.any(expected != _reference(put_sum, 0, skip))
    if with_skip:
        assert np.any(expected != _reference(put_sum, get_sum, np.zeros_like(skip)))
    a_put = x[:, :_HALF_IC] if with_skip else x
    a_get = x[:, _HALF_IC:] if with_skip else x
    inputs = [
        _pack_row(a_put),
        _pack_weights(weights[0]),
        _pack_row(a_get),
        _pack_weights(weights[1]),
    ]
    if with_skip:
        inputs[2] = np.concatenate([inputs[2].view(np.int8), _pack_row(skip)])
    return inputs, _pack_row(expected)


def _check_pair(design, with_skip, block_index):
    inputs, expected = _data(with_skip)
    tensors = [iron.tensor(x, dtype=x.dtype, device="npu") for x in inputs]
    poison = np.full(expected.shape, 0xA5, np.uint8).view(expected.dtype)
    out = iron.tensor(poison, dtype=expected.dtype, device="npu")
    design(*tensors, out, block_index=block_index)
    np.testing.assert_array_equal(out.numpy().copy(), expected)


@pytest.mark.supported_devices("npu1", "npu2")
@pytest.mark.kernel_check(
    "bn_conv2dk1_partial_put_i8", "bn_conv2dk1_partial_get_relu_i8"
)
@pytest.mark.parametrize("block_index", [13, 14])
def test_cascade_conv_relu_pair(block_index):
    _check_pair(_relu_design, False, block_index)


@pytest.mark.supported_devices("npu1", "npu2")
@pytest.mark.kernel_check(
    "bn_conv2dk1_input_split_partial_put_ui8",
    "bn_conv2dk1_input_split_partial_skip_get",
)
@pytest.mark.parametrize("block_index", [13, 14])
def test_cascade_conv_skip_pair(block_index):
    _check_pair(_skip_design, True, block_index)
