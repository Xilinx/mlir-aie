# test_algorithms.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np

import aie.iron as iron
from aie.iron import Compile, ExternalFunction, In, Out
from aie.iron.algorithms import (
    for_each_typed,
    transform_binary_typed,
    transform_parallel_binary_typed,
    transform_parallel_typed,
    transform_typed,
)

TILE_SIZE = 16


# =============================================================================
# @iron.jit wrappers using typed algorithm variants
# =============================================================================


@iron.jit
def run_transform(
    input: In,
    output: Out,
    *,
    func: Compile[object],
    N_in: Compile[int],
    N_out: Compile[int],
    dtype_in: Compile[object],
    dtype_out: Compile[object],
    tile_size: Compile[int] = 16,
):
    if N_in != N_out:
        raise ValueError(f"Tensor 1 shape ({N_out},) doesn't match expected ({N_in},)")
    if dtype_in != dtype_out:
        raise ValueError(
            f"Tensor 1 dtype {dtype_out} doesn't match expected {dtype_in}"
        )
    tensor_ty = np.ndarray[(N_in,), np.dtype[dtype_in]]
    return transform_typed(func, tensor_ty, tile_size=tile_size)


@iron.jit
def run_transform_binary(
    first: In,
    second: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[object],
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_binary_typed(func, tensor_ty, tile_size=tile_size)


@iron.jit
def run_transform_parallel(
    input: In,
    output: Out,
    *,
    func: Compile[object],
    N_in: Compile[int],
    N_out: Compile[int],
    dtype_in: Compile[object],
    dtype_out: Compile[object],
    tile_size: Compile[int] = 16,
):
    if N_in != N_out:
        raise ValueError(f"Tensor 1 shape ({N_out},) doesn't match expected ({N_in},)")
    if dtype_in != dtype_out:
        raise ValueError(
            f"Tensor 1 dtype {dtype_out} doesn't match expected {dtype_in}"
        )
    tensor_ty = np.ndarray[(N_in,), np.dtype[dtype_in]]
    return transform_parallel_typed(func, tensor_ty, tile_size=tile_size)


@iron.jit
def run_transform_parallel_with_scalar(
    input: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[object],
    scalar_param: Compile[int],
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_parallel_typed(func, tensor_ty, scalar_param, tile_size=tile_size)


@iron.jit
def run_transform_parallel_binary(
    first: In,
    second: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[object],
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_parallel_binary_typed(func, tensor_ty, tile_size=tile_size)


@iron.jit
def run_for_each(
    data: In,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[object],
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return for_each_typed(func, tensor_ty, tile_size=tile_size)


# =============================================================================
# transform tests
# =============================================================================


def test_transform_add():
    """Test transform algorithm with simple add_one operation"""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    original = input.numpy().copy()
    run_transform(
        input,
        output,
        func=lambda a: a + 1,
        N_in=input.shape[0],
        N_out=output.shape[0],
        dtype_in=input.dtype,
        dtype_out=output.dtype,
        tile_size=TILE_SIZE,
    )

    assert np.allclose(original + 1, output.numpy())


@pytest.mark.parametrize("add_value", [-1, 0, 1, 2, 3])
def test_transform_add_parametrized(add_value):
    """Test transform algorithm with different add values."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    original = input.numpy().copy()
    run_transform(
        input,
        output,
        func=lambda a: a + add_value,
        N_in=input.shape[0],
        N_out=output.shape[0],
        dtype_in=input.dtype,
        dtype_out=output.dtype,
        tile_size=TILE_SIZE,
    )

    assert np.allclose(original + add_value, output.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_transform_different_datatypes_extern(dtype, c_type, skip_on_f32_failure):
    """Test transform algorithm with ExternalFunction on different datatypes."""
    with skip_on_f32_failure():
        add_one = ExternalFunction(
            "add_one",
            source_string=f"""extern "C" {{
                void add_one({c_type}* input, {c_type}* output, int tile_size) {{
                    for (int i = 0; i < tile_size; i++) {{
                        output[i] = input[i] + 1.0f;
                    }}
                }}
            }}""",
            arg_types=[
                np.ndarray[(TILE_SIZE,), np.dtype[dtype]],
                np.ndarray[(TILE_SIZE,), np.dtype[dtype]],
                np.int32,
            ],
        )
        if np.issubdtype(dtype, np.floating):
            input = iron.rand(1024, dtype=dtype, device="npu")
        else:
            input = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
        output = iron.zeros_like(input)
        run_transform(
            input,
            output,
            func=add_one,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )
        assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_different_num_elements(num_elements):
    """Test transform algorithm with different input size."""
    input = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    run_transform(
        input,
        output,
        func=lambda a: a + 1,
        N_in=input.shape[0],
        N_out=output.shape[0],
        dtype_in=input.dtype,
        dtype_out=output.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(input.numpy() + 1, output.numpy())


def test_transform_shape_mismatch():
    """Test transform when input and output shapes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((512,), dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        run_transform(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


def test_transform_dtype_mismatch():
    """Test transform when input and output dtypes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((1024,), dtype=np.float32, device="npu")
    with pytest.raises(ValueError, match="dtype.*doesn't match"):
        run_transform(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


def test_transform_tile_size_mismatch():
    """Test transform when full input/output is not a multiple of tile size."""
    # 1000 is not divisible by default tile size of 16
    input = iron.randint(0, 100, (1000,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    with pytest.raises(ValueError, match="must be a multiple of tile size"):
        run_transform(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


def test_transform_tile_arg_type_mismatch():
    """Test transform when tile size and ExternalFunction input arg_type don't match."""
    WRONG_TILE_SIZE = 32
    add_one = ExternalFunction(
        "add_one",
        source_string=f"""extern "C" {{
            void add_one(int* input, int* output, int tile_size) {{
                for (int i = 0; i < tile_size; i++) {{
                    output[i] = input[i] + 1;
                }}
            }}
        }}""",
        arg_types=[
            np.ndarray[(WRONG_TILE_SIZE,), np.dtype[np.int32]],
            np.ndarray[(WRONG_TILE_SIZE,), np.dtype[np.int32]],
            np.int32,
        ],
    )
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    with pytest.raises(ValueError, match="tile_size.*does not match"):
        run_transform(
            input,
            output,
            func=add_one,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


# =============================================================================
# transform_binary tests
# =============================================================================


def test_transform_binary_add():
    """Test transform_binary algorithm with simple add_one operation."""
    first = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    run_transform_binary(
        first,
        second,
        output,
        func=lambda a, b: a + b,
        N=first.shape[0],
        dtype=first.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_binary_different_datatypes(dtype, skip_on_f32_failure):
    """Test transform_binary algorithm with different datatypes."""
    with skip_on_f32_failure():
        if np.issubdtype(dtype, np.floating):
            first = iron.rand(1024, dtype=dtype, device="npu")
            second = iron.rand(1024, dtype=dtype, device="npu")
        else:
            first = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
            second = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
        output = iron.zeros_like(first)
        run_transform_binary(
            first,
            second,
            output,
            func=lambda a, b: a + b,
            N=first.shape[0],
            dtype=first.dtype,
            tile_size=TILE_SIZE,
        )
        assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_binary_different_num_elements(num_elements):
    """Test transform_binary algorithm with different input size."""
    first = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    run_transform_binary(
        first,
        second,
        output,
        func=lambda a, b: a + b,
        N=first.shape[0],
        dtype=first.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


# =============================================================================
# transform_parallel tests
# =============================================================================


def test_transform_parallel_add():
    """Test transform_parallel algorithm with simple add_one operation."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    run_transform_parallel(
        input,
        output,
        func=lambda a: a + 1,
        N_in=input.shape[0],
        N_out=output.shape[0],
        dtype_in=input.dtype,
        dtype_out=output.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("add_value", [-1, 0, 1, 2, 3])
def test_transform_parallel_add_parametrized(add_value):
    """Test transform_parallel algorithm with different add values."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    original = input.numpy().copy()
    run_transform_parallel(
        input,
        output,
        func=lambda a: a + add_value,
        N_in=input.shape[0],
        N_out=output.shape[0],
        dtype_in=input.dtype,
        dtype_out=output.dtype,
        tile_size=TILE_SIZE,
    )

    assert np.allclose(original + add_value, output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_parallel_different_datatypes(dtype, skip_on_f32_failure):
    """Test transform_parallel algorithm with add operation on different datatypes."""
    with skip_on_f32_failure():
        if np.issubdtype(dtype, np.floating):
            input = iron.rand(1024, dtype=dtype, device="npu")
        else:
            input = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
        output = iron.zeros_like(input)
        run_transform_parallel(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )
        assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_parallel_different_num_elements(num_elements):
    """Test transform_parallel algorithm with different input size."""
    input = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    run_transform_parallel(
        input,
        output,
        func=lambda a: a + 1,
        N_in=input.shape[0],
        N_out=output.shape[0],
        dtype_in=input.dtype,
        dtype_out=output.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(input.numpy() + 1, output.numpy())


def test_transform_parallel_extern():
    """Test transform_parallel algorithm with ExternalFunction and scalar param."""
    scale_factor = 3
    scale = ExternalFunction(
        "scale",
        source_string=f"""extern "C" {{
            void scale(int* input, int* output, int factor, int tile_size) {{
                for (int i = 0; i < tile_size; i++) {{
                    output[i] = input[i] * factor;
                }}
            }}
        }}""",
        arg_types=[
            np.ndarray[(TILE_SIZE,), np.dtype[np.int32]],
            np.ndarray[(TILE_SIZE,), np.dtype[np.int32]],
            np.int32,
            np.int32,
        ],
    )
    input = iron.randint(1, 10, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    run_transform_parallel_with_scalar(
        input,
        output,
        func=scale,
        N=input.shape[0],
        dtype=input.dtype,
        scalar_param=scale_factor,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(input.numpy() * scale_factor, output.numpy())


def test_transform_parallel_shape_mismatch():
    """Test transform parallel when input and output shapes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((512,), dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        run_transform_parallel(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


def test_transform_parallel_dtype_mismatch():
    """Test transform parallel when input and output dtypes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((1024,), dtype=np.float32, device="npu")
    with pytest.raises(ValueError, match="dtype.*doesn't match"):
        run_transform_parallel(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


def test_transform_parallel_tile_size_mismatch():
    """Test transform parallel when number of elements is not a multiple of tile size."""
    # 1000 is not divisible by default tile size of 16
    input = iron.randint(0, 100, (1000,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    with pytest.raises(ValueError, match="must be a multiple of tile size"):
        run_transform_parallel(
            input,
            output,
            func=lambda a: a + 1,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


def test_transform_parallel_tile_arg_type_mismatch():
    """Test transform_parallel when tile size and ExternalFunction arg_type don't match."""
    WRONG_TILE_SIZE = 32
    add_one = ExternalFunction(
        "add_one",
        source_string=f"""extern "C" {{
            void add_one(int* input, int* output, int tile_size) {{
                for (int i = 0; i < tile_size; i++) {{
                    output[i] = input[i] + 1;
                }}
            }}
        }}""",
        arg_types=[
            np.ndarray[(WRONG_TILE_SIZE,), np.dtype[np.int32]],
            np.ndarray[(WRONG_TILE_SIZE,), np.dtype[np.int32]],
            np.int32,
        ],
    )
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    with pytest.raises(ValueError, match="tile_size.*does not match"):
        run_transform_parallel(
            input,
            output,
            func=add_one,
            N_in=input.shape[0],
            N_out=output.shape[0],
            dtype_in=input.dtype,
            dtype_out=output.dtype,
            tile_size=TILE_SIZE,
        )


# =============================================================================
# transform_parallel_binary tests
# =============================================================================


def test_transform_parallel_binary_add():
    """Test transform_parallel_binary algorithm with add operation."""
    first = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    run_transform_parallel_binary(
        first,
        second,
        output,
        func=lambda a, b: a + b,
        N=first.shape[0],
        dtype=first.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_parallel_binary_different_datatypes(dtype, skip_on_f32_failure):
    """Test transform_parallel_binary algorithm with add operation on different datatypes."""
    with skip_on_f32_failure():
        if np.issubdtype(dtype, np.floating):
            first = iron.rand(1024, dtype=dtype, device="npu")
            second = iron.rand(1024, dtype=dtype, device="npu")
        else:
            first = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
            second = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
        output = iron.zeros_like(first)
        run_transform_parallel_binary(
            first,
            second,
            output,
            func=lambda a, b: a + b,
            N=first.shape[0],
            dtype=first.dtype,
            tile_size=TILE_SIZE,
        )
        assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_parallel_binary_different_num_elements(num_elements):
    """Test transform_parallel_binary algorithm with different input size."""
    first = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    run_transform_parallel_binary(
        first,
        second,
        output,
        func=lambda a, b: a + b,
        N=first.shape[0],
        dtype=first.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


# =============================================================================
# for_each tests
# =============================================================================


def test_for_each_add():
    """Test for_each algorithm with simple add_one operation."""
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    run_for_each(
        data,
        func=lambda a: a + 1,
        N=data.shape[0],
        dtype=data.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_for_each_different_datatypes(dtype, skip_on_f32_failure):
    """Test for_each algorithm on different datatypes."""
    with skip_on_f32_failure():
        if np.issubdtype(dtype, np.floating):
            data = iron.rand(1024, dtype=dtype, device="npu")
        else:
            data = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
        original = data.numpy().copy()
        run_for_each(
            data,
            func=lambda a: a + 1,
            N=data.shape[0],
            dtype=data.dtype,
            tile_size=TILE_SIZE,
        )
        assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_for_each_different_datatypes_extern(dtype, c_type, skip_on_f32_failure):
    """Test for_each algorithm with ExternalFunction on different datatypes."""
    with skip_on_f32_failure():
        add_one = ExternalFunction(
            "add_one",
            source_string=f"""extern "C" {{
                void add_one({c_type}* input, {c_type}* output, int tile_size) {{
                    for (int i = 0; i < tile_size; i++) {{
                        output[i] = input[i] + 1.0f;
                    }}
                }}
            }}""",
            arg_types=[
                np.ndarray[(TILE_SIZE,), np.dtype[dtype]],
                np.ndarray[(TILE_SIZE,), np.dtype[dtype]],
                np.int32,
            ],
        )
        if np.issubdtype(dtype, np.floating):
            data = iron.rand(1024, dtype=dtype, device="npu")
        else:
            data = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
        original = data.numpy().copy()
        run_for_each(
            data,
            func=add_one,
            N=data.shape[0],
            dtype=data.dtype,
            tile_size=TILE_SIZE,
        )
        assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_for_each_different_num_elements(num_elements):
    """Test for_each algorithm with different input sizes."""
    data = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    run_for_each(
        data,
        func=lambda a: a + 1,
        N=data.shape[0],
        dtype=data.dtype,
        tile_size=TILE_SIZE,
    )
    assert np.allclose(original + 1, data.numpy())


def test_for_each_tile_arg_type_mismatch():
    """Test for_each when tile size and ExternalFunction arg_type don't match."""
    WRONG_TILE_SIZE = 32
    add_one = ExternalFunction(
        "add_one",
        source_string=f"""extern "C" {{
            void add_one(int* input, int* output, int tile_size) {{
                for (int i = 0; i < tile_size; i++) {{
                    output[i] = input[i] + 1;
                }}
            }}
        }}""",
        arg_types=[
            np.ndarray[(WRONG_TILE_SIZE,), np.dtype[np.int32]],
            np.ndarray[(WRONG_TILE_SIZE,), np.dtype[np.int32]],
            np.int32,
        ],
    )
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="tile_size.*does not match"):
        run_for_each(
            data,
            func=add_one,
            N=data.shape[0],
            dtype=data.dtype,
            tile_size=TILE_SIZE,
        )
