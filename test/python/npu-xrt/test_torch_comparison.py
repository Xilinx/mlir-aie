# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings
# REQUIRES: pytorch

import pytest
import numpy as np
import torch
import aie.iron as iron
from aie.utils.hostruntime.tensor_class import CPUOnlyTensor, Tensor
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16

TENSOR_CLASSES = [CPUOnlyTensor, XRTTensor]
TEST_DTYPES = [np.float32, np.int32, bfloat16]
TORCH_DTYPES = [torch.float32, torch.int32, torch.bfloat16]
TEST_SHAPES = [(2, 3), (1, 5), (4, 1), (3, 3, 3), (10,), ()]


def bfloat16_safe_allclose(dtype, arr1, arr2):
    if not isinstance(arr1, torch.Tensor):
        arr1 = arr1.to_torch()
    if not isinstance(arr2, torch.Tensor):
        arr2 = arr2.to_torch()

    if dtype == bfloat16 or dtype == torch.bfloat16:
        arr1 = arr1.to(torch.float16)
        arr2 = arr2.to(torch.float16)

    return torch.allclose(arr1.to(arr2.dtype), arr2)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_zeros(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    iron_t = iron.zeros(shape, dtype=dtype)
    torch_t = torch.zeros(shape, dtype=torch_dtype)
    assert bfloat16_safe_allclose(dtype, iron_t, torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_ones(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    iron_t = iron.ones(shape, dtype=dtype)
    torch_t = torch.ones(shape, dtype=torch_dtype)
    assert bfloat16_safe_allclose(dtype, iron_t, torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.integer)],
        [d for d in TORCH_DTYPES if not d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_randint(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    low, high = 0, 32
    iron_t = iron.randint(low, high, shape, dtype=dtype)
    torch_t = torch.randint(low, high, shape, dtype=torch_dtype)
    # Just check that the values are in the right range, since the
    # actual values will be different.
    assert iron_t.shape == torch_t.shape
    if shape == ():
        assert low <= iron_t.numpy().item() < high
    else:
        assert torch.all(
            (torch.from_numpy(iron_t.numpy()) >= low)
            & (torch.from_numpy(iron_t.numpy()) < high)
        )


@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.integer)],
        [d for d in TORCH_DTYPES if not d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_arange_integer(dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    start, end = 3, 9
    iron_t = iron.arange(start, end, dtype=dtype)
    torch_t = torch.arange(start, end, dtype=torch_dtype)
    assert torch.equal(torch.from_numpy(iron_t.numpy()), torch_t)


@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.floating)],
        [d for d in TORCH_DTYPES if d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_arange_floats(dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    start, end, step = 1.0, 5.0, 1.5
    iron_t = iron.arange(start, end, step, dtype=dtype)
    torch_t = torch.arange(start, end, step, dtype=torch_dtype)
    assert bfloat16_safe_allclose(dtype, iron_t, torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.floating)],
        [d for d in TORCH_DTYPES if d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_rand(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    if shape == ():
        with pytest.raises(ValueError, match="rand.. received no arguments"):
            iron.rand(*shape, dtype=dtype)
        with pytest.raises(
            TypeError,
            match=r"rand\(\) missing 1 required positional arguments: \"size\"",
        ):
            torch.rand(*shape, dtype=torch_dtype)
    else:
        iron_t = iron.rand(*shape, dtype=dtype)
        torch_t = torch.rand(*shape, dtype=torch_dtype)
        assert iron_t.shape == torch_t.shape
        assert torch.all(
            (torch.from_numpy(iron_t.numpy()) >= 0)
            & (torch.from_numpy(iron_t.numpy()) < 1.0)
        )


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_zeros_like(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    iron_t_orig = iron.ones(shape, dtype=dtype)
    torch_t_orig = torch.ones(shape, dtype=torch_dtype)

    iron_t = iron.zeros_like(iron_t_orig)
    torch_t = torch.zeros_like(torch_t_orig)

    assert bfloat16_safe_allclose(dtype, iron_t, torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_fill(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    iron_t = iron.zeros(shape, dtype=dtype)
    torch_t = torch.zeros(shape, dtype=torch_dtype)

    fill_value = 42 if np.issubdtype(dtype, np.integer) else 42.5
    iron_t.fill_(fill_value)
    torch_t.fill_(fill_value)

    assert bfloat16_safe_allclose(dtype, iron_t, torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_len(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    iron_t = iron.zeros(shape, dtype=dtype)
    torch_t = torch.zeros(shape, dtype=torch_dtype)
    if not shape:
        # len of a 0-d tensor is a TypeError
        with pytest.raises(TypeError):
            len(iron_t)
        with pytest.raises(TypeError):
            len(torch_t)
    else:
        assert len(iron_t) == len(torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_to_torch(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    iron_t = iron.ones(shape, dtype=dtype)
    torch_t = iron_t.to_torch()
    assert isinstance(torch_t, torch.Tensor)
    assert iron_t.shape == torch_t.shape
    if dtype == bfloat16:
        # torch doesn't support bfloat16 numpy conversion, so we convert to float32
        assert torch.allclose(
            torch_t.to(torch.float32), torch.ones(shape, dtype=torch.float32)
        )
    else:
        assert torch.allclose(torch_t, torch.ones(shape, dtype=torch_dtype))


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype, torch_dtype", zip(TEST_DTYPES, TORCH_DTYPES))
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_from_torch(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    torch_t = torch.ones(shape, dtype=torch_dtype)
    iron_t = tensorclass.from_torch(torch_t)
    assert isinstance(iron_t, Tensor)
    assert iron_t.shape == torch_t.shape
    assert bfloat16_safe_allclose(dtype, iron_t, torch_t)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.integer)],
        [d for d in TORCH_DTYPES if not d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_iron_torch_iron(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    low, high = 0, 100
    iron_t_orig = iron.randint(low, high, shape, dtype=dtype)
    torch_t = iron_t_orig.to_torch()
    iron_t_new = tensorclass.from_torch(torch_t)
    assert bfloat16_safe_allclose(dtype, iron_t_orig, iron_t_new)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.integer)],
        [d for d in TORCH_DTYPES if not d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_iron_torch(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    low, high = 0, 100
    torch_t_orig = torch.randint(low, high, shape, dtype=torch_dtype)
    iron_t = tensorclass.from_torch(torch_t_orig)
    torch_t_new = iron_t.to_torch()
    assert bfloat16_safe_allclose(dtype, torch_t_orig, torch_t_new)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.floating)],
        [d for d in TORCH_DTYPES if d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_iron_torch_iron_float(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    if shape == ():
        return
    iron_t_orig = iron.rand(*shape, dtype=dtype)
    torch_t = iron_t_orig.to_torch()
    iron_t_new = tensorclass.from_torch(torch_t)
    assert bfloat16_safe_allclose(dtype, iron_t_orig, iron_t_new)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    zip(
        [d for d in TEST_DTYPES if np.issubdtype(d, np.floating)],
        [d for d in TORCH_DTYPES if d.is_floating_point],
    ),
)
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_iron_torch_float(shape, dtype, torch_dtype, tensorclass):
    iron.set_tensor_class(tensorclass)
    if shape == ():
        return
    torch_t_orig = torch.rand(*shape, dtype=torch_dtype)
    iron_t = tensorclass.from_torch(torch_t_orig)
    torch_t_new = iron_t.to_torch()
    assert bfloat16_safe_allclose(dtype, torch_t_orig, torch_t_new)
