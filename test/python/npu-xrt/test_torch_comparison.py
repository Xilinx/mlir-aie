# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

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


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_to_torch_bfloat16(tensorclass):
    data = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
    tensor = tensorclass(data, dtype=bfloat16)
    torch_tensor = tensor.to_torch()
    assert (
        torch_tensor.dtype == torch.bfloat16
    ), f"Expected torch.bfloat16, got {torch_tensor.dtype}"
    assert torch.allclose(
        torch_tensor.float(), torch.tensor([1.0, 2.0, 3.0]).float()
    ), "Values mismatch in to_torch"


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_from_torch_bfloat16(tensorclass):
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    tensor = tensorclass.from_torch(data, device="cpu")
    assert tensor.dtype == bfloat16, f"Expected bfloat16, got {tensor.dtype}"
    np_data = tensor.numpy()
    expected = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
    assert np.array_equal(np_data, expected), "Values mismatch in from_torch"


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_roundtrip_bfloat16(tensorclass):
    # Round trip: numpy(bfloat16) -> Tensor -> torch(bfloat16) -> Tensor -> numpy(bfloat16)
    data = np.array([1.5, 2.5, 3.5], dtype=bfloat16)
    tensor = tensorclass(data, dtype=bfloat16)
    torch_tensor = tensor.to_torch()
    assert torch_tensor.dtype == torch.bfloat16
    tensor_back = tensorclass.from_torch(torch_tensor, device="cpu")
    assert tensor_back.dtype == bfloat16
    assert np.array_equal(tensor.numpy(), tensor_back.numpy())


# ---------------------------------------------------------------------------
# torch_view(): zero-copy write path without FROM_DEVICE sync
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_view_returns_bfloat16(tensorclass):
    """torch_view() returns a torch.bfloat16 tensor for bfloat16 buffers."""
    data = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
    tensor = tensorclass(data, dtype=bfloat16)
    view = tensor.torch_view()
    assert view.dtype == torch.bfloat16
    assert view.shape == (3,)


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_view_marks_device_cpu(tensorclass):
    """torch_view() marks the buffer as CPU-resident so to('npu') will sync."""
    data = np.zeros((2, 3), dtype=bfloat16)
    tensor = tensorclass(data, dtype=bfloat16)
    # Buffer starts on npu (XRTTensor default) or cpu (CPUOnlyTensor)
    _ = tensor.torch_view()
    assert tensor.device == "cpu"


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_view_zero_copy(tensorclass):
    """Writes through the torch_view() tensor are visible in the underlying buffer."""
    data = np.zeros(4, dtype=bfloat16)
    tensor = tensorclass(data, dtype=bfloat16)
    view = tensor.torch_view()
    view[0] = 7.0
    view[1] = 8.0
    # The write must be visible when reading host memory back
    assert float(tensor.data[0]) == pytest.approx(7.0, abs=0.5)
    assert float(tensor.data[1]) == pytest.approx(8.0, abs=0.5)


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_view_2d_shape_preserved(tensorclass):
    """torch_view() preserves 2D shape for ND arrays."""
    data = np.ones((4, 8), dtype=bfloat16)
    tensor = tensorclass(data, dtype=bfloat16)
    view = tensor.torch_view()
    assert view.shape == (4, 8)
    assert view.dtype == torch.bfloat16


@pytest.mark.parametrize("shape", [(10,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_view_write_then_to_torch(shape, tensorclass):
    """Write via torch_view then read back via to_torch gives same values."""
    tensor = tensorclass(shape, dtype=bfloat16)
    # Write known values through torch_view
    view = tensor.torch_view()
    view.fill_(5.0)
    # to_torch syncs from device (if on NPU) then returns host view
    # Since torch_view marked device=cpu, to_torch just returns the host data
    result = tensor.to_torch()
    assert result.dtype == torch.bfloat16
    assert torch.all(result.float() == pytest.approx(5.0, abs=0.5))


@pytest.mark.parametrize("tensorclass", TENSOR_CLASSES)
def test_torch_view_native_dtype(tensorclass):
    """torch_view() works for native numpy dtypes too."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = tensorclass(data, dtype=np.float32)
    view = tensor.torch_view()
    assert view.dtype == torch.float32
    assert tensor.device == "cpu"


# ---------------------------------------------------------------------------
# _array_to_torch: zero-copy routing correctness
# ---------------------------------------------------------------------------


from aie.utils.hostruntime.tensor_class import _array_to_torch


@pytest.mark.parametrize("shape", [(8,), (4, 4), (2, 3, 4)])
def test_array_to_torch_bfloat16_zero_copy(shape):
    """_array_to_torch is zero-copy for bfloat16 at any shape."""
    arr = np.ones(shape, dtype=bfloat16)
    t = _array_to_torch(arr)
    assert t.dtype == torch.bfloat16
    assert t.shape == tuple(shape)
    # Mutation check
    arr.flat[0] = bfloat16(99.0)
    assert t.flat[0].item() == pytest.approx(99.0, abs=1.0)


@pytest.mark.parametrize("dtype,torch_dtype", [
    (np.float32, torch.float32),
    (np.float16, torch.float16),
    (np.int32, torch.int32),
])
def test_array_to_torch_native_zero_copy(dtype, torch_dtype):
    """_array_to_torch is zero-copy for native dtypes."""
    arr = np.array([1.0, 2.0, 3.0], dtype=dtype)
    t = _array_to_torch(arr)
    assert t.dtype == torch_dtype
    arr[0] = dtype(99)
    assert t[0].item() == pytest.approx(99, rel=0.01)
