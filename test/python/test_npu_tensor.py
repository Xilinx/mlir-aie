# test_npu_tensor.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""The host-tensor base class and the name it used to have."""

import numpy as np
import pytest

from aie.utils.hostruntime.tensor_class import CPUOnlyTensor, NpuTensor, Tensor


def test_tensor_is_an_alias_of_npu_tensor():
    """``Tensor`` must stay usable, and stay the *same* class.

    An alias that is a separate class would silently break ``isinstance``
    checks in callers that still import the old name.
    """
    assert Tensor is NpuTensor


def test_old_name_still_works_for_isinstance_and_subclassing():
    tensor = CPUOnlyTensor((4,), dtype=np.int32)
    assert isinstance(tensor, Tensor)
    assert isinstance(tensor, NpuTensor)
    assert issubclass(CPUOnlyTensor, Tensor)


def test_backends_share_the_base():
    """The contract lives in one place, so every backend inherits it."""
    assert issubclass(CPUOnlyTensor, NpuTensor)
    for name in ("subview", "to", "fill_"):
        assert hasattr(NpuTensor, name)


# numpy_view(): the write path, without a sync the caller is about to discard


def test_numpy_view_writes_through_to_the_buffer():
    """What numpy_view() hands back is the buffer, not a copy of it."""
    tensor = CPUOnlyTensor((4,), dtype=np.float32)
    view = tensor.numpy_view()
    view[:] = [1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(tensor.numpy(), [1.0, 2.0, 3.0, 4.0])


def test_numpy_view_marks_the_buffer_cpu_resident():
    """The point of the view: a later to("npu") must actually push the write.

    numpy() syncs from the device first, which is wasted work when the caller
    is about to overwrite everything; numpy_view() skips that and instead
    marks the host copy dirty, exactly as torch_view() does.
    """
    tensor = CPUOnlyTensor((4,), dtype=np.int32)
    tensor.numpy_view()
    assert tensor.device == "cpu"


def test_numpy_view_and_torch_view_share_one_buffer():
    """The two views are peers over the same memory, not separate copies."""
    torch = pytest.importorskip("torch")
    tensor = CPUOnlyTensor((3,), dtype=np.float32)
    tensor.numpy_view()[:] = [5.0, 6.0, 7.0]
    assert torch.equal(tensor.torch_view(), torch.tensor([5.0, 6.0, 7.0]))
