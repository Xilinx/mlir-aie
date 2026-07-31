# test_npu_tensor.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""The host-tensor base class and the name it used to have."""

import numpy as np

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
