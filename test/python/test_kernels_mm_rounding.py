# test_kernels_mm_rounding.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""round_conv_even plumbing for the aie.iron.kernels mm factory.

mm_aie2.h selects conv_even rounding itself under -DROUND_CONV_EVEN;
mm_aie2p.h always does, so the flag is not passed there.

Sibling files:
  test_kernels_specs.py        — spec-table-driven coverage of every factory
  test_kernels_chess.py        — use_chess + emulated bf16 plumbing
  test_kernels_memoization.py  — memoization, independent zero, auto-prefix-on-collision

The npu2_device fixture comes from conftest.py at this directory level.
"""

import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1
from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16


@pytest.fixture
def npu1_device():
    previous = get_current_device(probe_runtime=False)
    set_current_device(NPU1Col1())
    try:
        yield
    finally:
        set_current_device(previous)


def _bf16_mm(**kwargs):
    return kernels.mm(
        dim_m=64,
        dim_k=64,
        dim_n=32,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        **kwargs,
    )


def test_kernels_mm_round_conv_even_carries_macro_on_aie2(npu1_device):
    """On aie2 the kernel rounds itself, so the contract drops its setup."""
    ef = _bf16_mm(round_conv_even=True)
    assert "-DROUND_CONV_EVEN" in ef._compile_flags
    assert ef.contract.setup is None
    assert _bf16_mm().contract.setup is not None
    assert ef.object_file_name != _bf16_mm().object_file_name


def test_kernels_mm_round_conv_even_ignored_on_aie2p(npu2_device):
    """mm_aie2p.h always rounds conv_even: no macro, one shared object."""
    ef = _bf16_mm(round_conv_even=True)
    assert "-DROUND_CONV_EVEN" not in ef._compile_flags
    assert ef == _bf16_mm()
