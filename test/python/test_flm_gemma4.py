# test_flm_gemma4.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The flm_gemma4 factories refuse the builds their kernels do not support."""

import dataclasses

import pytest
from aie.iron import kernels


@pytest.fixture(autouse=True)
def _aie2p_device(npu2_device):
    # The factories build only for AIE2P.
    yield


def test_the_lm_head_takes_whole_k_tiles():
    """A partial last block would read past the token and its sums."""
    with pytest.raises(ValueError, match="multiple of k_tile"):
        kernels.flm_gemma4_q4nx_lm_head(dim=272)


def test_decode_kernels_build_only_the_gemma4_presets():
    custom = dataclasses.replace(kernels.FLM_GEMMA4_E2B_DECODE, num_attn_heads=16)
    with pytest.raises(ValueError, match="custom geometry"):
        kernels.flm_gemma4_decode_glu(geometry=custom)
    renamed = dataclasses.replace(kernels.FLM_GEMMA4_E4B_DECODE, name="mine")
    kernels.flm_gemma4_decode_glu(geometry=renamed)
