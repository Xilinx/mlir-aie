# test_kernels_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compile every distinct kernel case's design on the host, as far as a host can.

``test_kernel_contracts`` lowers each design to MLIR; that does not run the
AIE passes, so a tile set that overflows core memory or a DMA pattern the
hardware cannot express is only caught at compile time. This sweep drives
``aiecc`` through buffer allocation, routing, the core compile and CDO
generation with the pinned Peano. Only the last step, packaging the xclbin,
needs XRT's ``xclbinutil``; on a host without XRT that failure is the
expected end and counts as a pass.

About six seconds per design, so it is marked ``extensive`` and excluded from
lit; the static kernel-check workflow runs it:

    KERNEL_TEST_DEVICE=npu2 pytest test/python/npu/test_kernels_compile.py -m extensive

Every design compiles into the test's own temporary directory, never into
the user's kernel cache.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from aie.iron import kernels
from aie.utils import kernel_harness as kh
from aie.utils.kernel_harness.cases import device_for, inputs_for
from kernel_cases import CASES

# Which generation to build for; the static workflow sets it from its matrix.
_DEVICE = os.environ.get("KERNEL_TEST_DEVICE", "npu2")


def _distinct_designs():
    seen: set[str] = set()
    for case in CASES:
        if not case.supported_on(_DEVICE):
            continue
        key = f"{case.factory}{sorted(case.kwargs.items())!r}{case.shape}{case.calls}"
        if key in seen:
            continue
        seen.add(key)
        yield pytest.param(case, id=case.name)


@pytest.mark.extensive
@pytest.mark.parametrize("case", list(_distinct_designs()))
def test_design_compiles_through_cdo(case, tmp_path):
    with device_for((_DEVICE,)):
        fn = case.fn()
        inputs = inputs_for(case, "random", np.random.default_rng(0))
        design = kh.design(
            getattr(kernels, case.factory),
            **case.harness_opts(),
            params=kh.param_values(fn, inputs),
            **case.kwargs,
        )
        try:
            design.compile(
                xclbin_path=str(tmp_path / "d.xclbin"),
                inst_path=str(tmp_path / "d.bin"),
            )
        except RuntimeError as ex:
            msg = str(ex)
            reached_packaging = "cdo_" in msg and "xclbinutil" in msg
            assert reached_packaging, msg[-3000:]
