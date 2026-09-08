# benchmarks/kernels/test_designs_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compile every distinct registry design on the host, as far as a host can.

``test_kernel_contracts`` lowers each design to MLIR; that does not run the
AIE passes, so a tile set that overflows core memory or a DMA pattern the
hardware cannot express is only caught at compile time. This test drives
``aiecc`` through buffer allocation, routing, the core compile and CDO
generation with the pinned Peano. Only the last step, packaging the xclbin,
needs XRT's ``xclbinutil``; on a host without XRT that step's failure is the
expected end and counts as a pass.

About six seconds per design; runs in the static workflow, not in lit.
``KERNEL_TEST_ARCH`` (``aie2p`` by default, ``aie2``) picks the device the
designs are generated for; cases bound to the other arch are skipped.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_kernels = pytest.importorskip("aie.iron.kernels", reason="requires the aie package")
if not getattr(_kernels, "__file__", None):
    pytest.skip("aie.iron.kernels is a stub", allow_module_level=True)

from aie.utils import kernel_harness as kh  # noqa: E402

from . import registry  # noqa: E402
from .registry import _device_for  # noqa: E402

# Which target to build for; the static workflow sets it from its matrix.
_ARCH = os.environ.get("KERNEL_TEST_ARCH", "aie2p")
_SEEN: set[str] = set()


def _cases():
    for case in registry.CASES:
        if case.arch and case.arch != _ARCH:
            continue
        key = f"{case.factory}{sorted(case.kwargs.items())!r}{case.shape}{case.calls}"
        if key in _SEEN:
            continue
        _SEEN.add(key)
        yield pytest.param(case, id=case.name)


@pytest.mark.parametrize("case", list(_cases()))
def test_design_compiles_through_cdo(case, tmp_path):
    with _device_for(_ARCH):
        fn = case.fn()
        inputs = registry.inputs_for(case, "random", np.random.default_rng(0))
        design = kh.design(
            getattr(_kernels, case.factory),
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
