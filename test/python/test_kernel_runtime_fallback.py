# test_kernel_runtime_fallback.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""Kernels sized by an ``*_ELEMS`` define still build on their runtime count."""

import re

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels import DTYPES
from aie.utils.compile.utils import compile_cxx_core_function
from aie.utils.hostruntime import set_current_device

_SIZED = (
    "passthrough",
    "scale",
    "reduce_add",
    "reduce_min",
    "reduce_max",
    "bitwise_and",
    "bitwise_or",
    "threshold",
    "add_weighted",
)
_VARIANTS = [(name, kw) for name in _SIZED for kw in DTYPES.get(name, ({},))]


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
@pytest.mark.parametrize(
    "name,kwargs",
    _VARIANTS,
    ids=[
        f"{name}-{np.dtype(kw['dtype']).name}" if kw else name for name, kw in _VARIANTS
    ],
)
def test_kernel_builds_without_its_size_define(tmp_path, arch, name, kwargs):
    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    fn = getattr(kernels, name)(**kwargs)
    flags = [f for f in fn.compile_flags if not re.fullmatch(r"-D\w+_ELEMS=\d+", f)]
    assert len(flags) == len(fn.compile_flags) - 1
    compile_cxx_core_function(
        fn.source_file,
        arch,
        str(tmp_path / "kernel.o"),
        include_dirs=fn.include_dirs,
        compile_args=flags,
    )
