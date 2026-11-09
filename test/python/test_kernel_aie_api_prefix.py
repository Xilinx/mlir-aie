# test_kernel_aie_api_prefix.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""Injecting aie_api ahead of a library kernel preserves its compiled object."""

import os

import pytest

import aie.iron.kernels as kernels
from aie.utils.compile.utils import AIE_API_PREFIX_FLAGS, compile_cxx_core_function


# Library kernels with macros, system headers or kernel headers ahead of their
# own aie_api include, which the prefix moves the header in front of.
@pytest.mark.parametrize(
    "factory",
    [kernels.axpy, kernels.argmax, kernels.limbs_f32, kernels.threshold],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_library_kernel_reading_aie_api_first_builds_the_same_object(
    tmp_path, monkeypatch, factory, arch
):
    monkeypatch.setenv("AIE_KERNEL_PCH", "0")
    ext = factory()
    flags = ext.compile_flags
    prefix = list(AIE_API_PREFIX_FLAGS)
    at = next(i for i in range(len(flags)) if flags[i : i + len(prefix)] == prefix)
    include_dirs = [*ext.include_dirs, os.path.dirname(ext.source_file)]
    objects = []
    for name, args in (
        ("prefixed", flags),
        ("own_order", flags[:at] + flags[at + len(prefix) :]),
    ):
        (tmp_path / name).mkdir()
        out = tmp_path / name / "kernel.o"
        compile_cxx_core_function(
            ext.source_file,
            arch,
            str(out),
            include_dirs=include_dirs,
            compile_args=args,
        )
        objects.append(out.read_bytes())
    assert objects[0] == objects[1]
