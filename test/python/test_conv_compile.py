# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""Convolution dimension specialization and raw-ABI compilation (no NPU)."""

from pathlib import Path
import os
import re
import subprocess

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import get_current_device
from aie.utils.compile.utils import cxx_core_compile_command
from aie.utils.hostruntime import set_current_device

_ROOT = Path(__file__).resolve().parents[2]
_CASES = [
    ("aie2", "conv2dk1", np.int8),
    ("aie2", "conv2dk1", np.uint8),
    ("aie2", "conv2dk1_skip", np.int8),
    ("aie2", "conv2dk1_skip", np.uint8),
    ("aie2", "conv2dk3", np.int8),
    ("aie2", "conv2dk3", np.uint8),
    ("aie2", "conv2dk1_i8", None),
    ("aie2p", "conv2dk1_i8", None),
    ("aie2", "conv2dk14", None),
    ("aie2p", "conv2dk14", None),
]


def _factory(arch, name, dtype, small):
    previous = get_current_device(probe_runtime=False)
    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    try:
        kwargs = dict(act_dtype=dtype) if dtype is not None else {}
        if small:
            kwargs.update(input_channels=16 if "skip" in name else 8, output_channels=8)
        if name == "conv2dk3":
            # The bound is this call's OC, not the shared weights' OC.
            kwargs["weight_output_channels"] = 128
        return getattr(kernels, name)(**kwargs)
    finally:
        set_current_device(previous)


@pytest.mark.parametrize("arch,name,dtype", _CASES)
def test_factory_dimensions_and_identity(arch, name, dtype):
    default = _factory(arch, name, dtype, False)
    small = _factory(arch, name, dtype, True)
    ic = 16 if "skip" in name else 8
    assert f"-DCONV_INPUT_CHANNELS={ic}" in small.compile_flags
    assert "-DCONV_OUTPUT_CHANNELS=8" in small.compile_flags
    width = 224 if name == "conv2dk14" else 32
    assert f"-DCONV_INPUT_WIDTH={width}" in small.compile_flags
    if name == "conv2dk3":
        assert "-DCONV_KERNEL_WIDTH=3" in small.compile_flags
        assert "-DCONV_KERNEL_HEIGHT=3" in small.compile_flags
    if name == "conv2dk14":
        assert "-DCONV_KERNEL_WIDTH=14" in small.compile_flags
    assert default is not small
    assert default.object_file_name != small.object_file_name
    assert len(default.arg_types()) == len(small.arg_types())


@pytest.mark.parametrize("arch,name,dtype", _CASES)
@pytest.mark.parametrize("mode", ["raw", "default", "small"])
@pytest.mark.parametrize("scalar", [False, True], ids=["vector", "scalar"])
def test_convolution_compile_and_bounds(arch, name, dtype, mode, scalar):
    fn = _factory(arch, name, dtype, mode == "small")
    flags = list(fn.compile_flags)
    if mode == "raw":
        flags = [f for f in flags if not f.startswith("-DCONV_")]
    if scalar:
        flags.append("-DSCALAR")
    # Keep compiler output in memory, including in environments without a
    # writable toolchain directory.
    flags += ["-MF", os.devnull]
    command = cxx_core_compile_command(
        fn.source_file,
        arch,
        "-",
        include_dirs=[str(_ROOT / "third_party/aie_api/include")],
        compile_args=[*flags, "-fno-discard-value-names"],
        inline=True,
    )
    ir = subprocess.run(command, capture_output=True, text=True)
    assert ir.returncode == 0, ir.stderr
    bodies = re.findall(r"^define [^\n]+\{\n(.*?)^}", ir.stdout, re.M | re.S)
    assert bodies
    if mode != "raw":
        # Check the optimized helpers as well as the public wrappers: accepting
        # a -D flag is insufficient if a loop still consumes its runtime arg.
        for body in bodies:
            assert not re.search(
                r"%(?:runtime_)?(?:input_width|input_channels|output_channels|"
                r"kernel_width|kernel_height)\b",
                body,
            )
    else:
        assert any("%runtime_output_channels" in body for body in bodies)
    assert "llvm.loop.itercount.range" not in ir.stdout
    # Exercise instruction selection and scheduling, not just the IR frontend.
    command = cxx_core_compile_command(
        fn.source_file,
        arch,
        "-",
        include_dirs=[str(_ROOT / "third_party/aie_api/include")],
        compile_args=flags,
    )
    obj = subprocess.run(command, capture_output=True)
    assert obj.returncode == 0, obj.stderr.decode()
    assert obj.stdout.startswith(b"\x7fELF")
