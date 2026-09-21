# test_stack_sizes_kernel_compile.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
# REQUIRES: peano

"""compile_cxx_core_function emits a `.stack_sizes` section for an object
compile, which matches the flag that the core-object `llc` invocation of aiecc
passes (tools/aiecc/aiecc.cpp). That section carries the stack accounting of a
kernel object, where the large frames of a design sit. These tests run the
Peano compile alone and need no NPU.
"""

import os
import subprocess
import tempfile

from aie.utils.compile.utils import compile_cxx_core_function

_KERNEL_SOURCE = """
extern "C" void add_one(int *a, int *b, int n) {
  for (int i = 0; i < n; i++)
    b[i] = a[i] + 1;
}
"""


def _has_stack_sizes_section(obj_path):
    out = subprocess.run(
        ["llvm-readelf", "-S", obj_path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return ".stack_sizes" in out


def test_compile_cxx_core_function_emits_stack_sizes_section():
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = os.path.join(tmp_dir, "add_one.cc")
        with open(src, "w") as f:
            f.write(_KERNEL_SOURCE)
        obj = os.path.join(tmp_dir, "add_one.o")
        compile_cxx_core_function(src, "aie2p", obj)
        assert os.path.exists(obj)
        assert _has_stack_sizes_section(obj)


def test_compile_cxx_core_function_inline_ir_has_no_stack_sizes_section():
    """The inline path emits textual LLVM IR ahead of codegen, so it fixes no
    frame layout and writes no `.stack_sizes` section."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = os.path.join(tmp_dir, "add_one.cc")
        with open(src, "w") as f:
            f.write(_KERNEL_SOURCE)
        ir = os.path.join(tmp_dir, "add_one.ll")
        compile_cxx_core_function(src, "aie2p", ir, inline=True, symbol_name="add_one")
        assert os.path.exists(ir)
        with open(ir) as f:
            assert "stack_sizes" not in f.read()
