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

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.utils import set_current_device
from aie.utils.compile.jit._object_cache import KernelObjectCache
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


def _table_kernel(words):
    """Return a kernel whose only static data is a ``words``-int table."""
    return ExternalFunction(
        f"table_{words}",
        source_string=f"""
static int table[{words}];
extern "C" void table_{words}(int *a, int n) {{
  for (int i = 0; i < n; i++)
    table[i % {words}] += a[i % 64];
  a[0] = table[n % {words}];
}}
""",
        arg_types=[np.ndarray[(64,), np.dtype[np.int32]], np.int32],
    )


def test_measure_reads_a_kernels_static_data(tmp_path):
    """Doubling a 4 KiB table adds exactly its 4096 bytes to the data.

    The stack is unchanged.
    """
    small, large = (
        _table_kernel(words).measure(
            "aie2p", KernelObjectCache(tmp_path / str(words), 60)
        )
        for words in (1024, 2048)
    )
    assert small.data >= 4096
    assert large.data - small.data == 4096
    assert large.stack == small.stack


def test_frame_bytes_is_data_and_stack_or_its_declared_override():
    kernel = _table_kernel(1024)
    found = kernel.measure("aie2p")
    assert kernel.frame_bytes("aie2p") == found.data + (found.stack or 0)
    declared = ExternalFunction(
        "table_1024",
        source_string=kernel.source_string,
        arg_types=kernel.arg_types(),
        stack_size_override=96,
    )
    assert declared.frame_bytes("aie2p") == found.data + 96


def test_measure_reads_each_entry_of_one_object(tmp_path):
    """Two symbols of one object: each entry's own stack, the object's data."""
    small = ExternalFunction(
        "small_frame",
        source_string="""
static int table[256];
extern "C" void small_frame(int *a, int n) {
  for (int i = 0; i < n; i++)
    table[i % 256] += a[i % 64];
}
extern "C" void large_frame(int *a, int n) {
  volatile int scratch[512];
  for (int i = 0; i < 512; i++)
    scratch[i] = a[i % 64] + n;
  a[0] = scratch[n % 512] + table[n % 256];
}
""",
        arg_types=[np.ndarray[(64,), np.dtype[np.int32]], np.int32],
    )
    large = small.object_file.bind("large_frame", small.arg_types())
    cache = KernelObjectCache(tmp_path, 60)
    own = small.measure("aie2p", cache)
    other = small.measure("aie2p", cache, entry=large.name)
    assert other.stack - (own.stack or 0) >= 2048
    assert other.data == own.data >= 1024
    assert small.frame_bytes("aie2p", entry=large.name) == other.data + other.stack


def test_measure_refuses_an_arch_the_recipe_was_not_built_for():
    """A factory's recipe is its bound device's; measuring it for another
    arch is refused before Peano compiles the wrong source.
    """
    with set_current_device(NPU2()):
        zero = kernels.zero(1024, np.int32)
        fused = kernels.fused_mm(
            dim_m=64, dim_k=256, dim_n=64, band_m=16, chunk_k=128, out_chunk=512
        )
    for kernel in (zero, fused):
        assert kernel.built_for_arch == "aie2p"
        with pytest.raises(ValueError, match="was built for aie2p"):
            kernel.measure("aie2")
        with pytest.raises(ValueError, match="was built for aie2p"):
            kernel.frame_bytes("aie2")
    assert zero.measure("aie2p") is not None
