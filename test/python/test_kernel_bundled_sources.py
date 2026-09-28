# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""A bundled source is compiled into the kernel's object, prefixed with it (no NPU)."""

from aie.iron.kernel import ExternalFunction
from aie.utils.compile import utils


def test_bundled_definitions_land_in_the_kernel_object_under_its_prefix(tmp_path):
    table = tmp_path / "lib" / "table.cpp"
    table.parent.mkdir()
    table.write_text('extern "C" { int table[4] = {1, 2, 3, 4}; }')
    kernel = tmp_path / "kernel.cc"
    kernel.write_text(
        'extern "C" { extern int table[4]; int kernel(int i) { return table[i]; } }'
    )
    fn = ExternalFunction(
        "kernel",
        source_file=str(kernel),
        arg_types=[],
        bundled_sources=[str(table)],
        digest_prefix=True,
    )
    out = tmp_path / "out"
    out.mkdir()
    utils.compile_external_kernels([fn], out, "aie2p")
    symbols = utils._defined_symbols(str(out / fn.object_file_name))
    prefix = fn._symbol_prefix
    assert sorted(symbols) == [f"{prefix}_kernel", f"{prefix}_table"]
