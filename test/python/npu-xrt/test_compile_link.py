# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import os
import tempfile

from aie.utils.compile import compile_cxx_core_function

SOURCE_STRING1 = """
extern "C" {
void add_one(int* input, int* output, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output[i] = input[i] + 1;
    }
}
}"""


def test_compile():
    """Test compilation of a C++ source file to an object file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.cpp")
        output_path = os.path.join(tmpdir, "output.o")

        with open(source_path, "w") as f:
            f.write(SOURCE_STRING1)

        assert os.path.getsize(source_path) > 0
        assert not os.path.exists(output_path)

        compile_cxx_core_function(
            source_path=source_path,
            target_arch="aie2",
            output_path=output_path,
            compile_args=["-DTEST"],
        )
        assert os.path.getsize(output_path) > 0
