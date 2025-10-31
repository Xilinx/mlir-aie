# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import os
import tempfile

from aie.iron.compile import compile_cxx_core_function
from aie.iron.compile import merge_object_files

SOURCE_STRING1 = """
extern "C" {
void add_one(int* input, int* output, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output[i] = input[i] + 1;
    }
}
}"""

SOURCE_STRING2 = """
extern "C" {
void add_two(int* input, int* output, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output[i] = input[i] + 2;
    }
}
}"""


def test_compile():
    """Test compilation of a C++ source file to an object file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cpp", delete_on_close=False, delete=True
    ) as source_file, tempfile.NamedTemporaryFile(
        mode="r", suffix=".o", delete_on_close=True
    ) as output_file:
        source_file.write(SOURCE_STRING1)
        source_file.close()
        assert os.path.getsize(source_file.name) > 0

        assert os.path.getsize(output_file.name) == 0
        compile_cxx_core_function(
            source_path=source_file.name,
            target_arch="aie2",
            output_path=output_file.name,
            compile_args=["-DTEST"],
        )
        assert os.path.getsize(output_file.name) > 0


def test_compile_and_link():
    """Test compilation of two C++ source files and link them."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cpp", delete_on_close=False, delete=True
    ) as source_file1, tempfile.NamedTemporaryFile(
        mode="w", suffix=".cpp", delete_on_close=False, delete=True
    ) as source_file2, tempfile.NamedTemporaryFile(
        mode="r", suffix=".o", delete_on_close=True
    ) as output_file1, tempfile.NamedTemporaryFile(
        mode="r", suffix=".o", delete_on_close=True
    ) as output_file2, tempfile.NamedTemporaryFile(
        mode="r", suffix=".o", delete_on_close=True
    ) as combined_output_file:

        source_file1.write(SOURCE_STRING1)
        source_file1.close()
        assert os.path.getsize(source_file1.name) > 0

        source_file2.write(SOURCE_STRING2)
        source_file2.close()
        assert os.path.getsize(source_file2.name) > 0

        assert os.path.getsize(output_file1.name) == 0
        compile_cxx_core_function(
            source_path=source_file1.name,
            target_arch="aie2",
            output_path=output_file1.name,
        )
        assert os.path.getsize(output_file1.name) > 0

        assert os.path.getsize(output_file2.name) == 0
        compile_cxx_core_function(
            source_path=source_file2.name,
            target_arch="aie2",
            output_path=output_file2.name,
        )
        assert os.path.getsize(output_file2.name) > 0

        assert os.path.getsize(combined_output_file.name) == 0
        merge_object_files(
            object_paths=[output_file1.name, output_file2.name],
            output_path=combined_output_file.name,
        )
        assert os.path.getsize(combined_output_file.name) > 0
