# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import inspect
import os
import tempfile

import pytest

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


def test_compile_signature_preserves_positional_parameters():
    """New inline parameters must not shift the established positional API."""
    assert list(inspect.signature(compile_cxx_core_function).parameters) == [
        "source_path",
        "target_arch",
        "output_path",
        "include_dirs",
        "compile_args",
        "cwd",
        "use_chess",
        "inline",
        "symbol_name",
    ]


@pytest.mark.parametrize("suffix", [".ll", ".bc"])
def test_compile_inline_ir(suffix):
    """Inline compilation emits textual .ll or real binary .bc as requested."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.cpp")
        output_path = os.path.join(tmpdir, f"output{suffix}")

        with open(source_path, "w") as f:
            f.write(SOURCE_STRING1)

        compile_cxx_core_function(
            source_path=source_path,
            target_arch="aie2",
            output_path=output_path,
            inline=True,
            symbol_name="add_one",
        )

        with open(output_path, "rb") as f:
            contents = f.read()
        if suffix == ".ll":
            assert b"define linkonce_odr" in contents
            assert b"alwaysinline" in contents
        else:
            assert contents.startswith(b"BC\xc0\xde")
