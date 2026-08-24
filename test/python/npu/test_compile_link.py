# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import inspect
import os
import subprocess
import tempfile

import pytest

import aie.utils.config as config
from aie.utils.compile import compile_cxx_core_function, prefix_symbols_in_object

SOURCE_STRING1 = """
extern "C" {
void add_one(int* input, int* output, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output[i] = input[i] + 1;
    }
}
}"""

SOURCE_STRING_MULTI = """
extern "C" {
void add_one(int* input, int* output, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output[i] = input[i] + 1;
    }
}
void add_two(int* input, int* output, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output[i] = input[i] + 2;
    }
}
}"""


def _defined_extern_symbols(object_path):
    result = subprocess.run(
        [config.nm_path(), "--defined-only", "--extern-only", object_path],
        capture_output=True,
        check=True,
    )
    return {line.split()[-1] for line in result.stdout.decode().splitlines() if line}


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


def test_prefix_symbols_in_object():
    """Every defined, external symbol is renamed; none are missed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.cpp")
        output_path = os.path.join(tmpdir, "output.o")

        with open(source_path, "w") as f:
            f.write(SOURCE_STRING_MULTI)

        compile_cxx_core_function(
            source_path=source_path,
            target_arch="aie2",
            output_path=output_path,
        )

        original_symbols = _defined_extern_symbols(output_path)
        assert {"add_one", "add_two"} <= original_symbols

        prefix_symbols_in_object(output_path, "op0_")

        renamed_symbols = _defined_extern_symbols(output_path)
        assert "add_one" not in renamed_symbols
        assert "add_two" not in renamed_symbols
        assert "op0_add_one" in renamed_symbols
        assert "op0_add_two" in renamed_symbols
        # No symbols lost or spuriously added in the rename.
        assert renamed_symbols == {f"op0_{s}" for s in original_symbols}


def test_prefix_symbols_in_object_is_idempotent():
    """Calling prefix_symbols_in_object again with the same prefix must not
    re-prefix an already-prefixed symbol.

    This matters for compile_external_kernel's symbol_prefix handling, which
    re-applies the prefix on every cache hit (a disk-cached object may have
    already been prefixed by an earlier process run).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.cpp")
        output_path = os.path.join(tmpdir, "output.o")

        with open(source_path, "w") as f:
            f.write(SOURCE_STRING_MULTI)

        compile_cxx_core_function(
            source_path=source_path,
            target_arch="aie2",
            output_path=output_path,
        )

        prefix_symbols_in_object(output_path, "op0_")
        once = _defined_extern_symbols(output_path)

        prefix_symbols_in_object(output_path, "op0_")
        twice = _defined_extern_symbols(output_path)

        assert twice == once
        assert "op0_op0_add_one" not in twice


def test_prefix_symbols_in_object_raises_on_nm_failure():
    """A real llvm-nm failure (invalid input) must raise, not silently no-op.

    No object file is compiled here: pointing nm at a nonexistent path is a
    genuine, unmocked way to make the real llvm-nm binary exit nonzero, which
    is exactly the case the `&&`-chaining in the original (IRON) version
    guarded against -- an ignored nm failure would otherwise produce an empty
    rename map and turn this into a silent no-op.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        bogus_object = os.path.join(tmpdir, "does-not-exist.o")
        with pytest.raises(RuntimeError, match="Symbol listing failed"):
            prefix_symbols_in_object(bogus_object, "op0_")
