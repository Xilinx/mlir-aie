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
from aie.iron import kernels
from aie.iron.kernels import _common, linalg
from aie.utils.compile import compile_cxx_core_function, prefix_symbols_in_object
from aie.utils.compile.utils import compile_external_kernel

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


def test_prefix_symbols_in_object_renames_symbols_even_if_already_prefixed():
    """The rename is literal: every defined external symbol gets the prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.cpp")
        output_path = os.path.join(tmpdir, "output.o")

        with open(source_path, "w") as f:
            f.write("""extern "C" {
                void op0_helper() {}
                void add_one() { op0_helper(); }
            }""")

        compile_cxx_core_function(
            source_path=source_path,
            target_arch="aie2",
            output_path=output_path,
        )

        prefix_symbols_in_object(output_path, "op0_")
        renamed = _defined_extern_symbols(output_path)

        assert "add_one" not in renamed
        assert "op0_helper" not in renamed
        assert "op0_add_one" in renamed
        assert "op0_op0_helper" in renamed


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
@pytest.mark.parametrize("input_dtype,output_dtype", linalg._MM_COMBOS)
def test_mm_object_exports_matmul_and_zero(
    tmp_path, monkeypatch, arch, input_dtype, output_dtype
):
    monkeypatch.setattr(_common, "_detect_arch", lambda: arch)
    monkeypatch.setattr(linalg, "_detect_arch", lambda: arch)
    matmul = kernels.mm(input_dtype=input_dtype, output_dtype=output_dtype)
    compile_external_kernel(matmul, tmp_path, arch)

    symbols = _defined_extern_symbols(str(tmp_path / matmul.object_file_name))
    suffix, _ = linalg._MM_COMBOS[(input_dtype, output_dtype)]
    zero_suffix = linalg._ZERO_SUFFIX[output_dtype]
    assert {
        matmul._name,
        matmul.also.zero._name,
        matmul.object_file.resolve_symbol(f"matmul_scalar_{suffix}"),
        matmul.object_file.resolve_symbol(f"zero_scalar_{zero_suffix}"),
    } <= symbols


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
