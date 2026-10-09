# test_extern_function_accessors.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
# REQUIRES: peano
"""Unit tests for ExternalFunction's compile-recipe accessors (no NPU required).

Also covers ``cxx_core_compile_command``, the factored Peano command builder,
and prefixing a Peano-compiled object's symbols.
"""

import numpy as np
import pytest
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils.compile import utils
from aie.utils.compile.utils import cxx_core_compile_command

# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


def test_name_is_the_symbol():
    assert Kernel("scale", "scale.o").name == "scale"


def test_file_backed_recipe_is_readable(tmp_path):
    src = tmp_path / "k.cc"
    src.write_text("void k(int*){}")
    # A host-absolute dir: on Windows a driveless "/inc/a" is drive-relative,
    # so the constructor would (correctly) anchor it to the current drive.
    inc = str(tmp_path / "inc")
    ef = ExternalFunction(
        "k",
        source_file=str(src),
        arg_types=[np.ndarray[(16,), np.dtype[np.int32]]],
        include_dirs=[inc],
        compile_flags=["-DBIT_WIDTH=32"],
    )
    assert ef.source_file == str(src)
    assert ef.source_string is None
    assert ef.include_dirs == [inc]
    assert ef.compile_flags == ["-DBIT_WIDTH=32"]
    assert ef.use_chess is False


def test_inline_recipe_is_readable():
    ef = ExternalFunction("k", source_string="void k(){}", arg_types=[])
    assert ef.source_file is None
    assert ef.source_string == "void k(){}"
    assert ef.include_dirs == [] and ef.compile_flags == []


def test_accessors_return_copies(tmp_path):
    inc = str(tmp_path / "i")
    ef = ExternalFunction(
        "k", source_string="void k(){}", include_dirs=[inc], compile_flags=["-O3"]
    )
    ef.include_dirs.append("/mutated")
    ef.compile_flags.append("-mutated")
    assert ef.include_dirs == [inc]
    assert ef.compile_flags == ["-O3"]


# ---------------------------------------------------------------------------
# cxx_core_compile_command
# ---------------------------------------------------------------------------


def _cmd(**kw):
    return cxx_core_compile_command("k.cc", "aie2p", "k.o", **kw)


def test_command_targets_the_requested_arch_and_emits_an_object():
    cmd = _cmd()
    assert "--target=aie2p-none-unknown-elf" in cmd
    assert "-c" in cmd and "-S" not in cmd
    assert cmd[cmd.index("-o") + 1] == "k.o"


def test_inline_emits_textual_ir_without_section_flags():
    cmd = _cmd(inline=True)
    assert "-S" in cmd and "-emit-llvm" in cmd
    # Object-only stack/section accounting flags must not leak into IR mode.
    assert "-fstack-size-section" not in cmd


def test_include_dirs_and_compile_args_are_appended_in_order():
    cmd = _cmd(include_dirs=["/a", "/b"], compile_args=["-DX=1", "-Rpass=pipeliner"])
    ia = cmd.index("/a")
    assert cmd[ia - 1] == "-I" and cmd[ia + 2] == "/b"
    # Caller flags come last, so they can add diagnostics without being
    # overridden by the library's own -W / -D set.
    assert cmd[-2:] == ["-DX=1", "-Rpass=pipeliner"]


def test_command_carries_the_library_defaults():
    cmd = _cmd()
    for flag in ("-std=c++20", "-O2", "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__"):
        assert flag in cmd


def test_chess_path_needs_the_wrapper_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="xchesscc_wrapper"):
        _cmd(use_chess=True)


# ---------------------------------------------------------------------------
# Symbol prefixing: a parameterized kernel's whole object travels with it
# ---------------------------------------------------------------------------


def test_prefixing_renames_every_defined_symbol(tmp_path):
    """A prefix covers the object's siblings, not just the declared symbol.

    Leaving additional entry points bare makes parameterizations collide at
    link.
    """
    src = tmp_path / "k.cc"
    src.write_text(
        'extern "C" void matmul_i16_i16() {}\n'
        'extern "C" void matmul_scalar_i16_i16() {}\n'
        'extern "C" void zero_i16() {}\n'
    )
    obj = tmp_path / "k.o"
    utils.compile_cxx_core_function(str(src), "aie2p", str(obj))

    utils.prefix_symbols_in_object(str(obj), "d00d_")
    assert sorted(utils._defined_symbols(str(obj))) == [
        "d00d_matmul_i16_i16",
        "d00d_matmul_scalar_i16_i16",
        "d00d_zero_i16",
    ]
    # Prefixing is literal; compile_external_kernel tracks cache state.
    utils.prefix_symbols_in_object(str(obj), "d00d_")
    assert sorted(utils._defined_symbols(str(obj))) == [
        "d00d_d00d_matmul_i16_i16",
        "d00d_d00d_matmul_scalar_i16_i16",
        "d00d_d00d_zero_i16",
    ]


def test_bind_other_symbols_from_the_same_object():
    """Explicit bindings follow the artifact's prefix."""
    tile = [np.ndarray[(16,), np.dtype[np.int32]]]
    prefixed = ExternalFunction(
        "matmul", source_string="void matmul(){}", arg_types=[], symbol_prefix="d00d"
    )
    first = prefixed.object_file.bind("first", tile)
    second = prefixed.object_file.bind("second", tile)
    assert first.name == "d00d_first"
    assert second.name == "d00d_second"
    assert first.object_file_name == prefixed.object_file_name
    assert first.arg_types() == tile
    # An unprefixed kernel binds the bare name.
    plain = ExternalFunction("k", source_string="void k(){}", arg_types=[])
    assert plain.object_file.bind("first", tile).name == "first"
