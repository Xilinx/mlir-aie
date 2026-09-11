# test_extern_function_accessors.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for ExternalFunction's compile-recipe accessors (no NPU required).

Also covers ``cxx_core_compile_command``, the factored Peano command builder;
no compiler is invoked, only the argument list is inspected.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils.compile.utils import cxx_core_compile_command

# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


def test_name_is_the_symbol():
    assert Kernel("scale", "scale.o").name == "scale"


def test_file_backed_recipe_is_readable(tmp_path):
    src = tmp_path / "k.cc"
    src.write_text("void k(int*){}")
    ef = ExternalFunction(
        "k",
        source_file=str(src),
        arg_types=[np.ndarray[(16,), np.dtype[np.int32]]],
        include_dirs=["/inc/a"],
        compile_flags=["-DBIT_WIDTH=32"],
    )
    assert ef.source_file == str(src)
    assert ef.source_string is None
    assert ef.include_dirs == ["/inc/a"]
    assert ef.compile_flags == ["-DBIT_WIDTH=32"]
    assert ef.use_chess is False


def test_inline_recipe_is_readable():
    ef = ExternalFunction("k", source_string="void k(){}", arg_types=[])
    assert ef.source_file is None
    assert ef.source_string == "void k(){}"
    assert ef.include_dirs == [] and ef.compile_flags == []


def test_accessors_return_copies():
    ef = ExternalFunction(
        "k", source_string="void k(){}", include_dirs=["/i"], compile_flags=["-O3"]
    )
    ef.include_dirs.append("/mutated")
    ef.compile_flags.append("-mutated")
    assert ef.include_dirs == ["/i"]
    assert ef.compile_flags == ["-O3"]


# ---------------------------------------------------------------------------
# cxx_core_compile_command
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_peano_needed(monkeypatch):
    """Stub the Peano and header paths so no Peano install is needed.

    Building the command resolves both through ``aie.utils.config``; a pure
    host test must not depend on the llvm-aie wheel being present.
    """
    monkeypatch.setattr("aie.utils.config.peano_cxx_path", lambda: "/peano/bin/clang++")
    monkeypatch.setattr("aie.utils.config.cxx_header_path", lambda: "/mlir_aie/include")


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


def test_chess_path_needs_the_wrapper_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="xchesscc_wrapper"):
        _cmd(use_chess=True)


# ---------------------------------------------------------------------------
# Symbol prefixing: a parameterized kernel's whole object travels with it
# ---------------------------------------------------------------------------


def test_prefixing_renames_every_defined_symbol_and_is_idempotent(tmp_path):
    """A prefix covers the object's siblings, not just the declared symbol.

    ``mm.cc`` exports the ``zero_*`` that ``.zero`` binds beside ``matmul_*``;
    leaving those bare made two parameterizations of one kernel collide at
    link. Uses a stub object so the test needs no Peano.
    """
    from aie.utils.compile.utils import _prefix_symbols_in_object

    obj = tmp_path / "k.o"
    obj.write_bytes(b"")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--defined-only" in cmd:  # llvm-nm
            listed = "\n".join(
                f"00000000 T {s}" for s in getattr(fake_run, "symbols", [])
            )
            return SimpleNamespace(returncode=0, stdout=listed.encode(), stderr=b"")
        redefines = [a for a in cmd if a.startswith("--redefine-syms=")]
        assert redefines, cmd
        mapping = dict(
            line.split()
            for line in Path(redefines[0].split("=", 1)[1]).read_text().splitlines()
        )
        fake_run.symbols = [mapping.get(s, s) for s in fake_run.symbols]
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    fake_run.symbols = ["matmul_i16_i16", "matmul_scalar_i16_i16", "zero_i16"]
    with patch("subprocess.run", fake_run):
        renamed = _prefix_symbols_in_object(str(obj), "d00d")
        assert sorted(renamed) == [
            "matmul_i16_i16",
            "matmul_scalar_i16_i16",
            "zero_i16",
        ]
        assert fake_run.symbols == [
            "d00d_matmul_i16_i16",
            "d00d_matmul_scalar_i16_i16",
            "d00d_zero_i16",
        ]
        # Re-applying on a cache hit must not double-prefix.
        before = list(fake_run.symbols)
        assert _prefix_symbols_in_object(str(obj), "d00d") == []
        assert fake_run.symbols == before


def test_sibling_binds_another_symbol_from_the_same_object():
    """``.zero`` and friends follow the parent's prefix (ExternalFunction.sibling)."""
    tile = [np.ndarray[(16,), np.dtype[np.int32]]]
    prefixed = ExternalFunction(
        "matmul", source_string="void matmul(){}", arg_types=[], symbol_prefix="d00d"
    )
    sib = prefixed.sibling("zero_i16", tile)
    assert sib.name == "d00d_zero_i16"
    assert sib.object_file_name == prefixed.object_file_name
    assert sib.arg_types() == tile
    # An unprefixed kernel binds the bare name.
    plain = ExternalFunction("k", source_string="void k(){}", arg_types=[])
    assert plain.sibling("zero_i16", tile).name == "zero_i16"
