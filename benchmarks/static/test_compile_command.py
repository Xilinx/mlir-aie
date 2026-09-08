# benchmarks/static/test_compile_command.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Host-only tests of how the static checker builds its compile command.

The factories resolve kernel sources from the installed wheel; on a pull
request the checker must compile the checkout instead. ``relocate`` is the
whole of that mechanism, so it is pinned here for every place a wheel path
can appear: a ``source_file``, an include directory and the aie2 LUT
factories' ``source_string``.
"""

from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("aie.iron.kernels", reason="run.py imports the aie package")

from . import run as run_mod  # noqa: E402

WHEEL = os.path.join(os.sep, "site", "mlir_aie")
CHECKOUT = os.path.join(os.sep, "work", "mlir-aie")


def _ext_fn(**over):
    base = dict(
        name="k",
        use_chess=False,
        include_dirs=[
            os.path.join(WHEEL, "include"),
            os.path.join(WHEEL, "include", "aie_kernels", "aie2"),
        ],
        source_file=os.path.join(WHEEL, "include", "aie_kernels", "aie2", "k.cc"),
        source_string=None,
        compile_flags=["-DX=1"],
    )
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_cmd(
        source_path, target, output_path, include_dirs=None, compile_args=None
    ):
        seen.update(src=source_path, inc=list(include_dirs), args=list(compile_args))
        return ["clang", source_path]

    monkeypatch.setattr(run_mod, "cxx_core_compile_command", fake_cmd)
    from aie.utils import config

    monkeypatch.setattr(config, "root_path", lambda: WHEEL)
    return seen


def test_relocate_rewrites_only_whole_directory_prefixes():
    text = (
        f"{WHEEL}/include/aie_kernels/aie2/k.cc "
        f"{WHEEL}/aie_runtime_lib/AIE2/lut_based_ops.cpp "
        f"{WHEEL}/include/aie_kernels_other/x.h {WHEEL}/include/aie_api/aie.hpp"
    )
    out = run_mod.relocate(text, WHEEL, CHECKOUT)
    assert out == (
        f"{CHECKOUT}/aie_kernels/aie2/k.cc "
        f"{CHECKOUT}/aie_runtime_lib/AIE2/lut_based_ops.cpp "
        f"{WHEEL}/include/aie_kernels_other/x.h {WHEEL}/include/aie_api/aie.hpp"
    )


def test_source_file_and_include_dirs_follow_source_root(captured, tmp_path):
    run_mod.compile_command(_ext_fn(), "aie2", tmp_path, source_root=CHECKOUT)
    assert captured["src"] == os.path.join(CHECKOUT, "aie_kernels", "aie2", "k.cc")
    # The wheel's headers stay (aie_api lives only there); the kernel dir moves.
    assert captured["inc"][0] == os.path.join(WHEEL, "include")
    assert os.path.join(CHECKOUT, "aie_kernels", "aie2") in captured["inc"]
    assert not any("site" in d and "aie_kernels" in d for d in captured["inc"])
    assert "-DX=1" in captured["args"]


def test_inline_lut_source_string_follows_source_root(captured, tmp_path):
    lut = os.path.join(WHEEL, "aie_runtime_lib", "AIE2")
    ef = _ext_fn(
        source_file=None,
        source_string=f'#include "{WHEEL}/include/aie_kernels/aie2/gelu.cc"\n'
        f'#include "{lut}/lut_based_ops.cpp"\n',
        include_dirs=[os.path.join(WHEEL, "include"), lut],
    )
    run_mod.compile_command(ef, "aie2", tmp_path, source_root=CHECKOUT)
    written = (tmp_path / "k.cc").read_text()
    assert f'"{CHECKOUT}/aie_kernels/aie2/gelu.cc"' in written
    assert f'"{CHECKOUT}/aie_runtime_lib/AIE2/lut_based_ops.cpp"' in written
    assert os.path.join(CHECKOUT, "aie_runtime_lib", "AIE2") in captured["inc"]


def test_without_source_root_the_wheel_is_compiled(captured, tmp_path):
    run_mod.compile_command(_ext_fn(), "aie2", tmp_path)
    assert captured["src"] == os.path.join(
        WHEEL, "include", "aie_kernels", "aie2", "k.cc"
    )


def test_source_root_must_hold_aie_kernels(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_mod.main(["--out", "x.json", "--source-root", str(tmp_path)])
    assert "no aie_kernels/" in capsys.readouterr().err
