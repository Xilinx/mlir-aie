# test_kernel_sources_override.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""``MLIR_AIE_KERNEL_SOURCES`` moves every kernel header, not just the source.

The override is how a checkout's kernels are compiled against an installed
wheel, and how a before/after measurement compiles two trees. An include
directory still pointing into the install mixes the two: a header missing
beside its includer falls through to the installed copy, and the build
silently measures a hybrid.
"""

import os
import shutil
import subprocess
from pathlib import Path

import aie.iron as iron
import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import config
from aie.utils.compile import remarks

_LUT_VARIANTS = ("silu", "swiglu", "tanh", "sigmoid")


def _peano_available() -> bool:
    try:
        return os.path.isfile(config.peano_cxx_path())
    except RuntimeError:
        return False


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Copy the kernel sources and select the copy with the override."""
    shutil.copytree(config.aie_kernels_dir(), tmp_path / "aie_kernels")
    shutil.copytree(config.aie_runtime_lib_dir(), tmp_path / "aie_runtime_lib")
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(tmp_path))
    return tmp_path


@pytest.fixture
def device(request):
    previous = iron.get_current_device(probe_runtime=False)
    iron.set_current_device(request.param())
    yield
    iron.set_current_device(previous)


def _builds():
    yield from remarks.kernel_builds()
    for name in _LUT_VARIANTS:
        try:
            yield f"{name}/use_lut=True", getattr(kernels, name)(use_lut=True)
        except NotImplementedError:
            continue


def _paths(ef) -> list[str]:
    paths = list(ef.include_dirs)
    for flag in ef.compile_flags:
        if flag.startswith("-I"):
            paths.append(flag[2:])
        elif flag.startswith("-DAIE_LUT_KERNEL_SOURCE="):
            paths.append(flag.split("=", 1)[1].strip('"'))
    if ef.source_file is not None:
        paths.append(ef.source_file)
    return paths


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1], indirect=True)
def test_every_build_resolves_inside_the_override(tree, device):
    # The installed include root stays for aie_api and the runtime headers;
    # anything kernel-specific must come from the selected tree.
    header_root = Path(config.cxx_header_path()).resolve()
    stray = [
        f"{name}: {p}"
        for name, ef in _builds()
        for p in _paths(ef)
        if not Path(p).resolve().is_relative_to(tree.resolve())
        and Path(p).resolve() != header_root
    ]
    assert not stray, f"include paths outside MLIR_AIE_KERNEL_SOURCES: {stray}"


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
@pytest.mark.parametrize("device", [NPU2Col1], indirect=True)
def test_a_header_only_in_the_override_is_the_one_compiled(tree, device, tmp_path):
    if not (tree / "aie_runtime_lib" / "AIE2P").is_dir():
        pytest.skip("this build has no aie_runtime_lib/AIE2P")
    # activation/tanh.cc reaching common/exp2_poly.h has no copy beside it,
    # and the stale install copy must not stand in for the override's.
    header = tree / "aie_kernels" / "common" / "exp2_poly.h"
    header.write_text("#define FROM_OVERRIDE 1\n" + header.read_text())
    with open(tree / "aie_kernels" / "activation" / "tanh.cc", "a") as f:
        f.write(
            '\n#include "../common/exp2_poly.h"\n#ifndef FROM_OVERRIDE\n'
            "#error exp2_poly.h resolved outside MLIR_AIE_KERNEL_SOURCES\n#endif\n"
        )
    out = tmp_path / "build"
    out.mkdir()
    cmd, _ = remarks.compile_command(kernels.tanh(use_lut=True), "aie2p", out)
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    deps = next(out.glob("*.d")).read_text().replace("\\\n", " ").split(":", 1)[1]
    kernel_deps = [
        Path(d).resolve()
        for d in deps.split()
        if "aie_kernels" in Path(d).parts or "aie_runtime_lib" in Path(d).parts
    ]
    assert header.resolve() in kernel_deps
    assert all(d.is_relative_to(tree.resolve()) for d in kernel_deps), kernel_deps
