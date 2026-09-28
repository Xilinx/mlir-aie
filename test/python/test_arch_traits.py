# test_arch_traits.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""``ARCH_TRAITS`` and ``aie_kernels/aie_arch.h`` describe the same machines.

The factories size tiles and pick references from the Python table, and the
sources they build read the header. A row edited on one side only would
compile a kernel for one width and judge it at another.

The header also decides what an architecture without a row of tuned code
gets: every kernel's untuned branch, which ``AIE_KERNELS_PORTABLE=1``
selects on the architectures that have one, so it is built for each.
"""

import concurrent.futures
import os
import subprocess
from pathlib import Path

import pytest
from aie.iron.device import from_name
from aie.iron.kernels._common import ARCH_TRAITS
from aie.utils import config, get_current_device
from aie.utils.compile.remarks import compile_command, kernel_builds
from aie.utils.compile.utils import resolve_target_arch
from aie.utils.hostruntime import set_current_device


def _peano_available() -> bool:
    try:
        return os.path.isfile(config.peano_cxx_path())
    except RuntimeError:
        return False


def _compile(source: str, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            config.peano_cxx_path(),
            "-std=c++20",
            "-fsyntax-only",
            f"-I{Path(config.aie_kernels_dir())}",
            *flags,
            "-x",
            "c++",
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
@pytest.mark.parametrize("arch", list(ARCH_TRAITS))
def test_header_row_matches_python_row(arch):
    t = ARCH_TRAITS[arch]
    expected = {
        "__AIE_ARCH__": t.aie_arch,
        f"AIE_ARCH_{arch.upper()}": 1,
        "AIE_BF16_LANES": t.bf16_lanes,
        "AIE_HAS_NATIVE_TANH": int(t.native_tanh),
        "AIE_HAS_NATIVE_EXP2": int(t.native_exp2),
        "AIE_HAS_BFP16": int(t.bfp16),
        "AIE_LUT_16B_RUN": t.lut_16b_run,
    }
    source = '#include "aie_arch.h"\n' + "".join(
        f'static_assert({name} == {value}, "{name}");\n'
        for name, value in expected.items()
    )
    p = _compile(source, f"--target={arch}-none-unknown-elf")
    assert p.returncode == 0, p.stderr


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_an_architecture_without_a_row_does_not_build():
    # Peano's AIE targets define __AIE_ARCH__ in a preincluded header, after
    # any -D, so the made-up architecture is a host build of the header alone.
    p = _compile('#include "aie_arch.h"\n', "-D__AIE_ARCH__=99")
    assert p.returncode != 0
    assert "no row for this __AIE_ARCH__" in p.stderr


@pytest.mark.parametrize("arch", list(ARCH_TRAITS))
def test_default_device_is_of_its_architecture(arch):
    assert resolve_target_arch(from_name(ARCH_TRAITS[arch].device)) == arch


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
@pytest.mark.parametrize("arch", list(ARCH_TRAITS))
def test_every_kernel_builds_from_its_portable_branch(arch, tmp_path, monkeypatch):
    if not (Path(config.aie_runtime_lib_dir()) / arch.upper()).is_dir():
        pytest.skip(f"this build has no aie_runtime_lib/{arch.upper()}")
    monkeypatch.setenv("AIE_KERNELS_PORTABLE", "1")
    previous = get_current_device(probe_runtime=False)
    set_current_device(from_name(ARCH_TRAITS[arch].device, n_cols=1))
    try:
        builds = list(kernel_builds())
    finally:
        set_current_device(previous)

    def one(indexed):
        i, (name, ef) = indexed
        cell = tmp_path / f"build{i}"
        cell.mkdir()
        cmd, _ = compile_command(ef, arch, cell)
        p = subprocess.run(
            [*cmd, "-fsyntax-only"],
            capture_output=True,
            text=True,
        )
        errors = [line for line in p.stderr.splitlines() if "error:" in line]
        return None if p.returncode == 0 else f"{name}: " + "\n".join(errors)

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        failed = [f for f in pool.map(one, enumerate(builds)) if f]
    assert not failed, "\n".join(failed)
