# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""The JIT cache key follows the tool that packages the image.

Each key is computed in a fresh process, so the tool is found on a real
``PATH`` the way aiecc finds it.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_KEY = textwrap.dedent("""
    import os, sys
    from pathlib import Path
    from aie.iron.device import NPU2Col1
    from aie.utils import set_current_device
    from aie.utils.compile.jit._hash import _compute_artifact_hash

    def design():
        pass

    set_current_device(NPU2Col1())
    flow = sys.argv[1]
    flags = sys.argv[3:]
    print(
        _compute_artifact_hash(
            Path("design.mlir") if sys.argv[2] == "path" else design,
            [],
            [],
            True,
            full_elf=flow == "full_elf",
            insts_only=flow == "insts_only",
            aiecc_flags=flags,
            emit_elf=flow == "xclbin+elf",
            work_dir=Path(os.environ["KEY_WORK_DIR"]) if "KEY_WORK_DIR" in os.environ else None,
        )
    )
    """)

_PACKAGERS = {"full_elf": "aiebu-asm", "xclbin": "xclbinutil"}


def _key(flow, bin_dir, generator="callable", flags=(), work_dir=None, **env):
    path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    if work_dir is not None:
        env["KEY_WORK_DIR"] = str(work_dir)
    result = subprocess.run(
        [sys.executable, "-c", _KEY, flow, generator, *flags],
        env={**os.environ, "PATH": path, **env},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _install(bin_dir, name, version):
    tool = bin_dir / name
    tool.write_text(f"#!/bin/sh\n# {version}\n")
    tool.chmod(0o755)
    return tool


@pytest.fixture
def bin_dir(tmp_path):
    path = tmp_path / "bin"
    path.mkdir()
    return path


@pytest.mark.parametrize("flow", _PACKAGERS)
@pytest.mark.parametrize("generator", ["callable", "path"])
def test_the_key_follows_the_flows_packaging_tool(bin_dir, flow, generator):
    before = _key(flow, bin_dir, generator)
    _install(bin_dir, _PACKAGERS[flow], "1")
    shadowed = _key(flow, bin_dir, generator)
    _install(bin_dir, _PACKAGERS[flow], "2.0")
    upgraded = _key(flow, bin_dir, generator)
    assert len({before, shadowed, upgraded}) == 3


@pytest.mark.parametrize("flow", [*_PACKAGERS, "insts_only"])
@pytest.mark.parametrize("generator", ["callable", "path"])
def test_the_key_ignores_other_flows_packaging_tools(bin_dir, flow, generator):
    before = _key(flow, bin_dir, generator)
    for other, tool in _PACKAGERS.items():
        if other != flow:
            _install(bin_dir, tool, "1")
    assert _key(flow, bin_dir, generator) == before


@pytest.mark.parametrize("generator", ["callable", "path"])
def test_aie_xclbinutil_picks_the_xclbinutil_the_key_follows(
    bin_dir, tmp_path, generator
):
    _install(bin_dir, "xclbinutil", "1")
    on_path = _key("xclbin", bin_dir, generator)
    chosen = _install(tmp_path, "xclbinutil", "1.0")
    overridden = _key("xclbin", bin_dir, generator, AIE_XCLBINUTIL=str(chosen))
    _install(tmp_path, "xclbinutil", "2.0")
    upgraded = _key("xclbin", bin_dir, generator, AIE_XCLBINUTIL=str(chosen))
    assert len({on_path, overridden, upgraded}) == 3


@pytest.mark.parametrize("generator", ["callable", "path"])
def test_aiecc_flag_overrides_aie_xclbinutil(bin_dir, tmp_path, generator):
    env_tool = _install(bin_dir, "xclbinutil", "env")
    selected = _install(tmp_path, "xclbinutil", "selected")
    flags = [f"--xclbinutil-path={selected}"]
    before = _key(
        "xclbin",
        bin_dir,
        generator,
        flags=flags,
        AIE_XCLBINUTIL=str(env_tool),
    )
    _install(tmp_path, "xclbinutil", "selected v2")
    after = _key(
        "xclbin",
        bin_dir,
        generator,
        flags=flags,
        AIE_XCLBINUTIL=str(env_tool),
    )
    assert before != after


@pytest.mark.parametrize("generator", ["callable", "path"])
def test_relative_aiecc_override_resolves_from_work_dir(bin_dir, tmp_path, generator):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    selected = _install(work_dir, "xclbinutil", "selected")
    flags = ["--xclbinutil-path", "./xclbinutil"]
    before = _key("xclbin", bin_dir, generator, flags=flags, work_dir=work_dir)
    _install(work_dir, "xclbinutil", "selected v2")
    after = _key("xclbin", bin_dir, generator, flags=flags, work_dir=work_dir)
    assert selected.exists()
    assert before != after


def test_relative_aiecc_override_requires_a_work_dir():
    from pathlib import Path

    from aie.utils.compile.jit._hash import _compute_artifact_hash

    with pytest.raises(ValueError, match="relative xclbinutil path"):
        _compute_artifact_hash(
            Path("design.mlir"),
            [],
            [],
            True,
            aiecc_flags=["--xclbinutil-path=./xclbinutil"],
        )


@pytest.mark.parametrize("generator", ["callable", "path"])
def test_xclbin_and_elf_key_tracks_both_packagers(bin_dir, generator):
    _install(bin_dir, "xclbinutil", "1")
    _install(bin_dir, "aiebu-asm", "1")
    before = _key("xclbin+elf", bin_dir, generator)
    _install(bin_dir, "xclbinutil", "2")
    xclbin_changed = _key("xclbin+elf", bin_dir, generator)
    _install(bin_dir, "aiebu-asm", "2")
    both_changed = _key("xclbin+elf", bin_dir, generator)
    assert len({before, xclbin_changed, both_changed}) == 3


@pytest.mark.parametrize("tool", ["nm", "objcopy"])
@pytest.mark.parametrize("generator", ["callable", "path"])
def test_design_key_tracks_object_tools(bin_dir, tmp_path, tool, generator):
    selected = _install(bin_dir, f"llvm-{tool}", "1")
    env = {f"AIE_{tool.upper()}_PATH": str(selected)}
    before = _key("insts_only", bin_dir, generator, **env)
    _install(bin_dir, f"llvm-{tool}", "2.0")
    upgraded = _key("insts_only", bin_dir, generator, **env)
    other = _install(tmp_path, f"llvm-{tool}", "2.0")
    env[f"AIE_{tool.upper()}_PATH"] = str(other)
    redirected = _key("insts_only", bin_dir, generator, **env)
    assert len({before, upgraded, redirected}) == 3
