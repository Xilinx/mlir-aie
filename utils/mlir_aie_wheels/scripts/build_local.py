#!/usr/bin/env python3
##===------ build_local.py - Local wheel build orchestration (cross-platform)------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===------------------------------------------------------------------------------===##
# Python shim for scripts/build_local.sh (cross-platform, Windows-friendly).
#
# Workflow:
#   1) Detect platform (linux/macos/windows)
#   2) Export the same env vars as build_local.sh
#   3) Run cibuildwheel for the main wheel project
#   4) Rename the main wheel's python tag (cpXYZ-cpXYZ -> py3-none)
#   5) Stage inputs into python_bindings/, unzip the mlir_aie wheel into it
#   6) Run cibuildwheel for python_bindings, outputting into ../wheelhouse
##===------------------------------------------------------------------------------===##


from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PIP_FIND_LINKS = "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _run(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
    pretty = " ".join(cmd)
    print(f"[run] $ {pretty}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _cibuildwheel_cmd() -> list[str]:
    return [sys.executable, "-m", "cibuildwheel"]


def _copy_tree_merge(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _unzip_overwrite(zip_path: Path, dst_dir: Path) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


# --------------------------------------------------------------------------------------
# Platform detection + env
# --------------------------------------------------------------------------------------


def _detect_machine() -> str:
    sysname = platform.system().lower()
    if sysname.startswith("linux"):
        return "linux"
    if sysname.startswith("darwin"):
        return "macos"
    if sysname.startswith("windows"):
        return "windows"
    # Fall back to the uname-style string for debugging.
    return f"UNKNOWN:{platform.system()}"


def _apply_env_for_machine(machine: str) -> None:
    os.environ.setdefault("APPLY_PATCHES", "true")
    os.environ.setdefault("CIBW_BUILD", "cp312-* cp313-* cp314-*")

    if machine == "linux":
        os.environ.setdefault("MATRIX_OS", "ubuntu-22.04")
        os.environ.setdefault("CIBW_ARCHS", "x86_64")
        os.environ.setdefault("ARCH", "x86_64")
        os.environ.setdefault("PARALLEL_LEVEL", "15")
    elif machine == "macos":
        os.environ.setdefault("MATRIX_OS", "macos-14")
        os.environ.setdefault("CIBW_ARCHS", "arm64")
        os.environ.setdefault("ARCH", "arm64")
        os.environ.setdefault("PARALLEL_LEVEL", "32")
    else:
        # Treat everything else as Windows, mirroring the bash script.
        os.environ.setdefault("MATRIX_OS", "windows-2022")
        os.environ.setdefault("CIBW_ARCHS", "AMD64")
        os.environ.setdefault("ARCH", "AMD64")


# --------------------------------------------------------------------------------------
# Wheelhouse mutation
# --------------------------------------------------------------------------------------


# Rename cpXYZ-cpXYZ -> py3-none in mlir*.whl.
def _rename_main_wheel_python_tag(wheelhouse: Path) -> None:
    tag_re = re.compile(r"cp\d{2,3}-cp\d{2,3}")

    for whl in wheelhouse.glob("mlir*whl"):
        new_name = tag_re.sub("py3-none", whl.name, count=1)
        if new_name == whl.name:
            continue
        dst = whl.with_name(new_name)
        print(f"[rename] {whl.name} -> {dst.name}")
        whl.rename(dst)


def _copy_ccache_from_wheelhouse(wheelhouse: Path, host_ccache_dir: Path) -> None:
    src = wheelhouse / ".ccache"
    if not src.is_dir():
        return
    host_ccache_dir.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        dst = host_ccache_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = (here / "..").resolve()  # utils/mlir_aie_wheels
    wheelhouse = project_root / "wheelhouse"
    python_bindings = project_root / "python_bindings"

    # If a nested checkout doesn't exist and the user didn't set MLIR_AIE_SOURCE_DIR,
    # point to the repo root so we can find python/requirements_dev.txt and CMake source.
    src_env = os.environ.get("MLIR_AIE_SOURCE_DIR", "").strip()
    if src_env:
        repo_root = Path(src_env).resolve()
    else:
        nested_checkout = project_root / "mlir-aie"
        if (nested_checkout / "utils" / "iron_setup.py").exists():
            repo_root = nested_checkout.resolve()
        else:
            repo_root = project_root.parent.parent.resolve()
            os.environ["MLIR_AIE_SOURCE_DIR"] = str(repo_root)
            print(f"[info] MLIR_AIE_SOURCE_DIR not set; defaulting to {repo_root}")

    machine = _detect_machine()
    print(machine)

    _apply_env_for_machine(machine)

    # For local builds, default to the official mlir-distro unless the user overrides.
    if not os.environ.get("PIP_FIND_LINKS", "").strip():
        os.environ["PIP_FIND_LINKS"] = DEFAULT_PIP_FIND_LINKS
        print(f"[info] PIP_FIND_LINKS not set; defaulting to {DEFAULT_PIP_FIND_LINKS}")

    # Ccache stat/config dumps when ccache is available.
    host_ccache_dir: Optional[Path] = None
    if _which("ccache"):
        _run(["ccache", "--show-stats"])
        _run(["ccache", "--print-stats"])
        _run(["ccache", "--show-config"])

        proc = subprocess.run(
            ["ccache", "--get-config", "cache_dir"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cache_dir_str = proc.stdout.strip()
        if cache_dir_str:
            host_ccache_dir = Path(cache_dir_str)
            os.environ["HOST_CCACHE_DIR"] = cache_dir_str

    # cibuildwheel "$HERE"/.. --platform "$machine"
    _run(
        _cibuildwheel_cmd()
        + [
            str(project_root),
            "--platform",
            machine,
            "--output-dir",
            str(wheelhouse),
        ],
        cwd=project_root,
    )

    _rename_main_wheel_python_tag(wheelhouse)

    # if [ -d wheelhouse/.ccache ], copy back to HOST_CCACHE_DIR
    if host_ccache_dir is not None:
        _copy_ccache_from_wheelhouse(wheelhouse, host_ccache_dir)

    # Stage inputs into python_bindings
    shutil.copy2(project_root / "requirements.txt", python_bindings / "requirements.txt")

    # Prefer a wheel-local requirements_dev.txt if present, fallback to repo_root/python/.
    dev_req_dst = python_bindings / "requirements_dev.txt"
    for src in [project_root / "requirements_dev.txt", repo_root / "python" / "requirements_dev.txt"]:
        if src.is_file():
            shutil.copy2(src, dev_req_dst)
            break

    _copy_tree_merge(project_root / "scripts", python_bindings / "scripts")

    for whl in wheelhouse.glob("mlir_aie*.whl"):
        shutil.copy2(whl, python_bindings / whl.name)

    # unzip -o -q mlir_aie\*.whl ; rm -rf mlir_aie*.whl
    for whl in python_bindings.glob("mlir_aie*.whl"):
        _unzip_overwrite(whl, python_bindings)

    for whl in python_bindings.glob("mlir_aie*.whl"):
        whl.unlink()

    # cibuildwheel --platform "$machine" --output-dir ../wheelhouse
    _run(_cibuildwheel_cmd() + ["--platform", machine, "--output-dir", "../wheelhouse"], cwd=python_bindings)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
