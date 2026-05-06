#!/usr/bin/env python3
##===----- download_mlir.py - Download/unpack host MLIR wheel (cross-platform)-----===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===------------------------------------------------------------------------------===##
# Python shim for scripts/download_mlir.sh (cross-platform, Windows-friendly).
#
# Env:
#   ENABLE_RTTI: ON/OFF (default: ON)
#   CIBW_ARCHS:  x86_64/aarch64/arm64/AMD64
#   MATRIX_OS:   ubuntu-20.04/macos-12/macos-14/windows-*
#
# Optional:
#   MLIR_AIE_WHEEL_VERSION: override wheel version (bypass clone-llvm.sh parsing)
#   MLIR_AIE_SOURCE_DIR:    explicit repo root (for clone-llvm.sh lookup)
##===------------------------------------------------------------------------------===##

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _run(cmd):
    cmd = [str(c) for c in cmd]
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd)


def _pip(args):
    _run([sys.executable, "-m", "pip", *args])


def _try_rmtree(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[warn] failed to remove '{path}': {exc}")


def _find_clone_llvm(script_dir):
    src = (os.environ.get("MLIR_AIE_SOURCE_DIR") or "").strip()
    if src:
        return Path(src) / "utils" / "clone-llvm.sh"

    # Assume we are in utils/mlir_aie_wheels/scripts.
    p = script_dir.parent / "clone-llvm.sh"
    return p if p.is_file() else script_dir.parent.parent / "clone-llvm.sh"


def _sh_var(text, name):
    m = re.search(rf"^{re.escape(name)}=(.*)$", text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"{name} not found")
    return m.group(1).split("#", 1)[0].strip().strip('"').strip("'")


def _expand_sh(template, vars):
    # Supports the subset clone-llvm.sh uses for WHEEL_VERSION.
    pat = re.compile(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::0:([0-9]+))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
    )

    def repl(m):
        name = m.group(1) or m.group(3)
        val = vars[name]
        if m.group(2):
            val = val[: int(m.group(2))]
        return val

    return pat.sub(repl, template)


def _wheel_version():
    direct = (os.environ.get("MLIR_AIE_WHEEL_VERSION") or "").strip()
    if direct:
        return direct

    clone_llvm = _find_clone_llvm(Path(__file__).resolve().parent)
    if not clone_llvm.is_file():
        raise FileNotFoundError(f"clone-llvm.sh not found at: {clone_llvm}")

    text = clone_llvm.read_text(encoding="utf-8", errors="replace")
    vars = {
        "DATETIME": _sh_var(text, "DATETIME"),
        "LLVM_PROJECT_COMMIT": _sh_var(text, "LLVM_PROJECT_COMMIT"),
    }
    return _expand_sh(_sh_var(text, "WHEEL_VERSION"), vars).strip()


# --------------------------------------------------------------------------------------
# Windows patch
# --------------------------------------------------------------------------------------


def _find_vs_install_dir():
    vsinstalldir = (os.environ.get("VSINSTALLDIR") or "").strip()
    if vsinstalldir:
        return Path(vsinstalldir)

    pf86 = os.environ.get("ProgramFiles(x86)")
    if not pf86:
        return None

    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    try:
        install = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None

    return Path(install) if install else None


def _find_diaguids_lib():
    if os.name != "nt":
        return None

    install_dir = _find_vs_install_dir()
    if install_dir is None:
        return None

    diaguids = install_dir / "DIA SDK" / "lib" / "amd64" / "diaguids.lib"
    return diaguids if diaguids.is_file() else None


def _fixup_llvm_diaguids(mlir_prefix):
    # The MLIR wheel's LLVM CMake exports can contain an absolute diaguids.lib path
    # from the machine that built the wheel. Rewrite it to the current machine's
    # DIA SDK path.
    if os.name != "nt":
        return

    llvm_dir = mlir_prefix / "lib" / "cmake" / "llvm"
    if not llvm_dir.is_dir():
        return

    cmake_files = [llvm_dir / "LLVMExports.cmake", llvm_dir / "LLVMTargets.cmake"]
    cmake_files = [cmake_file for cmake_file in cmake_files if cmake_file.is_file()]
    if not cmake_files:
        return

    pat = re.compile(r"([A-Za-z]:[\\/][^\n\"']*diaguids\.lib)", flags=re.IGNORECASE)
    file_data = {}
    found_reference = False

    for cmake_file in cmake_files:
        data = cmake_file.read_text(encoding="utf-8", errors="ignore")
        file_data[cmake_file] = data
        if pat.search(data):
            found_reference = True

    if not found_reference:
        return

    diaguids = _find_diaguids_lib()
    if diaguids is None:
        raise FileNotFoundError(
            "Could not locate diaguids.lib for LLVM CMake export fixup"
        )

    replacement = diaguids.as_posix()
    patched = 0

    for cmake_file, data in file_data.items():
        new_data, count = pat.subn(replacement, data)
        if count:
            cmake_file.write_text(new_data, encoding="utf-8")
            patched += count

    if patched:
        print(f"[fixup] Patched diaguids.lib path -> {replacement}")


def main():
    enable_rtti = (os.environ.get("ENABLE_RTTI") or "ON").upper()
    no_rtti = enable_rtti == "OFF"

    # Mirror: rm -rf mlir || true (also clear the sibling to avoid stale state).
    _try_rmtree(Path("mlir"))
    _try_rmtree(Path("mlir_no_rtti"))

    version = _wheel_version()
    print(f"Using MLIR version: {version}")

    _pip(["install", "-U", "--force-reinstall", f"mlir-native-tools=={version}"])

    pkg = f"mlir{'-no-rtti' if no_rtti else ''}=={version}"
    cibw_archs = (os.environ.get("CIBW_ARCHS") or "").strip()
    matrix_os = (os.environ.get("MATRIX_OS") or "").strip()

    if cibw_archs in {"arm64", "aarch64"}:
        if matrix_os in {"macos-12", "macos-14"} and cibw_archs == "arm64":
            plat = "macosx_12_0_arm64"
        elif matrix_os == "ubuntu-20.04" and cibw_archs == "aarch64":
            plat = "linux_aarch64"
        else:
            raise SystemExit(
                f"Unsupported CIBW_ARCHS/MATRIX_OS: {cibw_archs}/{matrix_os}"
            )

        _pip(["-q", "download", pkg, "--platform", plat, "--only-binary=:all:"])
    else:
        _pip(["-q", "download", pkg])

    wheels = sorted(Path.cwd().glob("mlir*whl"))
    if not wheels:
        raise FileNotFoundError("No wheels matched pattern: 'mlir*whl'")

    for wheel in wheels:
        print(f"[unzip] {wheel.name}")
        with zipfile.ZipFile(wheel, "r") as zf:
            zf.extractall(Path.cwd())

    for d in (Path("mlir"), Path("mlir_no_rtti")):
        if d.is_dir():
            _fixup_llvm_diaguids(d)

    print(Path.cwd())
    for p in sorted(Path.cwd().glob("mlir*")):
        print(p.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
