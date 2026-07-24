#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

##===----- download_mlir.py - Download/unpack host MLIR wheel (cross-platform)-----===##
#
##===------------------------------------------------------------------------------===##
# Python shim for scripts/download_mlir.sh (cross-platform, Windows-friendly).
# Reuses matching downloads, native-tools installs, and extracted trees.
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
from importlib import metadata
from pathlib import Path


def _run(cmd):
    cmd = [str(c) for c in cmd]
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd)


def _pip(args):
    _run([sys.executable, "-m", "pip", *args])


def _remove_tree(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


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


def _installed_version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _install_native_tools(version):
    if _installed_version("mlir-native-tools") == version:
        print(f"[reuse] mlir-native-tools {version}")
        return
    _pip(["install", "--no-deps", "--upgrade", f"mlir-native-tools=={version}"])


def _target_platform_args():
    cibw_archs = (os.environ.get("CIBW_ARCHS") or "").strip()
    matrix_os = (os.environ.get("MATRIX_OS") or "").strip()

    if cibw_archs not in {"arm64", "aarch64"}:
        return []
    if matrix_os in {"macos-12", "macos-14"} and cibw_archs == "arm64":
        return ["--platform", "macosx_12_0_arm64", "--only-binary=:all:"]
    if matrix_os == "ubuntu-20.04" and cibw_archs == "aarch64":
        return ["--platform", "linux_aarch64", "--only-binary=:all:"]
    raise SystemExit(f"Unsupported CIBW_ARCHS/MATRIX_OS: {cibw_archs}/{matrix_os}")


def _wheel_arch_marker():
    cibw_archs = (os.environ.get("CIBW_ARCHS") or "").strip().lower()
    return {
        "amd64": "win_amd64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "aarch64",
    }.get(cibw_archs)


def _matching_wheels(directory, distribution, version):
    package_name = distribution.replace("-", "_")
    prefix = f"{package_name}-{version}-".lower()
    wheels = sorted(
        wheel
        for wheel in directory.glob("*.whl")
        if wheel.name.lower().startswith(prefix)
    )

    # A shared wheel directory may contain the same version for several hosts.
    if marker := _wheel_arch_marker():
        wheels = [wheel for wheel in wheels if marker in wheel.name.lower()]
    return wheels


def _download_wheel(directory, distribution, version, platform_args):
    wheels = _matching_wheels(directory, distribution, version)
    if not wheels:
        _pip(
            [
                "-q",
                "download",
                "--no-deps",
                "--dest",
                str(directory),
                *platform_args,
                f"{distribution}=={version}",
            ]
        )
        wheels = _matching_wheels(directory, distribution, version)

    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one cached {distribution} {version} wheel in {directory}, "
            f"found {len(wheels)}"
        )
    print(f"[reuse] {wheels[0].name}")
    return wheels[0]


def _extraction_matches(wheel, dist_info):
    # RECORD identifies the exact wheel without invalidating the Windows path fixup.
    record = dist_info / "RECORD"
    if not record.is_file():
        return False
    try:
        with zipfile.ZipFile(wheel) as archive:
            archived_record = archive.read(f"{dist_info.name}/RECORD")
        return record.read_bytes() == archived_record
    except (KeyError, OSError, zipfile.BadZipFile):
        return False


def _ensure_extracted(wheel, directory, distribution, version):
    package_name = distribution.replace("-", "_")
    package_dir = directory / package_name
    dist_info = directory / f"{package_name}-{version}.dist-info"

    if (
        package_dir.is_dir()
        and dist_info.is_dir()
        and _extraction_matches(wheel, dist_info)
    ):
        print(f"[reuse] extracted {wheel.name}")
        return package_dir

    if package_dir.exists() or dist_info.exists():
        print(f"[refresh] extracted {wheel.name}")
    _remove_tree(package_dir)
    for old_dist_info in directory.glob(f"{package_name}-*.dist-info"):
        _remove_tree(old_dist_info)

    print(f"[unzip] {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory)

    if not package_dir.is_dir() or not dist_info.is_dir():
        raise RuntimeError(
            f"Wheel did not contain {package_dir.name} and {dist_info.name}"
        )
    return package_dir


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
    distribution = "mlir-no-rtti" if no_rtti else "mlir"
    directory = Path.cwd()

    version = _wheel_version()
    print(f"Using MLIR version: {version}")
    _install_native_tools(version)

    wheel = _download_wheel(directory, distribution, version, _target_platform_args())
    mlir_prefix = _ensure_extracted(wheel, directory, distribution, version)
    _fixup_llvm_diaguids(mlir_prefix)

    print(directory)
    for p in sorted(directory.glob("mlir*")):
        print(p.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
