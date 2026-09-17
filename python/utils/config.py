# compile.py -*- Python -*-
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import os
import shutil
from pathlib import Path

import aie.utils.configure as config  # pyright: ignore[reportMissingImports]


def _executable_name(name):
    return f"{name}.exe" if os.name == "nt" else name


def peano_install_dir():
    """Return the Peano install directory."""
    if not os.path.isdir(config.peano_install_dir):
        raise RuntimeError(
            f"Invalid Peano install directory: {config.peano_install_dir}"
        )
    return config.peano_install_dir


def peano_cxx_path():
    """Return the path to the Peano C++ compiler."""
    install_dir = peano_install_dir()
    peano_cxx = os.path.join(install_dir, "bin", _executable_name("clang++"))
    if not os.path.isfile(peano_cxx):
        raise RuntimeError(f"Peano compiler not found in {peano_cxx}")
    return peano_cxx


def peano_linker_path():
    """Return the path to the Peano linker."""
    install_dir = peano_install_dir()
    peano_ld = os.path.join(install_dir, "bin", _executable_name("ld.lld"))
    if not os.path.isfile(peano_ld):
        raise RuntimeError(f"Peano linker not found in {peano_ld}")
    return peano_ld


def root_path():
    """Return the root path of the MLIR-AIE project."""
    root_dir = config.install_path()
    if not os.path.isdir(root_dir):
        raise RuntimeError(f"Invalid MLIR-AIE root directory: {root_dir}")
    return root_dir


def aiecc_path():
    """Return the aiecc executable used by JIT compilation.

    Resolution order: AIECC_PATH, then the MLIR-AIE bin directory, then PATH.
    """
    override = os.environ.get("AIECC_PATH")
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(
                f"AIECC_PATH is set to {override}, but no such file exists."
            )
        return override

    bundled = os.path.join(root_path(), "bin", _executable_name("aiecc"))
    if os.path.isfile(bundled):
        return bundled

    found = shutil.which(_executable_name("aiecc"))
    if found:
        return found

    raise RuntimeError(
        "Could not find aiecc. Resolves in the order of the AIECC_PATH "
        "environment variable, MLIR-AIE bin directory, then PATH."
    )


def host_cxx_path():
    """Return a host C++ compiler: ``CXX``, then ``c++``/``g++``/``clang++``.

    Exclude Peano's bin directory from automatic discovery: lit prepends it
    to PATH, but its bundled headers do not support host compilation.
    """
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        found = shutil.which(env_cxx)
        if not found:
            raise RuntimeError(f"CXX is set to {env_cxx!r}, but it was not found.")
        return found

    peano_bin = os.path.realpath(os.path.join(config.peano_install_dir, "bin"))
    host_path = os.pathsep.join(
        entry
        for entry in os.get_exec_path()
        if os.path.normcase(os.path.realpath(entry)) != os.path.normcase(peano_bin)
    )
    for candidate in ("c++", "g++", "clang++"):
        found = shutil.which(candidate, path=host_path)
        if found:
            return found

    raise RuntimeError(
        "Could not find a host C++ compiler (checked CXX env var, then "
        "c++/g++/clang++ on PATH). Required to compile the dynamic dispatch "
        "bridge for DispatchTime[T] designs."
    )


def objcopy_path():
    """Return the llvm-objcopy used to rename symbols in compiled objects.

    AIE objects use the AIEngine ELF e_machine, which GNU binutils objcopy
    cannot parse; llvm-objcopy renames symbols structurally regardless of
    target. The wheel bundles llvm-objcopy under the MLIR-AIE bin directory;
    fall back to one on PATH for source/dev installs.
    """
    bundled_objcopy = os.path.join(root_path(), "bin", _executable_name("llvm-objcopy"))
    if os.path.isfile(bundled_objcopy):
        return bundled_objcopy

    path_objcopy = shutil.which(_executable_name("llvm-objcopy"))
    if path_objcopy:
        return path_objcopy

    raise RuntimeError(
        "Could not find llvm-objcopy. Expected it under the MLIR-AIE bin "
        "directory or on PATH. GNU binutils objcopy cannot process AIE "
        "objects, so an LLVM objcopy is required."
    )


def cxx_header_path():
    """Return the path to the MLIR-AIE C++ headers."""
    include_dir = os.path.join(root_path(), "include")
    if not os.path.isdir(include_dir):
        raise RuntimeError(f"MLIR-AIE C++ headers not found in {include_dir}")
    return include_dir


def runtime_header_path():
    """Return the include directory holding ``aie/Runtime/TxnEncoding.h``.

    Installed headers (including wheel headers) always take precedence. Only a
    CMake build tree may fall back to its source headers; installed packages do
    not retain or consult paths from the machine that built them.
    """
    sentinel = os.path.join("aie", "Runtime", "TxnEncoding.h")
    root = Path(root_path())
    candidates = [root / "include"]
    if (candidates[0] / sentinel).is_file():
        return str(candidates[0])
    cache = root / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                candidates.append(Path(line.split("=", 1)[1]) / "include")
                break
    for include_dir in candidates:
        if os.path.isfile(os.path.join(include_dir, sentinel)):
            return str(include_dir)
    raise RuntimeError(
        f"Could not find {sentinel} in any of: {', '.join(map(str, candidates))}."
    )
