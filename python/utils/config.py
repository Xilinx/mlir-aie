# compile.py -*- Python -*-
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import os
import shutil

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

    Resolution order: the AIECC_PATH environment variable (for consumers,
    e.g. IRON, that need to point at a specific aiecc without relying on
    PATH search order), then the MLIR-AIE bin directory, then PATH.
    """
    env_aiecc = os.environ.get("AIECC_PATH")
    if env_aiecc:
        if not os.path.isfile(env_aiecc):
            raise RuntimeError(
                f"AIECC_PATH is set to {env_aiecc}, but no such file exists."
            )
        return env_aiecc

    bundled_aiecc = os.path.join(root_path(), "bin", _executable_name("aiecc"))
    if os.path.isfile(bundled_aiecc):
        return bundled_aiecc

    path_aiecc = shutil.which(_executable_name("aiecc"))
    if path_aiecc:
        return path_aiecc

    raise RuntimeError(
        "Could not find aiecc. Resolves in the order of the AIECC_PATH "
        "environment variable, MLIR-AIE bin directory, then PATH."
    )


def aie_opt_path():
    """Return the aie-opt executable used to lower a dynamic runtime sequence.

    Resolution order: the AIE_OPT_PATH environment variable, then the
    MLIR-AIE bin directory, then PATH. Mirrors :func:`aiecc_path`.
    """
    env_aie_opt = os.environ.get("AIE_OPT_PATH")
    if env_aie_opt:
        if not os.path.isfile(env_aie_opt):
            raise RuntimeError(
                f"AIE_OPT_PATH is set to {env_aie_opt}, but no such file exists."
            )
        return env_aie_opt

    bundled_aie_opt = os.path.join(root_path(), "bin", _executable_name("aie-opt"))
    if os.path.isfile(bundled_aie_opt):
        return bundled_aie_opt

    path_aie_opt = shutil.which(_executable_name("aie-opt"))
    if path_aie_opt:
        return path_aie_opt

    raise RuntimeError(
        "Could not find aie-opt. Resolves in the order of the AIE_OPT_PATH "
        "environment variable, MLIR-AIE bin directory, then PATH."
    )


def aie_translate_path():
    """Return the aie-translate executable used to emit a dynamic TXN builder.

    Resolution order: the AIE_TRANSLATE_PATH environment variable, then the
    MLIR-AIE bin directory, then PATH. Mirrors :func:`aiecc_path`.
    """
    env_aie_translate = os.environ.get("AIE_TRANSLATE_PATH")
    if env_aie_translate:
        if not os.path.isfile(env_aie_translate):
            raise RuntimeError(
                f"AIE_TRANSLATE_PATH is set to {env_aie_translate}, but no "
                f"such file exists."
            )
        return env_aie_translate

    bundled_aie_translate = os.path.join(
        root_path(), "bin", _executable_name("aie-translate")
    )
    if os.path.isfile(bundled_aie_translate):
        return bundled_aie_translate

    path_aie_translate = shutil.which(_executable_name("aie-translate"))
    if path_aie_translate:
        return path_aie_translate

    raise RuntimeError(
        "Could not find aie-translate. Resolves in the order of the "
        "AIE_TRANSLATE_PATH environment variable, MLIR-AIE bin directory, "
        "then PATH."
    )


def host_cxx_path():
    """Return a HOST-target (x86_64) C++ compiler.

    Used to compile the dynamic dispatch bridge shared library that Python
    loads via ``ctypes``. This is deliberately NOT :func:`peano_cxx_path` --
    Peano's clang++ only
    targets ``aie2*-none-unknown-elf`` (AIE core object files for linking
    into an xclbin); it cannot produce a host-loadable ``.so``.

    Resolution order: the CXX environment variable, then ``c++``/``g++``/
    ``clang++`` on PATH.
    """
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        found = shutil.which(env_cxx)
        if not found:
            raise RuntimeError(f"CXX is set to {env_cxx!r}, but it was not found.")
        return found

    for candidate in ("c++", "g++", "clang++"):
        found = shutil.which(candidate)
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
