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


def _resolve_tool(tool: str, env_var: str) -> str:
    """Locate *tool*: ``$env_var``, then the MLIR-AIE bin directory, then PATH.

    The same order ``aiecc_path()`` uses, so every tool the dispatch bridge
    shells out to is found the same way aiecc itself is.
    """
    override = os.environ.get(env_var)
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(
                f"{env_var} is set to {override}, but no such file exists."
            )
        return override

    bundled = os.path.join(root_path(), "bin", _executable_name(tool))
    if os.path.isfile(bundled):
        return bundled

    found = shutil.which(_executable_name(tool))
    if found:
        return found

    raise RuntimeError(
        f"Could not find {tool}. Resolves in the order of the {env_var} "
        f"environment variable, MLIR-AIE bin directory, then PATH."
    )


def aie_opt_path():
    """Return the aie-opt executable used to lower a dynamic runtime sequence."""
    return _resolve_tool("aie-opt", "AIE_OPT_PATH")


def aie_translate_path():
    """Return the aie-translate executable used to emit a dynamic TXN builder."""
    return _resolve_tool("aie-translate", "AIE_TRANSLATE_PATH")


def host_cxx_path():
    """Return a HOST-target (x86_64) C++ compiler.

    Used to compile the dynamic dispatch bridge shared library that Python
    loads via ``ctypes``. This is deliberately NOT ``peano_cxx_path()`` --
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


def runtime_header_path():
    """Return the include directory holding ``aie/Runtime/TxnEncoding.h``.

    The dispatch bridge compiles generated host C++ that includes that header.
    Unlike the device-kernel headers ``cxx_header_path()`` serves, it is not
    staged into a build area's ``include/`` -- only into an install area -- so
    fall back to the source tree, the way ``configure.py`` resolves
    ``peano_install_dir``.
    """
    sentinel = os.path.join("aie", "Runtime", "TxnEncoding.h")
    candidates = [os.path.join(root_path(), "include")]
    source_dir = getattr(config, "aie_source_dir", "")
    if source_dir:
        candidates.append(os.path.join(source_dir, "include"))
    for include_dir in candidates:
        if os.path.isfile(os.path.join(include_dir, sentinel)):
            return include_dir
    raise RuntimeError(f"Could not find {sentinel} in any of: {', '.join(candidates)}.")
