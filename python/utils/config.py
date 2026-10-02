# compile.py -*- Python -*-
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import os
import re
import shutil
import subprocess
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


def aiecc_tool_path(name, *, override=None, cwd=None):
    """Return the executable aiecc runs for ``name``, found the way aiecc finds it.

    An explicit override takes precedence over ``AIE_XCLBINUTIL`` for
    ``xclbinutil``. Otherwise Peano's bin directory is searched ahead of PATH.
    """
    if override is None and name == "xclbinutil":
        override = os.environ.get("AIE_XCLBINUTIL")
    if override:
        override_path = Path(override)
        if (
            cwd is not None
            and not override_path.is_absolute()
            and any(sep and sep in override for sep in (os.sep, os.altsep))
        ):
            override = str(Path(cwd) / override_path)
        found = shutil.which(override)
    else:
        search = os.pathsep.join(
            [os.path.join(config.peano_install_dir, "bin"), *os.get_exec_path()]
        )
        found = shutil.which(_executable_name(name), path=search)
    if not found:
        raise RuntimeError(f"aiecc cannot find {override or name}.")
    return found


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


def _tool_runs(path):
    """Return True if the binary at ``path`` actually executes.

    Guards against a tool that is present on disk but cannot run -- e.g. one
    whose shared-library dependency fails to load, so it exits nonzero and
    emits nothing. Such a binary is worse than a missing one: a caller reading
    its output sees an empty symbol listing rather than a broken toolchain.
    """
    try:
        return (
            subprocess.run(
                [path, "--version"], capture_output=True, timeout=5
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _llvm_tool_dirs():
    """Return the bundled bin directories that may hold LLVM binutils.

    The MLIR-AIE and Peano installs are complementary rather than redundant:
    the MLIR-AIE wheel bundles llvm-objcopy, while the Peano (llvm-aie) wheel
    ships llvm-ar and llvm-nm. Searching only one of them leaves a stock
    install unable to find a tool that is sitting on disk in the other.
    A build tree bundles nothing (llvm-objcopy is copied at install time),
    so the bin directory of the LLVM it was configured against comes last.
    """
    dirs = []
    for get_dir in (root_path, peano_install_dir):
        try:
            dirs.append(os.path.join(get_dir(), "bin"))
        except RuntimeError:
            # A source or dev install may configure only one of the two.
            continue
    # Absent from a configure.py generated before it was recorded; a wheel
    # records its build machine's directory, which is skipped for not existing.
    llvm_bin = getattr(config, "llvm_tools_binary_dir", "")
    if llvm_bin and os.path.isdir(llvm_bin):
        dirs.append(llvm_bin)
    return dirs


def _path_candidates(name):
    """Yield every PATH match for ``name``, unsuffixed spellings first.

    All bare-name matches are yielded, in PATH order, rather than just the
    first: an earlier entry may be present yet unable to run, in which case a
    later one is the right answer. Distros also package LLVM binutils with a
    release suffix (``llvm-nm-18``), so those follow, highest version first.
    They are still LLVM tools, so they read AIE objects fine.
    """
    exe = _executable_name(name)
    suffix = ".exe" if os.name == "nt" else ""
    pattern = re.compile(rf"{re.escape(name)}-(\d+){re.escape(suffix)}$")

    directories = list(
        dict.fromkeys(directory or os.curdir for directory in os.get_exec_path())
    )
    for directory in directories:
        candidate = os.path.join(directory, exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            yield candidate

    versioned = []
    for directory in directories:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    match = pattern.fullmatch(entry.name)
                    if match and entry.is_file() and os.access(entry.path, os.X_OK):
                        versioned.append((int(match.group(1)), entry.path))
        except OSError:
            # PATH routinely names directories that do not exist.
            continue

    for _, candidate in sorted(versioned, key=lambda item: -item[0]):
        yield candidate


def _find_llvm_tool(name, env_var):
    """Resolve an LLVM binutil, preferring a candidate that actually runs.

    Resolution order: ``env_var``, the bundled MLIR-AIE and Peano bin
    directories, the configured LLVM's bin directory, then PATH. Candidates
    that fail to execute are passed over in favour of a later one; if every
    candidate is broken the first is returned anyway, so the caller surfaces
    that tool's own error rather than a misleading "not found".
    """
    override = os.environ.get(env_var)
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(
                f"{env_var} is set to {override}, but no such file exists."
            )
        return override

    searched = []
    broken = None
    for directory in _llvm_tool_dirs():
        candidate = os.path.join(directory, _executable_name(name))
        searched.append(candidate)
        if os.path.isfile(candidate):
            if _tool_runs(candidate):
                return candidate
            broken = broken or candidate

    for candidate in _path_candidates(name):
        if _tool_runs(candidate):
            return candidate
        broken = broken or candidate

    if broken is not None:
        return broken

    raise RuntimeError(
        f"Could not find {name}. Resolves in the order of the {env_var} "
        f"environment variable, the MLIR-AIE and Peano bin directories, the "
        "LLVM bin directory the build was configured against, then "
        f"PATH (including versioned spellings such as {name}-18). Searched: "
        + (", ".join(searched) if searched else "(no bundled bin directories)")
        + ". PATH directories: "
        + ", ".join(directory or os.curdir for directory in os.get_exec_path())
    )


def objcopy_path():
    """Return the llvm-objcopy used to rename symbols in compiled objects.

    The objects themselves are well formed -- plain ELF32/little-endian
    relocatables -- but they carry the AIEngine e_machine (0x108), which no GNU
    BFD backend claims. GNU binutils 2.42 objcopy therefore declines to pick an
    input target and fails with "Unable to recognise the format of the input
    file", while its own nm, ar and objdump read the same object fine: those
    tolerate an unknown architecture, objcopy insists on a definite one.

    GNU objcopy can be coerced with an explicit ``-I elf32-little``, but that
    pins a BFD target name from the outside and silently assumes the object is
    32-bit little-endian. llvm-objcopy needs no such hint, so it is what we
    resolve here.
    """
    return _find_llvm_tool("llvm-objcopy", "AIE_OBJCOPY_PATH")


def nm_path():
    """Return the llvm-nm used to list the symbols a compiled object defines.

    Paired with objcopy_path() to bulk-rename symbols in a compiled object: list
    every defined external symbol with nm, then bulk-``--redefine-syms`` with
    objcopy.

    Unlike objcopy, this is a preference rather than a hard requirement -- a
    symbol table is machine-agnostic, and GNU nm does list AIE objects. Pinning
    the LLVM spelling keeps the listing format matched to the parser that reads
    it, and keeps the pair on one toolchain, since the objcopy half has no GNU
    equivalent that works at all.
    """
    return _find_llvm_tool("llvm-nm", "AIE_NM_PATH")


def ar_path():
    """Return the llvm-ar used to bundle compiled objects into a static archive.

    As with nm_path(), a preference rather than a hard requirement: archiving
    only indexes the ELF symbol table, so GNU ar handles AIE objects too.
    Resolved here so a build draws its archiver from the same toolchain as the
    compiler that produced the objects.
    """
    return _find_llvm_tool("llvm-ar", "AIE_AR_PATH")


def readobj_path():
    """Return the llvm-readobj the static checks read kernel objects with.

    Its JSON output (sections, symbols, relocations) is what tells which
    functions a linked kernel keeps and which runtime helpers it calls; GNU
    readelf has no JSON form and does not decode the AIE relocations.
    """
    return _find_llvm_tool("llvm-readobj", "AIE_READOBJ_PATH")


def aie_kernels_dir():
    """Return the ``aie_kernels/`` directory the kernel factories compile from.

    The installed tree's copy (``<root>/include/aie_kernels``) unless
    ``MLIR_AIE_KERNEL_SOURCES`` names a checkout, in which case that
    checkout's ``aie_kernels/`` is used. The override lets a checked-out
    kernel source be compiled against an installed wheel, which is how the
    static kernel checks run on a pull request.
    """
    override = os.environ.get("MLIR_AIE_KERNEL_SOURCES")
    if override:
        return os.path.join(override, "aie_kernels")
    return os.path.join(cxx_header_path(), "aie_kernels")


def aie_runtime_lib_dir():
    """Return ``aie_runtime_lib/`` (the LUT sources), honoring ``MLIR_AIE_KERNEL_SOURCES``."""
    override = os.environ.get("MLIR_AIE_KERNEL_SOURCES")
    if override:
        return os.path.join(override, "aie_runtime_lib")
    return os.path.join(root_path(), "aie_runtime_lib")


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
