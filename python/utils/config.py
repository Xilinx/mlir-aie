# compile.py -*- Python -*-
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import os
import re
import shutil
import subprocess

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


def _tool_runs(path):
    """Return True if the binary at ``path`` actually executes.

    Guards against a tool that is present on disk but cannot run -- e.g. one
    whose shared-library dependency fails to load, so it exits nonzero and
    emits nothing. Such a binary is worse than a missing one: a caller reading
    its output sees an empty symbol listing rather than a broken toolchain.
    """
    try:
        return subprocess.run([path, "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


def _llvm_tool_dirs():
    """Return the bundled bin directories that may hold LLVM binutils.

    The MLIR-AIE and Peano installs are complementary rather than redundant:
    the MLIR-AIE wheel bundles llvm-objcopy, while the Peano (llvm-aie) wheel
    ships llvm-ar and llvm-nm. Searching only one of them leaves a stock
    install unable to find a tool that is sitting on disk in the other.
    """
    dirs = []
    for get_dir in (root_path, peano_install_dir):
        try:
            dirs.append(os.path.join(get_dir(), "bin"))
        except RuntimeError:
            # A source or dev install may configure only one of the two.
            continue
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

    bare = []
    versioned = []
    seen = set()
    for directory in os.get_exec_path():
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            # PATH routinely names directories that do not exist.
            continue
        for entry in entries:
            candidate = os.path.join(directory, entry)
            if candidate in seen or not os.access(candidate, os.X_OK):
                continue
            if entry == exe:
                seen.add(candidate)
                bare.append(candidate)
                continue
            match = pattern.match(entry)
            if match:
                seen.add(candidate)
                versioned.append((int(match.group(1)), candidate))

    yield from bare
    for _, candidate in sorted(versioned, key=lambda item: -item[0]):
        yield candidate


def _find_llvm_tool(name, env_var):
    """Resolve an LLVM binutil, preferring a candidate that actually runs.

    Resolution order: ``env_var``, the bundled MLIR-AIE and Peano bin
    directories, then PATH. Candidates that fail to execute are passed over in
    favour of a later one; if every candidate is broken the first is returned
    anyway, so the caller surfaces that tool's own error rather than a
    misleading "not found".
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
        f"environment variable, the MLIR-AIE and Peano bin directories, then "
        f"PATH (including versioned spellings such as {name}-18). Searched: "
        + (", ".join(searched) if searched else "(no bundled bin directories)")
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
    """Return the llvm-nm used to list defined external symbols in compiled objects.

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


def cxx_header_path():
    """Return the path to the MLIR-AIE C++ headers."""
    include_dir = os.path.join(root_path(), "include")
    if not os.path.isdir(include_dir):
        raise RuntimeError(f"MLIR-AIE C++ headers not found in {include_dir}")
    return include_dir
