# discovery.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Filesystem discovery for the HSA/ROCR runtime (no ctypes / no dlopen).

Locates ``libhsa-runtime64.so`` inside a ROCm installation. Performs no
``dlopen`` so it can serve as a cheap capability probe before committing to the
heavier ctypes bindings import.

ROCm installations are looked for in three places, in this order:

1. ``$ROCM_PATH`` -- an explicit installation root chosen by the user. It names
   a ROCm tree, not a library file, so the same variable serves every ROCm
   component.
2. A pip-installed ROCm from TheRock (``pip install "rocm[libraries,...]"``,
   see https://github.com/ROCm/TheRock/blob/main/RELEASES.md), whose runtime
   tree lives in a platform package inside site-packages.
3. A system install, conventionally ``/opt/rocm``.
"""

import importlib.util
import os
from pathlib import Path
from typing import Iterator, Optional

__all__ = ["find_libhsa", "find_hsa_include_dir", "hsa_available"]

# Explicit user-chosen root. ROCM_PATH is the ROCm-wide convention; TheRock's
# docs additionally suggest ROCM_HOME for a user's own builds, so honor both.
_ENV_ROOT_VARS = ("ROCM_PATH", "ROCM_HOME")

# Conventional system roots. /usr and /usr/local cover distro packages that
# install straight into <prefix>/lib.
_SYSTEM_ROOTS = (Path("/opt/rocm"), Path("/usr"), Path("/usr/local"))

# TheRock's pure-python shim, whose platform sibling holds the actual ROCm tree.
_WHEEL_CORE_PACKAGE = "_rocm_sdk_core"


def _wheel_root() -> Optional[Path]:
    """Root of a pip-installed ROCm from TheRock, if one is importable.

    The runtime tree ships as a platform package (``_rocm_sdk_core``) beside the
    pure-python ``rocm_sdk`` shim. Its name is computed and may carry a suffix
    nonce, so ask ``rocm_sdk`` for it and only fall back to the current literal
    name. Resolution stays at the spec level -- importing the package is not
    needed, and the ``rocm-sdk path --root`` CLI is deliberately avoided: it
    reports the *devel* tree, requires the multi-GB ``rocm[devel]`` package, and
    expands it on first call.
    """
    name = _WHEEL_CORE_PACKAGE
    try:
        from rocm_sdk import _dist_info  # cheap: pure python, no dlopen

        name = _dist_info.ALL_PACKAGES["core"].get_py_package_name()
    except Exception:
        pass  # not installed, or a private API moved; fall back to the literal
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def _rocm_roots() -> Iterator[Path]:
    """Candidate ROCm installation roots, most specific first."""
    for var in _ENV_ROOT_VARS:
        value = os.environ.get(var)
        if value:
            yield Path(value)
    wheel = _wheel_root()
    if wheel is not None:
        yield wheel
    yield from _SYSTEM_ROOTS


def _libhsa_under(root: Path) -> Optional[str]:
    """The HSA runtime inside one ROCm root, or None.

    Prefers the unversioned developer symlink, but falls back to the SONAME:
    TheRock's runtime wheels contain no symlinks at all, so a pip-installed ROCm
    ships only ``libhsa-runtime64.so.1``.
    """
    lib_dir = root / "lib"
    unversioned = lib_dir / "libhsa-runtime64.so"
    if unversioned.exists():
        return str(unversioned)
    # Sorting puts the bare SONAME (.so.1) ahead of fully-versioned siblings
    # (.so.1.21.0), which is the one whose dependencies are expected to resolve.
    for candidate in sorted(lib_dir.glob("libhsa-runtime64.so.*")):
        if candidate.exists():
            return str(candidate)
    return None


def find_libhsa() -> Optional[str]:
    """Locate libhsa-runtime64.so in the first ROCm install that provides it."""
    for root in _rocm_roots():
        lib = _libhsa_under(root)
        if lib:
            return lib
    return None


def find_hsa_include_dir() -> Optional[str]:
    """Locate the directory containing hsa/hsa.h (for reference; not required at runtime)."""
    for root in _rocm_roots():
        include = root / "include"
        if (include / "hsa" / "hsa.h").is_file():
            return str(include.resolve())
    return None


def hsa_available() -> bool:
    """Cheap capability probe: True if libhsa-runtime64.so can be located."""
    return find_libhsa() is not None
