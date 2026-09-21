# discovery.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Filesystem discovery for the HSA/ROCR runtime (no ctypes / no dlopen).

Locates ``libhsa-runtime64.so`` inside a ROCm installation. Performs no
``dlopen`` so it can serve as a cheap capability probe before committing to the
heavier ctypes bindings import.

The recommended way to provide ROCm is TheRock's pip wheel (``pip install
--index-url https://rocm.nightlies.amd.com/whl-multi-arch/ rocm``, see
https://github.com/ROCm/TheRock/blob/main/RELEASES.md); the base ``rocm``
package pulls ``rocm-sdk-core``, which carries the HSA runtime with AIE
support. It needs no configuration -- it is found in site-packages.

ROCm installations are looked for in three places, in this order:

1. ``$ROCM_PATH`` -- an explicit installation root chosen by the user, which
   overrides the wheel. It names a ROCm tree, not a library file, so the same
   variable serves every ROCm component.
2. The pip-installed ROCm above, whose runtime tree lives in a platform package
   inside site-packages.
3. A system install, conventionally ``/opt/rocm``.

**Platform support.** ROCR's AIE agent exists only on Linux today, so that is
the only platform with a library layout to probe.
"""

import importlib.util
import os
import platform
from collections.abc import Iterator
from pathlib import Path

__all__ = ["find_hsa_include_dir", "find_libhsa", "hsa_available"]

_IS_LINUX = platform.system() == "Linux"

# Explicit user-chosen root. ROCM_PATH is the ROCm-wide convention.
_ENV_ROOT_VARS = ("ROCM_PATH",)

# Library file names to look for under ``<root>/lib``, most preferred first.
# The unversioned developer symlink is preferred; ``libhsa-runtime64.so.*``
# catches the SONAME, which is all TheRock's symlink-free wheels ship.
_LIBHSA_PATTERNS: tuple[str, ...] = (
    ("libhsa-runtime64.so", "libhsa-runtime64.so.*") if _IS_LINUX else ()
)

# Conventional system install locations. POSIX-only, hence platform-gated.
_SYSTEM_ROOTS: tuple[Path, ...] = (Path("/opt/rocm"),) if _IS_LINUX else ()

# TheRock's pure-python shim, whose platform sibling holds the actual ROCm tree.
_WHEEL_CORE_PACKAGE = "_rocm_sdk_core"


def _wheel_root() -> Path | None:
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
        # Optional dependency, present only when ROCm was pip-installed, and
        # cheap when it is: pure python, no dlopen.
        from rocm_sdk import (  # pyright: ignore[reportMissingImports]
            _dist_info,
        )

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


def _libhsa_under(root: Path) -> str | None:
    """Return the HSA runtime inside one ROCm root, or None.

    Patterns are tried in ``_LIBHSA_PATTERNS`` order, so the unversioned
    developer symlink wins over the SONAME: TheRock's runtime wheels contain no
    symlinks at all, so a pip-installed ROCm ships only
    ``libhsa-runtime64.so.1``. Within one pattern, sorting puts the bare SONAME
    (``.so.1``) ahead of fully-versioned siblings (``.so.1.21.0``), which is the
    one whose dependencies are expected to resolve.
    """
    lib_dir = root / "lib"
    for pattern in _LIBHSA_PATTERNS:
        for candidate in sorted(lib_dir.glob(pattern)):
            if candidate.exists():
                return str(candidate)
    return None


def find_libhsa() -> str | None:
    """Locate libhsa-runtime64.so in the first ROCm install that provides it."""
    for root in _rocm_roots():
        lib = _libhsa_under(root)
        if lib:
            return lib
    return None


def find_hsa_include_dir() -> str | None:
    """Locate the directory containing hsa/hsa.h (for reference; not required at runtime)."""
    for root in _rocm_roots():
        include = root / "include"
        if (include / "hsa" / "hsa.h").is_file():
            return str(include.resolve())
    return None


def hsa_available() -> bool:
    """Cheap capability probe: True if libhsa-runtime64.so can be located."""
    return find_libhsa() is not None
