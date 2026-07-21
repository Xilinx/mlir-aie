# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem discovery for the HRX runtime (no ctypes / no dlopen).

This module mirrors ``FindHRX.cmake`` for the Python host stack: it locates
``libhrx.so`` by probing standard install locations, treating the ``HRX_*``
environment variables only as high-priority *hints*. It deliberately performs no
``dlopen`` so it can be used as a cheap capability probe before committing to
the heavier ``hrxruntime`` import that actually loads the library.
"""

import os
from pathlib import Path
from typing import List, Optional

__all__ = [
    "find_libhrx",
    "find_hrx_dir",
    "hrx_available",
]

_HOME = Path(os.path.expanduser("~"))

# The mlir-aie source root, derived from this file's location:
#   <mlir-aie>/python/utils/hostruntime/hrxruntime/discovery.py
# parents[4] == <mlir-aie>. Used to probe a sibling ``../hrx-system`` install,
# which is the layout FindHRX.cmake also probes.
_MLIR_AIE_ROOT = Path(__file__).resolve().parents[4]

# Standard install/checkout roots probed when HRX_DIR is unset. Kept in sync with
# the hints in cmake/modules/FindHRX.cmake so C++ and Python discover the same
# tree: a sibling hrx-system install first, then $HOME and the system locations.
_HRX_ROOT_CANDIDATES = [
    _MLIR_AIE_ROOT.parent / "hrx-system" / "build" / "hrx-install",
    _MLIR_AIE_ROOT.parent / "hrx",
    _HOME / "hrx",
    Path("/opt/hrx"),
    Path("/usr/local/hrx"),
]


def _existing(paths: List[Optional[str]]) -> List[str]:
    out = []
    for p in paths:
        if p and Path(p).exists() and p not in out:
            out.append(p)
    return out


# Layouts under an HRX root that contain hrx_runtime.h:
#   install prefix   -> <root>/include/hrx/hrx_runtime.h  (packaged headers)
#   flat install     -> <root>/include/hrx_runtime.h
#   source checkout  -> <root>/libhrx/include/hrx_runtime.h
_HEADER_SUFFIXES = [
    os.path.join("include", "hrx", "hrx_runtime.h"),
    os.path.join("include", "hrx_runtime.h"),
    os.path.join("libhrx", "include", "hrx_runtime.h"),
]


def find_hrx_dir() -> Optional[str]:
    """Locate an HRX root (install prefix or source checkout) with hrx_runtime.h."""
    hints = [os.environ.get("HRX_DIR")] + [str(c) for c in _HRX_ROOT_CANDIDATES]
    for c in hints:
        if not c:
            continue
        for suf in _HEADER_SUFFIXES:
            if (Path(c) / suf).is_file():
                return str(Path(c).resolve())
    return None


def find_libhrx() -> Optional[str]:
    """Locate libhrx.so, honoring env hints then standard locations."""
    hrx_dir = find_hrx_dir()
    libhrx_dir = os.environ.get("LIBHRX_DIR")

    candidates: List[Optional[str]] = [
        os.environ.get("HRX_LIBHRX"),
        os.path.join(libhrx_dir, "libhrx.so") if libhrx_dir else None,
    ]
    if hrx_dir:
        # Install-prefix layout: <root>/lib/libhrx.so alongside include/.
        candidates.append(os.path.join(hrx_dir, "lib", "libhrx.so"))
        # Source-build layout: <root>/build/cmake/libhrx/src/libhrx/libhrx.so.
        candidates.append(
            os.path.join(
                hrx_dir, "build", "cmake", "libhrx", "src", "libhrx", "libhrx.so"
            )
        )
    candidates += ["/usr/lib/libhrx.so", "/usr/local/lib/libhrx.so"]

    found = _existing(candidates)
    return found[0] if found else None


def hrx_available() -> bool:
    """Cheap capability probe: True if libhrx.so can be located on this host."""
    return find_libhrx() is not None
