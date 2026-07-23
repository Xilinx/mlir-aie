# discovery.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Filesystem discovery for the HSA/ROCR runtime (no ctypes / no dlopen).

Locates ``libhsa-runtime64.so`` by probing env hints then standard install
locations. Performs no ``dlopen`` so it can serve as a cheap capability probe
before committing to the heavier ctypes bindings import.
"""

import os
from pathlib import Path
from typing import List, Optional

__all__ = ["find_libhsa", "find_hsa_include_dir", "hsa_available"]

_HOME = Path(os.path.expanduser("~"))

# <mlir-aie>/python/utils/hostruntime/hsaruntime/discovery.py -> parents[4] == <mlir-aie>
_MLIR_AIE_ROOT = Path(__file__).resolve().parents[4]

# Standard ROCm roots probed when ROCM_PATH is unset. A sibling ../opt/rocm
# checkout is probed first (matches the dev layout in this workspace).
_ROCM_ROOT_CANDIDATES = [
    _MLIR_AIE_ROOT.parent / "opt" / "rocm",
    Path("/opt/rocm"),
    _HOME / "rocm",
]

_LIB_SUFFIX = os.path.join("lib", "libhsa-runtime64.so")


def _existing(paths: List[Optional[str]]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if p and Path(p).exists() and p not in out:
            out.append(p)
    return out


def find_libhsa() -> Optional[str]:
    """Locate libhsa-runtime64.so, honoring env hints then standard locations."""
    rocm_dir = os.environ.get("ROCM_PATH")
    hsa_dir = os.environ.get("HSA_RUNTIME_DIR")

    candidates: List[Optional[str]] = [
        os.environ.get("HSA_RUNTIME_LIB"),
        os.path.join(hsa_dir, "libhsa-runtime64.so") if hsa_dir else None,
        os.path.join(rocm_dir, _LIB_SUFFIX) if rocm_dir else None,
    ]
    for root in _ROCM_ROOT_CANDIDATES:
        candidates.append(str(root / _LIB_SUFFIX))
    candidates += ["/usr/lib/libhsa-runtime64.so", "/usr/local/lib/libhsa-runtime64.so"]

    found = _existing(candidates)
    return found[0] if found else None


def find_hsa_include_dir() -> Optional[str]:
    """Locate the directory containing hsa/hsa.h (for reference; not required at runtime)."""
    rocm_dir = os.environ.get("ROCM_PATH")
    candidates: List[Optional[str]] = [
        os.path.join(rocm_dir, "include") if rocm_dir else None,
    ]
    for root in _ROCM_ROOT_CANDIDATES:
        candidates.append(str(root / "include"))
    for c in candidates:
        if c and (Path(c) / "hsa" / "hsa.h").is_file():
            return str(Path(c).resolve())
    return None


def hsa_available() -> bool:
    """Cheap capability probe: True if libhsa-runtime64.so can be located."""
    return find_libhsa() is not None
