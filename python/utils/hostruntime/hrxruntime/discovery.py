# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem discovery for the HRX runtime (no ctypes / no dlopen).

Locates ``libhrx`` for the Python host stack. Performs no ``dlopen`` so it can
serve as a cheap capability probe before committing to the heavier ctypes
bindings import.

Python looks for the library in this order (CMake ``FindHRX`` shares the env
hints and filesystem roots, but does not search site-packages):

1. ``$HRX_LIBHRX`` / ``$LIBHRX_DIR`` -- explicit user overrides.
2. A pip-installed package in site-packages whose layout is
   ``<package>/lib/libhrx.so*`` (Linux) or ``<package>/bin/hrx.dll`` (Windows).
   Filesystem only: no package import.
3. ``$HRX_DIR``, then the same filesystem roots as ``FindHRX.cmake`` (sibling
   checkout, ``$HOME/hrx``, ``/opt/hrx``, ``/usr/local/hrx``).
4. Linux loader fallbacks (``/usr/lib``, ``/usr/local/lib``).
"""

import os
import platform
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "find_libhrx",
    "find_hrx_dir",
    "hrx_available",
]

_IS_WINDOWS = platform.system() == "Windows"

# The shared library's file name(s) differ by platform: an ``.so`` on Linux, a
# ``.dll`` (packaged as ``hrx.dll``, with ``libhrx.dll`` tolerated) on Windows.
_LIBHRX_NAMES = ("hrx.dll", "libhrx.dll") if _IS_WINDOWS else ("libhrx.so",)

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

# pip-installed runtime: any site-packages package with the conventional
# libhrx layout. No distribution name is hardcoded.
_SKIP_SITE_DIRS = frozenset({"__pycache__", "bin", "lib", "include"})

# Loader fallbacks after env / wheel / install-prefix. Tests clear this so a
# host with /usr/lib/libhrx.so cannot leak into discovery unit tests.
_SYSTEM_LIBS: tuple[str, ...] = (
    () if _IS_WINDOWS else ("/usr/lib/libhrx.so", "/usr/local/lib/libhrx.so")
)


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
    """Locate an HRX root (install prefix or source checkout) with hrx_runtime.h.

    Honors the ``HRX_DIR`` hint first, then the standard candidate roots (kept in
    sync with ``FindHRX.cmake``).

    Returns:
        Optional[str]: The resolved HRX root path, or ``None`` if no root
        containing ``hrx_runtime.h`` was found.
    """
    hints = [os.environ.get("HRX_DIR")] + [str(c) for c in _HRX_ROOT_CANDIDATES]
    for c in hints:
        if not c:
            continue
        for suf in _HEADER_SUFFIXES:
            if (Path(c) / suf).is_file():
                return str(Path(c).resolve())
    return None


def _pip_roots() -> Tuple[Path, ...]:
    """Return site-packages directories that may contain a pip-installed libhrx."""
    try:
        import site
    except ImportError:
        return ()
    raw: List[str] = []
    try:
        raw.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user = site.getusersitepackages()
        if user:
            raw.append(user)
    except Exception:
        pass
    roots: List[Path] = []
    seen: set[Path] = set()
    for item in raw:
        if not item:
            continue
        path = Path(item)
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return tuple(roots)


def _libhrx_in_prefix(root: Path) -> Optional[str]:
    """Return libhrx under an install-prefix or pip-package layout, or None."""
    if _IS_WINDOWS:
        for sub in ("bin", "lib"):
            for name in _LIBHRX_NAMES:
                candidate = root / sub / name
                if candidate.is_file():
                    return str(candidate)
        return None
    lib_dir = root / "lib"
    direct = lib_dir / "libhrx.so"
    if direct.is_file():
        return str(direct)
    try:
        matches = sorted(lib_dir.glob("libhrx.so*"))
    except OSError:
        return None
    return str(matches[0]) if matches else None


def _wheel_libhrx() -> Optional[str]:
    """Locate libhrx inside a pip-installed package (any distribution name).

    Walks site-packages one package-directory deep for the conventional layout.
    No ``import`` and no ``dlopen``.
    """
    for site_root in _pip_roots():
        if not site_root.is_dir():
            continue
        try:
            packages = sorted(
                p
                for p in site_root.iterdir()
                if p.is_dir()
                and p.name not in _SKIP_SITE_DIRS
                and not p.name.endswith((".dist-info", ".egg-info"))
                and not p.name.startswith(".")
            )
        except OSError:
            continue
        for pkg in packages:
            found = _libhrx_in_prefix(pkg)
            if found:
                return found
    return None


def find_libhrx() -> Optional[str]:
    """Locate the HRX shared library, honoring env hints then standard locations.

    On Linux this looks for ``libhrx.so``; on Windows for ``hrx.dll`` /
    ``libhrx.dll`` (which the packaged release ships under ``bin/``, with the
    import lib ``hrx.lib`` under ``lib/``). ``HRX_LIBHRX`` (explicit full path)
    and ``LIBHRX_DIR`` (a directory to search) are honored on both. A
    pip-installed package that ships libhrx in the conventional layout is
    probed next, then the sibling / ``FindHRX`` roots.

    Returns:
        Optional[str]: The path to the first existing HRX shared library found,
        or ``None`` if none of the hints/standard locations resolve.
    """
    hrx_dir = find_hrx_dir()
    libhrx_dir = os.environ.get("LIBHRX_DIR")

    candidates: List[Optional[str]] = [os.environ.get("HRX_LIBHRX")]
    if libhrx_dir:
        candidates += [os.path.join(libhrx_dir, n) for n in _LIBHRX_NAMES]
    candidates.append(_wheel_libhrx())

    if _IS_WINDOWS:
        if hrx_dir:
            # Install-prefix layout: the DLL is in <root>/bin (import lib in
            # <root>/lib); check bin first, then lib.
            for sub in ("bin", "lib"):
                candidates += [os.path.join(hrx_dir, sub, n) for n in _LIBHRX_NAMES]
            # Source-build layout mirrors the Linux one but with .dll.
            candidates += [
                os.path.join(hrx_dir, "build", "cmake", "libhrx", "src", "libhrx", n)
                for n in _LIBHRX_NAMES
            ]
        # No canonical system path for a DLL; rely on env hints / bin/ above.
    else:
        if hrx_dir:
            # Install-prefix layout: <root>/lib/libhrx.so alongside include/.
            candidates.append(os.path.join(hrx_dir, "lib", "libhrx.so"))
            # Source-build layout: <root>/build/cmake/libhrx/src/libhrx/libhrx.so.
            candidates.append(
                os.path.join(
                    hrx_dir, "build", "cmake", "libhrx", "src", "libhrx", "libhrx.so"
                )
            )
    candidates += list(_SYSTEM_LIBS)

    found = _existing(candidates)
    return found[0] if found else None


def hrx_available() -> bool:
    """Cheap capability probe for HRX (no dlopen, no device init).

    Returns:
        bool: True if the HRX shared library can be located on this host.
    """
    return find_libhrx() is not None
