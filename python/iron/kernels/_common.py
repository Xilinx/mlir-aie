# kernels/_common.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Shared helpers for the kernels submodules."""

import logging
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction

_log = logging.getLogger(__name__)


def _detect_arch() -> str:
    """Return ``'aie2p'`` or ``'aie2'`` based on the active device.

    Falls back to ``'aie2'`` if no device is currently set.
    """
    try:
        import aie.iron as _iron
        from aie.utils.compile.utils import resolve_target_arch

        device = _iron.get_current_device()
        return resolve_target_arch(device)
    except (ImportError, RuntimeError, AttributeError, ValueError):
        # ImportError: iron not built; RuntimeError: no active device set;
        # AttributeError/ValueError: unrecognised device.  Anything else (e.g.
        # OSError from a misconfigured install) bubbles up so the user sees it.
        _log.warning(
            "_detect_arch: no active device or unrecognised device; "
            "falling back to 'aie2'",
            exc_info=True,
        )
        return "aie2"


def _kernel_source(arch: str, subdir: str, filename: str) -> Path:
    """Return the absolute path to a kernel source file.

    Args:
        arch: Target architecture string (``'aie2'`` or ``'aie2p'``).
        subdir: Subdirectory under ``aie_kernels/`` (e.g. ``'aie2'``).
        filename: Source file name (e.g. ``'scale.cc'``).

    Returns:
        Path to the source file.

    Raises:
        FileNotFoundError: When the source file cannot be found.
    """
    from aie.utils import config

    base = Path(config.cxx_header_path()) / "aie_kernels"
    candidate = base / subdir / filename
    if candidate.exists():
        return candidate
    if subdir != "aie2":
        aie2_fallback = base / "aie2" / filename
        if aie2_fallback.exists():
            return aie2_fallback
    generic = base / "generic" / filename
    if generic.exists():
        return generic
    raise FileNotFoundError(
        f"Kernel source '{filename}' not found under {base}/{subdir}/, "
        f"{base}/aie2/, or {base}/generic/"
    )


def _include_dirs() -> list[str]:
    """Return the standard include directory list for kernel compilation."""
    from aie.utils import config

    return [config.cxx_header_path()]


_DTYPE_BIT_WIDTHS = {
    np.dtype(np.uint8): 8,
    np.dtype(np.int16): 16,
    np.dtype(np.int32): 32,
}


def _dtype_to_bit_width(dtype, *, factory_name: str) -> int:
    """Map ``np.uint8 | np.int16 | np.int32`` to 8/16/32.

    Raises:
        ValueError: When *dtype* is not one of the three supported types.
    """
    bit_width = _DTYPE_BIT_WIDTHS.get(np.dtype(dtype))
    if bit_width is None:
        raise ValueError(
            f"{factory_name}: unsupported dtype {dtype}. "
            "Use np.uint8, np.int16, or np.int32."
        )
    return bit_width


def _conv_act_dtype_info(
    base_name: str, act_dtype, *, factory_name: str
) -> tuple[str, list[str]]:
    """Map ``act_dtype`` to ``(func_name, compile_flags)`` for conv kernels.

    Raises:
        ValueError: When *act_dtype* is not ``np.int8`` or ``np.uint8``.
    """
    if act_dtype == np.int8:
        return f"{base_name}_i8", ["-DINT8_ACT"]
    elif act_dtype == np.uint8:
        return f"{base_name}_ui8", []
    else:
        raise ValueError(
            f"{factory_name}(): act_dtype must be np.int8 or np.uint8, "
            f"got {act_dtype}"
        )


def _require_fixed_tile_size(
    factory_name: str, tile_size: int, expected: int = 1024
) -> None:
    """Raise ValueError when ``tile_size`` does not match a hard-coded C++ loop bound."""
    if tile_size != expected:
        raise ValueError(
            f"{factory_name}() tile_size must be {expected} to match the "
            f"hard-coded C++ loop bound, got {tile_size}."
        )


def _min_dma_aligned_elems(dtype, align: int = 4) -> int:
    """Return the minimum element count whose byte size is a multiple of *align*.

    The NPU shim DMA requires a 4-byte alignment.  A 1-element output tile is
    fine for ``int32`` (4 bytes) but only 2 bytes for ``bfloat16`` — kernels
    whose C++ side writes a single value still need a Python tile type with
    enough elements to satisfy the alignment.
    """
    itemsize = np.dtype(dtype).itemsize
    return max(1, (align + itemsize - 1) // itemsize)


def _default_source_path(filename: str, subdir: str | None = None) -> Path:
    """Return ``_kernel_source(arch, subdir or arch, filename)`` using the active arch."""
    arch = _detect_arch()
    return _kernel_source(arch, subdir or arch, filename)


def _arg_type_key(t):
    """Hashable key for one entry of ``arg_types`` (used by ``_EXTERN_CACHE``)."""
    if hasattr(t, "__args__"):
        # np.ndarray[(shape,), np.dtype[T]]
        shape = t.__args__[0]
        inner = t.__args__[1]
        dtype = inner.__args__[0] if hasattr(inner, "__args__") else inner
        return ("ndarray", tuple(shape), str(dtype))
    return repr(t)


# Cache keyed on the full input parameter tuple.  Identical helper calls
# (kernels.mm(...) twice with same kwargs) should return the SAME
# ExternalFunction instance — otherwise both end up in
# ExternalFunction._instances, both get JIT-compiled, and (because they
# share the default ``<name>.o`` output filename) the second compilation
# overwrites the first's object file with whichever just-rebuilt copy
# wins the race.  The whole_array port hit exactly this footgun: a
# default-flag kernels.mm() call (just to fetch .mac_dims) and a
# c_col_maj=True kernels.mm() call for the actual binding produced two
# differently-flagged ExternalFunctions whose .o files collided on disk.
_EXTERN_CACHE: dict = {}


def _make_extern(
    func_name: str,
    source_path: "Path | str",
    arg_types: list,
    *,
    compile_flags: list[str] | None = None,
    use_chess: bool = False,
    shared_object_file_name: str | None = None,
) -> ExternalFunction:
    """Construct (or reuse) an ExternalFunction with the standard include_dirs.

    Memoized on (func_name, source_path, arg_types, compile_flags,
    use_chess) so repeated calls with identical parameters return the
    SAME ExternalFunction instance (see ``_EXTERN_CACHE`` for rationale).

    Different parameterizations get distinct instances AND distinct
    ``object_file_name``s — the latter is auto-suffixed with a short
    digest of the cache key so per-parameterization .o files don't
    overwrite each other on disk.  The default ``<name>.o`` is preserved
    when ``compile_flags`` is empty AND ``use_chess`` is False (no
    parameterization to disambiguate).

    ``use_chess`` selects the Chess (xchesscc) compiler instead of Peano
    for this kernel's .o build.  See
    :class:`aie.iron.kernel.ExternalFunction` for the design-level
    contract: all EFs in a single ``@iron.jit`` design must share the
    same toolchain choice (mixed peano/chess is rejected at compile
    time).

    ``shared_object_file_name`` pins the output ``.o`` filename so
    multiple factories targeting the SAME source file (e.g. companion
    symbols like ``reduce_max_vector`` + ``compute_max`` both in
    ``reduce_max.cc``) can share one compile.  The first call builds
    the ``.o``; subsequent calls with the same ``shared_object_file_name``
    skip the build and link against the existing one.  Without this,
    each factory would produce a distinct ``.o`` each carrying ALL
    symbols from the ``.cc``, tripping a duplicate-symbol link error.
    """
    flags_tuple = tuple(compile_flags or [])
    arg_keys = tuple(_arg_type_key(t) for t in arg_types)
    cache_key = (func_name, str(source_path), arg_keys, flags_tuple, use_chess)
    cached = _EXTERN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # The object_file_name suffix must distinguish every distinct cache_key,
    # not just compile_flags — otherwise two helper calls with the same
    # name + flags but different source / arg_types would generate
    # ExternalFunctions with identical .o filenames and trip the collision
    # check in ExternalFunction.__init__ (or, if both passed it, would
    # silently overwrite each other on disk).
    if shared_object_file_name is not None:
        # Caller explicitly pinned the .o filename so companion symbols from
        # the same .cc share one compile.  Skip the digest-suffix path so the
        # ExternalFunction lands at the pinned name; the second-and-later
        # callers' compiles short-circuit via compile_external_kernel's
        # "skip if .o exists" check.
        digest = None
        object_file_name = shared_object_file_name
    elif flags_tuple or arg_keys or str(source_path):
        import hashlib

        # 8 hex chars of sha256 — short enough not to bloat MLIR strings,
        # wide enough that the chance of two distinct cache_keys colliding
        # is vanishingly small (~2^-32).
        digest = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:8]
        object_file_name = f"{func_name}_{digest}.o"
    else:
        digest = None
        object_file_name = None  # ExternalFunction default → ``<name>.o``

    # Auto-prefix the SYMBOL name when an existing ExternalFunction with the
    # same _original_name is already registered.  Without this, two helper
    # calls with different parameterizations produce two ExternalFunctions
    # whose compiled .o files BOTH export the same C symbol — MLIR rejects
    # the duplicate `func.func` declaration; the linker rejects the duplicate
    # symbol.  The first call keeps the unprefixed name (preserves byte-
    # identity for the common single-version case).  Subsequent calls get
    # `<digest>_<name>` so each parameterization lives at a unique symbol.
    # The .o file is built and the symbol renamed via the existing
    # `symbol_prefix` plumbing in ExternalFunction.
    symbol_prefix = None
    if digest is not None:
        for existing in ExternalFunction._instances:
            if existing._original_name == func_name:
                symbol_prefix = digest
                # The auto-suffixed object_file_name we built above already
                # embeds the same digest; once symbol_prefix is in play,
                # ExternalFunction.__init__ rebuilds object_file_name from
                # `<prefix>_<name>.o` if we leave it None — keep the
                # explicit name so we control its layout.
                object_file_name = f"{digest}_{func_name}.o"
                break

    extern = ExternalFunction(
        func_name,
        object_file_name=object_file_name,
        source_file=str(source_path),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=list(flags_tuple),
        symbol_prefix=symbol_prefix,
        use_chess=use_chess,
    )
    _EXTERN_CACHE[cache_key] = extern
    return extern
