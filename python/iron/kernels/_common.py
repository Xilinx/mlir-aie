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


def _default_source_path(filename: str, subdir: str | None = None) -> Path:
    """Return ``_kernel_source(arch, subdir or arch, filename)`` using the active arch."""
    arch = _detect_arch()
    return _kernel_source(arch, subdir or arch, filename)


def _make_extern(
    func_name: str,
    source_path: Path | str,
    arg_types: list,
    *,
    compile_flags: list[str] | None = None,
) -> ExternalFunction:
    """Construct an ExternalFunction with the standard include_dirs."""
    return ExternalFunction(
        func_name,
        source_file=str(source_path),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=compile_flags or [],
    )
