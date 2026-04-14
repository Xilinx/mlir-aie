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
    except Exception:
        _log.debug("_detect_arch: falling back to aie2", exc_info=True)
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
