# kernels/__init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Factory functions for AIE kernel ExternalFunctions.

Each factory locates the appropriate C++ source in the installed
``aie_kernels/`` directory, builds the correct ``arg_types`` for a
given tile configuration, and returns a fully-configured
:class:`~aie.iron.ExternalFunction`.

Arch selection
--------------
Each factory detects the current device target architecture via
``iron.get_current_device()`` and selects either the ``aie2/`` or
``aie2p/`` kernel directory.  When no device is active the default
falls back to ``aie2``.

Available factories
-------------------
- :func:`passthrough` — copies an input tile to an output tile.
- :func:`scale` — multiplies each element by a scalar factor.
- :func:`add` — element-wise addition of two bf16 tiles.
"""

from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction


def _detect_arch() -> str:
    """Return ``'aie2p'`` or ``'aie2'`` based on the active device.

    Falls back to ``'aie2'`` if no device is currently set or if the
    device cannot be inspected.
    """
    try:
        import aie.iron as _iron
        from aie.utils.compile.utils import resolve_target_arch

        device = _iron.get_current_device()
        return resolve_target_arch(device)
    except Exception:
        return "aie2"


def _kernel_source(arch: str, subdir: str, filename: str) -> Path:
    """Return the absolute path to a kernel source file.

    Searches ``<cxx_header_path>/aie_kernels/<subdir>/<filename>`` first,
    then falls back to ``generic/<filename>``.

    Args:
        arch: Target architecture string (``'aie2'`` or ``'aie2p'``).
        subdir: Subdirectory under ``aie_kernels/`` (e.g. ``'aie2'``).
        filename: Source file name (e.g. ``'scale.cc'``).

    Returns:
        Path to the source file.

    Raises:
        FileNotFoundError: When the source file cannot be found in either
            location.
    """
    from aie.utils import config

    base = Path(config.cxx_header_path()) / "aie_kernels"
    candidate = base / subdir / filename
    if candidate.exists():
        return candidate
    generic = base / "generic" / filename
    if generic.exists():
        return generic
    raise FileNotFoundError(
        f"Kernel source '{filename}' not found under {base}/{subdir}/ "
        f"or {base}/generic/"
    )


def _include_dirs() -> list[str]:
    """Return the standard include directory list for kernel compilation."""
    from aie.utils import config

    return [config.cxx_header_path()]


# ---------------------------------------------------------------------------
# passthrough
# ---------------------------------------------------------------------------


def passthrough(tile_size: int = 4096, dtype=np.int32) -> ExternalFunction:
    """Element-wise passthrough kernel: copies input tile to output tile.

    Maps to ``passThroughLine`` exported from
    ``aie_kernels/generic/passThrough.cc``.

    Signature::

        void passThroughLine(T *in, T *out, int32_t lineWidth)

    Args:
        tile_size: Number of elements per tile.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for ``passThroughLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``.
    """
    dtype_np = np.dtype(dtype)
    if dtype_np == np.dtype(np.uint8):
        bit_width = 8
    elif dtype_np == np.dtype(np.int16):
        bit_width = 16
    elif dtype_np == np.dtype(np.int32):
        bit_width = 32
    else:
        raise ValueError(
            f"passthrough: unsupported dtype {dtype}. Use np.uint8, np.int16, or np.int32."
        )

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    source = _kernel_source(arch, arch, "passThrough.cc")
    return ExternalFunction(
        "passThroughLine",
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


def scale(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Scalar-multiply kernel: multiplies each element of an input tile by a factor.

    Maps to ``vector_scalar_mul_vector`` (vectorized) or
    ``vector_scalar_mul_scalar`` (scalar) exported from
    ``aie_kernels/aie2/scale.cc``.

    Signature::

        void vector_scalar_mul_vector(T *a_in, T *c_out,
                                      int32_t *factor, int32_t N)

    The ``factor`` argument is a 1-element ``int32`` array passed as a pointer
    so the caller can share it via an :class:`~aie.iron.ObjectFifo`.

    Args:
        tile_size: Number of elements per tile.
        dtype: Element data type.  Must be ``np.int16`` or ``np.int32``
            (controlled by the ``BIT_WIDTH`` macro).
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the scale kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int16`` or ``np.int32``.
    """
    if dtype not in (np.int16, np.int32):
        raise ValueError(f"scale() dtype must be np.int16 or np.int32, got {dtype}")

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"vector_scalar_mul_{func_variant}"

    bit_width = 16 if dtype == np.int16 else 32
    source = _kernel_source(arch, arch, "scale.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def add(
    tile_size: int = 1024, dtype=bfloat16, vectorized: bool = True
) -> ExternalFunction:
    """Element-wise addition kernel for two bf16 tiles.

    Maps to ``eltwise_add_bf16_vector`` (vectorized) or
    ``eltwise_add_bf16_scalar`` (scalar) exported from
    ``aie_kernels/aie2/add.cc``.

    Signature::

        void eltwise_add_bf16_vector(bfloat16 *a_in,
                                     bfloat16 *b_in,
                                     bfloat16 *c_out)

    .. note::
        The C++ source hard-codes ``N = 1024`` elements.  The ``tile_size``
        parameter is used to build the Python ``arg_types`` and **must**
        match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024 to match the
            hard-coded kernel loop bound).
        dtype: Element data type.  Only ``bfloat16`` is supported by the
            installed kernel.
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the add kernel.

    Raises:
        ValueError: When ``dtype`` is not ``bfloat16``.
    """
    if dtype is not bfloat16 and dtype != bfloat16:
        raise ValueError(
            f"add() dtype must be bfloat16, got {dtype}. "
            "Only the bf16 variant is available in the installed aie_kernels."
        )

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"eltwise_add_bf16_{func_variant}"

    source = _kernel_source(arch, arch, "add.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )


# saxpy is intentionally omitted: saxpy.cc lives only in
# programming_examples/, not in the installed aie_kernels/ tree.
# Locating it by walking up from __file__ is fragile for installed packages.


__all__ = [
    "passthrough",
    "scale",
    "add",
]
