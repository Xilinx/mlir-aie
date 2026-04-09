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

Compile-time parameters
-----------------------
Factory parameters such as ``tile_size``, ``line_width``, and ``dtype``
are **compile-time**: they determine function names, array shapes, and
compiler flags baked into the generated MLIR.  In an ``@iron.jit``
design, annotate them with ``Compile[T]`` so the JIT cache correctly
recompiles when their values change::

    from aie.iron import Compile, In, Out, iron

    @iron.jit
    def my_design(a: In, b: In, c: Out, *,
                  N: Compile[int], dtype: Compile[type] = bfloat16):
        tile_size = N
        add_kernel = kernels.add(tile_size=tile_size, dtype=dtype)
        # ... use add_kernel in workers ...

    # call-site — dtype can change, triggering a recompile
    my_design(a, b, c, N=1024, dtype=bfloat16)

Available factories
-------------------
- :func:`passthrough` — copies an input tile to an output tile.
- :func:`scale` — multiplies each element by a scalar factor.
- :func:`add` — element-wise addition of two bf16 tiles.
- :func:`mul` — element-wise multiplication of two bf16 tiles.
- :func:`reduce_add` — reduces a tile to a scalar sum.
- :func:`reduce_min` — reduces a tile to a scalar minimum.
- :func:`reduce_max` — reduces a tile to a scalar maximum.
- :func:`relu` — element-wise ReLU activation on bf16 tiles.
- :func:`rgba2hue` — converts RGBA pixels to hue values.
- :func:`threshold` — applies a threshold operation to a line.
- :func:`bitwiseOR` — element-wise bitwise OR of two lines.
- :func:`bitwiseAND` — element-wise bitwise AND of two lines.
- :func:`gray2rgba` — converts grayscale pixels to RGBA.
- :func:`rgba2gray` — converts RGBA pixels to grayscale.
- :func:`filter2d` — applies a 3x3 2D convolution filter.
- :func:`addWeighted` — weighted addition of two lines.
- :func:`softmax` — softmax activation on bf16 tiles.
- :func:`gelu` — GELU activation on bf16 tiles.
- :func:`silu` — SiLU (Swish) activation on bf16 tiles.
- :func:`swiglu` — SwiGLU gated activation on bf16 tiles.
- :func:`bf16_exp` — element-wise exponential on bf16 tiles.
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
    # For aie2p targets, fall back to aie2/ before generic/ — most aie2 kernels
    # are binary-compatible with aie2p and are not duplicated in aie2p/.
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


# ---------------------------------------------------------------------------
# Private helpers to reduce boilerplate across factory functions
# ---------------------------------------------------------------------------

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


def _reduce_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for :func:`reduce_add` and :func:`reduce_min`."""
    dtype_np = np.dtype(dtype)
    is_int32 = dtype_np == np.dtype(np.int32)
    if not is_int32:
        raise ValueError(
            f"reduce_{op}() dtype must be np.int32, got {dtype}. "
            "Only the int32 variant is available in the installed aie_kernels."
        )

    arch = _detect_arch()
    in_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    out_ty = np.ndarray[(1,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"reduce_{op}_{func_variant}"

    source = _kernel_source(arch, arch, f"reduce_{op}.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )


def _eltwise_bf16_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for :func:`add` and :func:`mul`."""
    if dtype is not bfloat16 and dtype != bfloat16:
        raise ValueError(
            f"{op}() dtype must be bfloat16, got {dtype}. "
            "Only the bf16 variant is available in the installed aie_kernels."
        )

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"eltwise_{op}_bf16_{func_variant}"

    source = _kernel_source(arch, arch, f"{op}.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )


def _bitwise_kernel(op: str, line_width: int, dtype) -> ExternalFunction:
    """Shared implementation for :func:`bitwiseOR` and :func:`bitwiseAND`."""
    bit_width = _dtype_to_bit_width(dtype, factory_name=f"bitwise{op}")

    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]

    source = _kernel_source(arch, arch, f"bitwise{op}.cc")
    return ExternalFunction(
        f"bitwise{op}Line",
        source_file=str(source),
        arg_types=[line_ty, line_ty, line_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


def _color_convert_kernel(
    func_name: str, filename: str, in_size: int, out_size: int
) -> ExternalFunction:
    """Shared implementation for color-space conversion line kernels.

    Used by :func:`rgba2hue`, :func:`gray2rgba`, and :func:`rgba2gray`.
    """
    arch = _detect_arch()
    in_ty = np.ndarray[(in_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]

    source = _kernel_source(arch, arch, filename)
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )


def _conv_act_dtype_info(
    base_name: str, act_dtype, *, factory_name: str
) -> tuple[str, list[str]]:
    """Map ``act_dtype`` to ``(func_name, compile_flags)`` for conv kernels.

    Used by :func:`conv2dk1`, :func:`conv2dk3`, :func:`conv2dk1_skip`,
    and :func:`conv2dk1_skip_init`.

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
        tile_size: Number of elements per tile.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type (``np.uint8``, ``np.int16``, or
            ``np.int32``).  Compile-time — determines ``BIT_WIDTH``; use
            ``Compile[type]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for ``passThroughLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="passthrough")

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
        tile_size: Number of elements per tile.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type.  Must be ``np.int16`` or ``np.int32``
            (controlled by the ``BIT_WIDTH`` macro).  Compile-time — use
            ``Compile[type]`` in ``@iron.jit`` designs.
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
            hard-coded kernel loop bound).  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type.  Only ``bfloat16`` is supported by the
            installed kernel.  Compile-time — use ``Compile[type]`` in
            ``@iron.jit`` designs.
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the add kernel.

    Raises:
        ValueError: When ``dtype`` is not ``bfloat16``.
    """
    return _eltwise_bf16_kernel("add", tile_size, dtype, vectorized)


# ---------------------------------------------------------------------------
# mul (element-wise multiply)
# ---------------------------------------------------------------------------


def mul(
    tile_size: int = 1024, dtype=bfloat16, vectorized: bool = True
) -> ExternalFunction:
    """Element-wise multiplication kernel for two bf16 tiles.

    Maps to ``eltwise_mul_bf16_vector`` (vectorized) or
    ``eltwise_mul_bf16_scalar`` (scalar) exported from
    ``aie_kernels/aie2/mul.cc``.

    Signature::

        void eltwise_mul_bf16_vector(bfloat16 *a_in,
                                     bfloat16 *b_in,
                                     bfloat16 *c_out)

    .. note::
        The C++ source hard-codes ``N = 1024`` elements.  The ``tile_size``
        parameter is used to build the Python ``arg_types`` and **must**
        match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024 to match the
            hard-coded kernel loop bound).  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type.  Only ``bfloat16`` is supported.
            Compile-time — use ``Compile[type]`` in ``@iron.jit`` designs.
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the mul kernel.

    Raises:
        ValueError: When ``dtype`` is not ``bfloat16``.
    """
    return _eltwise_bf16_kernel("mul", tile_size, dtype, vectorized)


# ---------------------------------------------------------------------------
# reduce_add
# ---------------------------------------------------------------------------


def reduce_add(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: sums all elements of a tile to a scalar.

    Maps to ``reduce_add_vector`` (vectorized) or ``reduce_add_scalar``
    (scalar) exported from ``aie_kernels/aie2/reduce_add.cc``.

    Signature::

        void reduce_add_vector(int32_t *a_in, int32_t *c_out,
                               int32_t input_size)

    Args:
        tile_size: Number of elements in the input tile.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type.  Only ``np.int32`` is currently supported
            by the installed kernel.  Compile-time — use ``Compile[type]``
            in ``@iron.jit`` designs.
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the reduce_add
        kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32``.
    """
    return _reduce_kernel("add", tile_size, dtype, vectorized)


# ---------------------------------------------------------------------------
# reduce_min
# ---------------------------------------------------------------------------


def reduce_min(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: finds the minimum element of a tile.

    Maps to ``reduce_min_vector`` (vectorized) or ``reduce_min_scalar``
    (scalar) exported from ``aie_kernels/aie2/reduce_min.cc``.

    Signature::

        void reduce_min_vector(int32_t *a_in, int32_t *c_out,
                               int32_t input_size)

    Args:
        tile_size: Number of elements in the input tile.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type.  Only ``np.int32`` is currently supported
            by the installed kernel.  Compile-time — use ``Compile[type]``
            in ``@iron.jit`` designs.
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the reduce_min
        kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32``.
    """
    return _reduce_kernel("min", tile_size, dtype, vectorized)


# ---------------------------------------------------------------------------
# reduce_max
# ---------------------------------------------------------------------------


def reduce_max(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: finds the maximum element of a tile.

    Maps to ``reduce_max_vector`` / ``reduce_max_vector_bfloat16``
    (vectorized) or ``reduce_max_scalar`` / ``reduce_max_scalar_bfloat16``
    (scalar) exported from ``aie_kernels/aie2/reduce_max.cc``.

    Signature::

        void reduce_max_vector(int32_t *a_in, int32_t *c_out,
                               int32_t input_size)
        void reduce_max_vector_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                        int32_t input_size)

    Args:
        tile_size: Number of elements in the input tile.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type.  ``np.int32`` or ``bfloat16``.
            Compile-time — selects the C++ function variant; use
            ``Compile[type]`` in ``@iron.jit`` designs.
        vectorized: If ``True`` use the vectorized variant; otherwise use the
            scalar reference variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the reduce_max
        kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    dtype_np = np.dtype(dtype)
    is_bf16 = dtype is bfloat16 or dtype_np == np.dtype(bfloat16)
    is_int32 = dtype_np == np.dtype(np.int32)
    if not is_bf16 and not is_int32:
        raise ValueError(
            f"reduce_max() dtype must be np.int32 or bfloat16, got {dtype}"
        )

    arch = _detect_arch()
    actual_dtype = bfloat16 if is_bf16 else np.int32
    in_ty = np.ndarray[(tile_size,), np.dtype[actual_dtype]]
    out_ty = np.ndarray[(1,), np.dtype[actual_dtype]]

    func_variant = "vector" if vectorized else "scalar"
    suffix = "_bfloat16" if is_bf16 else ""
    func_name = f"reduce_max_{func_variant}{suffix}"

    source = _kernel_source(arch, arch, "reduce_max.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


def relu(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise ReLU activation kernel for bf16 tiles.

    Maps to ``bf16_relu`` exported from ``aie_kernels/aie2/relu.cc``.

    Signature::

        void bf16_relu(bfloat16 *a_in, bfloat16 *c_out)

    .. note::
        The C++ source hard-codes the loop bound to 1024 elements.
        The ``tile_size`` parameter is used to build the Python
        ``arg_types`` and **must** match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024).  Compile-time
            — use ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the relu kernel.
    """
    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]

    source = _kernel_source(arch, arch, "relu.cc")
    return ExternalFunction(
        "bf16_relu",
        source_file=str(source),
        arg_types=[tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )


# ---------------------------------------------------------------------------
# Vision kernels
# ---------------------------------------------------------------------------


def rgba2hue(line_width: int = 1920) -> ExternalFunction:
    """Converts a line of RGBA pixels to hue values.

    Maps to ``rgba2hueLine`` exported from
    ``aie_kernels/aie2/rgba2hue.cc``.

    Signature::

        void rgba2hueLine(uint8_t *in, uint8_t *out, int32_t lineWidth)

    The input is ``lineWidth * 4`` bytes (RGBA), the output is
    ``lineWidth`` bytes (single-channel hue).

    Args:
        line_width: Number of pixels per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for ``rgba2hueLine``.
    """
    return _color_convert_kernel(
        "rgba2hueLine", "rgba2hue.cc", line_width * 4, line_width
    )


def threshold(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Applies a threshold operation to a line of pixels.

    Maps to ``thresholdLine`` exported from
    ``aie_kernels/aie2/threshold.cc``.

    Signature (BIT_WIDTH=8)::

        void thresholdLine(uint8_t *in, uint8_t *out, int32_t lineWidth,
                           uint8_t thresholdValue, uint8_t maxValue,
                           uint8_t thresholdType)

    Args:
        line_width: Number of elements per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type (``np.uint8``, ``np.int16``, or
            ``np.int32``).  Compile-time — determines ``BIT_WIDTH``; use
            ``Compile[type]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for ``thresholdLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="threshold")
    scalar_ty = np.int32 if bit_width == 32 else np.int16

    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]

    source = _kernel_source(arch, arch, "threshold.cc")
    return ExternalFunction(
        "thresholdLine",
        source_file=str(source),
        arg_types=[
            line_ty,
            line_ty,
            np.int32,
            scalar_ty,
            scalar_ty,
            np.int8,
        ],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


def bitwiseOR(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Element-wise bitwise OR of two lines.

    Maps to ``bitwiseORLine`` exported from
    ``aie_kernels/aie2/bitwiseOR.cc``.

    Signature (BIT_WIDTH=8)::

        void bitwiseORLine(uint8_t *in1, uint8_t *in2, uint8_t *out,
                           int32_t lineWidth)

    Args:
        line_width: Number of elements per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type (``np.uint8``, ``np.int16``, or
            ``np.int32``).  Compile-time — determines ``BIT_WIDTH``; use
            ``Compile[type]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for
        ``bitwiseORLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``.
    """
    return _bitwise_kernel("OR", line_width, dtype)


def bitwiseAND(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Element-wise bitwise AND of two lines.

    Maps to ``bitwiseANDLine`` exported from
    ``aie_kernels/aie2/bitwiseAND.cc``.

    Signature (BIT_WIDTH=8)::

        void bitwiseANDLine(uint8_t *in1, uint8_t *in2, uint8_t *out,
                            int32_t lineWidth)

    Args:
        line_width: Number of elements per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type (``np.uint8``, ``np.int16``, or
            ``np.int32``).  Compile-time — determines ``BIT_WIDTH``; use
            ``Compile[type]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for
        ``bitwiseANDLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``.
    """
    return _bitwise_kernel("AND", line_width, dtype)


def gray2rgba(line_width: int = 1920) -> ExternalFunction:
    """Converts a grayscale line to RGBA.

    Maps to ``gray2rgbaLine`` exported from
    ``aie_kernels/aie2/gray2rgba.cc``.

    Signature::

        void gray2rgbaLine(uint8_t *in, uint8_t *out, int32_t lineWidth)

    The input is ``lineWidth`` bytes (single-channel), the output is
    ``lineWidth * 4`` bytes (RGBA).

    Args:
        line_width: Number of pixels per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for
        ``gray2rgbaLine``.
    """
    return _color_convert_kernel(
        "gray2rgbaLine", "gray2rgba.cc", line_width, line_width * 4
    )


def rgba2gray(line_width: int = 1920) -> ExternalFunction:
    """Converts an RGBA line to grayscale.

    Maps to ``rgba2grayLine`` exported from
    ``aie_kernels/aie2/rgba2gray.cc``.

    Signature::

        void rgba2grayLine(uint8_t *in, uint8_t *out, int32_t lineWidth)

    The input is ``lineWidth * 4`` bytes (RGBA), the output is
    ``lineWidth`` bytes (single-channel grayscale).

    Args:
        line_width: Number of pixels per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for
        ``rgba2grayLine``.
    """
    return _color_convert_kernel(
        "rgba2grayLine", "rgba2gray.cc", line_width * 4, line_width
    )


def filter2d(line_width: int = 1920) -> ExternalFunction:
    """Applies a 3x3 2D convolution filter across three input lines.

    Maps to ``filter2dLine`` exported from
    ``aie_kernels/aie2/filter2d.cc``.

    Signature::

        void filter2dLine(uint8_t *lineIn0, uint8_t *lineIn1,
                          uint8_t *lineIn2, uint8_t *out,
                          int32_t lineWidth, int16_t *filterKernel)

    Args:
        line_width: Number of pixels per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for
        ``filter2dLine``.
    """
    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    kernel_ty = np.ndarray[(3, 3), np.dtype[np.int16]]

    source = _kernel_source(arch, arch, "filter2d.cc")
    return ExternalFunction(
        "filter2dLine",
        source_file=str(source),
        arg_types=[line_ty, line_ty, line_ty, line_ty, np.int32, kernel_ty],
        include_dirs=_include_dirs(),
    )


def addWeighted(line_width: int = 1920, dtype=np.uint8) -> ExternalFunction:
    """Weighted addition of two lines with a gamma offset.

    Maps to ``addWeightedLine`` exported from
    ``aie_kernels/aie2/addWeighted.cc``.

    Signature (BIT_WIDTH=8)::

        void addWeightedLine(uint8_t *in1, uint8_t *in2, uint8_t *out,
                             int32_t lineWidth, int16_t alpha,
                             int16_t beta, uint8_t gamma)

    Args:
        line_width: Number of elements per line.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.
        dtype: Element data type (``np.uint8``, ``np.int16``, or
            ``np.int32``).  Compile-time — determines ``BIT_WIDTH``; use
            ``Compile[type]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for
        ``addWeightedLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="addWeighted")
    _gamma_types = {8: np.int8, 16: np.int16, 32: np.int32}
    gamma_ty = _gamma_types[bit_width]

    arch = _detect_arch()
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]

    source = _kernel_source(arch, arch, "addWeighted.cc")
    return ExternalFunction(
        "addWeightedLine",
        source_file=str(source),
        arg_types=[
            line_ty,
            line_ty,
            line_ty,
            np.int32,
            np.int16,
            np.int16,
            gamma_ty,
        ],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


# saxpy is intentionally omitted: saxpy.cc lives only in
# programming_examples/, not in the installed aie_kernels/ tree.
# Locating it by walking up from __file__ is fragile for installed packages.


# ---------------------------------------------------------------------------
# LUT-based activation kernels (softmax, gelu, silu, swiglu, bf16_exp)
# ---------------------------------------------------------------------------
#
# On aie2, these kernels depend on lookup-table operations declared in
# ``lut_based_ops.h`` with the LUT data arrays defined in
# ``lut_based_ops.cpp`` (both under ``aie_runtime_lib/AIE2/``).  Because
# the functions in ``lut_based_ops.h`` are ``always_inline``, only the
# data arrays need to be compiled alongside the kernel.  We achieve this
# by creating a wrapper ``source_string`` that ``#include``s both the
# kernel source and ``lut_based_ops.cpp``, producing a single translation
# unit and a single ``.o`` file.
#
# On aie2p, these kernels use native hardware intrinsics (``aie::tanh``,
# ``aie::exp2``) and have no LUT dependency, so the source file is used
# directly.


def _create_lut_kernel(
    func_name: str,
    kernel_filename: str,
    arg_types: list,
    compile_flags: list[str] | None = None,
) -> ExternalFunction:
    """Create an :class:`ExternalFunction` for a LUT-dependent kernel.

    Handles the aie2/aie2p split automatically:

    * **aie2** — creates a combined ``source_string`` that ``#include``s
      the kernel source and ``lut_based_ops.cpp`` so the LUT data arrays
      are compiled into the same object file.
    * **aie2p** — uses ``source_file`` directly (no LUT dependency).

    Both paths add the kernel's arch sub-directory to ``include_dirs``
    so that relative includes like ``../aie_kernel_utils.h`` resolve
    correctly even after the source is copied to the build directory.
    """
    arch = _detect_arch()
    kernel_path = _kernel_source(arch, arch, kernel_filename)

    from aie.utils import config

    include = _include_dirs()
    # Add kernel arch dir so ../aie_kernel_utils.h resolves correctly.
    kernel_arch_dir = Path(config.cxx_header_path()) / "aie_kernels" / arch
    include.append(str(kernel_arch_dir))

    flags = compile_flags or []

    if arch == "aie2":
        # aie2 kernels need LUT data from lut_based_ops.cpp.
        runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
        lut_cpp = runtime_dir / "lut_based_ops.cpp"
        include.append(str(runtime_dir))
        # Use preprocessor #include to combine sources — preserves
        # relative include resolution within each original file.
        source = f'#include "{kernel_path}"\n#include "{lut_cpp}"\n'
        return ExternalFunction(
            func_name,
            source_string=source,
            arg_types=arg_types,
            include_dirs=include,
            compile_flags=flags,
        )
    else:
        return ExternalFunction(
            func_name,
            source_file=str(kernel_path),
            arg_types=arg_types,
            include_dirs=include,
            compile_flags=flags,
        )


def softmax(tile_size: int = 1024) -> ExternalFunction:
    """Softmax activation kernel for bf16 tiles.

    Maps to ``softmax_bf16`` exported from
    ``aie_kernels/<arch>/softmax.cc``.

    Signature::

        void softmax_bf16(bfloat16 *input, bfloat16 *output,
                          int32_t input_size)

    Args:
        tile_size: Number of elements per tile.  Compile-time — use
            ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the softmax
        kernel.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "softmax_bf16",
        "softmax.cc",
        [tile_ty, tile_ty, np.int32],
    )


def gelu(tile_size: int = 1024) -> ExternalFunction:
    """GELU activation kernel (tanh approximation) for bf16 tiles.

    Maps to ``gelu_bf16`` exported from
    ``aie_kernels/<arch>/gelu.cc``.

    Signature::

        void gelu_bf16(bfloat16 *input, bfloat16 *output)

    .. note::
        The C++ source hard-codes the loop bound to 1024 elements.
        The ``tile_size`` parameter is used to build the Python
        ``arg_types`` and **must** match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024).  Compile-time
            — use ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the gelu kernel.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "gelu_bf16",
        "gelu.cc",
        [tile_ty, tile_ty],
    )


def silu(tile_size: int = 1024) -> ExternalFunction:
    """SiLU (Swish) activation kernel for bf16 tiles.

    Maps to ``silu_bf16`` exported from
    ``aie_kernels/<arch>/silu.cc``.

    Signature::

        void silu_bf16(bfloat16 *input, bfloat16 *output)

    .. note::
        The C++ source hard-codes the loop bound to 1024 elements.
        The ``tile_size`` parameter is used to build the Python
        ``arg_types`` and **must** match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024).  Compile-time
            — use ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the silu kernel.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "silu_bf16",
        "silu.cc",
        [tile_ty, tile_ty],
    )


def swiglu(tile_size: int = 1024) -> ExternalFunction:
    """SwiGLU gated activation kernel for bf16 tiles.

    Maps to ``swiglu_bf16`` exported from
    ``aie_kernels/<arch>/swiglu.cc``.

    Signature::

        void swiglu_bf16(bfloat16 *input, bfloat16 *weights_1,
                         bfloat16 *weights_2, bfloat16 *output)

    .. note::
        The C++ source hard-codes the loop bound to 1024 elements.
        The ``tile_size`` parameter is used to build the Python
        ``arg_types`` and **must** match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024).  Compile-time
            — use ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the swiglu
        kernel.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "swiglu_bf16",
        "swiglu.cc",
        [tile_ty, tile_ty, tile_ty, tile_ty],
    )


def bf16_exp(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise exponential kernel for bf16 tiles.

    Maps to ``exp_bf16_1024`` exported from
    ``aie_kernels/<arch>/bf16_exp.cc``.

    Signature::

        void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out)

    .. note::
        The C++ template is instantiated only for ``N = 1024``.  The
        ``tile_size`` parameter is used to build the Python ``arg_types``
        and **must** match ``1024`` for correct results.

    Args:
        tile_size: Number of elements per tile (must be 1024).  Compile-time
            — use ``Compile[int]`` in ``@iron.jit`` designs.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the bf16_exp
        kernel.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "exp_bf16_1024",
        "bf16_exp.cc",
        [tile_ty, tile_ty],
    )


# ---------------------------------------------------------------------------
# Matrix multiply (mm)
# ---------------------------------------------------------------------------

# Mapping from (input_dtype, output_dtype) to the C++ function name suffix
# and the compile-time -D flag that selects only that combo.
_MM_COMBOS = {
    (np.int8, np.int8): ("i8_i8", "i8_i8_ONLY"),
    (np.int8, np.int16): ("i8_i16", "i8_i16_ONLY"),
    (np.int8, np.int32): ("i8_i32", "i8_i32_ONLY"),
    (np.int16, np.int16): ("i16_i16", "i16_i16_ONLY"),
    (np.int16, np.int32): ("i16_i32", "i16_i32_ONLY"),
    (bfloat16, bfloat16): ("bf16_bf16", "bf16_bf16_ONLY"),
    (bfloat16, np.float32): ("bf16_f32", "bf16_f32_ONLY"),
}


def mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    vectorized: bool = True,
) -> ExternalFunction:
    """Matrix-multiply kernel: C += A * B.

    Maps to ``matmul_{in}_{out}`` (vectorized) or
    ``matmul_scalar_{in}_{out}`` (scalar) exported from
    ``aie_kernels/<arch>/mm.cc``, plus the companion ``zero_{out}``
    function for clearing the accumulator.

    Signature::

        void matmul_i16_i16(int16 *a_in, int16 *b_in, int16 *c_out)

    Tile sizes are controlled by ``-DDIM_M``, ``-DDIM_K``, ``-DDIM_N``
    compile flags (default 64 each).

    Args:
        dim_m: Number of rows of A / C.
        dim_k: Number of columns of A / rows of B.
        dim_n: Number of columns of B / C.
        input_dtype: Input element type (``np.int8``, ``np.int16``, or
            ``bfloat16``).
        output_dtype: Output element type.
        vectorized: If ``True`` use the vectorized variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the matmul
        kernel.

    Raises:
        ValueError: When ``(input_dtype, output_dtype)`` is not a supported
            combination.
    """
    key = (input_dtype, output_dtype)
    if key not in _MM_COMBOS:
        supported = ", ".join(
            f"({k[0].__name__ if hasattr(k[0], '__name__') else k[0]}, "
            f"{k[1].__name__ if hasattr(k[1], '__name__') else k[1]})"
            for k in _MM_COMBOS
        )
        raise ValueError(
            f"mm(): unsupported (input_dtype, output_dtype) = {key}. "
            f"Supported combos: {supported}"
        )

    suffix, only_flag = _MM_COMBOS[key]
    prefix = "matmul" if vectorized else "matmul_scalar"
    func_name = f"{prefix}_{suffix}"

    arch = _detect_arch()
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[input_dtype]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[input_dtype]]
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]

    source = _kernel_source(arch, arch, "mm.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
            f"-D{only_flag}",
        ],
    )


def mm_zero(
    dim_m: int = 64,
    dim_n: int = 64,
    output_dtype=np.int16,
    vectorized: bool = True,
) -> ExternalFunction:
    """Zero-fill kernel companion for :func:`mm`.

    Maps to ``zero_{out}`` (vectorized) or ``zero_scalar_{out}`` (scalar)
    exported from ``aie_kernels/<arch>/mm.cc``.

    Signature::

        void zero_i16(int16 *c_out)

    Args:
        dim_m: Number of rows.
        dim_n: Number of columns.
        output_dtype: Element type of the output matrix.
        vectorized: If ``True`` use the vectorized variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the zero kernel.

    Raises:
        ValueError: When ``output_dtype`` is not supported.
    """
    _dtype_suffix_map = {
        np.int8: "i8",
        np.int16: "i16",
        np.int32: "i32",
        np.float32: "f32",
        bfloat16: "bf16",
    }
    if output_dtype not in _dtype_suffix_map:
        raise ValueError(
            f"mm_zero(): unsupported output_dtype {output_dtype}. "
            f"Supported: {list(_dtype_suffix_map.keys())}"
        )

    suffix = _dtype_suffix_map[output_dtype]
    prefix = "zero" if vectorized else "zero_scalar"
    func_name = f"{prefix}_{suffix}"

    arch = _detect_arch()
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]

    # We need to pick a valid combo flag; use a matching input dtype.
    _combo_for_out = {
        np.int8: "i8_i8_ONLY",
        np.int16: "i16_i16_ONLY",
        np.int32: "i16_i32_ONLY",
        np.float32: "bf16_f32_ONLY",
        bfloat16: "bf16_bf16_ONLY",
    }
    only_flag = _combo_for_out[output_dtype]

    source = _kernel_source(arch, arch, "mm.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K=64",
            f"-DDIM_N={dim_n}",
            f"-D{only_flag}",
        ],
    )


# ---------------------------------------------------------------------------
# Matrix-vector multiply (mv)
# ---------------------------------------------------------------------------


def mv(
    dim_m: int = 32,
    dim_k: int = 32,
    input_dtype=np.int16,
    output_dtype=np.int32,
    vectorized: bool = True,
) -> ExternalFunction:
    """Matrix-vector multiply kernel: c += A * b.

    Maps to ``matvec_vectorized_i16_i32`` (vectorized) or
    ``matvec_scalar_i16_i32`` (scalar) exported from
    ``aie_kernels/aie2/mv.cc``.

    Signature::

        void matvec_vectorized_i16_i32(int16 *a_in, int16 *b_in, int32 *c_out)

    Args:
        dim_m: Number of rows of A (output vector length).
        dim_k: Number of columns of A (input vector length).
        input_dtype: Input element type. Only ``np.int16`` is supported.
        output_dtype: Output element type. Only ``np.int32`` is supported.
        vectorized: If ``True`` use the vectorized variant.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the matvec
        kernel.

    Raises:
        ValueError: When the dtype combination is not supported.
    """
    if input_dtype != np.int16 or output_dtype != np.int32:
        raise ValueError(
            f"mv(): only (np.int16, np.int32) is supported, "
            f"got ({input_dtype}, {output_dtype})"
        )

    prefix = "matvec_vectorized" if vectorized else "matvec_scalar"
    func_name = f"{prefix}_i16_i32"

    arch = _detect_arch()
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[np.int16]]
    b_ty = np.ndarray[(dim_k,), np.dtype[np.int16]]
    c_ty = np.ndarray[(dim_m,), np.dtype[np.int32]]

    source = _kernel_source(arch, arch, "mv.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}"],
    )


# ---------------------------------------------------------------------------
# Cascade matrix multiply (cascade_mm)
# ---------------------------------------------------------------------------


def cascade_mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    cascade_mode: str = "get_only",
) -> ExternalFunction:
    """Cascade matrix-multiply kernel for multi-core accumulation.

    Maps to ``matmul_scalar_cascade_{mode}_{in}_{out}`` exported from
    ``aie_kernels/aie2/cascade_mm.cc``.

    Available cascade modes:

    - ``"put_only"`` — computes partial result and sends via cascade.
    - ``"get_only"`` — receives cascade data, adds local result, writes out.
    - ``"put_get"``  — receives cascade, adds local result, sends cascade.

    Signature::

        void matmul_scalar_cascade_get_only_i16_i16(int16 *a, int16 *b, int16 *c)

    Args:
        dim_m: Number of rows of A / C.
        dim_k: Number of columns of A / rows of B.
        dim_n: Number of columns of B / C.
        input_dtype: Input element type.
        output_dtype: Output element type.
        cascade_mode: One of ``"put_only"``, ``"get_only"``, ``"put_get"``.

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the cascade
        matmul kernel.

    Raises:
        ValueError: When the cascade_mode or dtype combination is not
            supported.
    """
    _CASCADE_COMBOS = {
        (np.int16, np.int16): "i16_i16",
        (np.int16, np.int32): "i16_i32",
        (bfloat16, bfloat16): "bf16_bf16",
        (bfloat16, np.float32): "bf16_f32",
    }
    valid_modes = ("put_only", "get_only", "put_get")
    if cascade_mode not in valid_modes:
        raise ValueError(
            f"cascade_mm(): cascade_mode must be one of {valid_modes}, "
            f"got '{cascade_mode}'"
        )
    key = (input_dtype, output_dtype)
    if key not in _CASCADE_COMBOS:
        raise ValueError(
            f"cascade_mm(): unsupported (input_dtype, output_dtype) = {key}. "
            f"Supported: {list(_CASCADE_COMBOS.keys())}"
        )

    suffix = _CASCADE_COMBOS[key]
    func_name = f"matmul_scalar_cascade_{cascade_mode}_{suffix}"

    arch = _detect_arch()
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[input_dtype]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[input_dtype]]
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]

    source = _kernel_source(arch, arch, "cascade_mm.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
        ],
    )


# ---------------------------------------------------------------------------
# conv2dk1 — 1x1 convolution
# ---------------------------------------------------------------------------


def conv2dk1(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """1x1 convolution kernel.

    Maps to ``conv2dk1_i8`` (``INT8_ACT``) or ``conv2dk1_ui8``
    (``UINT8_ACT``) exported from ``aie_kernels/<arch>/conv2dk1.cc``.

    Signature::

        void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                         int32_t input_width, int32_t input_channels,
                         int32_t output_channels, int scale)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the conv2dk1
        kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1", act_dtype, factory_name="conv2dk1"
    )

    in_ty = np.ndarray[
        (input_width * input_channels,), np.dtype[act_dtype]
    ]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.uint8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, arch, "conv2dk1.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


# ---------------------------------------------------------------------------
# conv2dk3 — 3x3 convolution
# ---------------------------------------------------------------------------


def conv2dk3(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """3x3 convolution kernel.

    Maps to ``conv2dk3_i8`` (``INT8_ACT``) or ``conv2dk3_ui8``
    (``UINT8_ACT``) exported from ``aie_kernels/<arch>/conv2dk3.cc``.

    Signature::

        void conv2dk3_i8(int8_t *line0, int8_t *line1, int8_t *line2,
                         int8_t *wts, uint8_t *output,
                         int32_t input_width, int32_t input_channels,
                         int32_t output_channels, int32_t kernel_width,
                         int32_t kernel_height, int32_t check,
                         int scale, int channel_offset)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        :class:`~aie.iron.ExternalFunction` configured for the conv2dk3
        kernel.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk3", act_dtype, factory_name="conv2dk3"
    )

    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[act_dtype]]
    wt_ty = np.ndarray[
        (3 * 3 * input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.uint8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, arch, "conv2dk3.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[
            line_ty, line_ty, line_ty, wt_ty, out_ty,
            np.int32, np.int32, np.int32, np.int32,
            np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


# ---------------------------------------------------------------------------
# conv2dk1_skip — 1x1 convolution with skip connection
# ---------------------------------------------------------------------------


def conv2dk1_skip(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """1x1 convolution kernel with skip (residual) connection.

    Maps to ``conv2dk1_skip_i8`` (``INT8_ACT``) or ``conv2dk1_skip_ui8``
    (``UINT8_ACT``) exported from ``aie_kernels/aie2/conv2dk1_skip.cc``.

    Signature::

        void conv2dk1_skip_i8(uint8_t *input0, uint8_t *input1,
                              int8_t *kernels, uint8_t *output,
                              int8_t *skip, int32_t input_width,
                              int32_t input_channels,
                              int32_t output_channels,
                              int scale, int skip_scale)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).
            ``np.int8`` selects the ``INT8_ACT`` variant where skip is
            ``int8_t``; ``np.uint8`` selects ``UINT8_ACT`` where skip is
            ``uint8_t``.

    Returns:
        :class:`~aie.iron.ExternalFunction`.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1_skip", act_dtype, factory_name="conv2dk1_skip"
    )

    half_ch = input_channels // 2
    in0_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    in1_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.uint8]
    ]
    skip_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[act_dtype]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "conv2dk1_skip.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[
            in0_ty, in1_ty, wt_ty, out_ty, skip_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


# ---------------------------------------------------------------------------
# conv2dk1_i8 — 1x1 convolution (int8 in, int8 out)
# ---------------------------------------------------------------------------


def conv2dk1_i8(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """1x1 convolution kernel with int8 activations/weights/output.

    Maps to ``conv2dk1_i8`` exported from
    ``aie_kernels/<arch>/conv2dk1_i8.cc``.  This kernel exists in both
    ``aie2/`` and ``aie2p/`` directories.

    Signature::

        void conv2dk1_i8(int8_t *input, int8_t *kernels, int8_t *output,
                         int32_t input_width, int32_t input_channels,
                         int32_t output_channels, int scale)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        :class:`~aie.iron.ExternalFunction`.
    """
    in_ty = np.ndarray[
        (input_width * input_channels,), np.dtype[np.int8]
    ]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.int8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, arch, "conv2dk1_i8.cc")
    return ExternalFunction(
        "conv2dk1_i8",
        source_file=str(source),
        arg_types=[in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=["-DINT8_ACT"],
    )


# ---------------------------------------------------------------------------
# conv2dk14 — 14x14 convolution (aie2p only)
# ---------------------------------------------------------------------------


def conv2dk14(
    input_width: int = 224,
    input_channels: int = 16,
    output_channels: int = 16,
    kernel_width: int = 14,
) -> ExternalFunction:
    """14x14 convolution kernel (aie2p only).

    Maps to ``conv2dk14_i8`` exported from
    ``aie_kernels/aie2p/conv2dk14.cc``.

    Signature::

        void conv2dk14_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                          int32_t input_width, int32_t input_channels,
                          int32_t output_channels, int32_t kernel_width,
                          int scale)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        kernel_width: Width (and height) of the convolution kernel.

    Returns:
        :class:`~aie.iron.ExternalFunction`.
    """
    tiles = input_width // kernel_width
    pixels = kernel_width * kernel_width
    in_ty = np.ndarray[
        (tiles * pixels * 4,), np.dtype[np.uint8]
    ]
    wt_ty = np.ndarray[
        (output_channels * pixels * 4,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (output_channels * tiles * 8,), np.dtype[np.int8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2p", "conv2dk14.cc")
    return ExternalFunction(
        "conv2dk14_i8",
        source_file=str(source),
        arg_types=[
            in_ty, wt_ty, out_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=_include_dirs(),
    )


# ---------------------------------------------------------------------------
# conv2dk1_skip_init — 1x1 conv with skip + initial accumulation
# ---------------------------------------------------------------------------


def conv2dk1_skip_init(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    act_dtype=np.int8,
) -> ExternalFunction:
    """1x1 convolution kernel with skip-init connection.

    Maps to ``conv2dk1_skip_init_i8`` (``INT8_ACT``) or
    ``conv2dk1_skip_init_ui8`` (``UINT8_ACT``) exported from
    ``aie_kernels/aie2/conv2dk1_skip_init.cc``.

    Signature::

        void conv2dk1_skip_init_i8(uint8_t *input0, uint8_t *input1,
                                   int8_t *kernels, uint8_t *output,
                                   int8_t *skip, int32_t input_width,
                                   int32_t input_channels,
                                   int32_t output_channels,
                                   int32_t input_channels_skip,
                                   int scale, int skip_scale,
                                   int scale_skip_conv)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        act_dtype: Activation data type (``np.int8`` or ``np.uint8``).

    Returns:
        :class:`~aie.iron.ExternalFunction`.

    Raises:
        ValueError: When ``act_dtype`` is not ``np.int8`` or ``np.uint8``.
    """
    func_name, flags = _conv_act_dtype_info(
        "conv2dk1_skip_init", act_dtype, factory_name="conv2dk1_skip_init"
    )

    half_ch = input_channels // 2
    in0_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    in1_ty = np.ndarray[(input_width * half_ch,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.uint8]
    ]
    skip_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[act_dtype]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "conv2dk1_skip_init.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[
            in0_ty, in1_ty, wt_ty, out_ty, skip_ty,
            np.int32, np.int32, np.int32, np.int32,
            np.int32, np.int32, np.int32,
        ],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


# ---------------------------------------------------------------------------
# Bottleneck convolution kernels (aie2/bottleneck/)
# ---------------------------------------------------------------------------


def bn_conv2dk1_relu(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """Bottleneck 1x1 conv + ReLU kernel (int8 in, uint8 out).

    Maps to ``conv2dk1_relu_i8_ui8`` exported from
    ``aie_kernels/aie2/bottleneck/bn_conv2dk1_relu.cc``.

    Signature::

        void conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels,
                                  uint8_t *output, int32_t input_width,
                                  int32_t input_channels,
                                  int32_t output_channels, int scale)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        :class:`~aie.iron.ExternalFunction`.
    """
    in_ty = np.ndarray[
        (input_width * input_channels,), np.dtype[np.int8]
    ]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.uint8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk1_relu.cc")
    return ExternalFunction(
        "conv2dk1_relu_i8_ui8",
        source_file=str(source),
        arg_types=[in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32],
        include_dirs=_include_dirs(),
    )


def bn_conv2dk3(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """Bottleneck 3x3 conv with stride-2 kernel (int8 in, uint8 out).

    Maps to ``conv2dk3_stride2_i8`` exported from
    ``aie_kernels/aie2/bottleneck/bn_conv2dk3.cc``.

    Signature::

        void conv2dk3_stride2_i8(int8_t *line0, int8_t *line1, int8_t *line2,
                                 int8_t *wts, uint8_t *output,
                                 int32_t input_width, int32_t input_channels,
                                 int32_t output_channels,
                                 int32_t kernel_width, int32_t kernel_height,
                                 int32_t check, int scale,
                                 int channel_offset)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        :class:`~aie.iron.ExternalFunction`.
    """
    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[np.int8]]
    wt_ty = np.ndarray[
        (3 * 3 * input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.uint8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk3.cc")
    return ExternalFunction(
        "conv2dk3_stride2_i8",
        source_file=str(source),
        arg_types=[
            line_ty, line_ty, line_ty, wt_ty, out_ty,
            np.int32, np.int32, np.int32, np.int32,
            np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=_include_dirs(),
    )


def bn_conv2dk1_i8(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
) -> ExternalFunction:
    """Bottleneck 1x1 conv kernel (uint8 in, int8 out).

    Maps to ``conv2dk1_ui8_i8`` exported from
    ``aie_kernels/aie2/bottleneck/bn_conv2dk1_i8.cc``.

    Signature::

        void conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels,
                             int8_t *output, int32_t input_width,
                             int32_t input_channels,
                             int32_t output_channels, int scale)

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.

    Returns:
        :class:`~aie.iron.ExternalFunction`.
    """
    in_ty = np.ndarray[
        (input_width * input_channels,), np.dtype[np.uint8]
    ]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.int8]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk1_i8.cc")
    return ExternalFunction(
        "conv2dk1_ui8_i8",
        source_file=str(source),
        arg_types=[in_ty, wt_ty, out_ty, np.int32, np.int32, np.int32, np.int32],
        include_dirs=_include_dirs(),
    )


def bn_conv2dk1_skip(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    skip_dtype=np.uint8,
) -> ExternalFunction:
    """Bottleneck 1x1 conv with skip connection (uint8 in).

    Maps to ``conv2dk1_skip_ui8_ui8_i8`` or ``conv2dk1_skip_ui8_i8_i8``
    exported from ``aie_kernels/aie2/bottleneck/bn_conv2dk1_skip.cc``.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        skip_dtype: Skip connection data type (``np.uint8`` or ``np.int8``).

    Returns:
        :class:`~aie.iron.ExternalFunction`.

    Raises:
        ValueError: When ``skip_dtype`` is not ``np.uint8`` or ``np.int8``.
    """
    if skip_dtype == np.uint8:
        func_name = "conv2dk1_skip_ui8_ui8_i8"
    elif skip_dtype == np.int8:
        func_name = "conv2dk1_skip_ui8_i8_i8"
    else:
        raise ValueError(
            f"bn_conv2dk1_skip(): skip_dtype must be np.uint8 or np.int8, "
            f"got {skip_dtype}"
        )

    in_ty = np.ndarray[
        (input_width * input_channels,), np.dtype[np.uint8]
    ]
    wt_ty = np.ndarray[
        (input_channels * output_channels,), np.dtype[np.int8]
    ]
    out_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[np.int8]
    ]
    skip_ty = np.ndarray[
        (input_width * output_channels,), np.dtype[skip_dtype]
    ]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk1_skip.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[
            in_ty, wt_ty, out_ty, skip_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        include_dirs=_include_dirs(),
    )


def bn_conv2dk3_dw(
    input_width: int = 32,
    input_channels: int = 64,
    output_channels: int = 64,
    stride: int = 1,
) -> ExternalFunction:
    """Bottleneck depthwise 3x3 conv + ReLU kernel (uint8 in/out).

    Maps to ``conv2dk3_dw_stride{1|2}_relu_ui8_ui8`` exported from
    ``aie_kernels/aie2/bottleneck/bn_conv2dk3_dw.cc``.

    Args:
        input_width: Spatial width of the input.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        stride: Convolution stride (1 or 2).

    Returns:
        :class:`~aie.iron.ExternalFunction`.

    Raises:
        ValueError: When ``stride`` is not 1 or 2.
    """
    if stride not in (1, 2):
        raise ValueError(
            f"bn_conv2dk3_dw(): stride must be 1 or 2, got {stride}"
        )

    func_name = f"conv2dk3_dw_stride{stride}_relu_ui8_ui8"

    line_size = input_width * input_channels
    line_ty = np.ndarray[(line_size,), np.dtype[np.uint8]]
    wt_ty = np.ndarray[
        (3 * 3 * input_channels,), np.dtype[np.int8]
    ]
    out_size = (input_width // stride) * output_channels
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]

    arch = _detect_arch()
    source = _kernel_source(arch, "aie2", "bottleneck/bn_conv2dk3_dw.cc")

    if stride == 1:
        # stride1 has output_channels_split args
        return ExternalFunction(
            func_name,
            source_file=str(source),
            arg_types=[
                line_ty, line_ty, line_ty, wt_ty, out_ty, out_ty,
                np.int32, np.int32, np.int32, np.int32,
                np.int32, np.int32, np.int32, np.int32,
            ],
            include_dirs=_include_dirs(),
        )
    else:
        return ExternalFunction(
            func_name,
            source_file=str(source),
            arg_types=[
                line_ty, line_ty, line_ty, wt_ty, out_ty,
                np.int32, np.int32, np.int32, np.int32,
                np.int32, np.int32, np.int32, np.int32,
            ],
            include_dirs=_include_dirs(),
        )


__all__ = [
    "passthrough",
    "scale",
    "add",
    "mul",
    "reduce_add",
    "reduce_min",
    "reduce_max",
    "relu",
    "rgba2hue",
    "threshold",
    "bitwiseOR",
    "bitwiseAND",
    "gray2rgba",
    "rgba2gray",
    "filter2d",
    "addWeighted",
    "softmax",
    "gelu",
    "silu",
    "swiglu",
    "bf16_exp",
    "mm",
    "mm_zero",
    "mv",
    "cascade_mm",
    "conv2dk1",
    "conv2dk3",
    "conv2dk1_skip",
    "conv2dk1_i8",
    "conv2dk14",
    "conv2dk1_skip_init",
    "bn_conv2dk1_relu",
    "bn_conv2dk3",
    "bn_conv2dk1_i8",
    "bn_conv2dk1_skip",
    "bn_conv2dk3_dw",
]
