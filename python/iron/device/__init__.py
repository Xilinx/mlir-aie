# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Device representations for supported AMD Ryzen AI NPU targets."""

from . import device as _device_module
from .device import (
    Device,
    NPU1,
    NPU1Col1,
    NPU1Col2,
    NPU1Col3,
    NPU2,
    NPU2Col1,
    NPU2Col2,
    NPU2Col3,
    NPU2Col4,
    NPU2Col5,
    NPU2Col6,
    NPU2Col7,
    XCVC1902,
)
from .tile import AnyShimTile, AnyMemTile, AnyComputeTile, Tile

# Map from CLI device name → base class name (max-column variant).
_NAME_TO_BASE: dict[str, str] = {
    "npu": "NPU1",
    "npu1": "NPU1",
    "npu2": "NPU2",
    "xcvc1902": "XCVC1902",
    "xcve2302": "XCVE2302",
    "xcve2802": "XCVE2802",
}

# Max-column count per NPU family.  Versal AIE1 devices (xcvc1902 etc.)
# have no per-column variants and are not listed here.
_MAX_COLS: dict[str, int] = {"NPU1": 4, "NPU2": 8}


def from_name(name: str, *, n_cols: int | None = 1) -> Device:
    """Resolve a device-name string (and optional column count) to a Device.

    Designed to consume the ``--dev`` CLI flag that almost every
    programming example exposes.  Replaces the per-example
    ``_device_for(dev_str)`` boilerplate.

    Args:
        name: One of:

          - ``"npu"`` / ``"npu1"`` — Phoenix/Hawk Point (AIE2)
          - ``"npu2"`` — Strix/Krackan (AIE2P)
          - ``"xcvc1902"`` / ``"xcve2302"`` / ``"xcve2802"`` — Versal AIE1

          Case-insensitive.
        n_cols: Number of columns to use for NPU devices.  ``1`` (the
          default) selects the single-column variant
          (``NPU1Col1`` / ``NPU2Col1``).  ``2..N-1`` selects
          ``NPU{1,2}Col{n_cols}``.  Passing the device's max
          (``4`` for NPU1, ``8`` for NPU2) — or ``None`` as a
          family-agnostic "max" sentinel — selects the unrestricted
          multi-column device (``NPU1`` / ``NPU2``).  Ignored for
          Versal devices.

    Returns:
        Device instance.

    Raises:
        ValueError: ``name`` unrecognized, or ``n_cols`` out of range
            for the chosen NPU family.
    """
    base = _NAME_TO_BASE.get(name.lower())
    if base is None:
        raise ValueError(
            f"Unknown device name {name!r}; expected one of " f"{sorted(_NAME_TO_BASE)}"
        )
    if base not in _MAX_COLS:
        # Versal AIE1 devices have no per-column variants.
        return getattr(_device_module, base)()
    max_cols = _MAX_COLS[base]
    if n_cols is None or n_cols == max_cols:
        cls_name = base  # e.g. "NPU1" / "NPU2"
    elif 1 <= n_cols < max_cols:
        cls_name = f"{base}Col{n_cols}"
    else:
        raise ValueError(
            f"n_cols must be in 1..{max_cols} (or None for max) for "
            f"{base}, got {n_cols}"
        )
    return getattr(_device_module, cls_name)()


def device_from_args(args, *, dev_attr: str = "dev") -> Device:
    """Resolve a parsed-argparse namespace to a Device.

    Collapses the boilerplate that every example repeats::

        dev = from_name(args.dev, n_cols=1 if args.dev == "npu" else None)

    into::

        dev = device_from_args(args)

    The single-column-for-npu1, full-width-for-npu2 default reflects the
    historic Phoenix vs Strix behaviour of the example suite.  Override by
    setting ``args.n_cols`` to a specific integer.

    Args:
        args: An ``argparse.Namespace`` with a ``.dev`` attribute (or
            whatever ``dev_attr`` names).  Optionally also a
            ``.n_cols`` attribute that overrides the default.
        dev_attr: Name of the device-string attribute on ``args``.
            Defaults to ``"dev"``.

    Returns:
        Device instance.
    """
    dev = getattr(args, dev_attr)
    n_cols = getattr(args, "n_cols", None)
    if n_cols is None:
        n_cols = 1 if dev == "npu" else None
    return from_name(dev, n_cols=n_cols)
