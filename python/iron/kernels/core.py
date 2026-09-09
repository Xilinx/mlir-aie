# core.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Core state kernels: ``set_rounding``.

The AIE core narrows accumulators (an ``srs`` shift, a bf16 store) in the
rounding mode its mode register holds, and a fresh core boots in ``floor``.
A kernel whose contract names a ``rounding_mode`` expects the design to have
set that mode before its first call; ``set_rounding`` is the kernel that
does so, and ``aie.utils.kernel_harness`` calls it for such contracts.
"""

from aie.iron.kernel import ExternalFunction

from ._common import ROUNDING_MODES, _default_source_path, _make_extern


def set_rounding(mode: str = "conv_even") -> ExternalFunction:
    """Kernel that sets the core's rounding mode to ``mode`` and returns.

    Call it once in a Worker before the first kernel whose contract's
    ``rounding_mode`` is ``mode``; the mode persists on that core until
    another kernel changes it. ``mode`` is an ``aie::rounding_mode`` name:
    ``floor``, ``ceil``, ``positive_inf``, ``negative_inf``,
    ``symmetric_inf``, ``symmetric_zero``, ``conv_even`` or ``conv_odd``.

    Args:
        mode: The ``aie::rounding_mode`` to set.

    Returns:
        ExternalFunction ``set_rounding_<mode>``, which takes no arguments.

    Raises:
        ValueError: When ``mode`` is not an ``aie::rounding_mode`` name.
    """
    if mode not in ROUNDING_MODES or mode in ("unspecified", "sets_own"):
        raise ValueError(
            f"set_rounding() mode must be an aie::rounding_mode name, got {mode!r}."
        )
    return _make_extern(
        f"set_rounding_{mode}",
        _default_source_path("set_rounding.cc", subdir="generic"),
        [],
        compile_flags=[f"-DROUNDING_MODE={mode}"],
    )
