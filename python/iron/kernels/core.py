# core.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``set_rounding``: the core's rounding-mode register.

The AIE core narrows accumulators (an ``srs`` shift, a bf16 store) in the
rounding mode its mode register holds, and a fresh core boots in ``floor``.
A kernel that needs another mode names a setter as its contract's ``setup``,
and a design calls it once before the kernel's first call; the mode persists
on that core until something changes it.
"""

from enum import Enum
from functools import partial

from aie.iron.kernel import ExternalFunction

from ._common import KernelContract, Trace, _kernel_source, _make_extern


class RoundingMode(str, Enum):
    """An ``aie::rounding_mode``, named as the C++ enumerator is."""

    FLOOR = "floor"
    CEIL = "ceil"
    POSITIVE_INF = "positive_inf"
    NEGATIVE_INF = "negative_inf"
    SYMMETRIC_INF = "symmetric_inf"
    SYMMETRIC_ZERO = "symmetric_zero"
    CONV_EVEN = "conv_even"
    CONV_ODD = "conv_odd"

    def __str__(self):
        return self.value


def set_rounding(mode: RoundingMode = RoundingMode.CONV_EVEN) -> ExternalFunction:
    """Set the core's rounding mode using always-inline, merge-linked LLVM IR.

    Args:
        mode: The [`RoundingMode`][iron.kernels.core.RoundingMode] to set.

    Returns:
        ExternalFunction ``set_rounding_<mode>``, which takes no arguments.
    """
    mode = RoundingMode(mode)
    return _make_extern(
        f"set_rounding_{mode}",
        _kernel_source("core/set_rounding.cc"),
        [],
        compile_flags=[f"-DROUNDING_MODE={mode}"],
        inline=True,
        # Sets core state and has no data arguments.
        contract=KernelContract(
            roles=(), trace=Trace.none("runs once before the calls; nothing to time")
        ),
    )


#: Round ties to even, the mode numpy's float casts use. The ``setup`` of
#: every contract judged against a numpy reference that rounds that way.
conv_even = partial(set_rounding, RoundingMode.CONV_EVEN)
