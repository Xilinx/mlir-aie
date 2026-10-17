#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A router hazard case router_hw.py runs on an NPU.

A case module in hw_cases/ defines `Case`, an HwCase: one design with its
packet flows left to the router, the same design with them pinned as
switchbox configuration in ways that should or should not deadlock, and the
output every run that completes has to produce.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Variant:
    """One way to route a case's design.

    Args:
        name: Names the variant, and its verdict file `<case>_<name>.mlir`.
        design: The design as the router gets it, with every flow requested.
        pinned: The design with some packet trees as switchbox configuration
            in place of their flows; None to let the router route `design`.
        expect: "PASS" or "HANG", what the variant should do on hardware.
        options: --aie-create-pathfinder-flows options to route it with.
        note: One sentence on what the routing does, for the verdict file.
    """

    name: str
    design: str
    pinned: str | None
    expect: str
    options: str
    note: str


class HwCase:
    """A design, the ways to route it, and the output it has to produce.

    Subclasses set `name`, `description` and `outputs`, the element count and
    dtype of each runtime sequence argument, all outputs, in order.
    """

    name: str
    description: str
    outputs: list[tuple[int, type]]

    def variants(self) -> list[Variant]:
        """Returns:
        Every variant of the case, the router's first.
        """
        raise NotImplementedError

    def check(self, outputs: list[np.ndarray]) -> list[str]:
        """Compare one completed run's outputs with the reference.

        Args:
            outputs: Each runtime sequence argument as the run left it.

        Returns:
            One line per mismatch; empty when the outputs are right.
        """
        raise NotImplementedError
