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

from collections import Counter
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

    def check_packets(
        self, outputs: list[np.ndarray], wants: list[list], chunk: int
    ) -> list[str]:
        """Compare each output with the packets it has to hold, in any order.

        Packets from different sources reach a receiver in either order, so
        only the set of packets an output holds is fixed.

        Args:
            outputs: Each runtime sequence argument as the run left it.
            wants: For each output, the packets it has to hold once each.
            chunk: The element count of one packet.

        Returns:
            One line per output that differs; empty when the outputs are right.
        """
        problems = []
        for k, (got, packets) in enumerate(zip(outputs, wants, strict=True)):
            chunks = Counter(tuple(c) for c in got.reshape(-1, chunk))
            want = Counter(tuple(p) for p in packets)
            if chunks != want:
                missing = (want - chunks).total()
                unexpected = (chunks - want).total()
                problems.append(
                    f"output {k}: packets differ from the expected set "
                    f"({missing} missing, {unexpected} unexpected)"
                )
        return problems


class PacketCase(HwCase):
    """A case whose receivers forward the packets they get to the host, so
    each output holds a set of packets in any order, and whose one variant is
    the router's routing.

    Subclasses set `name`, `description`, `design` (the design as the router
    gets it), `meets` (what the router's routing does, for the verdict note),
    `chunk` (the element count of one packet) and `wants` (for each output,
    the packets it has to hold once each); `outputs` follows from `wants`.
    """

    design: str
    meets: str
    chunk: int
    wants: list[list[np.ndarray]]

    def __init__(self):
        self.outputs = [(len(w) * self.chunk, np.int32) for w in self.wants]

    def variants(self) -> list[Variant]:
        return [
            Variant(
                "router",
                self.design,
                None,
                "PASS",
                "",
                f"the router's routing without allow-deadlock-prone: {self.meets}.",
            )
        ]

    def check(self, outputs: list[np.ndarray]) -> list[str]:
        return self.check_packets(outputs, self.wants, self.chunk)
