# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .tap import TensorAccessPattern


@dataclass(frozen=True)
class BdLimits:
    """What one DMA buffer descriptor holds, on one tile.

    A descriptor has four dimensions, outermost first ``[iteration, d2, d1,
    d0]``. Whether an access pattern fits one is ``AIEX::verifyStridesWraps``,
    stated here over the pattern so a design can choose its transfers before
    it builds them. A constant pattern that does not fit is split by the
    compiler (``aie-decompose-large-dma-bd``); one with a runtime offset, size
    or repeat must fit as given.

    A device gives the limits of each of its tiles (``Device.bd_limits``).
    """

    wrap: int
    """The largest ``d0`` (in granules) and ``d1`` (in elements)."""
    step: int
    """The largest stride, in granules."""
    iterations: int
    """The most times the iteration dimension runs."""
    granule_bytes: int
    """The unit of addressing: offsets, ``d0`` and strides are whole granules."""
    linear: bool
    """Whether a contiguous pattern is one transfer whatever its length (a
    shim tile's buffer length field), exempt from ``wrap``."""

    @staticmethod
    def slots(sizes: Sequence[Any], strides: Sequence[Any] | None):
        """``sizes`` and ``strides`` padded to a descriptor's four dimensions.

        Unit dimensions go in front, except after a leading re-read (stride
        0, size above 1): a stride of 0 is only encodable in the iteration
        dimension, so a re-read stays outermost. More than four dimensions
        are returned as they are. A size or stride may be a runtime value,
        which is taken as no re-read.

        Returns:
            tuple[list, list | None]: The sizes and strides, outermost first
        """

        def constant(v):
            return isinstance(v, (int, np.integer))

        sizes = list(sizes)
        strides = None if strides is None else list(strides)
        reread = (
            strides is not None
            and 2 <= len(sizes) < 4
            and constant(sizes[0])
            and sizes[0] > 1
            and constant(strides[0])
            and strides[0] == 0
        )
        at = 1 if reread else 0
        while len(sizes) < 4:
            sizes.insert(at, 1)
            if strides is not None:
                strides.insert(at, 0)
        return sizes, strides

    def granule(self, dtype) -> int:
        """Elements of ``dtype`` in one granule.

        Raises:
            ValueError: An element does not divide the granule
        """
        itemsize = np.dtype(dtype).itemsize
        if self.granule_bytes % itemsize:
            raise ValueError(
                f"{np.dtype(dtype)} does not divide the {self.granule_bytes}-byte granule"
            )
        return self.granule_bytes // itemsize

    def factor(self, run: int, granule: int = 1) -> tuple[int, int] | None:
        """A contiguous run of ``run`` elements as ``(d1, d0)``, or None.

        ``d0`` is a whole number of ``granule``-element granules, at most
        ``wrap`` of them; ``d1`` is at most ``wrap``. With the default granule
        of one element, ``d0`` is bounded in elements, as ``d1`` is. The
        largest ``d0`` that divides the run is taken.
        """
        largest = self.wrap * granule
        if run <= largest and run % granule == 0:
            return (1, run)
        for d0 in range(largest - largest % granule, 0, -granule):
            if run % d0 == 0 and run // d0 <= self.wrap:
                return (run // d0, d0)
        return None

    def fits(self, tap: TensorAccessPattern, dtype) -> bool:
        """Whether one descriptor holds ``tap`` over elements of ``dtype``.

        The pattern is taken as the compiler lowers it: in the four
        dimensions ``shim_dma_single_bd_task`` gives it (:meth:`slots`), and,
        when it does not iterate, with its unit dimensions dropped and a
        contiguous remainder made one linear transfer
        (``aie-normalize-dma-bd-dims``).
        """
        itemsize = np.dtype(dtype).itemsize

        def granules(elements):
            if elements * itemsize % self.granule_bytes:
                return None
            return elements * itemsize // self.granule_bytes

        if granules(tap.offset) is None:
            return False
        sizes, strides = self.slots(tap.sizes, tap.strides)
        assert strides is not None
        if int(np.prod(sizes[:-3])) == 1:
            if tap.contiguous:
                return granules(int(np.prod(sizes))) is not None
            kept = [(n, s) for n, s in zip(sizes, strides) if n != 1]
            kept = [(1, 0)] * (4 - len(kept)) + kept
            sizes, strides = [n for n, _ in kept], [s for _, s in kept]
        if len(sizes) > 4:
            return False
        (it, it_s), (d2, d2_s), (d1, d1_s), (d0, d0_s) = zip(sizes, strides)

        if granules(d0) is None:
            return False
        # Only the iteration dimension may step by 0 (a re-read).
        if any(n > 1 and s < 1 for n, s in zip(sizes[1:], strides[1:])):
            return False
        # Every stride is whole granules, even one never applied, except a
        # unit innermost stride: d0 is whole granules already.
        if d0_s != 1 and granules(d0_s) is None:
            return False
        if any(granules(s) is None for s in (it_s, d2_s, d1_s)):
            return False

        linear = d1 == 1 and d2 == 1 and d0_s == 1
        contiguous = (
            d0_s == 1 and (d1 == 1 or d1_s == d0) and (d2 == 1 or d2_s == d0 * d1)
        )
        if not (linear or (self.linear and contiguous)):
            if granules(d0) > self.wrap or d1 > self.wrap:
                return False
        if it > self.iterations:
            return False
        applied = [(d2, d2_s), (d1, d1_s), (it, it_s)]
        # d0 steps by one granule in hardware when its stride is within one,
        # or when an element is wider than a granule.
        if self.granule_bytes <= d0_s * itemsize and itemsize <= self.granule_bytes:
            applied.append((d0, d0_s))
        return all(granules(s) <= self.step for n, s in applied if n > 1 and s > 0)
