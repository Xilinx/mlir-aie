# external_buffer.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Off-chip memory declared at device scope, at a fixed address."""

import copy
import itertools
from typing import Sequence

import numpy as np

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects.aie import external_buffer
from ..helpers.taplib import TensorAccessPattern
from ..helpers.util import (
    NpuDType,
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from .device import Tile
from .resolvable import NotResolvedError, Resolvable


class ExternalBuffer(Resolvable):
    """A region of off-chip (DDR) memory, named and addressed in the design itself.

    Peer of [`Buffer`][iron.Buffer], which names on-chip tile memory. Lowers to
    one `aie.external_buffer` op at device scope.

    The usual way to reach DDR is to make it a runtime-sequence argument, so the
    host supplies a buffer per dispatch and its address is patched in then. An
    ExternalBuffer is the opposite trade: the address is written into the design
    and fixed for every run. That suits a static `aie.shim_dma` program -- a
    shim DMA whose descriptors are configured once, with no runtime sequence
    issuing transfers -- and designs reproducing a specific hardware
    configuration. It does mean the design is bound to whatever is at that
    address, so nothing checks that a host allocation lives there.

    Register it with the [`Runtime`][iron.Runtime] (``rt.add_external_buffer``)
    so the Program emits it at device scope, the same way explicit
    [`Lock`][iron.Lock] and [`TileDma`][iron.TileDma] objects are registered.
    """

    # Used to generate unique names when none is provided during construction.
    _gbuf_index = itertools.count()

    def __init__(
        self,
        type: type[np.ndarray],
        address: int | None = None,
        name: str | None = None,
    ):
        """Declare a region of off-chip memory at the top level of the design.

        Args:
            type (type[np.ndarray]): The type of the buffer.
            address (int | None, optional): The address the buffer lives at.
                Defaults to None, which leaves the address off the op.
            name (str | None, optional): The symbol name of the buffer. If none
                is given, a unique name will be generated. Defaults to None.

        Raises:
            ValueError: If ``address`` is given but is not a non-negative int.
        """
        if address is not None:
            if not isinstance(address, int) or isinstance(address, bool):
                raise ValueError(
                    f"ExternalBuffer address must be an int, but got "
                    f"{address.__class__.__name__}"
                )
            if address < 0:
                raise ValueError(
                    f"ExternalBuffer address must be >= 0, but got {address}"
                )
        self._arr_type = type
        self._address = address
        self._name = name or f"ext_buf_{next(ExternalBuffer._gbuf_index)}"
        self._op = None
        # Off-chip memory sits on no tile. The attribute exists because a
        # TileDma's BDs are placed against their buffer's tile, and this is how
        # such a buffer says it needs no placing.
        self._tile: Tile | None = None
        # Set on a view returned by __getitem__: the buffer it is part of, which
        # owns the declaration both share.
        self._whole: "ExternalBuffer | None" = None
        self._tap: TensorAccessPattern | None = None

    def __getitem__(self, key) -> "ExternalBuffer":
        """Return the part of this buffer a numpy-style slice names.

        A slice of a buffer is still a buffer -- it can be copied from or to
        wherever the whole one can -- so it comes back as an ExternalBuffer
        carrying the access pattern the slice describes, sharing the one
        declaration with the buffer it came from.

        Nothing is allocated: the geometry is read off a zero-storage numpy view
        (see ``TensorAccessPattern.from_slice``).
        """
        if self._whole is not None:
            raise ValueError(
                f"{self._name} is already a slice. A second [] would measure "
                "against the whole buffer's shape rather than composing with "
                "the first, so slice the whole buffer once instead."
            )
        # Everything but the access pattern is the buffer being sliced -- type,
        # address, name, and the declaration itself (see op/resolve below).
        view = copy.copy(self)
        view._whole = self
        view._tap = TensorAccessPattern.from_slice(self.shape, key)
        return view

    @property
    def tap(self) -> TensorAccessPattern:
        """The part of the buffer this refers to; all of it unless sliced."""
        if self._tap is None:
            self._tap = TensorAccessPattern.from_slice(self.shape, np.s_[...])
        return self._tap

    @property
    def tile(self) -> Tile | None:
        """The tile this buffer is on, which for off-chip memory is none."""
        return self._tile

    def tiles(self) -> list:
        """Tile dependency for Program.resolve tile discovery: none."""
        return []

    @property
    def name(self) -> str:
        """The symbol name of the buffer."""
        return self._name

    @property
    def shape(self) -> Sequence[int]:
        """The shape of the buffer."""
        return np_ndarray_type_get_shape(self._arr_type)

    @property
    def dtype(self) -> NpuDType:
        """The per-element datatype of the buffer."""
        return np_ndarray_type_get_dtype(self._arr_type)

    @property
    def op(self):
        if self._whole is not None:
            return self._whole.op
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        # A view declares nothing of its own; the buffer it is part of does.
        if self._whole is not None:
            self._whole.resolve(loc, ip)
            return
        if not self._op:
            self._op = external_buffer(
                self._arr_type,  # pyright: ignore[reportCallIssue]
                name=self._name,
                address=self._address,
            )
