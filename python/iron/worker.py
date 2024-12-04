# worker.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import contextvars
import sys
from typing import Callable

from .. import ir  # type: ignore
from ..dialects.aie import core
from ..helpers.dialects.ext.scf import _for as range_
from .phys.tile import PlacementTile, AnyComputeTile, Tile
from .dataflow.objectfifo import ObjectFifoHandle, ObjectFifo
from .dataflow.endpoint import ObjectFifoEndpoint
from .kernels.binkernel import BinKernel
from .kernels.kernel import Kernel
from .globalbuffer import GlobalBuffer
from .placeable import Placeable


class Worker(ObjectFifoEndpoint):
    current_core_placement = contextvars.ContextVar(
        "current_core_placement", default=None
    )

    def __init__(
        self,
        core_fn: Callable[[ObjectFifoHandle | Kernel], None] | None,
        fn_args: list[ObjectFifoHandle | Kernel] = [],
        placement: PlacementTile | None = AnyComputeTile,
        while_true: bool = True,
    ):
        self._tile = placement
        self._while_true = while_true
        if core_fn is None:

            def do_nothing_core_fun(*args) -> None:
                for _ in range_(sys.maxsize):
                    pass

            self.core_fn = do_nothing_core_fun
        else:
            self.core_fn = core_fn
        self.link_with: str | None = None
        self.fn_args = fn_args
        bin_names = set()
        self._fifos = []
        self._buffers = []

        for arg in self.fn_args:
            if isinstance(arg, BinKernel):
                bin_names.add(arg.bin_name)
            elif isinstance(arg, ObjectFifoHandle):
                arg.set_endpoint(self)
                self._fifos.append(arg)
            elif isinstance(arg, GlobalBuffer):
                self._buffers.append(arg)
            elif isinstance(arg, ObjectFifo):
                # This is an easy error to make, so we catch it early
                raise ValueError(
                    "Cannot give an ObjectFifo directly to a worker; "
                    "must give an ObjectFifoHandle obtained through "
                    "ObjectFifo.prod() or ObjectFifo.cons()"
                )
            # We assume other arguments are metaprogramming (e.g, Python args)
            # This could allow some errors to sink through, but we allow it for now.
            # TODO: this could be cleaned up through creation of a MetaArgs struct, so you
            # could access values through meta.my_var within the function.

        assert len(bin_names) <= 1, "Right now only link with one bin"
        if len(bin_names) == 1:
            self.link_with = list(bin_names)[0]

    def place(self, tile: Tile) -> None:
        ObjectFifoEndpoint.place(self, tile)
        for buffer in self._buffers:
            buffer.place(tile)

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self._fifos.copy()

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        assert self._tile != None
        my_tile = self._tile.op
        self.current_core_placement.set(my_tile)
        my_link = self.link_with

        @core(my_tile, my_link)
        def core_body():
            for _ in range_(sys.maxsize) if self._while_true else range(1):
                self.core_fn(*self.fn_args)

        self.current_core_placement.set(None)
