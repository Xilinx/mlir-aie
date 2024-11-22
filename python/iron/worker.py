"""
TODO: 
* docs
* types
* logical?
"""

import sys
from typing import Callable

from .. import ir  # type: ignore
from ..dialects.aie import core
from ..helpers.dialects.ext.scf import _for as range_
from .phys.tile import PlacementTile, AnyComputeTile, Tile
from .dataflow.objectfifo import ObjectFifoHandle
from .dataflow.endpoint import ObjectFifoEndpoint
from .kernels.binkernel import BinKernel
from .kernels.kernel import Kernel


class Worker(ObjectFifoEndpoint):
    def __init__(
        self,
        core_fn: Callable[[ObjectFifoHandle | Kernel], None] | None,
        fn_args: list[ObjectFifoHandle | Kernel] = [],
        placement: PlacementTile | None = AnyComputeTile,
        while_true: bool = False,
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

        for arg in self.fn_args:
            if isinstance(arg, BinKernel):
                bin_names.add(arg.bin_name)
            elif isinstance(arg, ObjectFifoHandle):
                arg.set_endpoint(self)
                self._fifos.append(arg)

        assert len(bin_names) <= 1, "Right now only link with one bin"
        if len(bin_names) == 1:
            self.link_with = list(bin_names)[0]

    @property
    def tile(self) -> PlacementTile:
        return self._tile

    def place(self, tile: Tile) -> None:
        assert not isinstance(
            self._tile, Tile
        ), f"Worker already placed at {self.tile}, cannot place {tile}"
        self._tile = tile

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self._fifos.copy()

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        assert self._tile != None
        my_tile = self._tile.op
        my_link = self.link_with

        @core(my_tile, my_link)
        def core_body():
            for _ in range_(sys.maxsize) if self._while_true else range(1):
                self.core_fn(*self.fn_args)
