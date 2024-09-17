"""
TODO: 
* docs
* types
* logical?
"""

import sys
from typing import Callable, Optional, Union

from .. import ir
from ..dialects.aie import core
from ..dialects.scf import yield_, for_
from .phys.tile import MyTile
from .dataflow.objectfifo import ObjectFifoHandle
from .dataflow.endpoint import MyObjectFifoEndpoint
from .kernels.kernel import MyKernel


class MyWorker(MyObjectFifoEndpoint):
    def __init__(
        self,
        core_fn: Optional[Callable[[Union[ObjectFifoHandle, MyKernel]], None]],
        fn_args: list[Union[ObjectFifoHandle, MyKernel]] = [],
        coords: tuple[int, int] = None,
    ):
        column, row = coords
        self.tile = MyTile(column, row)
        if core_fn is None:

            def do_nothing_core_fun() -> None:
                for _ in for_(sys.maxsize):
                    yield_([])

            self.core_fn = do_nothing_core_fun
        else:
            self.core_fn = core_fn
        self.link_with: Optional[str] = None
        self.fn_args = fn_args
        bin_names = set()

        for arg in self.fn_args:
            if isinstance(arg, MyKernel):
                bin_names.add(arg.bin_name)
            elif isinstance(arg, ObjectFifoHandle):
                arg.set_endpoint(self)

        assert len(bin_names) <= 1, "Right now only link with one bin"
        if len(bin_names) == 1:
            self.link_with = list(bin_names)[0]

    def get_tile(self) -> MyTile:
        assert self.tile != None
        return self.tile

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        my_tile = self.tile.op
        my_link = self.link_with

        @core(my_tile, my_link)
        def core_body():
            self.core_fn(*self.fn_args)
