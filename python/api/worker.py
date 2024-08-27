"""
TODO: 
* docs
* types
* producer/consumer
* join/distribute
* logical?
"""

from .. import ir
from ..dialects.aie import core
from .resolvable import Resolvable
from .phys.tile import MyTile
from .dataflow.objectfifo import MyObjectFifo
from .kernels.kernel import MyKernel


class MyWorker(Resolvable):
    def __init__(
        self,
        core_fn,
        ofs_end1: list[MyObjectFifo] = [],
        ofs_end2: list[MyObjectFifo] = [],
        external_functions: list[MyKernel] = [],
        coords: tuple[int, int] = None,
    ):
        column, row = coords
        self.tile = MyTile(column, row)
        self.core_fn = core_fn

        assert isinstance(external_functions, list)
        bin_names = set()
        for e in external_functions:
            assert isinstance(e, MyKernel)
            bin_names.add(e.bin_name)
        assert len(bin_names) <= 1, "Right now only link with one bin"
        if len(bin_names) == 1:
            self.link_with = list(bin_names)[0]
        self.external_functions = external_functions

        self.ofs_end1 = ofs_end1
        for of in self.ofs_end1:
            assert isinstance(of, MyObjectFifo), "ofs_end1 must be List[ObjectFifo]"
            of.set_endpoint(self, True)

        self.ofs_end2 = ofs_end2
        for of in self.ofs_end2:
            assert isinstance(of, MyObjectFifo), "ofs_end1 must be List[ObjectFifo]"
            of.set_endpoint(self, False)

    def get_tile(self):
        assert self.tile != None
        return self.tile.op

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        my_tile = self.tile.op
        my_link = self.link_with

        @core(my_tile, my_link)
        def core_body():
            self.core_fn(self.ofs_end1, self.ofs_end2, self.external_functions)
