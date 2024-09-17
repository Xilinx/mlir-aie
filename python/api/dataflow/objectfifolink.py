from typing import Optional

from ... import ir
from ...dialects._aie_ops_gen import ObjectFifoLinkOp
from ...dialects.aie import object_fifo_link

from ..tensor import MyTensorType
from ..phys.tile import MyTile
from .endpoint import MyObjectFifoEndpoint
from .objectfifo import ObjectFifoHandle

class MyObjectFifoLink(MyObjectFifoEndpoint):
    def __init__(
        self,
        seconds: list[ObjectFifoHandle] = [],
        firsts: list[ObjectFifoHandle] = [],
        coords: Optional[tuple[int, int]] = None,
    ):
        column, row = coords
        self.tile = MyTile(column, row)

        self.__seconds = []
        self.__firsts = []
        self.__op = None

        self.__obj_type = seconds[0].obj_type
        for s in seconds:
            assert s.obj_type == self.__obj_type
            s.set_endpoint(self)
            self.__seconds.append(s)
        for f in firsts:
            assert f.obj_type == self.__obj_type
            f.set_endpoint(self)
            self.__firsts.append(f)

    def get_tile(self) -> MyTile:
        assert self.tile != None
        return self.tile

    @property
    def op(self) -> ObjectFifoLinkOp:
        assert self.__op != None
        return self.__op

    @property
    def obj_type(self) -> MyTensorType:
        return self.__obj_type

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        if self.__op == None:
            self.__op = object_fifo_link(
                [s.op for s in self.__seconds],
                [f.op for f in self.__firsts],
                loc=loc,
                ip=ip,
            )
