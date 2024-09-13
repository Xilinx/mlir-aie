from ... import ir

from ...dialects._aie_ops_gen import ObjectFifoLinkOp
from ...dialects.aie import object_fifo_link

from ..phys.tile import MyTile
from .endpoint import MyObjectFifoEndpoint
from .objectfifo import ObjectFifoHandle


class MyObjectFifoLink(MyObjectFifoEndpoint):
    def __init__(
        self,
        seconds: list[ObjectFifoHandle] = None,
        firsts: list[ObjectFifoHandle] = None,
        coords: tuple[int, int] = None,
    ):
        column, row = coords
        self.tile = MyTile(column, row)

        self.__seconds = []
        self.__firsts = []
        self.__op = None

        self.__memref_type = seconds[0].obj_type
        for s in seconds:
            assert isinstance(s, ObjectFifoHandle)
            assert s.obj_type == self.__memref_type
            s.set_endpoint(self)
            self.__seconds.append(s)
        for f in firsts:
            assert isinstance(f, ObjectFifoHandle)
            assert f.obj_type == self.__memref_type
            f.set_endpoint(self)
            self.__firsts.append(f)

    def get_tile(self) -> MyTile:
        assert self.tile != None
        return self.tile.op

    @property
    def op(self) -> ObjectFifoLinkOp:
        assert self.__op != None
        return self.__op

    # TODO: type this
    @property
    def obj_type(self):
        return self.__memref_type

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        if self.__op != None:
            return
        self.__op = object_fifo_link(
            [s.op for s in self.__seconds],
            [f.op for f in self.__firsts],
            loc=loc,
            ip=ip,
        )
