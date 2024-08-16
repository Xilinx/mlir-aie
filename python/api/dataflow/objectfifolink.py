import numpy as np
from collections.abc import Sequence

from ... import ir
from ...dialects._aie_ops_gen import ObjectFifoLinkOp
from ...dialects.aie import object_fifo_link

from ..phys.tile import MyTile
from .endpoint import MyObjectFifoEndpoint
from .objectfifo import ObjectFifoHandle
from ...extras.util import single_elem_or_list_to_list


class MyObjectFifoLink(MyObjectFifoEndpoint):
    def __init__(
        self,
        seconds: Sequence[ObjectFifoHandle] | ObjectFifoHandle = [],
        firsts: Sequence[ObjectFifoHandle] | ObjectFifoHandle = [],
        coords: tuple[int, int] | None = None,
    ):
        column, row = coords
        self.__tile = MyTile(column, row)

        self.__seconds = single_elem_or_list_to_list(seconds)
        self.__firsts = single_elem_or_list_to_list(firsts)
        self.__op = None

        assert len(self.__firsts) > 0
        assert len(self.__seconds) > 0

        self.__obj_type = self.__seconds[0].obj_type
        for f in self.__firsts:
            # TODO: need to check size not exactness
            assert f.obj_type == self.__obj_type
            f.set_endpoint(self)
        for s in self.__seconds:
            assert s.obj_type == self.__obj_type
            s.set_endpoint(self)

    @property
    def tile(self) -> MyTile:
        assert self.__tile != None
        return self.__tile

    @property
    def op(self) -> ObjectFifoLinkOp:
        assert self.__op != None
        return self.__op

    @property
    def obj_type(self) -> np.ndarray[np.generic.dtype, np.generic.shape]:
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
