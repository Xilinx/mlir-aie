"""
TODO: 
* docs
* types
* join/distribute
"""

# Address circular dependency between ObjectFifo and ObjectFifoHandle
from __future__ import annotations
import numpy as np

from ... import ir  # type: ignore
from ..._mlir_libs._aie import ObjectFifoSubviewType  # type: ignore
from ...dialects._aie_enum_gen import ObjectFifoPort  # type: ignore
from ...dialects._aie_ops_gen import (
    ObjectFifoCreateOp,
    ObjectFifoSubviewAccessOp,
    ObjectFifoAcquireOp,
    objectfifo_release,
)  # type: ignore
from ...dialects.aie import object_fifo
from ...helpers.util import np_ndarray_type_to_memref_type

from ..resolvable import Resolvable
from .endpoint import ObjectFifoEndpoint
from ..phys.tile import Tile


class ObjectFifo(Resolvable):
    __of_index = 0

    def __init__(
        self,
        depth: int,
        obj_type: type[np.ndarray],
        name: str | None = None,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    ):
        self.__depth = depth
        self.__obj_type = obj_type
        self.end1 = None
        self.end2 = []
        self.dimensionToStream = dimensionsToStream
        self.dimensionsFromStreamPerConsumer = dimensionsFromStreamPerConsumer

        if name is None:
            self.name = f"of{ObjectFifo.__get_index()}"
        else:
            self.name = name
        self.__op: ObjectFifoCreateOp | None = None
        self.__first: ObjectFifoHandle = ObjectFifoHandle(self, True)
        self.__second: ObjectFifoHandle = ObjectFifoHandle(self, False)

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__of_index
        cls.__of_index += 1
        return idx

    @property
    def op(self) -> ObjectFifoCreateOp:
        assert self.__op != None
        return self.__op

    @property
    def first(self) -> ObjectFifoHandle:
        return self.__first

    @property
    def second(self) -> ObjectFifoHandle:
        return self.__second

    def end1_tile(self) -> Tile | None:
        if self.end1 == None:
            return None
        return self.end1.tile

    def end2_tiles(self) -> list[Tile | None] | None:
        if self.end2 == []:
            return None
        return [e.tile for e in self.end2]

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self.__obj_type

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self.__op == None:
            tile1 = self.end1_tile()
            tiles2 = self.end2_tiles()
            assert tile1 != None
            assert tiles2 != None and len(tiles2) >= 1
            for t in tiles2:
                assert t != None

            self.__op = object_fifo(
                self.name,
                tile1.op,
                [t.op for t in tiles2],
                self.__depth,
                np_ndarray_type_to_memref_type(self.__obj_type),
                dimensionsToStream=self.dimensionToStream,
                dimensionsFromStreamPerConsumer=self.dimensionsFromStreamPerConsumer,
                loc=loc,
                ip=ip,
            )

    def _set_endpoint(self, endpoint: ObjectFifoEndpoint, first: bool = True) -> None:
        if first:
            assert (
                self.end1 == None or self.end1 == endpoint
            ), f"ObjectFifo already assigned endpoint 1 ({self.end1})"
            self.end1 = endpoint
        else:
            # TODO: need rules about shim tiles here
            self.end2.append(endpoint)

    def _acquire(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        memref_ty = np_ndarray_type_to_memref_type(self.__obj_type)
        subview_t = ObjectFifoSubviewType.get(memref_ty)
        acq = ObjectFifoAcquireOp(subview_t, port, self.name, num_elem, loc=loc, ip=ip)

        objects = []
        if acq.size.value == 1:
            return ObjectFifoSubviewAccessOp(
                memref_ty,
                acq.subview,
                acq.size.value - 1,
                loc=loc,
                ip=ip,
            )
        for i in range(acq.size.value):
            objects.append(
                ObjectFifoSubviewAccessOp(memref_ty, acq.subview, i, loc=loc, ip=ip)
            )
        return objects

    def _release(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        objectfifo_release(port, self.name, num_elem, loc=loc, ip=ip)


class ObjectFifoHandle(Resolvable):
    def __init__(self, of: ObjectFifo, is_first: bool):
        self.__port: ObjectFifoPort = (
            ObjectFifoPort.Produce if is_first else ObjectFifoPort.Consume
        )
        self.__is_first = is_first
        self.__object_fifo = of

    def acquire(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        return self.__object_fifo._acquire(self.__port, num_elem, loc=loc, ip=ip)

    def release(
        self,
        num_elem: int,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ):
        return self.__object_fifo._release(self.__port, num_elem, loc=loc, ip=ip)

    @property
    def name(self) -> str:
        return self.__object_fifo.name

    @property
    def op(self) -> ObjectFifoCreateOp:
        return self.__object_fifo.op

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self.__object_fifo.obj_type

    def end1_tile(self) -> Tile | None:
        return self.__object_fifo.end1_tile()

    def end2_tiles(self) -> list[Tile | None] | None:
        return self.__object_fifo.end2_tiles()

    def set_endpoint(self, endpoint: ObjectFifoEndpoint) -> None:
        self.__object_fifo._set_endpoint(endpoint, first=self.__is_first)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return self.__object_fifo.resolve(loc=loc, ip=ip)