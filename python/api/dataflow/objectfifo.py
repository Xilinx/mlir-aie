"""
TODO: 
* docs
* types
* join/distribute
"""

# Address circular dependency between MyObjectFifo and ObjectFifoHandle
from __future__ import annotations
from typing import Optional

from ... import ir
from ..._mlir_libs._aie import ObjectFifoSubviewType
from ...extras.util import np_dtype_to_mlir_type

from ...dialects._aie_enum_gen import ObjectFifoPort
from ...dialects._aie_ops_gen import (
    ObjectFifoCreateOp,
    ObjectFifoSubviewAccessOp,
    ObjectFifoAcquireOp,
    objectfifo_release,
)
from ...dialects.aie import object_fifo

from ..resolvable import Resolvable
from .endpoint import MyObjectFifoEndpoint
from ..tensor import MyTensorType


class MyObjectFifo(Resolvable):
    __of_index = 0

    def __init__(
        self,
        depth: int,
        obj_type: np.ndarray[np.generic.dtype, np.generic.shape],
        name: str = None,
        end1: MyObjectFifoEndpoint = None,
        end2: MyObjectFifoEndpoint = None,
        dimensionsToStream=None,  # TODO(erika): needs a type
        dimensionsFromStreamPerConsumer=None,  # TODO(erika): needs a type
    ):
        self.__depth = depth
        self.__obj_type = MyTensorType(obj_type)
        self.end1: MyObjectFifoEndpoint = end1
        self.end2: MyObjectFifoEndpoint = end2
        self.dimensionToStream = dimensionsToStream
        self.dimensionsFromStreamPerConsumer = dimensionsFromStreamPerConsumer

        if name is None:
            self.name = f"myof{MyObjectFifo.__get_index()}"
        else:
            self.name = name
        self.__op: Optional[ObjectFifoCreateOp] = None
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

    @property
    def obj_type(self) -> MyTensorType:
        return self.__obj_type

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        if self.__op == None:
            assert self.end1 != None, "ObjectFifo missing endpoint 1"
            assert self.end2 != None, "ObjectFifo missing endpoint 2"
            assert self.__obj_type != None, "ObjectFifo missing object type"
            self.__op = object_fifo(
                self.name,
                self.end1.get_tile().op,
                self.end2.get_tile().op,
                self.__depth,
                self.__obj_type.memref_type,
                dimensionsToStream=self.dimensionToStream,
                dimensionsFromStreamPerConsumer=self.dimensionsFromStreamPerConsumer,
                loc=loc,
                ip=ip,
            )

    def _set_endpoint(self, endpoint: MyObjectFifoEndpoint, first: bool = True) -> None:
        if first:
            assert self.end1 == None, "ObjectFifo already assigned endpoint 1"
            self.end1 = endpoint
        else:
            assert self.end2 == None, "ObjectFifo already assigned endpoint 2"
            self.end2 = endpoint

    def _acquire(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"

        subview_t = ObjectFifoSubviewType.get(self.__obj_type.memref_type)
        acq = ObjectFifoAcquireOp(subview_t, port, self.name, num_elem, loc=loc, ip=ip)

        objects = []
        if acq.size.value == 1:
            return ObjectFifoSubviewAccessOp(
                self.__obj_type.memref_type,
                acq.subview,
                acq.size.value - 1,
                loc=loc,
                ip=ip,
            )
        for i in range(acq.size.value):
            objects.append(
                ObjectFifoSubviewAccessOp(
                    self.__obj_type.memref_type, acq.subview, i, loc=loc, ip=ip
                )
            )
        return objects

    def _release(
        self,
        port: ObjectFifoPort,
        num_elem: int,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        objectfifo_release(port, self.name, num_elem, loc=loc, ip=ip)


class ObjectFifoHandle(Resolvable):
    def __init__(self, of: MyObjectFifo, is_first: bool):
        self.__port: ObjectFifoPort = (
            ObjectFifoPort.Produce if is_first else ObjectFifoPort.Consume
        )
        self.__is_first = is_first
        self.__object_fifo = of

    def acquire(
        self, num_elem: int, loc: ir.Location = None, ip: ir.InsertionPoint = None
    ):
        return self.__object_fifo._acquire(self.__port, num_elem, loc=loc, ip=ip)

    def release(
        self, num_elem: int, loc: ir.Location = None, ip: ir.InsertionPoint = None
    ):
        return self.__object_fifo._release(self.__port, num_elem, loc=loc, ip=ip)

    @property
    def obj_type(self) -> MyTensorType:
        return self.__object_fifo.obj_type

    @property
    def name(self) -> str:
        return self.__object_fifo.name

    @property
    def op(self) -> ObjectFifoCreateOp:
        return self.__object_fifo.op

    def set_endpoint(self, endpoint: MyObjectFifoEndpoint) -> None:
        self.__object_fifo._set_endpoint(endpoint, first=self.__is_first)

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        return self.__object_fifo.resolve(loc=loc, ip=ip)
