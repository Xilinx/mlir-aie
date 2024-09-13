"""
TODO: 
* docs
* types
* producer/consumer
* join/distribute
"""

from abc import ABC, abstractmethod

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

from ...dialects.memref import MemRefType
from ..resolvable import Resolvable
from .endpoint import MyObjectFifoEndpoint


# TODO: maybe not First and Second types, maybe just have Handle type.
class ObjectFifoHandle(Resolvable):
    @abstractmethod
    def set_endpoint(self, endpoint: MyObjectFifoEndpoint) -> None: ...

    @abstractmethod
    def acquire(self, num_elem: int, loc=None, ip=None, context=None): ...

    @abstractmethod
    def release(self, num_elem: int, loc=None, ip=None, context=None): ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class ObjectFifoFirst(ObjectFifoHandle):
    def __init__(self, of):
        self.__object_fifo = of

    def acquire(self, num_elem: int, loc=None, ip=None, context=None):
        return self.__object_fifo._acquire(ObjectFifoPort.Produce, num_elem)

    def release(self, num_elem: int, loc=None, ip=None, context=None):
        return self.__object_fifo._release(ObjectFifoPort.Produce, num_elem)

    def set_endpoint(self, endpoint: MyObjectFifoEndpoint) -> None:
        self.__object_fifo._set_endpoint(endpoint, first=True)

    @property
    def name(self) -> str:
        return self.__object_fifo.name

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        return self.__object_fifo.resolve(loc, ip, context)


class ObjectFifoSecond(ObjectFifoHandle):
    def __init__(self, of):
        self.__object_fifo = of

    def acquire(self, num_elem: int, loc=None, ip=None, context=None):
        return self.__object_fifo._acquire(ObjectFifoPort.Consume, num_elem)

    def release(self, num_elem: int, loc=None, ip=None, context=None):
        return self.__object_fifo._release(ObjectFifoPort.Consume, num_elem)

    def set_endpoint(self, endpoint: MyObjectFifoEndpoint) -> None:
        self.__object_fifo._set_endpoint(endpoint, first=False)

    @property
    def name(self) -> str:
        return self.__object_fifo.name

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        return self.__object_fifo.resolve(loc, ip, context)


class MyObjectFifo(Resolvable):
    __of_index = 0

    def __init__(
        self,
        depth: int = 1,
        memref_type=None,
        name: str = None,
        end1: MyObjectFifoEndpoint = None,
        end2: MyObjectFifoEndpoint = None,
    ):
        self.__depth: int = depth
        self.__memref_type = memref_type
        self.__end1: MyObjectFifoEndpoint = end1
        self.__end2: MyObjectFifoEndpoint = end2
        if name is None:
            self.name = f"myof{MyObjectFifo.__get_index()}"
        else:
            self.name = name
        self.__op = None
        self.__first = None
        self.__second = None

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
    def first(self) -> ObjectFifoFirst:
        if self.__first == None:
            self.__first = ObjectFifoFirst(self)
        return self.__first

    @property
    def second(self) -> ObjectFifoSecond:
        if self.__second == None:
            self.__second = ObjectFifoSecond(self)
        return self.__second

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        if self.__op != None:
            pass
        assert self.__end1 != None, "ObjectFifo missing endpoint 1"
        assert self.__end2 != None, "ObjectFifo missing endpoint 2"
        assert self.__memref_type != None, "ObjectFifo missing memref_type"
        dtype = np_dtype_to_mlir_type(self.__memref_type[1])
        assert dtype != None
        memRef_ty = MemRefType.get(shape=self.__memref_type[0], element_type=dtype)
        self.__op = object_fifo(
            self.name,
            self.__end1.get_tile(),
            self.__end2.get_tile(),
            self.__depth,
            memRef_ty,
            loc=loc,
            ip=ip,
        )

    def _set_endpoint(self, endpoint, first=True):
        if first:
            assert self.__end1 == None, "ObjectFifo already assigned endpoint 1"
            self.__end1 = endpoint
        else:
            assert self.__end2 == None, "ObjectFifo already assigned endpoint 2"
            self.__end2 = endpoint

    def _acquire(
        self, port: ObjectFifoPort, num_elem: int, loc=None, ip=None, context=None
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        dtype = np_dtype_to_mlir_type(self.__memref_type[1])
        assert dtype != None
        memRef_ty = MemRefType.get(shape=self.__memref_type[0], element_type=dtype)
        subview_t = ObjectFifoSubviewType.get(memRef_ty)
        acq = ObjectFifoAcquireOp(subview_t, port, self.name, num_elem)

        objects = []
        if acq.size.value == 1:
            return ObjectFifoSubviewAccessOp(memRef_ty, acq.subview, acq.size.value - 1)
        for i in range(acq.size.value):
            objects.append(ObjectFifoSubviewAccessOp(memRef_ty, acq.subview, i))
        return objects

    def _release(
        self, port: ObjectFifoPort, num_elem: int, loc=None, ip=None, context=None
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.__depth
        ), "Cannot consume elements to exceed ObjectFifo depth"
        objectfifo_release(port, self.name, num_elem, loc=loc, ip=ip)
