"""
TODO: 
* docs
* types
* producer/consumer
* join/distribute
"""

from ... import ir
from ..._mlir_libs._aie import ObjectFifoSubviewType
from ...extras.util import np_dtype_to_mlir_type

from ...dialects._aie_enum_gen import ObjectFifoPort
from ...dialects._aie_ops_gen import (
    ObjectFifoSubviewAccessOp,
    ObjectFifoAcquireOp,
    objectfifo_release,
)
from ...dialects.aie import object_fifo

from ...dialects.memref import MemRefType
from ..resolvable import Resolvable
from .endpoint import MyObjectFifoEndpoint


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
        self.__name = name
        self.__op = None

    @classmethod
    def __get_index(cls):
        idx = cls.__of_index
        cls.__of_index += 1
        return idx

    @property
    def op(self):
        assert self.__op != None
        return self.__op

    @property
    def name(self):
        if not self.__name:
            self.__name = str(MyObjectFifo.__get_index())
        return self.__name

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
            self.__name,
            self.__end1.get_tile(),
            self.__end2.get_tile(),
            self.__depth,
            memRef_ty,
            loc=loc,
            ip=ip,
        )

    def set_endpoint(self, endpoint, first=True):
        if first:
            assert self.__end1 == None, "ObjectFifo already assigned endpoint 1"
            self.__end1 = endpoint
        else:
            assert self.__end2 == None, "ObjectFifo already assigned endpoint 2"
            self.__end2 = endpoint

    def acquire_produce(self, num_elem: int, loc=None, ip=None, context=None):
        return self._acquire(ObjectFifoPort.Produce, num_elem)

    def acquire_consume(self, num_elem: int, loc=None, ip=None, context=None):
        return self._acquire(ObjectFifoPort.Consume, num_elem)

    def release_produce(self, num_elem: int, loc=None, ip=None, context=None):
        self._release(ObjectFifoPort.Produce, num_elem)

    def release_consume(self, num_elem: int, loc=None, ip=None, context=None):
        self._release(ObjectFifoPort.Consume, num_elem)

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
