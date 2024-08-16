# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.meta import region_op, op_region_builder

"""
mycore = region_op(
    lambda tile, link_with, *, loc=None, ip=None: Core(tile, link_with),
    terminator=lambda *_: EndOp()
)
"""

"""
mycore = region_op(Core, terminator=lambda *_: EndOp())
device = region_op(Device)
mem = region_op(
    lambda tile, *, loc=None, ip=None: MemOp(T.index(), tile, loc=loc, ip=ip)
)
"""

import sys
from collections import defaultdict
#from bfloat16 import bfloat16

TYPE_MAP_DICT = defaultdict(
    lambda: None,
    {
        # Integer types
        np.int8: T.i8,
        np.int16: T.i16,
        np.int32: T.i32,
        np.int64: T.i64,
        # Unsigned Integer Types
        np.uint8: T.ui8,
        np.uint16: T.ui16,
        np.uint32: T.ui32,
        np.uint64: T.ui64,
        # Floating point types
        np.float16: T.f16,
        np.float32: T.f32,
        np.float64: T.f64,
        #bfloat16: T.bf16,
    },
)

def type_mapper(np_dtype):
    """
    This function is meant to run within a module context (e.g., with a function wrapped with @build_module)
    args:
        np_dtype: the numpy data type to map
    return:
        The data type to run on the npu
    """
    xrt_dtype = TYPE_MAP_DICT[np_dtype]()

    if xrt_dtype is None:
        raise AirBackendError(f"numpy data type {np_dtype} has no default mapping")
    elif xrt_dtype.width / 8 != np.dtype(np_dtype).itemsize:
        # This is a sanity check on the TYPE_MAP_DICT rather than a check on the user input
        raise AirBackendError(
            f"Python data type has width {xrt_dtype.width / 8} but numpy data type has width {np.dtype(np_dtype).itemsize}"
        )
    return xrt_dtype

class ObjectFifo:
    of_index = 0

    def __init__(self, size=1, memref_type=None, end1=None, end2=None):
        self.size = size
        self.memref_type = memref_type
        self._end1 = end1
        self._end2 = end2
        self.op = None

    @classmethod
    def get_index(cls):
        idx = cls.of_index
        cls.of_index += 1
        return idx

    def try_create_fifo(self, loc=None, ip=None, context=None):
        if self.op != None:
            pass
        self.create_fifo()

    def create_fifo(self, loc=None, ip=None, context=None):
        assert self._end1 != None, "ObjectFifo missing endpoint 1"
        assert self._end2 != None, "ObjectFifo missing endpoint 2"
        assert self.memref_type != None, "ObjectFifo missing memref_type"
        assert self.op == None, "Cannot resolve ObjectFifo more than once"

        dtype = type_mapper(self.memref_type[1])
        memRef_ty = MemRefType.get(shape=self.memref_type[0], element_type=dtype)
        self.op = object_fifo(str(self.get_index()), self._end1.get_tile(), self._end2.get_tile(), self.size, memRef_ty)

    def set_endpoint(self, endpoint, first=True):
        if first:
            assert self._end1 == None, "ObjectFifo already assigned endpoint 1"
            self._end1 = endpoint
        else:
            assert self._end2 == None, "ObjectFifo already assigned endpoint 2"
            self._end2 = endpoint

    def acquire_produce(self, num_elem: int, loc=None, ip=None, context=None):
        self._acquire(ObjectFifoPort.Produce, num_elem)

    def acquire_consume(self, num_elem: int, loc=None, ip=None, context=None):
        self._acquire(ObjectFifoPort.Consume, num_elem)

    def release_produce(self, num_elem: int, loc=None, ip=None, context=None):
        self._release(ObjectFifoPort.Produce, num_elem)

    def release_consume(self, num_elem: int, loc=None, ip=None, context=None):
        self._release(ObjectFifoPort.Consume, num_elem)

    def _acquire(self, port: ObjectFifoPort, num_elem: int, loc=None, ip=None, context=None):
        assert num_elem > 0, "Must consume at least one element"
        assert num_elem <= self.size, "Cannot consume elements to exceed ObjectFifo size"
        self.op.acquire(port, num_elem)

    def _release(self, port: ObjectFifoPort, num_elem: int, loc=None, ip=None, context=None):
        assert num_elem > 0, "Must consume at least one element"
        assert num_elem <= self.size, "Cannot consume elements to exceed ObjectFifo size"
        self.op.release(port, num_elem)

class Tile:
    def __init__(self, column, row):
        assert isinstance(column, int)
        assert isinstance(row, int)
        self.column = column
        self.row = row
        self.op = None

    def create_tile(self, loc=None, ip=None, context=None):
        assert self.op == None
        self.op = tile(self.column, self.row)

class CoreProgram:
    def __init__(self, column: int, row: int, core_fn, ofs_end1=[], ofs_end2=[], link_with=None):
        self.tile = Tile(column, row)
        self.core_fn = core_fn
        self.link_with = link_with

        self.ofs_end1 = ofs_end1
        for of in self.ofs_end1:
            assert isinstance(of, ObjectFifo), "ofs_end1 must be List[ObjectFifo]"
            of.set_endpoint(self, True)

        self.ofs_end2 = ofs_end2
        for of in self.ofs_end2:
            assert isinstance(of, ObjectFifo), "ofs_end1 must be List[ObjectFifo]"
            of.set_endpoint(self, False)

    def get_tile(self, loc=None, ip=None, context=None):
        assert self.tile != None
        return self.tile.op

    def resolve(self, loc=None, ip=None, context=None):
        my_tile = self.tile.op
        my_link = self.link_with
        of_out = self.ofs_end1[0].op
        of_in = self.ofs_end2[0].op

        @core(my_tile, my_link)
        def core_body():
            for _ in for_(sys.maxsize):
                self.core_fn(of_out, of_in)
                yield_([])


class HostProgram:
    def __init__(self, host_fn):
        self.inputs = []
        self.outputs = []
        self.tile = None

        # TODO: how to validate this
        self.host_fn = host_fn

        if self.host_fn is None:
            self.tile = Tile(0, 0) # Use a default

    def add_input(self, input_type, of: ObjectFifo):
        assert isinstance(of, ObjectFifo), "Wrong Type: Expected ObjectFifo"
        of.set_endpoint(self, False)
        self.inputs.append((input_type, of))
    
    def add_output(self, output_type, of: ObjectFifo):
        assert isinstance(of, ObjectFifo), "Wrong Type: Expected ObjectFifo"
        of.set_endpoint(self, True)
        self.outputs.append((output_type, of))

    def get_tile(self, loc=None, ip=None, context=None):
        assert self.tile != None
        return self.tile.op

    def resolve(self, loc=None, ip=None, context=None):
        pass

class AIEProgram:
    def __init__(self, device, core_programs, host_program):
        assert isinstance(device, AIEDevice)
        assert core_programs != None and len(core_programs) >= 1
        for c in core_programs:
            assert isinstance(c, CoreProgram)
        assert isinstance(host_program, HostProgram)
        self.device = device
        self.core_programs = core_programs
        self.host_program = host_program

    def resolve(self):
        with mlir_mod_ctx() as ctx:
            @device(self.device)
            def device_body():
                # generate tiles
                for c in self.core_programs:
                    c.tile.create_tile()
                host_program.tile.create_tile()
                
                # generate fifos
                ofs = set()
                for c in self.core_programs:
                    for of1 in c.ofs_end1:
                        ofs.add(of1)
                    for of2 in c.ofs_end2:
                        ofs.add(of2)
                for of in ofs:
                    of.create_fifo()

                # Generate core programs

                for c in self.core_programs:
                    c.resolve()

                # TODO: Host program
            print(ctx.module)

##############################################################################################################################
# Program Start
##############################################################################################################################

import numpy as np

if __name__ == "__main__":
    vector_size = 512
    assert vector_size % 4 == 0
    line_size = vector_size // 4

    inout_type = ((vector_size,), np.int8)
    fifo_memref_type = ((line_size,), np.int8)

    of0 = ObjectFifo(2, memref_type=fifo_memref_type)
    of1 = ObjectFifo(2, memref_type=fifo_memref_type)

    def core_fn(of_out, of_in):
        elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
        elemIn = of_in.acquire(ObjectFifoPort.Consume, 1)
        #call("passThroughLine", [elemIn, elemOut, line_size])
        of_in.release(ObjectFifoPort.Consume, 1)
        of_out.release(ObjectFifoPort.Produce, 1)

    core_program = CoreProgram(0, 2, core_fn, ofs_end1=[of0], ofs_end2=[of1], link_with="passThrough.cc.o")

    host_program = HostProgram(None)
    host_program.add_input(inout_type, of0)
    host_program.add_output(inout_type, of1)
    
    my_program = AIEProgram(AIEDevice.npu1_1col, core_programs=[core_program], host_program=host_program)
    my_program.resolve()
    #print(my_program.module())


    #input = np.arange(1, vector_size + 1, dtype=np.int8)
    #output = np.zeros(vector_size, dtype=np.int8)
    #my_program.run(input, output, expected_output=input)

