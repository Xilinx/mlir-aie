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

import sys
from collections import defaultdict

# from bfloat16 import bfloat16
range_ = for_

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
        # bfloat16: T.bf16,
    },
)


def type_mapper(np_dtype):
    xrt_dtype = TYPE_MAP_DICT[np_dtype]

    if xrt_dtype:
        xrt_dtype = xrt_dtype()
        if xrt_dtype.width / 8 != np.dtype(np_dtype).itemsize:
            # This is a sanity check on the TYPE_MAP_DICT rather than a check on the user input
            raise AttributeError(
                f"Python data type has width {xrt_dtype.width / 8} but numpy data type has width {np.dtype(np_dtype).itemsize}"
            )
        return xrt_dtype
    else:
        return None


def get_arg_types(objs):
    my_types = []
    for o in objs:
        if isinstance(o, Value):
            my_types.append(o.type)
        elif isinstance(o, OpView):
            if len(o.results.types) != 1:
                raise AttributeError(
                    f"Operation given to a region op as a parameter ({o}) has more "
                    "than one return type ({o.results.types}), which would lead to a mismatch "
                    "between number of operands and number of operand types"
                )
            my_types += o.results.types
        else:
            return None
    return my_types


class MyObjectFifo:
    of_index = 0

    def __init__(self, size=1, memref_type=None, name=None, end1=None, end2=None):
        self.size = size
        self.memref_type = memref_type
        self._end1 = end1
        self._end2 = end2
        if name:
            self.name = name
        else:
            self.name = str(MyObjectFifo.get_index())
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
        assert dtype != None
        memRef_ty = MemRefType.get(shape=self.memref_type[0], element_type=dtype)
        self.op = object_fifo(
            self.name,
            self._end1.get_tile(),
            self._end2.get_tile(),
            self.size,
            memRef_ty,
        )

    def set_endpoint(self, endpoint, first=True):
        if first:
            assert self._end1 == None, "ObjectFifo already assigned endpoint 1"
            self._end1 = endpoint
        else:
            assert self._end2 == None, "ObjectFifo already assigned endpoint 2"
            self._end2 = endpoint

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
            num_elem <= self.size
        ), "Cannot consume elements to exceed ObjectFifo size"
        return self.op.acquire(port, num_elem)

    def _release(
        self, port: ObjectFifoPort, num_elem: int, loc=None, ip=None, context=None
    ):
        assert num_elem > 0, "Must consume at least one element"
        assert (
            num_elem <= self.size
        ), "Cannot consume elements to exceed ObjectFifo size"
        self.op.release(port, num_elem)


class MyTile:
    def __init__(self, column, row):
        assert isinstance(column, int)
        assert isinstance(row, int)
        self.column = column
        self.row = row
        self.op = None

    def create_tile(self, loc=None, ip=None, context=None):
        assert self.op == None
        self.op = tile(self.column, self.row)


class MyExternalFunction:
    def __init__(self, name, bin_name, inout_types):
        assert isinstance(name, str)
        assert len(name) > 0
        assert isinstance(bin_name, str)
        assert len(bin_name) > 0
        assert isinstance(inout_types, list)
        self.name = name
        self.bin_name = bin_name
        self.inout_types = inout_types
        self.op = None

    def resolve(self):
        assert self.op == None
        resolved_inout_types = []
        for t in self.inout_types:
            dtype = type_mapper(t)
            if dtype is None:
                dtype = get_arg_types(t)
                if dtype is None:
                    # Interpret as a dummy memref
                    dtype = MemRefType.get(shape=t[0], element_type=type_mapper(t[1]))
            resolved_inout_types.append(dtype)
        self.op = external_func(self.name, inputs=resolved_inout_types)

    def call(self, *args, **kwargs):
        assert self.op
        call(self.name, args)


class CoreProgram:
    def __init__(
        self,
        column: int,
        row: int,
        core_fn,
        ofs_end1=[],
        ofs_end2=[],
        external_functions=[],
    ):
        self.tile = MyTile(column, row)
        self.core_fn = core_fn

        assert isinstance(external_functions, list)
        bin_names = set()
        for e in external_functions:
            assert isinstance(e, MyExternalFunction)
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

    def get_tile(self, loc=None, ip=None, context=None):
        assert self.tile != None
        return self.tile.op

    def resolve(self, loc=None, ip=None, context=None):
        my_tile = self.tile.op
        my_link = self.link_with

        @core(my_tile, my_link)
        def core_body():
            self.core_fn(self.ofs_end1, self.ofs_end2, self.external_functions)


class FifoInOutHostProgram:
    def __init__(self, fifo_in, bytes_in, fifo_out, bytes_out):
        assert bytes_in % np.prod(fifo_in.memref_type[0]) == 0
        assert bytes_out % np.prod(fifo_out.memref_type[0]) == 0
        self.fifo_in = fifo_in
        self.fifo_out = fifo_out
        self.bytes_in = bytes_in
        self.bytes_out = bytes_out
        fifo_in.set_endpoint(self, False)
        fifo_out.set_endpoint(self, True)

        self.tile = MyTile(0, 0)  # Use a default

    # TODO: remove this, add to resolve
    def get_tile(self, loc=None, ip=None, context=None):
        assert self.tile != None
        return self.tile.op

    def resolve(self, loc=None, ip=None, context=None):
        tensor_in_ty = T.memref(self.bytes_in, T.ui8())
        tensor_out_ty = T.memref(self.bytes_out, T.ui8())

        @runtime_sequence(tensor_in_ty, tensor_out_ty)
        def sequence(inTensor, outTensor):
            npu_dma_memcpy_nd(
                metadata=self.fifo_out.name,
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, self.bytes_out],
            )
            npu_dma_memcpy_nd(
                metadata=self.fifo_in.name,
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, self.bytes_in],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


class AIEProgram:
    def __init__(self, device, core_programs, host_program):
        assert isinstance(device, AIEDevice)
        assert core_programs != None and len(core_programs) >= 1
        for c in core_programs:
            assert isinstance(c, CoreProgram)
        # assert isinstance(host_program, HostProgram) # TODO: check via hierarchy
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
                self._print_verify(ctx)

                host_program.tile.create_tile()
                self._print_verify(ctx)

                # generate fifos (and external functions)
                ofs = set()
                external_functions = set()
                for c in self.core_programs:
                    for of1 in c.ofs_end1:
                        ofs.add(of1)
                    for of2 in c.ofs_end2:
                        ofs.add(of2)

                    for e in c.external_functions:
                        external_functions.add(e)
                for of in ofs:
                    of.create_fifo()
                    self._print_verify(ctx)
                for e in external_functions:
                    e.resolve()
                    self._print_verify(ctx)

                # Generate core programs
                for c in self.core_programs:
                    # c.resolve(passThroughLine)
                    c.resolve()
                    self._print_verify(ctx)

                # Host program
                self.host_program.resolve()
                self._print_verify(
                    ctx
                )  # TODO: This should happen at end of every resolve() type operation not just in this method

            print(ctx.module)

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            print(verify)

"""
Problems for clarify/conciseness:
* ObjectFifo needs (ordered) endpoints at instantiation
* Need introspection to declare functions/fifos on-the-fly so they still land in the symbol table
* Can remove type data if we're okay with inferring it through use (also required introspection) => but less verification if we go this route
    - Could we fix this somehow? e.g. loop emulation or something like that?
"""
##############################################################################################################################
# Program Start
##############################################################################################################################

import numpy as np

try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
except ValueError:
    print("Argument has inappropriate value")

assert vector_size % 4 == 0
line_size = vector_size // 4

inout_type = ((vector_size,), np.uint8)
fifo_memref_type = ((line_size,), np.uint8)

of0 = MyObjectFifo(2, memref_type=fifo_memref_type, name="out")
of1 = MyObjectFifo(2, memref_type=fifo_memref_type, name="in")

passthrough_fn = MyExternalFunction(
    "passThroughLine",
    "passThrough.cc.o",
    [fifo_memref_type, fifo_memref_type, np.int32],
)

def core_fn(ofs_end1, ofs_end2, external_functions):
    of_out = ofs_end1[0]
    of_in = ofs_end2[0]
    passThroughLine = external_functions[0]

    for _ in range_(vector_size // line_size):
        elemOut = of_out.acquire_produce(1)
        elemIn = of_in.acquire_consume(1)
        passThroughLine.call(elemIn, elemOut, line_size)
        of_in.release_consume(1)
        of_out.release_produce(1)
        yield_([])

core_program = CoreProgram(0, 2, core_fn, [of0], [of1], [passthrough_fn])
host_program = FifoInOutHostProgram(of0, vector_size, of1, vector_size)

my_program = AIEProgram(
    AIEDevice.npu1_1col, core_programs=[core_program], host_program=host_program
)
my_program.resolve()