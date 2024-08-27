# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import for_ as range_
from aie.dialects.scf import yield_
from aie.extras.context import mlir_mod_ctx

from aie.api.phys.tile import MyTile
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.kernels.binkernel import BinKernel
from aie.api.kernels.kernel import MyKernel
from aie.api.worker import MyWorker

import sys


class FifoInOutHostProgram:
    def __init__(self, fifo_in, bytes_in, fifo_out, bytes_out):
        # assert bytes_in % np.prod(fifo_in.__memref_type[0]) == 0
        # assert bytes_out % np.prod(fifo_out.__memref_type[0]) == 0
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
            assert isinstance(c, MyWorker)
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
                    c.tile.resolve()
                self._print_verify(ctx)

                host_program.tile.resolve()
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
                    of.resolve()
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

passthrough_fn = BinKernel(
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
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release_consume(1)
        of_out.release_produce(1)
        yield_([])


core_program = MyWorker(core_fn, [of0], [of1], [passthrough_fn], coords=(0, 2))
host_program = FifoInOutHostProgram(of0, vector_size, of1, vector_size)

my_program = AIEProgram(
    AIEDevice.npu1_1col, core_programs=[core_program], host_program=host_program
)
my_program.resolve()
