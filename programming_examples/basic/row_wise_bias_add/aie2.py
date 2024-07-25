#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

import sys


def row_wise_bias_add(M, N, m, n):

    assert M % m == 0
    assert N % n == 0

    @device(AIEDevice.npu1_1col)
    def device_body():

        complete_in_memref = T.memref(m * n, T.f32())
        complete_bias_memref = T.memref(n, T.f32())
        complete_out_memref = T.memref(m * n, T.f32())

        in_memref = T.memref(m * n, T.f32())
        bias_memref = T.memref(n, T.f32())
        out_memref = T.memref(m * n, T.f32())

        kernel_func = external_func(
            f"row_wise_bias_add_f32_f32", inputs=[in_memref, bias_memref, out_memref]
        )

        shim_tile = tile(0, 0)
        compute_tile = tile(0, 2)

        in_fifo = object_fifo("in_fifo", shim_tile, compute_tile, 2, in_memref)
        bias_fifo = object_fifo("bias_fifo", shim_tile, compute_tile, 2, bias_memref)
        out_fifo = object_fifo("out_fifo", compute_tile, shim_tile, 2, out_memref)

        @core(compute_tile, "kernel.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                for _ in for_(N // n):
                    elem_bias = bias_fifo.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(M // m):
                        elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                        call(kernel_func, [elem_in, elem_bias, elem_out])
                        out_fifo.release(ObjectFifoPort.Produce, 1)
                        in_fifo.release(ObjectFifoPort.Consume, 1)
                        yield_([])
                    bias_fifo.release(ObjectFifoPort.Consume, 1)
                    yield_([])
                yield_([])

        @runtime_sequence(complete_in_memref, complete_bias_memref, complete_out_memref)
        def sequence(inp, bias, out):
            npu_dma_memcpy_nd(
                metadata=in_fifo.sym_name.value,
                bd_id=0,
                mem=inp,
                sizes=[1, N // n, M, n],
                strides=[0, n, N, 1],
            )
            npu_dma_memcpy_nd(
                metadata=bias_fifo.sym_name.value,
                bd_id=1,
                mem=bias,
                sizes=[1, 1, N // n, n],
                strides=[0, 0, n, 1],
            )
            npu_dma_memcpy_nd(
                metadata=out_fifo.sym_name.value,
                bd_id=2,
                mem=out,
                sizes=[1, N // n, M, n],
                strides=[0, n, N, 1],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    row_wise_bias_add(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    )
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
