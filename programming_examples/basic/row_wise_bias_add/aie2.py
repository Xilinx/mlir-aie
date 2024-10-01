#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_


def row_wise_bias_add(M, N, m, n):

    assert M % m == 0
    assert N % n == 0

    @device(AIEDevice.npu1_1col)
    def device_body():

        tensor_ty = np.ndarray[(m * n,), np.dtype[np.float32]]
        bias_ty = np.ndarray[(n,), np.dtype[np.float32]]

        kernel_func = external_func(
            f"row_wise_bias_add_f32_f32", inputs=[tensor_ty, bias_ty, tensor_ty]
        )

        shim_tile = tile(0, 0)
        compute_tile = tile(0, 2)

        in_fifo = object_fifo("in_fifo", shim_tile, compute_tile, 2, tensor_ty)
        bias_fifo = object_fifo("bias_fifo", shim_tile, compute_tile, 2, bias_ty)
        out_fifo = object_fifo("out_fifo", compute_tile, shim_tile, 2, tensor_ty)

        @core(compute_tile, "kernel.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(N // n):
                    elem_bias = bias_fifo.acquire(ObjectFifoPort.Consume, 1)
                    for _ in range_(M // m):
                        elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                        kernel_func(elem_in, elem_bias, elem_out)
                        out_fifo.release(ObjectFifoPort.Produce, 1)
                        in_fifo.release(ObjectFifoPort.Consume, 1)
                    bias_fifo.release(ObjectFifoPort.Consume, 1)

        @runtime_sequence(tensor_ty, bias_ty, tensor_ty)
        def sequence(inp, bias, out):
            npu_dma_memcpy_nd(
                metadata=in_fifo,
                bd_id=0,
                mem=inp,
                sizes=[1, N // n, M, n],
                strides=[0, n, N, 1],
            )
            npu_dma_memcpy_nd(
                metadata=bias_fifo,
                bd_id=1,
                mem=bias,
                sizes=[1, 1, N // n, n],
                strides=[0, 0, n, 1],
            )
            npu_dma_memcpy_nd(
                metadata=out_fifo,
                bd_id=2,
                mem=out,
                sizes=[1, N // n, M, n],
                strides=[0, n, N, 1],
            )
            # of_out will only complete after of_in completes, so we just wait on of_out instead of both
            dma_wait(out_fifo)


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
