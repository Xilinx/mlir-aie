#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

tensorInW = 7
tensorInH = 7
tensorInC = 16

tensorL2InC = tensorInC
tensorL2OutC = tensorInC

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4


def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            # define types
            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)

            tensorLayer2In_ty = MemRefType.get(
                (
                    tensorInW,
                    1,
                    tensorL2InC,
                ),
                uint8_ty,
            )
            weightsLayer2_ty = MemRefType.get(
                (3 * 3 * tensorL2OutC * 1,), int8_ty
            )
            tensorLayer2Out_ty = MemRefType.get(
                (
                    tensorInW,
                    1,
                    tensorL2OutC,
                ),
                uint8_ty,
            )

            # AIE Core Function declarations
            conv2dk3_dw = external_func(
                "conv2dk3_ui8",
                inputs=[
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    weightsLayer2_ty,
                    tensorLayer2Out_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            compute_tile2_col, compute_tile2_row = 0, 2

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, tensorLayer2In_ty
            )
            of_act_2_3_5 = object_fifo("act_2_3_5", MemTile, ComputeTile2, 2, tensorLayer2In_ty)
            object_fifo_link(of_inOF_act_L3L2, of_act_2_3_5)

            # wts
            wts_buf_01 = object_fifo(
                "wts_buf_01", ShimTile, [ComputeTile2], 1, weightsLayer2_ty
            )

            # Output
            act_3_4 = object_fifo("act_3_4", ComputeTile2, [MemTile], 2, tensorLayer2Out_ty)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, tensorLayer2Out_ty)
            object_fifo_link(act_3_4, of_outOFL2L3)

            # Set up compute tiles

            rtp2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")

            # # Compute tile 2
            @core(ComputeTile2, "conv2dk3_dw.o")
            def core_body():
                scale = 8
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_01.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile3, 0)

                    # pre-amble: top row
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_3_4.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            1,
                            tensorL2OutC,
                            3,
                            3,
                            0,
                            scale,
                            0,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Produce, "act_3_4", 1)

                    # middle
                    for _ in for_(tensorInH - 2):
                        elementActivactionsIn = of_act_2_3_5.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = act_3_4.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            conv2dk3_dw,
                            [
                                elementActivactionsIn[0],
                                elementActivactionsIn[1],
                                elementActivactionsIn[2],
                                element0Weights,
                                element0ActivactionsOut,
                                tensorInW,
                                1,
                                tensorL2OutC,
                                3,
                                3,
                                1,
                                scale,
                                0,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "act_2_3_5", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "act_3_4", 1)
                        yield_([])

                    # last part
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_3_4.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            1,
                            tensorL2OutC,
                            3,
                            3,
                            2,
                            scale,
                            0,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "act_2_3_5", 2)
                    objectfifo_release(ObjectFifoPort.Produce, "act_3_4", 1)

                    objectfifo_release(ObjectFifoPort.Consume, "wts_buf_01", 1)
                    yield_([])


            # # instruction stream generation
            activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
            acitivationsOutSize32b = activationsInSize32b
            totalWeightsSize32b = (
               3 * 3 * tensorL2OutC * 1
            ) // 4

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsInL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                NpuWriteRTPOp("rtp2", col=0, row=2, index=0, value=8)

                npu_dma_memcpy_nd(
                    metadata="inOF_act_L3L2",
                    bd_id=0,
                    mem=inputFromL3,
                    sizes=[1, 1, 1, activationsInSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="outOFL2L3",
                    bd_id=2,
                    mem=outputToL3,
                    sizes=[1, 1, 1, acitivationsOutSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="wts_buf_01",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeightsSize32b],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


conv2dk1()
