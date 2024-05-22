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

tensorInW = 112
tensorInH = 112
tensorInC = 16

depthWiseStride = 2
depthWiseChannels = 64 
tensorOutC = 24

tensorOutW = 112 // depthWiseStride
tensorOutH = 112 // depthWiseStride

tensorL1InC = tensorInC
tensorL1OutC = depthWiseChannels

tensorL2InC = tensorL1OutC
tensorL2OutC = tensorL2InC

tensorL3InC = tensorL2InC
tensorL3OutC = tensorOutC

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4

tileRowIndex = 2
tileColIndex = 0

def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            # define types
            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int16_ty = IntegerType.get_signless(16)
            int32_ty = IntegerType.get_signless(32)

            tensorLayer1In_ty = MemRefType.get((tensorInW, 1, tensorL1InC), int8_ty)
            weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL1OutC * tensorL1InC,), int8_ty)
            tensorLayer1Out_ty = MemRefType.get((tensorInW, 1, tensorL1OutC),uint8_ty)

            tensorLayer2In_ty = MemRefType.get((tensorInW, 1, tensorL2InC), uint8_ty)
            weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL2OutC * 1,), int8_ty)
            tensorLayer2Out_ty = MemRefType.get((tensorInW//depthWiseStride, 1, tensorL2OutC),uint8_ty)

            tensorLayer3In_ty = MemRefType.get((tensorInW//depthWiseStride, 1, tensorL3InC), uint8_ty)
            weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
            tensorLayer3Out_ty = MemRefType.get((tensorInW//depthWiseStride, 1, tensorL3OutC),int8_ty)

            # AIE Core Function declarations
            conv2dk1_relu_i8_ui8 = external_func("conv2dk1_i8",inputs=[tensorLayer1In_ty, weightsLayer1_ty, tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
            conv2dk3_dw_relu_ui8_ui8 = external_func("conv2dk3_dw_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
            conv2dk1_ui8_i8 = external_func("conv2dk1_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

            # Tile declarations
            ShimTile = tile(tileColIndex, 0)
            MemTile = tile(tileColIndex, 1)
            ComputeTile = tile(tileColIndex, tileRowIndex)
            
            # AIE-array data movement with object fifos
            # Input
            #of_inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile, MemTile, 2, tensorLayer1In_ty)
            act_in = object_fifo("act_in", ShimTile, ComputeTile, 2, tensorLayer1In_ty)
            #object_fifo_link(of_inOF_act_L3L2, act_in)

            # wts
            wts_buf_01 = object_fifo("wts_buf_01", ShimTile, [ComputeTile], 1, weightsLayer2_ty)
            wts_buf_02 = object_fifo("wts_buf_02", ShimTile, [ComputeTile], 1, weightsLayer2_ty)
            wts_buf_03 = object_fifo("wts_buf_03", ShimTile, [ComputeTile], 1, weightsLayer3_ty)

            # Output
            act_3 = object_fifo("act_3_4", ComputeTile, [ShimTile], 2, tensorLayer3Out_ty)
            #of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, tensorLayer3Out_ty)
            #object_fifo_link(act_3, of_outOFL2L3)

            # Intermediate
            of_act_1_2 = object_fifo("act_1_2", ComputeTile, ComputeTile, 3, tensorLayer1Out_ty)
            of_act_2_3 = object_fifo("act_2_3", ComputeTile, ComputeTile, 1, tensorLayer2Out_ty)


            # Set up compute tiles

            rtp2 = Buffer(ComputeTile, [16], T.i32(), "rtp2")

            # # Compute tile 2
            @core(ComputeTile, "conv2dk1_conv2dk3_dw.a")
            def core_body():
                scale = 8
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_01.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile3, 0)

                    # pre-amble: top 2 rows
                    for _ in for_(2):
                        actInRow0 = act_in.acquire(ObjectFifoPort.Consume, 1)
                        actOutRow0 = of_act_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(conv2dk1_relu_i8_ui8, [actInRow0, element0Weights, actOutRow0, tensorInW, 1, tensorL1OutC, 1, scale])
                        objectfifo_release(ObjectFifoPort.Consume, "act_in", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "actOutRow", 1)

                    HIER BEZIG
                    
                    # middle
                    for _ in for_(tensorInH - 2):
                        elementActivactionsIn = of_act_2_3_5.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = act_3_4.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            conv2dk3,
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
                        conv2dk3,
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
