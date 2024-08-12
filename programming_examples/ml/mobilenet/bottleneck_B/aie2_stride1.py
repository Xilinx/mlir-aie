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

bneck_10_InW1 = 14
bneck_10_InH1 = 14
bneck_10_InC1 = 80
bneck_10_OutC1 = 480

bneck_10_InW2 = 14
bneck_10_InH2 = 14
bneck_10_OutC2 = bneck_10_OutC1

bneck_10_InW3 = 14
bneck_10_InH3 = 14
bneck_10_OutC3 = 112

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4


def mobilenetBottleneckB():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            # define types
            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)
            uint32_ty = IntegerType.get_unsigned(32)
            ty_bneck_10_layer1_in = MemRefType.get(
                (
                    bneck_10_InW1,
                    1,
                    bneck_10_InC1,
                ),
                int8_ty,
            )
            ty_bneck_10_layer2_in = MemRefType.get(
                (
                    bneck_10_InW2,
                    1,
                    bneck_10_OutC1,
                ),
                uint8_ty,
            )
            ty_bneck_10_layer3_in = MemRefType.get(
                (
                    bneck_10_InW3,
                    1,
                    bneck_10_OutC2,
                ),
                uint8_ty,
            )
            ty_bneck_10_layer1_wts = MemRefType.get(
                (bneck_10_InC1 * bneck_10_OutC1,), int8_ty
            )
            ty_bneck_10_layer2_wts = MemRefType.get(
                (3 * 3 * bneck_10_OutC2 * 1,), int8_ty
            )
            ty_bneck_10_layer3_wts = MemRefType.get(
                (bneck_10_OutC2 * bneck_10_OutC3,), int8_ty
            )
            ty_bneck_10_all_wts= MemRefType.get(
                (
                    bneck_10_InC1 * bneck_10_OutC1
                    + 3 * 3 * bneck_10_OutC2 * 1
                    + bneck_10_OutC2 * bneck_10_OutC3,
                ),
                int8_ty,
            )
            ty_bneck_10_layer1_out = MemRefType.get(
                (
                    bneck_10_InW2,
                    1,
                    bneck_10_OutC1,
                ),
                uint8_ty,
            )
            ty_bneck_10_layer2_out = MemRefType.get(
                (
                    bneck_10_InW3,
                    1,
                    bneck_10_OutC2,
                ),
                uint8_ty,
            )
            ty_bneck_10_layer3_out = MemRefType.get(
                (
                    bneck_10_InW3,
                    1,
                    bneck_10_OutC3,
                ),
                int8_ty,
            )

            # AIE Core Function declarations
            conv2dk1_fused_relu = external_func(
                "conv2dk1_i8",
                inputs=[
                    ty_bneck_10_layer1_in,
                    ty_bneck_10_layer1_wts,
                    ty_bneck_10_layer1_out,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )
            conv2dk3_dw = external_func(
                "conv2dk3_ui8",
                inputs=[
                    ty_bneck_10_layer2_in,
                    ty_bneck_10_layer2_in,
                    ty_bneck_10_layer2_in,
                    ty_bneck_10_layer2_wts,
                    ty_bneck_10_layer2_out,
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
            conv2dk1_ui8 = external_func(
                "conv2dk1_ui8",
                inputs=[
                    ty_bneck_10_layer3_in,
                    ty_bneck_10_layer3_wts,
                    ty_bneck_10_layer3_out,
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
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)


            # AIE-array data movement with object fifos
            # Input
            OF_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, ty_bneck_10_layer1_in
            )
            OF_bneck_10_memtile_layer1_act = object_fifo("OF_bneck_10_memtile_layer1_act", MemTile, ComputeTile2, 2, ty_bneck_10_layer1_in)
            object_fifo_link(OF_inOF_act_L3L2, OF_bneck_10_memtile_layer1_act)

            # wts
            OF_bneck_10_wts_L3L2 = object_fifo(
                "OF_bneck_10_wts_L3L2", ShimTile, MemTile, 1, ty_bneck_10_all_wts
            )
       
            OF_bneck_10_wts_memtile_layer1 = object_fifo(
                "OF_bneck_10_wts_memtile_layer1", MemTile, ComputeTile2, 1, ty_bneck_10_layer1_wts
            )
            OF_bneck_10_wts_memtile_layer2 = object_fifo(
                "OF_bneck_10_wts_memtile_layer2",
                MemTile,
                ComputeTile3,
                1,
                ty_bneck_10_layer2_wts,
            )
            OF_bneck_10_wts_memtile_layer3 = object_fifo(
                "OF_bneck_10_wts_memtile_layer3",
                MemTile,
                ComputeTile4,
                1,
                ty_bneck_10_layer3_wts,
            )
            object_fifo_link(OF_bneck_10_wts_L3L2, [OF_bneck_10_wts_memtile_layer1, OF_bneck_10_wts_memtile_layer2, OF_bneck_10_wts_memtile_layer3])

            # Output
            OF_bneck_10_act_layer1_layer2 = object_fifo("OF_bneck_10_act_layer1_layer2", ComputeTile2, [ComputeTile3], 4,ty_bneck_10_layer2_in,via_DMA=True)
            OF_bneck_10_act_layer2_layer3 = object_fifo("OF_bneck_10_act_layer2_layer3", ComputeTile3, [ComputeTile4], 2,ty_bneck_10_layer3_in)
            
            OF_bneck_10_layer3_final = object_fifo("OF_bneck_10_layer3_final", ComputeTile4, [MemTile], 2, ty_bneck_10_layer3_out)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, ty_bneck_10_layer3_out)
            object_fifo_link(OF_bneck_10_layer3_final, OF_outOFL2L3)


            # Set up compute tiles

            rtp2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")
            rtp3 = Buffer(ComputeTile3, [16], T.i32(), "rtp3")
            rtp4 = Buffer(ComputeTile4, [16], T.i32(), "rtp4")

             # 1x1 conv2d
            @core(ComputeTile2, "conv2dk1_fused_relu.o")
            def core_body():
                for _ in for_(sys.maxsize):

                    # acquire weights once
                    element0Weights = OF_bneck_10_wts_memtile_layer1.acquire(ObjectFifoPort.Consume, 1)
                    scale = memref.load(rtp2, [0])
                    for _ in for_(bneck_10_InH1):
                        element0ActivactionsIn = OF_bneck_10_memtile_layer1_act.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element0ActivactionsOut = OF_bneck_10_act_layer1_layer2.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            conv2dk1_fused_relu,
                            [
                                element0ActivactionsIn,
                                element0Weights,
                                element0ActivactionsOut,
                                bneck_10_InW1,
                                bneck_10_InC1,
                                bneck_10_OutC1,
                                scale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_memtile_layer1_act", 1)

                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_10_act_layer1_layer2", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_wts_memtile_layer1", 1)
                    yield_([])

            # # # # Compute tile 3
            @core(ComputeTile3, "conv2dk3_dw.o")
            def core_body():
                scale = 8
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = OF_bneck_10_wts_memtile_layer2.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile3, 0)

                    # pre-amble: top row
                    elementActivactionsIn = OF_bneck_10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_10_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            bneck_10_InW2,
                            1,
                            bneck_10_OutC2,
                            3,
                            3,
                            0,
                            scale,
                            0,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_10_act_layer2_layer3", 1)

                    # middle
                    for _ in for_(bneck_10_InH2 - 2):
                        elementActivactionsIn = OF_bneck_10_act_layer1_layer2.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = OF_bneck_10_act_layer2_layer3.acquire(
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
                                bneck_10_InW2,
                                1,
                                bneck_10_OutC2,
                                3,
                                3,
                                1,
                                scale,
                                0,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_act_layer1_layer2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_10_act_layer2_layer3", 1)
                        yield_([])

                    # last part
                    elementActivactionsIn = OF_bneck_10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_10_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            bneck_10_InW2,
                            1,
                            bneck_10_OutC2,
                            3,
                            3,
                            2,
                            scale,
                            0,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_act_layer1_layer2", 2)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_10_act_layer2_layer3", 1)

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_wts_memtile_layer2", 1)
                    yield_([])

            # Compute tile 4
            @core(ComputeTile4, "conv2dk1_ui8.o")
            def core_body():

                for _ in for_(0xFFFFFFFF):
                    elemWts = OF_bneck_10_wts_memtile_layer3.acquire(ObjectFifoPort.Consume, 1)

                    scale = memref.load(rtp4, [0])
                    # scale = memref.load(rtpComputeTile2, [0])

                    for _ in for_(bneck_10_InH3):
                        elemIn = OF_bneck_10_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_10_layer3_final.acquire(ObjectFifoPort.Produce, 1)

                        call(
                            conv2dk1_ui8,
                            [
                                elemIn,
                                elemWts,
                                elemOut0,
                                bneck_10_InW3,
                                bneck_10_OutC2,
                                bneck_10_OutC3,
                                scale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_act_layer2_layer3", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_10_layer3_final", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_wts_memtile_layer3", 1)
                    yield_([])

            # # instruction stream generation
            activationsInSize32b = (bneck_10_InW1 * bneck_10_InH1 * bneck_10_InC1) // 4
            acitivationsOutSize32b = (bneck_10_InW3 * bneck_10_InH3 * bneck_10_OutC3) // 4
            totalWeightsSize32b = (
            bneck_10_InC1*bneck_10_OutC1+
               3 * 3 * bneck_10_OutC2 * 1+
               bneck_10_OutC2*bneck_10_OutC3
            ) // 4

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                NpuWriteRTPOp("rtp2", col=0, row=2, index=0, value=9)
                NpuWriteRTPOp("rtp3", col=0, row=3, index=0, value=8)
                NpuWriteRTPOp("rtp4", col=0, row=4, index=0, value=12)
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
                    metadata="OF_bneck_10_wts_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeightsSize32b],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


mobilenetBottleneckB()