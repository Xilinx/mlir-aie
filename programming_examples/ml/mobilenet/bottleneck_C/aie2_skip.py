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

bneck_13_InW1 = 7
bneck_13_InH1 = 7
bneck_13_InC1 = 160
bneck_13_OutC1 = 960 // 4

bneck_13_InW2 = 7
bneck_13_InH2 = 7
bneck_13_OutC2 = bneck_13_OutC1

bneck_13_InW3 = 7
bneck_13_InH3 = 7
bneck_13_OutC3 = 160

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

        # ************************ bneck13 ************************
            # input
            ty_bneck_13_layer1_in = MemRefType.get(
                (
                    bneck_13_InW1,
                    1,
                    bneck_13_InC1,
                ),
                int8_ty,
            )
            ty_bneck_13_layer2_in = MemRefType.get(
                (
                    bneck_13_InW2,
                    1,
                    bneck_13_OutC1,
                ),
                uint8_ty,
            )
            ty_bneck_13_layer3_in = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC2,
                ),
                uint8_ty,
            )
         
            # define wts
            ty_bneck_13_layer1_wts = MemRefType.get(
                (bneck_13_InC1 * bneck_13_OutC1,), int8_ty
            )
            ty_bneck_13_layer2_wts = MemRefType.get(
                (3 * 3 * bneck_13_OutC2 * 1,), int8_ty
            )
            ty_bneck_13_layer3_wts = MemRefType.get(
                (bneck_13_OutC2 * bneck_13_OutC3,), int8_ty
            )
            ty_bneck_13_all_wts= MemRefType.get(
                (
                    bneck_13_InC1 * bneck_13_OutC1
                    + 3 * 3 * bneck_13_OutC2 * 1
                    + bneck_13_OutC2 * bneck_13_OutC3,
                ),
                int8_ty,
            )

            # output
            ty_bneck_13_layer1_out = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC1,
                ),
                uint8_ty,
            )
            ty_bneck_13_layer2_out = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC2,
                ),
                uint8_ty,
            )
            ty_bneck_13_layer3_out = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC3,
                ),
                int8_ty,
            )
            
# HERE
            
            # ************************ bneck13 ************************
            bn13_conv2dk1_fused_relu = external_func(
                "bn11_conv2dk1_i8",
                inputs=[
                    ty_bneck_13_layer1_in,
                    ty_bneck_13_layer1_wts,
                    ty_bneck_13_layer1_out,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )
            bn13_conv2dk3_dw = external_func(
                "bn11_conv2dk3_ui8",
                inputs=[
                    ty_bneck_13_layer2_in,
                    ty_bneck_13_layer2_in,
                    ty_bneck_13_layer2_in,
                    ty_bneck_13_layer2_wts,
                    ty_bneck_13_layer2_out,
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
            bn13_conv2dk1_skip = external_func(
                "conv2dk1_skip_ui8_i8_i8",
                inputs=[
                    ty_bneck_13_layer3_in,
                    ty_bneck_13_layer3_wts,
                    ty_bneck_13_layer3_out,
                    ty_bneck_13_layer1_in,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)

            MemTile01 = tile(0, 1)


            ComputeTile02 = tile(0, 2)
            ComputeTile03 = tile(0, 3)
            ComputeTile04 = tile(0, 4)
            # bn11
            # ComputeTile05 = tile(0, 5)
            # ComputeTile15 = tile(1, 5)


            # AIE-array data movement with object fifos
            # ************************ bneck13 ************************
            # Input
            inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2",
                ShimTile00,
                [ComputeTile02, MemTile01],
                [2, 2, 4],
                ty_bneck_13_layer1_in,
            )
            OF_bneck_13_skip = object_fifo(
                "OF_bneck_13_skip", MemTile01, ComputeTile04, 2, ty_bneck_13_layer1_in
            )
            object_fifo_link(inOF_act_L3L2, OF_bneck_13_skip)
            
            
            OF_bneck_13_act_layer1_layer2 = object_fifo("OF_bneck_13_act_layer1_layer2", ComputeTile02, [ComputeTile03], 4,ty_bneck_13_layer2_in,via_DMA=True)
            OF_bneck_13_act_layer2_layer3 = object_fifo("OF_bneck_13_act_layer2_layer3", ComputeTile03, [ComputeTile04], 2,ty_bneck_13_layer3_in)

            # # wts
            OF_bneck_13_wts_L3L2 = object_fifo(
                "OF_bneck_13_wts_L3L2", ShimTile00, MemTile01, 1, ty_bneck_13_all_wts
            )
       
            OF_bneck_13_wts_memtile_layer1 = object_fifo(
                "OF_bneck_13_wts_memtile_layer1", MemTile01, ComputeTile02, 1, ty_bneck_13_layer1_wts
            )
            OF_bneck_13_wts_memtile_layer2 = object_fifo(
                "OF_bneck_13_wts_memtile_layer2",
                MemTile01,
                ComputeTile03,
                1,
                ty_bneck_13_layer2_wts,
            )
            OF_bneck_13_wts_memtile_layer3 = object_fifo(
                "OF_bneck_13_wts_memtile_layer3",
                MemTile01,
                ComputeTile04,
                1,
                ty_bneck_13_layer3_wts,
            )
            object_fifo_link(OF_bneck_13_wts_L3L2, [OF_bneck_13_wts_memtile_layer1, OF_bneck_13_wts_memtile_layer2, OF_bneck_13_wts_memtile_layer3])

        
            # Set up compute tiles

            rtp02 = Buffer(ComputeTile02, [16], T.i32(), "rtp02")
            rtp03 = Buffer(ComputeTile03, [16], T.i32(), "rtp03")
            rtp04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")


            OF_bneck_13_layer3_bn_12_layer1 = object_fifo("OF_bneck_13_layer3_bn_12_layer1", ComputeTile04, [MemTile01], 2, ty_bneck_13_layer3_out)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile01, [ShimTile00], 2, ty_bneck_13_layer3_out)
            object_fifo_link(OF_bneck_13_layer3_bn_12_layer1, OF_outOFL2L3)
        # ************************ bneck13 ************************
             # 1x1 conv2d
            @core(ComputeTile02, "bn11_conv2dk1_fused_relu.o")
            def core_body():
                for _ in for_(sys.maxsize):

                    # acquire weights once
                    element0Weights = OF_bneck_13_wts_memtile_layer1.acquire(ObjectFifoPort.Consume, 1)
                    scale = memref.load(rtp02, [0])
                    for _ in for_(bneck_13_InH1):
                        element0ActivactionsIn = inOF_act_L3L2.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element0ActivactionsOut = OF_bneck_13_act_layer1_layer2.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            bn13_conv2dk1_fused_relu,
                            [
                                element0ActivactionsIn,
                                element0Weights,
                                element0ActivactionsOut,
                                bneck_13_InW1,
                                bneck_13_InC1,
                                bneck_13_OutC1,
                                scale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)

                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer1_layer2", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer1", 1)
                    yield_([])

            # # # Compute tile 3
            @core(ComputeTile03, "bn11_conv2dk3_dw.o")
            def core_body():
                scale = 8
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = OF_bneck_13_wts_memtile_layer2.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile03, 0)

                    # pre-amble: top row
                    elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_13_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn13_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            bneck_13_InW2,
                            1,
                            bneck_13_OutC2,
                            3,
                            3,
                            0,
                            scale,
                            0,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3", 1)

                    # middle
                    for _ in for_(bneck_13_InH2 - 2):
                        elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = OF_bneck_13_act_layer2_layer3.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            bn13_conv2dk3_dw,
                            [
                                elementActivactionsIn[0],
                                elementActivactionsIn[1],
                                elementActivactionsIn[2],
                                element0Weights,
                                element0ActivactionsOut,
                                bneck_13_InW2,
                                1,
                                bneck_13_OutC2,
                                3,
                                3,
                                1,
                                scale,
                                0,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer1_layer2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3", 1)
                        yield_([])

                    # last part
                    elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_13_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn13_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            bneck_13_InW2,
                            1,
                            bneck_13_OutC2,
                            3,
                            3,
                            2,
                            scale,
                            0,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer1_layer2", 2)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3", 1)

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer2", 1)
                    yield_([])

            # Compute tile 4
            @core(ComputeTile04, "bn_conv2dk1_skip.o")
            def core_body():

                for _ in for_(0xFFFFFFFF):
                    elemWts = OF_bneck_13_wts_memtile_layer3.acquire(ObjectFifoPort.Consume, 1)

                    scale = memref.load(rtp04, [0])
                    skipScale = memref.load(rtp04, [1])
                    # scale = memref.load(rtpComputeTile02, [0])

                    for _ in for_(bneck_13_InH3):
                        elemIn = OF_bneck_13_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_13_layer3_bn_12_layer1.acquire(ObjectFifoPort.Produce, 1)
                        elementSkipsIn = OF_bneck_13_skip.acquire(
                                ObjectFifoPort.Consume, 1
                            )

                        call(
                            bn13_conv2dk1_skip,
                            [
                                elemIn,
                                elemWts,
                                elemOut0,
                                elementSkipsIn,
                                bneck_13_InW3,
                                bneck_13_OutC2,
                                bneck_13_OutC3,
                                scale,
                                skipScale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer2_layer3", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_layer3_bn_12_layer1", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_skip", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer3", 1)
                    yield_([])
            

            # # instruction stream generation
            activationsInSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_InC1) // 4
            # acitivationsOutSize32b = (bneck_12_InW2 * bneck_12_InH2 * bneck_12_OutC3) // 4
            acitivationsOutSize32b = (bneck_13_InW2 * bneck_13_InW2 * bneck_13_OutC3) // 4

            bn13_totalWeightsSize32b = (
            bneck_13_InC1*bneck_13_OutC1+
               3 * 3 * bneck_13_OutC2 * 1+
               bneck_13_OutC2*bneck_13_OutC3
            ) // 4

            bn13_totalWeightsSize32b = (
            bneck_13_OutC3*bneck_13_OutC1+
               3 * 3 * bneck_13_OutC2 * 1+
               bneck_13_OutC2*bneck_13_OutC3
            ) // 4

            bn12_totalWeightsSize32b = (
            bneck_13_OutC3*bneck_13_OutC1+
               3 * 3 * bneck_13_OutC2 * 1+
               bneck_13_OutC2*bneck_13_OutC3
            ) // 4


            bn12_Offset_32b = bn13_totalWeightsSize32b+bn13_totalWeightsSize32b



            totalWeightsSize32b_complete = (
                bn13_totalWeightsSize32b + bn13_totalWeightsSize32b + bn12_totalWeightsSize32b
            )

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

            @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                NpuWriteRTPOp("rtp02", index=0, value=9)
                NpuWriteRTPOp("rtp03", index=0, value=8)
                NpuWriteRTPOp("rtp04", index=0, value=11)
                NpuWriteRTPOp("rtp04", index=1, value=0)
                
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
                    metadata="OF_bneck_13_wts_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, bn13_totalWeightsSize32b],
                )

                npu_sync(column=0, row=0, direction=0, channel=0)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


mobilenetBottleneckB()