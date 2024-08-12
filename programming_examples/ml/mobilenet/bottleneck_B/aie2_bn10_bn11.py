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

import json
def read_scale_factors(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Read the existing scale factors
file_path = 'scale_factors.json'
scale_factors = read_scale_factors(file_path)

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

bneck_11_OutC1 = 336
bneck_11_OutC2 = 336
bneck_11_OutC3 = 112

bneck_12_OutC1 = 336
bneck_12_OutC2 = 336
bneck_12_InW2 = 7
bneck_12_InH2 = 7
bneck_12_OutC3 = 80


OutC=bneck_11_OutC3
OutH=bneck_10_InH3
OutW=bneck_10_InW3

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4


def mobilenetBottleneckB(bn10_scaleFactor1=10,bn10_scaleFactor2=7,bn10_scaleFactor3=9,
                           bn11_scaleFactor1=9,bn11_scaleFactor2=8,bn11_scaleFactor3=12,bn11_scaleFactorAdd=1,
                           bn12_scaleFactor1=8,bn12_scaleFactor2=8,bn12_scaleFactor3=9):
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_3col)
        def device_body():

            # define types
            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)
            uint32_ty = IntegerType.get_unsigned(32)
            # ************************ bneck10 ************************
            ty_bneck_10_layer1_in = MemRefType.get((bneck_10_InW1, 1, bneck_10_InC1, ), int8_ty, ) 
            ty_bneck_10_layer2_in = MemRefType.get((bneck_10_InW2, 1, bneck_10_OutC1, ), uint8_ty, ) 
            ty_bneck_10_layer3_in = MemRefType.get((bneck_10_InW3, 1, bneck_10_OutC2, ), uint8_ty, ) 

            # define wts 
            ty_bneck_10_layer1_wts = MemRefType.get((bneck_10_InC1 * bneck_10_OutC1,), int8_ty ) 
            ty_bneck_10_layer2_wts = MemRefType.get((3 * 3 * bneck_10_OutC2 * 1,), int8_ty ) 
            ty_bneck_10_layer3_wts = MemRefType.get((bneck_10_OutC2 * bneck_10_OutC3,), int8_ty ) 
            ty_bneck_10_all_wts= MemRefType.get((bneck_10_InC1 * bneck_10_OutC1 + 3 * 3 * bneck_10_OutC2 * 1 + bneck_10_OutC2 * bneck_10_OutC3, ), int8_ty, ) 

            # output 
            ty_bneck_10_layer1_out = MemRefType.get((bneck_10_InW2, 1, bneck_10_OutC1, ), uint8_ty, ) 
            ty_bneck_10_layer2_out = MemRefType.get((bneck_10_InW3, 1, bneck_10_OutC2, ), uint8_ty, ) 
            ty_bneck_10_layer3_out = MemRefType.get((bneck_10_InW3, 1, bneck_10_OutC3, ), int8_ty, ) 
            # ************************ bneck11 ************************ 
            # input 
            ty_bneck_11_layer1_in = MemRefType.get((bneck_10_InW3, 1, bneck_10_OutC3, ), int8_ty, ) 
            ty_bneck_11_layer2_in = MemRefType.get((bneck_10_InW3, 1, bneck_11_OutC1, ), uint8_ty, ) 
            ty_bneck_11_layer3_in = MemRefType.get((bneck_10_InW3, 1, bneck_11_OutC2, ), uint8_ty, ) 
            # define wts 
            ty_bneck_11_layer1_wts = MemRefType.get((bneck_10_OutC3 * bneck_11_OutC1,), int8_ty ) 
            ty_bneck_11_layer2_wts = MemRefType.get((3 * 3 * bneck_11_OutC2 * 1,), int8_ty ) 
            ty_bneck_11_layer3_wts = MemRefType.get((bneck_11_OutC2 * bneck_11_OutC3,), int8_ty ) 
            ty_bneck_11_all_wts= MemRefType.get((bneck_10_OutC3 * bneck_11_OutC1 + 3 * 3 * bneck_11_OutC2 * 1 + bneck_11_OutC2 * bneck_11_OutC3, ), int8_ty, ) 
            # output 
            ty_bneck_11_layer1_out = MemRefType.get((bneck_10_InW3, 1, bneck_11_OutC1, ), uint8_ty, ) 
            ty_bneck_11_layer2_out = MemRefType.get((bneck_10_InW3, 1, bneck_11_OutC2, ), uint8_ty, ) 
            ty_bneck_11_layer3_out = MemRefType.get((bneck_10_InW3, 1, bneck_11_OutC3, ), int8_ty, ) 
            # ************************ bneck12 ************************ 
            ty_bneck_12_layer1_in = MemRefType.get((bneck_10_InW1, 1, bneck_11_OutC3, ), int8_ty, ) 
            ty_bneck_12_layer2_in = MemRefType.get((bneck_10_InW1, 1, bneck_12_OutC1, ), uint8_ty, ) 
            ty_bneck_12_layer3_in = MemRefType.get((bneck_12_InW2, 1, bneck_12_OutC2, ), uint8_ty, ) 
            # define wts 
            ty_bneck_12_layer1_wts = MemRefType.get((bneck_11_OutC3 * bneck_12_OutC1,), int8_ty ) 
            ty_bneck_12_layer2_wts = MemRefType.get((3 * 3 * bneck_12_OutC2 * 1,), int8_ty ) 
            ty_bneck_12_layer3_wts = MemRefType.get((bneck_12_OutC2 * bneck_12_OutC3,), int8_ty ) 
            ty_bneck_12_all_wts= MemRefType.get((bneck_11_OutC3 * bneck_12_OutC1 + 3 * 3 * bneck_12_OutC2 * 1 + bneck_12_OutC2 * bneck_12_OutC3, ), int8_ty, ) 
            # output 
            ty_bneck_12_layer1_out = MemRefType.get((bneck_10_InW3, 1, bneck_12_OutC1, ), uint8_ty, ) 
            ty_bneck_12_layer2_out = MemRefType.get((bneck_12_InW2, 1, bneck_12_OutC2, ), uint8_ty, ) 
            ty_bneck_12_layer3_out = MemRefType.get((bneck_12_InW2, 1, bneck_12_OutC3, ), int8_ty, )
            # AIE Core Function declarations
            # ************************ bneck10 ************************
            bn10_conv2dk1_fused_relu = external_func("bn10_conv2dk1_relu_i8_ui8", inputs=[ty_bneck_10_layer1_in, ty_bneck_10_layer1_wts, ty_bneck_10_layer1_out, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            bn10_conv2dk3_dw = external_func("bn10_conv2dk3_dw_stride1_relu_ui8_ui8", inputs=[ty_bneck_10_layer2_in, ty_bneck_10_layer2_in, ty_bneck_10_layer2_in, ty_bneck_10_layer2_wts, ty_bneck_10_layer2_out, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            bn10_conv2dk1_ui8 = external_func("bn10_conv2dk1_ui8_i8", inputs=[ty_bneck_10_layer3_in, ty_bneck_10_layer3_wts, ty_bneck_10_layer3_out, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            # ************************ bneck11 ************************
            bn11_conv2dk1_fused_relu = external_func("bn11_conv2dk1_relu_i8_ui8", inputs=[ty_bneck_11_layer1_in, ty_bneck_11_layer1_wts, ty_bneck_11_layer1_out, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            bn11_conv2dk3_dw = external_func("bn11_conv2dk3_dw_stride1_relu_ui8_ui8", inputs=[ty_bneck_11_layer2_in, ty_bneck_11_layer2_in, ty_bneck_11_layer2_in, ty_bneck_11_layer2_wts, ty_bneck_11_layer2_out, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            bn11_conv2dk1_skip = external_func("bn11_conv2dk1_skip_ui8_i8_i8", inputs=[ty_bneck_11_layer3_in, ty_bneck_11_layer3_wts, ty_bneck_11_layer3_out, ty_bneck_11_layer1_in, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], )

             # ************************ bneck12 ************************
            bn12_conv2dk1_fused_relu = external_func("bn12_conv2dk1_relu_i8_ui8", inputs=[ty_bneck_12_layer1_in, ty_bneck_12_layer1_wts, ty_bneck_12_layer1_out, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            bn12_conv2dk3_dw = external_func("bn12_conv2dk3_dw_stride2_relu_ui8_ui8", inputs=[ty_bneck_12_layer2_in, ty_bneck_12_layer2_in, ty_bneck_12_layer2_in, ty_bneck_12_layer2_wts, ty_bneck_12_layer2_out, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            bn12_conv2dk1_ui8 = external_func("bn12_conv2dk1_ui8_i8", inputs=[ty_bneck_12_layer3_in, ty_bneck_12_layer3_wts, ty_bneck_12_layer3_out, int32_ty, int32_ty, int32_ty, int32_ty, ], )
            
            # Tile declarations
            ShimTile00 = tile(0, 0)
            ShimTile10 = tile(1, 0)


            MemTile01 = tile(0, 1)
            MemTile11 = tile(1, 1)
            MemTile21 = tile(2, 1)

            ComputeTile02 = tile(0, 2)
            ComputeTile03 = tile(0, 3)
            ComputeTile04 = tile(0, 4)
            # bn11
            ComputeTile05 = tile(0, 5)
            ComputeTile15 = tile(1, 5)
            ComputeTile14 = tile(1, 4)

            # bn12
            ComputeTile13 = tile(1, 3)
            ComputeTile12 = tile(1, 2)
            ComputeTile22 = tile(2, 2)


            # AIE-array data movement with object fifos
            # ************************ bneck10 ************************
            # Input
            OF_inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile00, MemTile01, 2, ty_bneck_10_layer1_in )
            OF_bneck_10_memtile_layer1_act = object_fifo("OF_bneck_10_memtile_layer1_act", MemTile01, ComputeTile02, 2, ty_bneck_10_layer1_in)
            object_fifo_link(OF_inOF_act_L3L2, OF_bneck_10_memtile_layer1_act)

            # wts
            OF_bneck_10_wts_L3L2 = object_fifo("OF_bneck_10_wts_L3L2", ShimTile00, MemTile01, 1, ty_bneck_10_all_wts )
       
            OF_bneck_10_wts_memtile_layer1 = object_fifo("OF_bneck_10_wts_memtile_layer1", MemTile01, ComputeTile02, 1, ty_bneck_10_layer1_wts )
            OF_bneck_10_wts_memtile_layer2 = object_fifo("OF_bneck_10_wts_memtile_layer2", MemTile01, ComputeTile03, 1, ty_bneck_10_layer2_wts, )
            OF_bneck_10_wts_memtile_layer3 = object_fifo("OF_bneck_10_wts_memtile_layer3", MemTile01, ComputeTile04, 1, ty_bneck_10_layer3_wts, )
            object_fifo_link(OF_bneck_10_wts_L3L2, [OF_bneck_10_wts_memtile_layer1, OF_bneck_10_wts_memtile_layer2, OF_bneck_10_wts_memtile_layer3],[],[0,bneck_10_InC1 * bneck_10_OutC1,bneck_10_InC1 * bneck_10_OutC1+3 * 3 * bneck_10_OutC2 * 1])

            # Output
            OF_bneck_10_act_layer1_layer2 = object_fifo("OF_bneck_10_act_layer1_layer2", ComputeTile02, [ComputeTile03], 4,ty_bneck_10_layer2_in,via_DMA=True)
            OF_bneck_10_act_layer2_layer3 = object_fifo("OF_bneck_10_act_layer2_layer3", ComputeTile03, [ComputeTile04], 2,ty_bneck_10_layer3_in)
            
            # ************************ bneck11 ************************
            # OF_bneck_10_layer3_bn_11_layer1 = object_fifo("OF_bneck_10_layer3_bn_11_layer1", ComputeTile04, [ComputeTile05], 2, ty_bneck_11_layer1_in)
       
            OF_bneck_10_layer3_bn_11_layer1 = object_fifo("OF_bneck_10_layer3_bn_11_layer1", ComputeTile04, [ComputeTile05,MemTile11], [2, 2, 6], ty_bneck_11_layer1_in)
            OF_bneck_11_skip = object_fifo("OF_bneck_11_skip", MemTile11, [ComputeTile14], 2,ty_bneck_11_layer1_in)
            object_fifo_link(OF_bneck_10_layer3_bn_11_layer1,OF_bneck_11_skip )

            
            OF_bneck_11_act_layer1_layer2 = object_fifo("OF_bneck_11_act_layer1_layer2", ComputeTile05, [ComputeTile15], 4,ty_bneck_11_layer2_in,via_DMA=True)
            OF_bneck_11_act_layer2_layer3 = object_fifo("OF_bneck_11_act_layer2_layer3", ComputeTile15, [ComputeTile14], 2,ty_bneck_11_layer3_in)

            # # wts
            OF_bneck_11_wts_L3L2 = object_fifo("OF_bneck_11_wts_L3L2", ShimTile10, MemTile11, 1, ty_bneck_11_all_wts )
            OF_bneck_11_wts_memtile_layer1 = object_fifo("OF_bneck_11_wts_memtile_layer1", MemTile11, ComputeTile05, 1, ty_bneck_11_layer1_wts )
            OF_bneck_11_wts_memtile_layer2 = object_fifo("OF_bneck_11_wts_memtile_layer2", MemTile11, ComputeTile15, 1, ty_bneck_11_layer2_wts, )
            OF_bneck_11_wts_memtile_layer3 = object_fifo("OF_bneck_11_wts_memtile_layer3", MemTile11, ComputeTile14, 1, ty_bneck_11_layer3_wts, )
            object_fifo_link(OF_bneck_11_wts_L3L2, [OF_bneck_11_wts_memtile_layer1, OF_bneck_11_wts_memtile_layer2, OF_bneck_11_wts_memtile_layer3],[],[0,bneck_10_OutC3 * bneck_11_OutC1,bneck_10_OutC3 * bneck_11_OutC1+3 * 3 * bneck_11_OutC2 * 1])

            



            # ************************ bneck12 ************************
            # # wts
            OF_bneck_12_wts_L3L2 = object_fifo("OF_bneck_12_wts_L3L2", ShimTile10, MemTile21, 1, ty_bneck_12_all_wts )
            OF_bneck_12_wts_memtile_layer1 = object_fifo("OF_bneck_12_wts_memtile_layer1", MemTile21, ComputeTile13, 1, ty_bneck_12_layer1_wts )
            OF_bneck_12_wts_memtile_layer2 = object_fifo("OF_bneck_12_wts_memtile_layer2", MemTile21, ComputeTile12, 1, ty_bneck_12_layer2_wts, )
            OF_bneck_12_wts_memtile_layer3 = object_fifo("OF_bneck_12_wts_memtile_layer3", MemTile21, ComputeTile22, 1, ty_bneck_12_layer3_wts, )
            object_fifo_link(OF_bneck_12_wts_L3L2, [OF_bneck_12_wts_memtile_layer1, OF_bneck_12_wts_memtile_layer2, OF_bneck_12_wts_memtile_layer3],[],[0,bneck_11_OutC3 * bneck_12_OutC1,bneck_11_OutC3 * bneck_12_OutC1+3 * 3 * bneck_12_OutC2 * 1])

            
            OF_bneck_11_layer3_bn_12_layer1 = object_fifo("OF_bneck_11_layer3_bn_12_layer1", ComputeTile14, [MemTile21], 2, ty_bneck_12_layer1_in)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile21, [ShimTile10], 2, ty_bneck_12_layer1_in)
            object_fifo_link(OF_bneck_11_layer3_bn_12_layer1, OF_outOFL2L3)
            # Set up compute tiles

            # rtp02 = Buffer(ComputeTile02, [16], T.i32(), "rtp02")
            # rtp03 = Buffer(ComputeTile03, [16], T.i32(), "rtp03")
            # rtp04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")

            # rtp05 = Buffer(ComputeTile05, [16], T.i32(), "rtp05")
            # rtp15 = Buffer(ComputeTile15, [16], T.i32(), "rtp15")
            # rtp14 = Buffer(ComputeTile14, [16], T.i32(), "rtp14")

            # rtp13 = Buffer(ComputeTile13, [16], T.i32(), "rtp13")
            # rtp12 = Buffer(ComputeTile12, [16], T.i32(), "rtp12")
            # rtp22 = Buffer(ComputeTile22, [16], T.i32(), "rtp22")

        # ************************ bneck10 ************************
             # 1x1 conv2d
            @core(ComputeTile02, "bn10_conv2dk1_fused_relu.o")
            def core_body():
                for _ in for_(sys.maxsize):

                    # acquire weights once
                    element0Weights = OF_bneck_10_wts_memtile_layer1.acquire(ObjectFifoPort.Consume, 1)
                    scale = bn10_scaleFactor1
                    for _ in for_(bneck_10_InH1):
                        element0ActivactionsIn = OF_bneck_10_memtile_layer1_act.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element0ActivactionsOut = OF_bneck_10_act_layer1_layer2.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            bn10_conv2dk1_fused_relu,
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

            # # # Compute tile 3
            @core(ComputeTile03, "bn10_conv2dk3_dw.o")
            def core_body():
                scale = bn10_scaleFactor2
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = OF_bneck_10_wts_memtile_layer2.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile03, 0)

                    # pre-amble: top row
                    elementActivactionsIn = OF_bneck_10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_10_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn10_conv2dk3_dw,
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
                            bn10_conv2dk3_dw,
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
                        bn10_conv2dk3_dw,
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
            @core(ComputeTile04, "bn10_conv2dk1_ui8.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    elemWts = OF_bneck_10_wts_memtile_layer3.acquire(ObjectFifoPort.Consume, 1)

                    scale = bn10_scaleFactor3
                    # scale = memref.load(rtpComputeTile02, [0])

                    for _ in for_(bneck_10_InH3):
                        elemIn = OF_bneck_10_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_10_layer3_bn_11_layer1.acquire(ObjectFifoPort.Produce, 1)

                        call(
                            bn10_conv2dk1_ui8,
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
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_10_layer3_bn_11_layer1", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_wts_memtile_layer3", 1)
                    yield_([])
            
        # ************************ bneck11 ************************
            #     # 1x1 conv2d
            @core(ComputeTile05, "bn11_conv2dk1_fused_relu.o")
            def core_body():
                for _ in for_(sys.maxsize):

                    # acquire weights once
                    element0Weights = OF_bneck_11_wts_memtile_layer1.acquire(ObjectFifoPort.Consume, 1)
                    scale = bn11_scaleFactor1
                    for _ in for_(bneck_10_InH1):
                        element0ActivactionsIn = OF_bneck_10_layer3_bn_11_layer1.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element0ActivactionsOut = OF_bneck_11_act_layer1_layer2.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            bn11_conv2dk1_fused_relu,
                            [
                                element0ActivactionsIn,
                                element0Weights,
                                element0ActivactionsOut,
                                bneck_10_InW1,
                                bneck_10_OutC3,
                                bneck_11_OutC1,
                                scale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_10_layer3_bn_11_layer1", 1)

                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_11_act_layer1_layer2", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_wts_memtile_layer1", 1)
                    yield_([])

            # # # # # # Compute tile 3
            @core(ComputeTile15, "bn11_conv2dk3_dw.o")
            def core_body():
                scale = bn11_scaleFactor2
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = OF_bneck_11_wts_memtile_layer2.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile03, 0)

                    # pre-amble: top row
                    elementActivactionsIn = OF_bneck_11_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_11_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn11_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            bneck_10_InW2,
                            1,
                            bneck_11_OutC2,
                            3,
                            3,
                            0,
                            scale,
                            0,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_11_act_layer2_layer3", 1)

                    # middle
                    for _ in for_(bneck_10_InH2 - 2):
                        elementActivactionsIn = OF_bneck_11_act_layer1_layer2.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = OF_bneck_11_act_layer2_layer3.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        res = call(
                            bn11_conv2dk3_dw,
                            [
                                elementActivactionsIn[0],
                                elementActivactionsIn[1],
                                elementActivactionsIn[2],
                                element0Weights,
                                element0ActivactionsOut,
                                bneck_10_InW2,
                                1,
                                bneck_11_OutC2,
                                3,
                                3,
                                1,
                                scale,
                                0,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_act_layer1_layer2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_11_act_layer2_layer3", 1)
                        yield_([])

                    # last part
                    elementActivactionsIn = OF_bneck_11_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_11_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn11_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[1],
                            element0Weights,
                            element0ActivactionsOut,
                            bneck_10_InW2,
                            1,
                            bneck_11_OutC2,
                            3,
                            3,
                            2,
                            scale,
                            0,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_act_layer1_layer2", 2)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_11_act_layer2_layer3", 1)

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_wts_memtile_layer2", 1)
                    yield_([])

            # # Compute tile 4
            @core(ComputeTile14, "bn11_conv2dk1_skip.o")
            def core_body():

                for _ in for_(0xFFFFFFFF):
                    elemWts = OF_bneck_11_wts_memtile_layer3.acquire(ObjectFifoPort.Consume, 1)

                    scale = bn11_scaleFactor3
                    skipScale = bn11_scaleFactorAdd
                    # scale = memref.load(rtpComputeTile02, [0])

                    for _ in for_(bneck_10_InH3):
                        elemIn = OF_bneck_11_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_11_layer3_bn_12_layer1.acquire(ObjectFifoPort.Produce, 1)
                        elementSkipsIn = OF_bneck_11_skip.acquire(ObjectFifoPort.Consume, 1)

                        call(
                            bn11_conv2dk1_skip,
                            [
                                elemIn,
                                elemWts,
                                elemOut0,
                                elementSkipsIn,
                                bneck_10_InW3,
                                bneck_11_OutC2,
                                bneck_11_OutC3,
                                scale,
                                skipScale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_act_layer2_layer3", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_11_layer3_bn_12_layer1", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_skip", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_11_wts_memtile_layer3", 1)
                    yield_([])
      
            
            # # instruction stream generation
            activationsInSize32b = (bneck_10_InW1 * bneck_10_InH1 * bneck_10_InC1) // 4
            acitivationsOutSize32b = (OutW * OutH * OutC) // 4


            bn10_totalWeightsSize32b = (
            bneck_10_InC1*bneck_10_OutC1+
               3 * 3 * bneck_10_OutC2 * 1+
               bneck_10_OutC2*bneck_10_OutC3
            ) // 4

            bn11_totalWeightsSize32b = (
            bneck_10_OutC3*bneck_11_OutC1+
               3 * 3 * bneck_11_OutC2 * 1+
               bneck_11_OutC2*bneck_11_OutC3
            ) // 4

            bn12_totalWeightsSize32b = (
            bneck_11_OutC3*bneck_12_OutC1+
               3 * 3 * bneck_12_OutC2 * 1+
               bneck_12_OutC2*bneck_12_OutC3
            ) // 4


            bn12_Offset_32b = bn10_totalWeightsSize32b+bn11_totalWeightsSize32b



            totalWeightsSize32b_complete = (
                bn10_totalWeightsSize32b + bn11_totalWeightsSize32b + bn12_totalWeightsSize32b
            )

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                # NpuWriteRTPOp("rtp02", col=0, row=2, index=0, value=10)
                # NpuWriteRTPOp("rtp03", col=0, row=3, index=0, value=7)
                # NpuWriteRTPOp("rtp04", col=0, row=4, index=0, value=9)


                # NpuWriteRTPOp("rtp05", col=0, row=5, index=0, value=9)
                # NpuWriteRTPOp("rtp15", col=1, row=5, index=0, value=8)
                # NpuWriteRTPOp("rtp14", col=1, row=4, index=0, value=12)
                # NpuWriteRTPOp("rtp14", col=1, row=4, index=1, value=1)

                # NpuWriteRTPOp("rtp13", col=1, row=3, index=0, value=8)
                # NpuWriteRTPOp("rtp12", col=1, row=2, index=0, value=8)
                # NpuWriteRTPOp("rtp22", col=2, row=2, index=0, value=9)
                
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
                    sizes=[1, 1, 1, bn10_totalWeightsSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="OF_bneck_11_wts_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[0, 0, 0, bn10_totalWeightsSize32b],
                    sizes=[1, 1, 1, bn11_totalWeightsSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="OF_bneck_12_wts_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[0, 0, 0, bn12_Offset_32b],
                    sizes=[1, 1, 1, bn12_totalWeightsSize32b],
                )
                npu_sync(column=1, row=0, direction=0, channel=0)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


mobilenetBottleneckB(bn10_scaleFactor1=scale_factors["BN10"]["conv1x1_1"],bn10_scaleFactor2=scale_factors["BN10"]["conv3x3"],bn10_scaleFactor3=scale_factors["BN10"]["conv1x1_2"],
                           bn11_scaleFactor1=scale_factors["BN11"]["conv1x1_1"],bn11_scaleFactor2=scale_factors["BN11"]["conv3x3"],bn11_scaleFactor3=scale_factors["BN11"]["conv1x1_2"],bn11_scaleFactorAdd=scale_factors["BN11"]["skip_add"],
                           bn12_scaleFactor1=scale_factors["BN12"]["conv1x1_1"],bn12_scaleFactor2=scale_factors["BN12"]["conv3x3"],bn12_scaleFactor3=scale_factors["BN12"]["conv1x1_2"])


# mobilenetBottleneckB(bn10_scaleFactor1=9,bn10_scaleFactor2=8,bn10_scaleFactor3=12,
#                            bn11_scaleFactor1=9,bn11_scaleFactor2=scale_factors["BN11"]["conv3x3"],bn11_scaleFactor3=scale_factors["BN11"]["conv1x1_2"],bn11_scaleFactorAdd=scale_factors["BN11"]["skip_add"],
#                            bn12_scaleFactor1=scale_factors["BN12"]["conv1x1_1"],bn12_scaleFactor2=scale_factors["BN12"]["conv3x3"],bn12_scaleFactor3=scale_factors["BN12"]["conv1x1_2"])