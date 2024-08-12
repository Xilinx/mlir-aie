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

def create_tile(col, row):
    # Replace this with the appropriate constructor or conversion
    return aie.dialects.aie.tile(col, row)
    
def select_cores(start_col, start_row):
    # Initialize the list to store the selected cores
    selected_cores = []

    # Current position
    current_col = start_col
    current_row = start_row

    # Direction flag for snake-like pattern
    downward = True

    # Loop to select the next 9 cores
    for _ in range(9):
        # Add the current core to the list
        selected_cores.append((current_col, current_row))

        # Move to the next core based on the direction
        if downward:
            current_row += 1
            if current_row > 5:  # If we reach the bottom boundary
                current_row = 5
                current_col += 1
                downward = False  # Change direction
        else:
            current_row -= 1
            if current_row < 2:  # If we reach the top boundary
                current_row = 2
                current_col += 1
                downward = True  # Change direction

        # If the column index exceeds the limit, break the loop
        if current_col > 7:
            break

    return selected_cores


class bottleneckBCore:
    def __init__(self,_computeTileBN10_1,_computeTileBN10_2,_computeTileBN10_3,_computeTileBN11_1,_computeTileBN11_2,_computeTileBN11_3,_computeTileBN12_1,_computeTileBN12_2,_computeTileBN12_3, 
                 _weightsInBN10_1,_weightsInBN10_2,_weightsInBN10_3, _weightsInBN11_1,_weightsInBN11_2,_weightsInBN11_3, _weightsInBN12_1,_weightsInBN12_2,_weightsInBN12_3,
                 _rtpBN10_1,_rtpBN10_2,_rtpBN10_3,_rtpBN11_1,_rtpBN11_2,_rtpBN11_3,_rtpBN12_1,_rtpBN12_2,_rtpBN12_3,
                 _skipMemTile,
                 _actIn, _actOut, bn10_scaleFactor1,bn10_scaleFactor2,bn10_scaleFactor3,
                           bn11_scaleFactor1,bn11_scaleFactor2,bn11_scaleFactor3,bn11_scaleFactorAdd,
                           bn12_scaleFactor1,bn12_scaleFactor2,bn12_scaleFactor3,):

        self.computeTileBN10_1 = _computeTileBN10_1
        self.computeTileBN10_2 = _computeTileBN10_2
        self.computeTileBN10_3 = _computeTileBN10_3

        self.computeTileBN11_1 = _computeTileBN11_1
        self.computeTileBN11_2 = _computeTileBN11_2
        self.computeTileBN11_3 = _computeTileBN11_3

        self.computeTileBN12_1 = _computeTileBN12_1
        self.computeTileBN12_2 = _computeTileBN12_2
        self.computeTileBN12_3 = _computeTileBN12_3

        self.weightsInBN10_layer1 = _weightsInBN10_1
        self.weightsInBN10_layer2 = _weightsInBN10_2
        self.weightsInBN10_layer3 = _weightsInBN10_3

        self.weightsInBN11_layer1 = _weightsInBN11_1
        self.weightsInBN11_layer2 = _weightsInBN11_2
        self.weightsInBN11_layer3 = _weightsInBN11_3

        self.weightsInBN12_layer1 = _weightsInBN12_1
        self.weightsInBN12_layer2 = _weightsInBN12_2
        self.weightsInBN12_layer3 = _weightsInBN12_3

        self.rtpBN10_layer1 = _rtpBN10_1
        self.rtpBN10_layer2 = _rtpBN10_2
        self.rtpBN10_layer3 = _rtpBN10_3

        self.rtpBN11_layer1 = _rtpBN11_1
        self.rtpBN11_layer2 = _rtpBN11_2
        self.rtpBN11_layer3 = _rtpBN11_3

        self.rtpBN12_layer1 = _rtpBN12_1
        self.rtpBN12_layer2 = _rtpBN12_2
        self.rtpBN12_layer3 = _rtpBN12_3

        self.skipMemTile = _skipMemTile
        self.actIn = _actIn
        self.actOut = _actOut

        b10_InW1 = 14
        b10_InH1 = 14
        b10_InC1 = 80
        b10_OutC1 = 480

        b10_InW2 = 14
        b10_InH2 = 14
        b10_OutC2 = b10_OutC1

        b10_InW3 = 14
        b10_InH3 = 14
        b10_OutC3 = 112

        b11_OutC1 = 336
        b11_OutC2 = 336
        b11_OutC3 = 112

        b12_OutC1 = 336
        b12_OutC2 = 336
        b12_InW2 = 7
        b12_InH2 = 7
        b12_OutC3 = 80




        enableTrace = False
        trace_size = 16384
        traceSizeInInt32s = trace_size // 4


        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)
        # ************************ bneck10 ************************
        b10_layer1_in = MemRefType.get((b10_InW1, 1, b10_InC1, ), int8_ty, ) 
        b10_layer2_in = MemRefType.get((b10_InW2, 1, b10_OutC1, ), uint8_ty, )
        b10_layer3_in = MemRefType.get((b10_InW3, 1, b10_OutC2, ), uint8_ty, ) 

        # define wts 
        b10_layer1_wts = MemRefType.get((b10_InC1 * b10_OutC1,), int8_ty ) 
        b10_layer2_wts = MemRefType.get((3 * 3 * b10_OutC2 * 1,), int8_ty ) 
        b10_layer3_wts = MemRefType.get((b10_OutC2 * b10_OutC3,), int8_ty ) 
        b10_all_wts= MemRefType.get((b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1 + b10_OutC2 * b10_OutC3, ), int8_ty, ) 
        # output 
        b10_layer1_out = MemRefType.get((b10_InW2, 1, b10_OutC1, ), uint8_ty, ) 
        b10_layer2_out = MemRefType.get((b10_InW3, 1, b10_OutC2, ), uint8_ty, ) 
        b10_layer3_out = MemRefType.get((b10_InW3, 1, b10_OutC3, ), int8_ty, ) 
        # ************************ bneck11 ************************ 
        # input 
        b11_layer1_in = MemRefType.get((b10_InW3, 1, b10_OutC3, ), int8_ty, ) 
        b11_layer2_in = MemRefType.get((b10_InW3, 1, b11_OutC1, ), uint8_ty, ) 
        b11_layer3_in = MemRefType.get((b10_InW3, 1, b11_OutC2, ), uint8_ty, ) 

        # define wts 
        b11_layer1_wts = MemRefType.get((b10_OutC3 * b11_OutC1,), int8_ty ) 
        b11_layer2_wts = MemRefType.get((3 * 3 * b11_OutC2 * 1,), int8_ty ) 
        b11_layer3_wts = MemRefType.get((b11_OutC2 * b11_OutC3,), int8_ty ) 
        b11_all_wts= MemRefType.get((b10_OutC3 * b11_OutC1 + 3 * 3 * b11_OutC2 * 1 + b11_OutC2 * b11_OutC3, ), int8_ty, )
        # output
        b11_layer1_out = MemRefType.get((b10_InW3, 1, b11_OutC1, ), uint8_ty, ) 
        b11_layer2_out = MemRefType.get((b10_InW3, 1, b11_OutC2, ), uint8_ty, ) 
        b11_layer3_out = MemRefType.get((b10_InW3, 1, b11_OutC3, ), int8_ty, ) 
        # ************************ bneck12 ************************ 
        b12_layer1_in = MemRefType.get((b10_InW1, 1, b11_OutC3, ), int8_ty, ) 
        b12_layer2_in = MemRefType.get((b10_InW1, 1, b12_OutC1, ), uint8_ty, ) 
        b12_layer3_in = MemRefType.get((b12_InW2, 1, b12_OutC2, ), uint8_ty, ) 
        # define wts 
        b12_layer1_wts = MemRefType.get((b11_OutC3 * b12_OutC1,), int8_ty ) 
        b12_layer2_wts = MemRefType.get((3 * 3 * b12_OutC2 * 1,), int8_ty ) 
        b12_layer3_wts = MemRefType.get((b12_OutC2 * b12_OutC3,), int8_ty ) 
        b12_all_wts= MemRefType.get((b11_OutC3 * b12_OutC1 + 3 * 3 * b12_OutC2 * 1 + b12_OutC2 * b12_OutC3, ), int8_ty, ) 
        # output 
        b12_layer1_out = MemRefType.get((b10_InW3, 1, b12_OutC1, ), uint8_ty, ) 
        b12_layer2_out = MemRefType.get((b12_InW2, 1, b12_OutC2, ), uint8_ty, ) 
        b12_layer3_out = MemRefType.get((b12_InW2, 1, b12_OutC3, ), int8_ty, )
        # AIE Core Function declarations
        # ************************ bneck10 ************************
        bn10_conv2dk1_fused_relu = external_func("bn10_conv2dk1_relu_i8_ui8", inputs=[b10_layer1_in, b10_layer1_wts, b10_layer1_out, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        bn10_conv2dk3_dw = external_func("bn10_conv2dk3_dw_stride1_relu_ui8_ui8", inputs=[b10_layer2_in, b10_layer2_in, b10_layer2_in, b10_layer2_wts, b10_layer2_out, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        bn10_conv2dk1_ui8 = external_func("bn10_conv2dk1_ui8_i8", inputs=[b10_layer3_in, b10_layer3_wts, b10_layer3_out, int32_ty, int32_ty, int32_ty, int32_ty, ], )
        # ************************ bneck11 ************************
        bn11_conv2dk1_fused_relu = external_func("bn11_conv2dk1_relu_i8_ui8", inputs=[b11_layer1_in, b11_layer1_wts, b11_layer1_out, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        bn11_conv2dk3_dw = external_func("bn11_conv2dk3_dw_stride1_relu_ui8_ui8", inputs=[b11_layer2_in, b11_layer2_in, b11_layer2_in, b11_layer2_wts, b11_layer2_out, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        bn11_conv2dk1_skip = external_func("bn11_conv2dk1_skip_ui8_i8_i8", inputs=[b11_layer3_in, b11_layer3_wts, b11_layer3_out, b11_layer1_in, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        
        # ************************ bneck12 ************************ 
        bn12_conv2dk1_fused_relu = external_func("bn12_conv2dk1_relu_i8_ui8", inputs=[b12_layer1_in, b12_layer1_wts, b12_layer1_out, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        bn12_conv2dk3_dw = external_func("bn12_conv2dk3_dw_stride2_relu_ui8_ui8", inputs=[b12_layer2_in, b12_layer2_in, b12_layer2_in, b12_layer2_wts, b12_layer2_out, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, ], ) 
        bn12_conv2dk1_ui8 = external_func("bn12_conv2dk1_ui8_i8", inputs=[b12_layer3_in, b12_layer3_wts, b12_layer3_out, int32_ty, int32_ty, int32_ty, int32_ty, ],)
        
        # Tile declarations
        # ShimTile00 = tile(0, 0)
        # ShimTile10 = tile(1, 0)


        # MemTile01 = tile(0, 1)
        # MemTile11 = tile(1, 1)
        # MemTile21 = tile(2, 1)

    
        # AIE-array data movement with object fifos
        # ************************ bneck10 ************************
        # Input
        # OF_inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile00, MemTile01, 2, b10_layer1_in ) 
        # self.actIn = object_fifo("self.actIn", MemTile01, self.computeTileBN10_1, 2, b10_layer1_in) 
        # object_fifo_link(OF_inOF_act_L3L2, self.actIn) 
        # wts 
        
        # Output 
        OF_b10_act_layer1_layer2 = object_fifo("OF_b10_act_layer1_layer2", self.computeTileBN10_1, [self.computeTileBN10_2], 4,b10_layer2_in,via_DMA=True) 
        OF_b10_act_layer2_layer3 = object_fifo("OF_b10_act_layer2_layer3", self.computeTileBN10_2, [self.computeTileBN10_3], 2,b10_layer3_in) 
        # ************************ bneck11 ************************ 
        OF_b10_layer3_bn_11_layer1 = object_fifo("OF_b10_layer3_bn_11_layer1", self.computeTileBN10_3, [self.computeTileBN11_1,self.skipMemTile], [2, 2, 6], b11_layer1_in) 
        OF_b11_skip = object_fifo("OF_b11_skip", self.skipMemTile, [self.computeTileBN11_3], 2,b11_layer1_in) 
        object_fifo_link(OF_b10_layer3_bn_11_layer1,OF_b11_skip ) 
        OF_b11_act_layer1_layer2 = object_fifo("OF_b11_act_layer1_layer2", self.computeTileBN11_1, [self.computeTileBN11_2], 4,b11_layer2_in,via_DMA=True) 
        OF_b11_act_layer2_layer3 = object_fifo("OF_b11_act_layer2_layer3", self.computeTileBN11_2, [self.computeTileBN11_3], 2,b11_layer3_in) # 
        # ************************ bneck12 ************************ 
        OF_b11_layer3_bn_12_layer1 = object_fifo("OF_b11_layer3_bn_12_layer1", self.computeTileBN11_3, [self.computeTileBN12_1 ], 2, b12_layer1_in) 
        OF_b12_act_layer1_layer2 = object_fifo("OF_b12_act_layer1_layer2", self.computeTileBN12_1 , [self.computeTileBN12_2], 4,b12_layer1_out,via_DMA=True) 
        OF_b12_act_layer2_layer3 = object_fifo("OF_b12_act_layer2_layer3", self.computeTileBN12_2, [self.computeTileBN12_3], 2,b12_layer2_out)
        
        # self.actOut = object_fifo("self.actOut", self.computeTileBN12_3, [MemTile21], 2, b12_layer3_out)
        # OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile21, [ShimTile10], 2, b12_layer3_out)
        # object_fifo_link(self.actOut, OF_outOFL2L3)
        # Set up compute tiles

        

    # ************************ bneck10 ************************
        # 1x1 conv2d
        @core(self.computeTileBN10_1, "bn10_conv2dk1_fused_relu.o")
        def core_body():
            for _ in for_(sys.maxsize):

                # acquire weights once
                element0Weights = self.weightsInBN10_layer1.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(self.rtpBN10_layer1, [0])
                scale = bn10_scaleFactor1
                for _ in for_(b10_InH1):
                    element0ActivactionsIn = self.actIn.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    element0ActivactionsOut = OF_b10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    call(
                        bn10_conv2dk1_fused_relu,
                        [
                            element0ActivactionsIn,
                            element0Weights,
                            element0ActivactionsOut,
                            b10_InW1,
                            b10_InC1,
                            b10_OutC1,
                            scale,
                        ],
                    )
                    self.actIn.release(ObjectFifoPort.Consume,1)
                    OF_b10_act_layer1_layer2.release(ObjectFifoPort.Produce,1)
                    yield_([])
                self.weightsInBN10_layer1.release(ObjectFifoPort.Consume,1)
                yield_([])

        # # # Compute tile 3
        @core(self.computeTileBN10_2, "bn10_conv2dk3_dw.o")
        def core_body():
            scale = bn10_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                element0Weights = self.weightsInBN10_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN10_2, 0)

                # pre-amble: top row
                elementActivactionsIn = OF_b10_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b10_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn10_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b10_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                OF_b10_act_layer2_layer3.release(ObjectFifoPort.Produce,1)

                # middle
                for _ in for_(b10_InH2 - 2):
                    elementActivactionsIn = OF_b10_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = OF_b10_act_layer2_layer3.acquire(
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
                            b10_InW2,
                            1,
                            b10_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )

                    OF_b10_act_layer1_layer2.release(ObjectFifoPort.Consume,1)
                    OF_b10_act_layer2_layer3.release(ObjectFifoPort.Produce,1)
                    yield_([])

                # last part
                elementActivactionsIn = OF_b10_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b10_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn10_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b10_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )

                OF_b10_act_layer1_layer2.release(ObjectFifoPort.Consume,2)
                OF_b10_act_layer2_layer3.release(ObjectFifoPort.Produce,1)
                self.weightsInBN10_layer2.release(ObjectFifoPort.Consume,1)

                yield_([])

        # Compute tile 4
        @core(self.computeTileBN10_3, "bn10_conv2dk1_ui8.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                elemWts = self.weightsInBN10_layer3.acquire(ObjectFifoPort.Consume, 1)

                # scale = memref.load(self.rtpBN10_layer3, [0])
                scale = bn10_scaleFactor3
                # scale = memref.load(rtpself.computeTileBN10_1, [0])

                for _ in for_(b10_InH3):
                    elemIn = OF_b10_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = OF_b10_layer3_bn_11_layer1.acquire(ObjectFifoPort.Produce, 1)

                    call(
                        bn10_conv2dk1_ui8,
                        [
                            elemIn,
                            elemWts,
                            elemOut0,
                            b10_InW3,
                            b10_OutC2,
                            b10_OutC3,
                            scale,
                        ],
                    )
                    OF_b10_act_layer2_layer3.release(ObjectFifoPort.Consume,1)
                    OF_b10_layer3_bn_11_layer1.release(ObjectFifoPort.Produce,1)
                    yield_([])
                self.weightsInBN10_layer3.release(ObjectFifoPort.Consume,1)
                yield_([])
        
    # # # ************************ bneck11 ************************
    # #     #     # 1x1 conv2d
        @core(self.computeTileBN11_1, "bn11_conv2dk1_fused_relu.o")
        def core_body():
            for _ in for_(sys.maxsize):

                # acquire weights once
                element0Weights = self.weightsInBN11_layer1.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(self.rtpBN11_layer1, [0])
                scale = bn11_scaleFactor1
                for _ in for_(b10_InH1):
                    element0ActivactionsIn = OF_b10_layer3_bn_11_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    element0ActivactionsOut = OF_b11_act_layer1_layer2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn11_conv2dk1_fused_relu,
                        [
                            element0ActivactionsIn,
                            element0Weights,
                            element0ActivactionsOut,
                            b10_InW1,
                            b10_OutC3,
                            b11_OutC1,
                            scale,
                        ],
                    )
                    OF_b10_layer3_bn_11_layer1.release(ObjectFifoPort.Consume,1)
                    OF_b11_act_layer1_layer2.release(ObjectFifoPort.Produce,1)
                    yield_([])
                self.weightsInBN11_layer1.release(ObjectFifoPort.Consume,1)
                yield_([])

        # # # # # # Compute tile 3
        @core(self.computeTileBN11_2, "bn11_conv2dk3_dw.o")
        def core_body():
            scale = bn11_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                element0Weights = self.weightsInBN11_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN10_2, 0)

                # pre-amble: top row
                elementActivactionsIn = OF_b11_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b11_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn11_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b11_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                OF_b11_act_layer2_layer3.release(ObjectFifoPort.Produce,1)

                # middle
                for _ in for_(b10_InH2 - 2):
                    elementActivactionsIn = OF_b11_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = OF_b11_act_layer2_layer3.acquire(
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
                            b10_InW2,
                            1,
                            b11_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )
                    OF_b11_act_layer1_layer2.release(ObjectFifoPort.Consume,1)
                    OF_b11_act_layer2_layer3.release(ObjectFifoPort.Produce,1)
                    yield_([])

                # last part
                elementActivactionsIn = OF_b11_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b11_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn11_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        b10_InW2,
                        1,
                        b11_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )
                OF_b11_act_layer1_layer2.release(ObjectFifoPort.Consume,2)
                OF_b11_act_layer2_layer3.release(ObjectFifoPort.Produce,1)
    
                self.weightsInBN11_layer2.release(ObjectFifoPort.Consume,1)

                yield_([])

        # # Compute tile 4
        @core(self.computeTileBN11_3, "bn11_conv2dk1_skip.o")
        def core_body():

            for _ in for_(0xFFFFFFFF):
                elemWts = self.weightsInBN11_layer3.acquire(ObjectFifoPort.Consume, 1)
                scale= bn11_scaleFactor3
                skipScale=bn11_scaleFactorAdd
                # scale = memref.load(self.rtpBN11_layer3 , [0])
                # skipScale = memref.load(self.rtpBN11_layer3 , [1])


                for _ in for_(b10_InH3):
                    elemIn = OF_b11_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = OF_b11_layer3_bn_12_layer1.acquire(ObjectFifoPort.Produce, 1)
                    elementSkipsIn = OF_b11_skip.acquire(
                            ObjectFifoPort.Consume, 1
                        )

                    call(
                        bn11_conv2dk1_skip,
                        [
                            elemIn,
                            elemWts,
                            elemOut0,
                            elementSkipsIn,
                            b10_InW3,
                            b11_OutC2,
                            b11_OutC3,
                            scale,
                            skipScale,
                        ],
                    )

                    OF_b11_act_layer2_layer3.release(ObjectFifoPort.Consume,1)
                    OF_b11_layer3_bn_12_layer1.release(ObjectFifoPort.Produce,1)
                    OF_b11_skip.release(ObjectFifoPort.Consume,1)
                    yield_([])
                self.weightsInBN11_layer3.release(ObjectFifoPort.Consume,1)
                yield_([])
    # # # ************************ bneck12 ************************
    #     # 1x1 conv2d
        @core(self.computeTileBN12_1 , "bn12_conv2dk1_fused_relu.o")
        def core_body():
            for _ in for_(sys.maxsize):

                # acquire weights once
                element0Weights = self.weightsInBN12_layer1.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(self.rtpBN12_layer1, [0])
                scale=bn12_scaleFactor1
                for _ in for_(b10_InH1):
                    element0ActivactionsIn = OF_b11_layer3_bn_12_layer1.acquire(
                        ObjectFifoPort.Consume, 1
                    )
                    element0ActivactionsOut = OF_b12_act_layer1_layer2.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn12_conv2dk1_fused_relu,
                        [
                            element0ActivactionsIn,
                            element0Weights,
                            element0ActivactionsOut,
                            b10_InW1,
                            b11_OutC3,
                            b12_OutC1,
                            scale,
                        ],
                    )
                    OF_b11_layer3_bn_12_layer1.release(ObjectFifoPort.Consume,1)
                    OF_b12_act_layer1_layer2.release(ObjectFifoPort.Produce,1)
                    yield_([])
                self.weightsInBN12_layer1.release(ObjectFifoPort.Consume,1)
                yield_([])

        @core(self.computeTileBN12_2, "bn12_conv2dk3_dw_stride2.o")
        def core_body():
            scale = bn12_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                element0Weights = self.weightsInBN12_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpComputeTile3, 0)

                # pre-amble: top row
                elementActivactionsIn = OF_b12_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = OF_b12_act_layer2_layer3.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn12_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        b10_InW3,
                        1,
                        b12_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                OF_b12_act_layer2_layer3.release(ObjectFifoPort.Produce,1)
                OF_b12_act_layer1_layer2.release(ObjectFifoPort.Consume,1)

                # middle
                for _ in for_(b12_InH2 - 1):
                    elementActivactionsIn = OF_b12_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = OF_b12_act_layer2_layer3.acquire(
                        ObjectFifoPort.Produce, 1
                    )
                    res = call(
                        bn12_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            b10_InW3,
                            1,
                            b12_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )
                    OF_b12_act_layer1_layer2.release(ObjectFifoPort.Consume,2)
                    OF_b12_act_layer2_layer3.release(ObjectFifoPort.Produce,1)
                    
                    yield_([])

    
                OF_b12_act_layer1_layer2.release(ObjectFifoPort.Consume,1)
                self.weightsInBN12_layer2.release(ObjectFifoPort.Consume,1)
                yield_([])

    #     # # Compute tile 4
        @core(self.computeTileBN12_3, "bn12_conv2dk1_ui8.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                elemWts = self.weightsInBN12_layer3.acquire(ObjectFifoPort.Consume, 1)

                # scale = memref.load(self.rtpBN12_layer3, [0])
                scale = bn12_scaleFactor3
                # scale = memref.load(rtpself.computeTileBN10_1, [0])

                for _ in for_(b12_InH2):
                    elemIn = OF_b12_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = self.actOut.acquire(ObjectFifoPort.Produce, 1)

                    call(
                        bn12_conv2dk1_ui8,
                        [
                            elemIn,
                            elemWts,
                            elemOut0,
                            b12_InW2,
                            b12_OutC2,
                            b12_OutC3,
                            scale,
                        ],
                    )
                    OF_b12_act_layer2_layer3.release(ObjectFifoPort.Consume,1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                self.weightsInBN12_layer3.release(ObjectFifoPort.Consume,1)
                yield_([])

def mobilenetV3_bn_10_11_12(start_row = 2, start_col = 0, bn10_scaleFactor1=10,bn10_scaleFactor2=7,bn10_scaleFactor3=9,
                           bn11_scaleFactor1=9,bn11_scaleFactor2=8,bn11_scaleFactor3=12,bn11_scaleFactorAdd=1,
                           bn12_scaleFactor1=8,bn12_scaleFactor2=8,bn12_scaleFactor3=9):
    b10_InW1 = 14
    b10_InH1 = 14
    b10_InC1 = 80
    b10_OutC1 = 480

    b10_InW2 = 14
    b10_InH2 = 14
    b10_OutC2 = b10_OutC1

    b10_InW3 = 14
    b10_InH3 = 14
    b10_OutC3 = 112

    b11_OutC1 = 336
    b11_OutC2 = 336
    b11_OutC3 = 112

    b12_OutC1 = 336
    b12_OutC2 = 336
    b12_InW2 = 7
    b12_InH2 = 7
    b12_OutC3 = 80

    enableTrace = False
    trace_size = 16384
    traceSizeInInt32s = trace_size // 4

    @device(AIEDevice.npu1_3col)
    def device_body():
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)

                # Start core
        b_start_col = 0
        b_start_row = 2

        # Get the selected cores
        selected_cores= select_cores(b_start_col, b_start_row)
        # Assign the selected cores to variables
        bn10_tile_1 = tile(selected_cores[0][0], selected_cores[0][1])
        bn10_tile_2 = tile(selected_cores[1][0], selected_cores[1][1])
        bn10_tile_3 = tile(selected_cores[2][0], selected_cores[2][1])

        # Moving to the next group, starting from the 4th core
        bn11_tile_1 = tile(selected_cores[3][0], selected_cores[3][1])
        bn11_tile_2 = tile(selected_cores[4][0], selected_cores[4][1])
        bn11_tile_3 = tile(selected_cores[5][0], selected_cores[5][1])

        # Moving to the next group, starting from the 7th core
        bn12_tile_1 = tile(selected_cores[6][0], selected_cores[6][1])
        bn12_tile_2 = tile(selected_cores[7][0], selected_cores[7][1])
        bn12_tile_3 = tile(selected_cores[8][0], selected_cores[8][1])
        
        ShimTile00 = tile(0, 0)
        ShimTile10 = tile(1, 0)

        MemTile01 = tile(0, 1)
        MemTile11 = tile(1, 1)
        MemTile21 = tile(2, 1)
        
        b10_layer1_in = MemRefType.get((b10_InW1, 1, b10_InC1, ), int8_ty, ) 
        b12_layer3_out = MemRefType.get((b12_InW2, 1, b12_OutC3, ), int8_ty, )
         # define wts 
        b10_layer1_wts = MemRefType.get((b10_InC1 * b10_OutC1,), int8_ty ) 
        b10_layer2_wts = MemRefType.get((3 * 3 * b10_OutC2 * 1,), int8_ty ) 
        b10_layer3_wts = MemRefType.get((b10_OutC2 * b10_OutC3,), int8_ty ) 
        b10_all_wts= MemRefType.get((b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1 + b10_OutC2 * b10_OutC3, ), int8_ty, ) 
        # output 
        b10_layer1_out = MemRefType.get((b10_InW2, 1, b10_OutC1, ), uint8_ty, ) 
        b10_layer2_out = MemRefType.get((b10_InW3, 1, b10_OutC2, ), uint8_ty, ) 
        b10_layer3_out = MemRefType.get((b10_InW3, 1, b10_OutC3, ), int8_ty, ) 
        # ************************ bneck11 ************************ 
        # input 
        b11_layer1_in = MemRefType.get((b10_InW3, 1, b10_OutC3, ), int8_ty, ) 
        b11_layer2_in = MemRefType.get((b10_InW3, 1, b11_OutC1, ), uint8_ty, ) 
        b11_layer3_in = MemRefType.get((b10_InW3, 1, b11_OutC2, ), uint8_ty, ) 

        # define wts 
        b11_layer1_wts = MemRefType.get((b10_OutC3 * b11_OutC1,), int8_ty ) 
        b11_layer2_wts = MemRefType.get((3 * 3 * b11_OutC2 * 1,), int8_ty ) 
        b11_layer3_wts = MemRefType.get((b11_OutC2 * b11_OutC3,), int8_ty ) 
        b11_all_wts= MemRefType.get((b10_OutC3 * b11_OutC1 + 3 * 3 * b11_OutC2 * 1 + b11_OutC2 * b11_OutC3, ), int8_ty, )
        # output
        b11_layer1_out = MemRefType.get((b10_InW3, 1, b11_OutC1, ), uint8_ty, ) 
        b11_layer2_out = MemRefType.get((b10_InW3, 1, b11_OutC2, ), uint8_ty, ) 
        b11_layer3_out = MemRefType.get((b10_InW3, 1, b11_OutC3, ), int8_ty, ) 
        # ************************ bneck12 ************************ 
        b12_layer1_in = MemRefType.get((b10_InW1, 1, b11_OutC3, ), int8_ty, ) 
        b12_layer2_in = MemRefType.get((b10_InW1, 1, b12_OutC1, ), uint8_ty, ) 
        b12_layer3_in = MemRefType.get((b12_InW2, 1, b12_OutC2, ), uint8_ty, ) 
        # define wts 
        b12_layer1_wts = MemRefType.get((b11_OutC3 * b12_OutC1,), int8_ty ) 
        b12_layer2_wts = MemRefType.get((3 * 3 * b12_OutC2 * 1,), int8_ty ) 
        b12_layer3_wts = MemRefType.get((b12_OutC2 * b12_OutC3,), int8_ty ) 
        b12_all_wts= MemRefType.get((b11_OutC3 * b12_OutC1 + 3 * 3 * b12_OutC2 * 1 + b12_OutC2 * b12_OutC3, ), int8_ty, ) 
        # output 
        b12_layer1_out = MemRefType.get((b10_InW3, 1, b12_OutC1, ), uint8_ty, ) 
        b12_layer2_out = MemRefType.get((b12_InW2, 1, b12_OutC2, ), uint8_ty, ) 
        b12_layer3_out = MemRefType.get((b12_InW2, 1, b12_OutC3, ), int8_ty, )
        # Input
        act_in = object_fifo("act_in", ShimTile00, bn10_tile_1, 2, b10_layer1_in ) 
        act_out = object_fifo("act_out", bn12_tile_3, ShimTile10, 2, b12_layer3_out)

        # wts 
        wts_b10_L3L2 = object_fifo("wts_b10_L3L2", ShimTile00, MemTile01, 1, b10_all_wts ) 
        bn10_1_wts_OF_L3L1 = object_fifo("weightsInBN10_layer1", MemTile01, bn10_tile_1, 1, b10_layer1_wts ) 
        bn10_2_wts_OF_L3L1 = object_fifo("weightsInBN10_layer2", MemTile01, bn10_tile_2, 1, b10_layer2_wts, ) 
        bn10_3_wts_OF_L3L1 = object_fifo("weightsInBN10_layer3", MemTile01, bn10_tile_3, 1, b10_layer3_wts, ) 
        object_fifo_link(wts_b10_L3L2, [bn10_1_wts_OF_L3L1, bn10_2_wts_OF_L3L1, bn10_3_wts_OF_L3L1],[],[0,b10_InC1 * b10_OutC1,b10_InC1 * b10_OutC1+3 * 3 * b10_OutC2 * 1]) 
        
        # wts 
        wts_b11_L3L2 = object_fifo("wts_b11_L3L2", ShimTile10, MemTile11, 1, b11_all_wts ) 
        bn11_1_wts_OF_L3L1 = object_fifo("weightsInBN11_layer1", MemTile11, bn11_tile_1, 1, b11_layer1_wts ) 
        bn11_2_wts_OF_L3L1 = object_fifo("weightsInBN11_layer2", MemTile11, bn11_tile_2, 1, b11_layer2_wts, ) 
        bn11_3_wts_OF_L3L1 = object_fifo("weightsInBN11_layer3", MemTile11, bn11_tile_3, 1, b11_layer3_wts, ) 
        object_fifo_link(wts_b11_L3L2, [bn11_1_wts_OF_L3L1, bn11_2_wts_OF_L3L1, bn11_3_wts_OF_L3L1],[],[0,b10_OutC3 * b11_OutC1,b10_OutC3 * b11_OutC1+3 * 3 * b11_OutC2 * 1]) 
        
         # # wts 
        wts_b12_L3L2 = object_fifo("wts_b12_L3L2", ShimTile10, MemTile21, 1, b12_all_wts ) 
        bn12_1_wts_OF_L3L1 = object_fifo("weightsInBN12_layer1", MemTile21, bn12_tile_1 , 1, b12_layer1_wts ) 
        bn12_2_wts_OF_L3L1 = object_fifo("weightsInBN12_layer2", MemTile21, bn12_tile_2, 1, b12_layer2_wts, ) 
        bn12_3_wts_OF_L3L1 = object_fifo("weightsInBN12_layer3", MemTile21, bn12_tile_3, 1, b12_layer3_wts, ) 
        object_fifo_link(wts_b12_L3L2, [bn12_1_wts_OF_L3L1, bn12_2_wts_OF_L3L1, bn12_3_wts_OF_L3L1],[],[0,b11_OutC3 * b12_OutC1,b11_OutC3 * b12_OutC1+3 * 3 * b12_OutC2 * 1])
        

        bn10_1_rtp = Buffer(bn10_tile_1, [16], T.i32(), "bn10_1_rtp")
        bn10_2_rtp = Buffer(bn10_tile_2, [16], T.i32(), "bn10_2_rtp")
        bn10_3_rtp = Buffer(bn10_tile_3, [16], T.i32(), "bn10_3_rtp")

        bn11_1_rtp = Buffer(bn11_tile_1, [16], T.i32(), "bn11_1_rtp")
        bn11_2_rtp = Buffer(bn11_tile_2, [16], T.i32(), "bn11_2_rtp")
        bn11_3_rtp = Buffer(bn11_tile_3, [16], T.i32(), "bn11_3_rtp")


        bn12_1_rtp = Buffer(bn12_tile_1 , [16], T.i32(), "bn12_1_rtp")
        bn12_2_rtp = Buffer(bn12_tile_2, [16], T.i32(), "bn12_2_rtp")
        bn12_3_rtp = Buffer(bn12_tile_3, [16], T.i32(), "bn12_3_rtp")

        bottleneckBCore(bn10_tile_1,bn10_tile_2,bn10_tile_3,bn11_tile_1,bn11_tile_2,bn11_tile_3,bn12_tile_1,bn12_tile_2,bn12_tile_3,
                        bn10_1_wts_OF_L3L1,bn10_2_wts_OF_L3L1,bn10_3_wts_OF_L3L1,bn11_1_wts_OF_L3L1,bn11_2_wts_OF_L3L1,bn11_3_wts_OF_L3L1,bn12_1_wts_OF_L3L1,bn12_2_wts_OF_L3L1,bn12_3_wts_OF_L3L1,
                        bn10_1_rtp,bn10_2_rtp,bn10_3_rtp,bn11_1_rtp,bn11_2_rtp,bn11_3_rtp,bn12_1_rtp,bn12_2_rtp,bn12_3_rtp,
                        MemTile01,act_in,act_out,bn10_scaleFactor1,bn10_scaleFactor2,bn10_scaleFactor3,
                           bn11_scaleFactor1,bn11_scaleFactor2,bn11_scaleFactor3,bn11_scaleFactorAdd,
                           bn12_scaleFactor1,bn12_scaleFactor2,bn12_scaleFactor3, )


        # # instruction stream generation
        activationsInSize32b = (b10_InW1 * b10_InH1 * b10_InC1) // 4
        # acitivationsOutSize32b = (b12_InW2 * b12_InH2 * b12_OutC3) // 4
        acitivationsOutSize32b = (b12_InW2 * b12_InW2 * b12_OutC3) // 4

        bn10_totalWeightsSize32b = (
        b10_InC1*b10_OutC1+
            3 * 3 * b10_OutC2 * 1+
            b10_OutC2*b10_OutC3
        ) // 4

        bn11_totalWeightsSize32b = (
        b10_OutC3*b11_OutC1+
            3 * 3 * b11_OutC2 * 1+
            b11_OutC2*b11_OutC3
        ) // 4

        bn12_totalWeightsSize32b = (
        b11_OutC3*b12_OutC1+
            3 * 3 * b12_OutC2 * 1+
            b12_OutC2*b12_OutC3
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
            NpuWriteRTPOp("bn10_1_rtp", col=0, row=2, index=0, value=bn10_scaleFactor1)
            NpuWriteRTPOp("bn10_2_rtp", col=0, row=3, index=0, value=bn10_scaleFactor2)
            NpuWriteRTPOp("bn10_3_rtp", col=0, row=4, index=0, value=bn10_scaleFactor3)


            NpuWriteRTPOp("bn11_1_rtp", col=0, row=5, index=0, value=bn11_scaleFactor1)
            NpuWriteRTPOp("bn11_2_rtp", col=1, row=5, index=0, value=bn11_scaleFactor2)
            NpuWriteRTPOp("bn11_3_rtp", col=1, row=4, index=0, value=bn11_scaleFactor3)
            NpuWriteRTPOp("bn11_3_rtp", col=1, row=4, index=1, value=bn11_scaleFactorAdd)

            NpuWriteRTPOp("bn12_1_rtp", col=1, row=3, index=0, value=bn12_scaleFactor1)
            NpuWriteRTPOp("bn12_2_rtp", col=1, row=2, index=0, value=bn12_scaleFactor2)
            NpuWriteRTPOp("bn12_3_rtp", col=2, row=2, index=0, value=bn12_scaleFactor3)
            
            npu_dma_memcpy_nd(
                metadata="act_in",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="act_out",
                bd_id=2,
                mem=outputToL3,
                sizes=[1, 1, 1, acitivationsOutSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_b10_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                sizes=[1, 1, 1, bn10_totalWeightsSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_b11_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, bn10_totalWeightsSize32b],
                sizes=[1, 1, 1, bn11_totalWeightsSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_b12_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, bn12_Offset_32b],
                sizes=[1, 1, 1, bn12_totalWeightsSize32b],
            )
            npu_sync(column=1, row=0, direction=0, channel=0)

with mlir_mod_ctx() as ctx:
    mobilenetV3_bn_10_11_12(bn10_scaleFactor1=scale_factors["BN10"]["conv1x1_1"],bn10_scaleFactor2=scale_factors["BN10"]["conv3x3"],bn10_scaleFactor3=scale_factors["BN10"]["conv1x1_2"],
                           bn11_scaleFactor1=scale_factors["BN11"]["conv1x1_1"],bn11_scaleFactor2=scale_factors["BN11"]["conv3x3"],bn11_scaleFactor3=scale_factors["BN11"]["conv1x1_2"],bn11_scaleFactorAdd=scale_factors["BN11"]["skip_add"],
                           bn12_scaleFactor1=scale_factors["BN12"]["conv1x1_1"],bn12_scaleFactor2=scale_factors["BN12"]["conv3x3"],bn12_scaleFactor3=scale_factors["BN12"]["conv1x1_2"])

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
