#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, bneck_13_InC1.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx
import math



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


class bottleneckCCore:
    def __init__(self,_computeTileBN13_1,_computeTileBN13_2,_computeTileBN13_3,_computeTileBN13_4,_computeTileBN13_5,_computeTileBN14_1,_computeTileBN14_2,_computeTileBN14_3,_computeTileBN14_4,_computeTileBN14_5,
                 _weightsInBN13_1,_weightsInBN13_2,_weightsInBN13_3, _weightsInBN13_4,_weightsInBN13_5,_weightsInBN14_1,_weightsInBN14_2,_weightsInBN14_3, _weightsInBN14_4,_weightsInBN14_5,
                _rtp_bn13_tile_layer1_get,_rtp_bn13_tile_layer3_get,
                 _bn13_scaleFactor1,_bn13_scaleFactor2,_bn13_scaleFactor3,_bn13_scaleFactorAdd,
                  _bn14_scaleFactor1,_bn14_scaleFactor2,_bn14_scaleFactor3,_bn14_scaleFactorAdd,
                 _skipMemTile,
                 
                 _actIn, _actOut,_bn13_skip,):

        self.computeTileBN13_layer1_put=_computeTileBN13_1
        self.computeTileBN13_layer1_get=_computeTileBN13_2
        self.computeTileBN13_layer2=_computeTileBN13_3
        self.computeTileBN13_layer3_put=_computeTileBN13_4
        self.computeTileBN13_layer3_get=_computeTileBN13_5

        self.computeTileBN14_layer1_put=_computeTileBN14_1
        self.computeTileBN14_layer1_get=_computeTileBN14_2
        self.computeTileBN14_layer2=_computeTileBN14_3
        self.computeTileBN14_layer3_put=_computeTileBN14_4
        self.computeTileBN14_layer3_get=_computeTileBN14_5

        # wts

        self.weightsInBN13_layer1_put=_weightsInBN13_1
        self.weightsInBN13_layer1_get=_weightsInBN13_2
        self.weightsInBN13_layer2=_weightsInBN13_3
        self.weightsInBN13_layer3_put=_weightsInBN13_4
        self.weightsInBN13_layer3_get=_weightsInBN13_5

        self.weightsInBN14_layer1_put=_weightsInBN14_1
        self.weightsInBN14_layer1_get=_weightsInBN14_2
        self.weightsInBN14_layer2=_weightsInBN14_3
        self.weightsInBN14_layer3_put=_weightsInBN14_4
        self.weightsInBN14_layer3_get=_weightsInBN14_5

        self.skipMemTile = _skipMemTile

        self.actIn = _actIn
        self.actOut = _actOut
        self.bn13_skip=_bn13_skip

        bneck_13_InW1 = 7
        bneck_13_InH1 = 7
        bneck_13_InC1 = 160
        bneck_13_OutC1 = 960
        InputSplit=2
        OutputSplit=bneck_13_OutC1//8 #calculate 8 OCs at a time, should bneck_13_InC1rease to more


        RepeatChannels=math.floor(bneck_13_InH1)

        bneck_13_InW2 = bneck_13_InW1
        bneck_13_InH2 = bneck_13_InH1
        bneck_13_OutC2 = bneck_13_OutC1

        bneck_13_InW3 = bneck_13_InW2
        bneck_13_InH3 = bneck_13_InH2
        bneck_13_OutC3 = 160
        OutputSplit2=bneck_13_OutC3//8 #calculate 8 OCs at a time, should bneck_13_InC1rease to more

        # second block
        bneck_14_InW1 = bneck_13_InW1
        bneck_14_InH1 = bneck_13_InH1
        bneck_14_InC1 = bneck_13_OutC3
        bneck_14_OutC1 = 960

        bneck_14_InW2 = bneck_14_InW1
        bneck_14_InH2 = bneck_14_InH1
        bneck_14_OutC2 = bneck_14_OutC1

        bneck_14_InW3 = bneck_14_InW2
        bneck_14_InH3 = bneck_14_InH2
        bneck_14_OutC3 = 160
        OutputSplit3 = bneck_14_OutC3 // 8  # Calculate 8 OCs at a time, should increase to more

        self.rtp_bn13_tile_layer1_get=_rtp_bn13_tile_layer1_get
        self.rtp_bn13_tile_layer3_get=_rtp_bn13_tile_layer3_get

        self.bn13_scaleFactor1=_bn13_scaleFactor1
        self.bn13_scaleFactor2=_bn13_scaleFactor2
        self.bn13_scaleFactor3=_bn13_scaleFactor3
        self.bn13_scaleFactorAdd=_bn13_scaleFactorAdd


        self.bn14_scaleFactor1=_bn14_scaleFactor1
        self.bn14_scaleFactor2=_bn14_scaleFactor2
        self.bn14_scaleFactor3=_bn14_scaleFactor3
        self.bn14_scaleFactorAdd=_bn14_scaleFactorAdd



        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)

    # ************************ bneck13 ************************

        ty_bneck_13_layer1_in = MemRefType.get((bneck_13_InW1,1,bneck_13_InC1,),int8_ty,)
        ty_bneck_13_layer2_in = MemRefType.get((bneck_13_InW2,1,bneck_13_OutC1,),uint8_ty,)
    
        # define wts
        #layer1
        ty_bneck_13_layer1_wts_split = MemRefType.get(((bneck_13_InC1 * 8)//(InputSplit),), int8_ty)
        ty_bneck_13_layer1_wts_full= MemRefType.get((bneck_13_InC1 * bneck_13_OutC1,),int8_ty,)
        # layer2
        ty_bneck_13_layer2_wts = MemRefType.get((3 * 3 * bneck_13_OutC2 * 1,), int8_ty )
        # layer3
        ty_bneck_13_layer3_wts_split = MemRefType.get(((bneck_13_OutC2 * 8)//(InputSplit),), int8_ty)
        ty_bneck_13_layer3_wts_full= MemRefType.get((bneck_13_OutC2 * bneck_13_OutC3,),int8_ty,)

        # OUTPUT
        ty_bneck_13_layer1_out = MemRefType.get((bneck_13_InW1,1,bneck_13_OutC1,),uint8_ty,)
        ty_bneck_13_layer2_out = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC2, ), uint8_ty, )
        ty_bneck_13_layer2_out_split = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC2//InputSplit, ), uint8_ty, )
        # layer3
        ty_bneck_13_layer3_out = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC3, ), int8_ty, )
        
# HERE

# ************************ bneck14 ************************

        ty_bneck_14_layer1_in = MemRefType.get((bneck_14_InW1, 1, bneck_14_InC1,), int8_ty,)
        ty_bneck_14_layer2_in = MemRefType.get((bneck_14_InW2, 1, bneck_14_OutC1,), uint8_ty,)

        # define wts
        # layer1
        ty_bneck_14_layer1_wts_split = MemRefType.get(((bneck_14_InC1 * 8) // (InputSplit),), int8_ty)
        ty_bneck_14_layer1_wts_full = MemRefType.get((bneck_14_InC1 * bneck_14_OutC1,), int8_ty,)
        # layer2
        ty_bneck_14_layer2_wts = MemRefType.get((3 * 3 * bneck_14_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_14_layer3_wts_split = MemRefType.get(((bneck_14_OutC2 * 8) // (InputSplit),), int8_ty)
        ty_bneck_14_layer3_wts_full = MemRefType.get((bneck_14_OutC2 * bneck_14_OutC3,), int8_ty,)

        # OUTPUT
        ty_bneck_14_layer1_out = MemRefType.get((bneck_14_InW1, 1, bneck_14_OutC1,), uint8_ty,)
        ty_bneck_14_layer2_out = MemRefType.get((bneck_14_InW3, 1, bneck_14_OutC2,), uint8_ty,)
        ty_bneck_14_layer2_out_split = MemRefType.get((bneck_14_InW3, 1, bneck_14_OutC2 // InputSplit,), uint8_ty,)
        # layer3
        ty_bneck_14_layer3_out = MemRefType.get((bneck_14_InW3, 1, bneck_14_OutC3,), int8_ty,)

        

        
        bn13_conv2dk1_fused_relu_get = external_func(
            "bn13_1_conv2dk1_i8_ui8_partial_width_get",
            inputs=[
                ty_bneck_13_layer1_in,
                ty_bneck_13_layer1_wts_split,
                ty_bneck_13_layer1_out,
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
        bn13_conv2dk1_fused_relu_put = external_func(
            "bn13_1_conv2dk1_i8_ui8_partial_width_put",
            inputs=[
                ty_bneck_13_layer1_in,
                ty_bneck_13_layer1_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn13_conv2dk3_dw = external_func(
            "bn13_conv2dk3_ui8_out_split",
            inputs=[
                ty_bneck_13_layer2_in,
                ty_bneck_13_layer2_in,
                ty_bneck_13_layer2_in,
                ty_bneck_13_layer2_wts,
                ty_bneck_13_layer2_out_split,
                ty_bneck_13_layer2_out_split,
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

        bn13_layer3_conv2dk1_put = external_func(
            "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put",
            inputs=[
                ty_bneck_13_layer2_out_split,
                ty_bneck_13_layer3_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn13_layer3_conv2dk1_skip_get = external_func(
            "bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get",
            inputs=[
                ty_bneck_13_layer2_out_split,
                ty_bneck_13_layer3_wts_split,
                ty_bneck_13_layer3_out,
                ty_bneck_13_layer1_in,
                int32_ty,
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
        bn14_conv2dk1_fused_relu_get = external_func(
            "bn14_1_conv2dk1_i8_ui8_partial_width_get",
            inputs=[
                ty_bneck_14_layer1_in,
                ty_bneck_14_layer1_wts_split,
                ty_bneck_14_layer1_out,
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
        bn14_conv2dk1_fused_relu_put = external_func(
            "bn14_1_conv2dk1_i8_ui8_partial_width_put",
            inputs=[
                ty_bneck_14_layer1_in,
                ty_bneck_14_layer1_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )
        bn14_conv2dk3_dw = external_func(
            "bn14_conv2dk3_ui8_out_split",
            inputs=[
                ty_bneck_14_layer2_in,
                ty_bneck_14_layer2_in,
                ty_bneck_14_layer2_in,
                ty_bneck_14_layer2_wts,
                ty_bneck_14_layer2_out_split,
                ty_bneck_14_layer2_out_split,
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

        bn14_layer3_conv2dk1_put = external_func(
            "bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put",
            inputs=[
                ty_bneck_14_layer2_out_split,
                ty_bneck_14_layer3_wts_split,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
                int32_ty,
            ],
        )

        bn14_layer3_conv2dk1_skip_get = external_func(
            "bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get",
            inputs=[
                ty_bneck_14_layer2_out_split,
                ty_bneck_14_layer3_wts_split,
                ty_bneck_14_layer3_out,
                ty_bneck_14_layer1_in,
                int32_ty,
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

        

        # AIE-array data movement with object fifos
        # ************************ bneck13 ************************
        
    
        

        # OUTPUT
        bn13_act_layer1_layer2 = object_fifo("bn13_act_layer1_layer2", self.computeTileBN13_layer1_get, [self.computeTileBN13_layer2], 4,ty_bneck_13_layer2_in,via_DMA=True)
        
        bn13_act_layer2_layer3_first = object_fifo("bn13_act_layer2_layer3_first", self.computeTileBN13_layer2, [self.computeTileBN13_layer3_put], 2, ty_bneck_13_layer2_out_split)
        bn13_act_layer2_layer3_second = object_fifo("bn13_act_layer2_layer3_second", self.computeTileBN13_layer2, [self.computeTileBN13_layer3_get], 2, ty_bneck_13_layer2_out_split)
        
        # ************************ bneck14 ************************
        # Input

        bn13_act_layer3_bn_14_layer1 = object_fifo("bn13_act_layer3_bn_14_layer1", self.computeTileBN13_layer3_get, [self.computeTileBN14_layer1_put,self.computeTileBN14_layer1_get,self.skipMemTile], [2, 2, 2, 6], ty_bneck_13_layer3_out)
        bn14_skip = object_fifo("bn14_skip", self.skipMemTile, self.computeTileBN14_layer3_get, 2, ty_bneck_13_layer3_out)
        object_fifo_link(bn13_act_layer3_bn_14_layer1, bn14_skip)
        
        

        # Object FIFO for b14 block results
        bn14_act_layer1_layer2 = object_fifo("bn14_act_layer1_layer2", self.computeTileBN14_layer1_get, self.computeTileBN14_layer2, 4, ty_bneck_14_layer1_out,via_DMA=True)

        bn14_act_layer2_layer3_first = object_fifo("bn14_act_layer2_layer3_first", self.computeTileBN14_layer2, self.computeTileBN14_layer3_put, 2, ty_bneck_14_layer2_out_split)
        bn14_act_layer2_layer3_second = object_fifo("bn14_act_layer2_layer3_second", self.computeTileBN14_layer2, [self.computeTileBN14_layer3_get], 2, ty_bneck_14_layer2_out_split)

        
        
        # object_fifo_link(bn13_act_layer2_layer3_first, [OF_outOFL2L3],[],[0])
        # object_fifo_link([bn13_act_layer2_layer3_first,bn13_act_layer2_layer3_second],[OF_outOFL2L3],[0,(bneck_13_InW3 *  bneck_13_OutC2//2)])

        
        # ************************ bneck13 ************************
        # conv1x1_first put
        @core(self.computeTileBN13_layer1_put, "bn13_1_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_13_InH1):
                    elemIn = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit):
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN13_layer1_put.acquire(ObjectFifoPort.Consume, 1)
                            for x_start in range(0,bneck_13_InW1,7):
                                call(
                                    bn13_conv2dk1_fused_relu_put,
                                    [
                                        elemIn,
                                        elemWts,
                                        arith.constant(bneck_13_InW1),
                                        arith.constant(bneck_13_InC1),
                                        arith.constant(bneck_13_OutC1),
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                            self.weightsInBN13_layer1_put.release(ObjectFifoPort.Consume,1)
                            yield_([])
                        yield_([])
                    self.actIn.release(ObjectFifoPort.Consume,1)                    
                    yield_([])
                yield_([])

        # conv1x1_first get
        @core(self.computeTileBN13_layer1_get, "bn13_1_conv2dk1_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_13_InH1):
                    elemIn = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = bn13_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)
                    
                    # scale = memref.load(rtp04, [0])
                    scale = self.bn13_scaleFactor1
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit):
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        for WeightIndex in for_(InputSplit//2, InputSplit):
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN13_layer1_get.acquire(ObjectFifoPort.Consume, 1)
                            for x_start in range(0,bneck_13_InW1,7):
                                call(
                                    bn13_conv2dk1_fused_relu_get,
                                    [
                                        elemIn,
                                        elemWts,
                                        elemOut0,
                                        arith.constant(bneck_13_InW1),
                                        arith.constant(bneck_13_InC1),
                                        arith.constant(bneck_13_OutC1),
                                        scale,
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                            self.weightsInBN13_layer1_get.release(ObjectFifoPort.Consume,1)
                            yield_([])
                        yield_([])
                    self.actIn.release(ObjectFifoPort.Consume,1)  
                    bn13_act_layer1_layer2.release(ObjectFifoPort.Produce,1)                      
                    yield_([])
                yield_([])
        
        # conv3x3
        @core(self.computeTileBN13_layer2, "bn13_conv2dk3_dw.o")
        def core_body():
            scale = self.bn13_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                element0Weights = self.weightsInBN13_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN13_layer1_get, 0)

                # pre-amble: top row
                elementActivactionsIn = bn13_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn13_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                element1ActivactionsOut = bn13_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn13_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
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
                bn13_act_layer2_layer3_first.release(ObjectFifoPort.Produce,1)   
                bn13_act_layer2_layer3_second.release(ObjectFifoPort.Produce,1)   
                # middle
                for _ in for_(bneck_13_InH2 - 2):
                    elementActivactionsIn = bn13_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = bn13_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                    element1ActivactionsOut = bn13_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn13_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            element1ActivactionsOut,
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
                    bn13_act_layer1_layer2.release(ObjectFifoPort.Consume,1)   
                    bn13_act_layer2_layer3_first.release(ObjectFifoPort.Produce,1)   
                    bn13_act_layer2_layer3_second.release(ObjectFifoPort.Produce,1)
                    yield_([])

                # last part
                elementActivactionsIn = bn13_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn13_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                element1ActivactionsOut = bn13_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn13_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
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
                bn13_act_layer1_layer2.release(ObjectFifoPort.Consume,2)   
                bn13_act_layer2_layer3_first.release(ObjectFifoPort.Produce,1)   
                bn13_act_layer2_layer3_second.release(ObjectFifoPort.Produce,1)
                self.weightsInBN13_layer2.release(ObjectFifoPort.Consume,1)
                yield_([])
        
        # conv1x1_second put
        @core(self.computeTileBN13_layer3_put, "bn13_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_13_InH3):
                    elemIn = bn13_act_layer2_layer3_first.acquire(ObjectFifoPort.Consume, 1)
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit2):
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        # for WeightIndex in range (0,InputSplit//2):
                        for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN13_layer3_put.acquire(ObjectFifoPort.Consume, 1)
                            x_start=0  
                            call(
                                bn13_layer3_conv2dk1_put,
                                [
                                    elemIn,
                                    elemWts,
                                    arith.constant(bneck_13_InW3),
                                    arith.constant(bneck_13_OutC2),
                                    arith.constant(bneck_13_OutC3),
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast
                                ],
                            )
                            self.weightsInBN13_layer3_put.release(ObjectFifoPort.Consume,1)
                            yield_([])
                        yield_([])
                    bn13_act_layer2_layer3_first.release(ObjectFifoPort.Consume,1)                    
                    yield_([])
                yield_([])

        # conv1x1_second get
        @core(self.computeTileBN13_layer3_get, "bn13_conv2dk1_skip_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_13_InH3):
                    
                    elemIn = bn13_act_layer2_layer3_second.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = bn13_act_layer3_bn_14_layer1.acquire(ObjectFifoPort.Produce, 1)
                    elementSkipsIn = self.bn13_skip.acquire(ObjectFifoPort.Consume, 1)
                    
                    scale = self.bn13_scaleFactor3
                    scale_skip = self.bn13_scaleFactorAdd
                    # scale = memref.load(rtp04, [0])
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit2):
                        
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        for WeightIndex in for_(0,InputSplit//2):
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN13_layer3_get.acquire(ObjectFifoPort.Consume, 1)
                            x_start=0 

                            call(
                                bn13_layer3_conv2dk1_skip_get,
                                [
                                    elemIn,
                                    elemWts,
                                    elemOut0,
                                    elementSkipsIn,
                                    arith.constant(bneck_13_InW3),
                                    arith.constant(bneck_13_OutC2),
                                    arith.constant(bneck_13_OutC3),
                                    scale,
                                    scale_skip,
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast
                                ],
                            )
                            self.weightsInBN13_layer3_get.release(ObjectFifoPort.Consume,1)
                            yield_([])
                        yield_([])
                    bn13_act_layer2_layer3_second.release(ObjectFifoPort.Consume,1)
                    bn13_act_layer3_bn_14_layer1.release(ObjectFifoPort.Produce,1)
                    self.bn13_skip.release(ObjectFifoPort.Consume,1)                    
                    yield_([])
                yield_([])


        # ************************ bneck14 ************************
        # conv1x1_first put
        @core(self.computeTileBN14_layer1_put, "bn14_1_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_13_InH1):
                    elemIn = bn13_act_layer3_bn_14_layer1.acquire(ObjectFifoPort.Consume, 1)
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit):
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN14_layer1_put.acquire(ObjectFifoPort.Consume, 1)
                            for x_start in range(0,bneck_14_InW1,7):
                                call(
                                    bn14_conv2dk1_fused_relu_put,
                                    [
                                        elemIn,
                                        elemWts,
                                        arith.constant(bneck_14_InW1),
                                        arith.constant(bneck_14_InC1),
                                        arith.constant(bneck_14_OutC1),
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                            self.weightsInBN14_layer1_put.release(ObjectFifoPort.Consume,1)
                            yield_([])
                        yield_([])
                    bn13_act_layer3_bn_14_layer1.release(ObjectFifoPort.Consume,1)                    
                    yield_([])
                yield_([])

        # conv1x1_first get
        @core(self.computeTileBN14_layer1_get, "bn14_1_conv2dk1_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_13_InH1):
                    elemIn = bn13_act_layer3_bn_14_layer1.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = bn14_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)
                    
                    # scale = memref.load(rtp04, [0])
                    scale = self.bn14_scaleFactor1
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit):
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        for WeightIndex in for_(InputSplit//2, InputSplit):
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN14_layer1_get.acquire(ObjectFifoPort.Consume, 1)
                            for x_start in range(0,bneck_14_InW1,7):
                                call(
                                    bn14_conv2dk1_fused_relu_get,
                                    [
                                        elemIn,
                                        elemWts,
                                        elemOut0,
                                        arith.constant(bneck_14_InW1),
                                        arith.constant(bneck_14_InC1),
                                        arith.constant(bneck_14_OutC1),
                                        scale,
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                            self.weightsInBN14_layer1_get.release(ObjectFifoPort.Consume,1)
                            yield_([])
                        yield_([])
                    bn13_act_layer3_bn_14_layer1.release(ObjectFifoPort.Consume,1)   
                    bn14_act_layer1_layer2.release(ObjectFifoPort.Produce,1)   
                    
                    yield_([])
                yield_([])
        
        # conv3x3
        @core(self.computeTileBN14_layer2, "bn14_conv2dk3_dw.o")
        def core_body():
            scale = self.bn14_scaleFactor2
            for _ in for_(sys.maxsize):

                # acquire weights and rtps once
                element0Weights = self.weightsInBN14_layer2.acquire(ObjectFifoPort.Consume, 1)
                # scale = memref.load(rtpself.computeTileBN13_layer1_get, 0)

                # pre-amble: top row
                elementActivactionsIn = bn14_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn14_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                element1ActivactionsOut = bn14_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn14_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
                        bneck_14_InW2,
                        1,
                        bneck_14_OutC2,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    ],
                )
                bn14_act_layer2_layer3_first.release(ObjectFifoPort.Produce,1) 
                bn14_act_layer2_layer3_second.release(ObjectFifoPort.Produce,1) 

                # middle
                for _ in for_(bneck_14_InH2 - 2):
                    elementActivactionsIn = bn14_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 3
                    )
                    element0ActivactionsOut = bn14_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                    element1ActivactionsOut = bn14_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
                    res = call(
                        bn14_conv2dk3_dw,
                        [
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            element1ActivactionsOut,
                            bneck_14_InW2,
                            1,
                            bneck_14_OutC2,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        ],
                    )
                    bn14_act_layer1_layer2.release(ObjectFifoPort.Consume,1) 
                    bn14_act_layer2_layer3_first.release(ObjectFifoPort.Produce,1) 
                    bn14_act_layer2_layer3_second.release(ObjectFifoPort.Produce,1) 

                    yield_([])

                # last part
                elementActivactionsIn = bn14_act_layer1_layer2.acquire(
                    ObjectFifoPort.Consume, 2
                )
                element0ActivactionsOut = bn14_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                element1ActivactionsOut = bn14_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
                res = call(
                    bn14_conv2dk3_dw,
                    [
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        element1ActivactionsOut,
                        bneck_14_InW2,
                        1,
                        bneck_14_OutC2,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    ],
                )

                bn14_act_layer1_layer2.release(ObjectFifoPort.Consume,2) 
                bn14_act_layer2_layer3_first.release(ObjectFifoPort.Produce,1) 
                bn14_act_layer2_layer3_second.release(ObjectFifoPort.Produce,1) 
                self.weightsInBN14_layer2.release(ObjectFifoPort.Consume,1) 
                yield_([])
        
        # conv1x1_second put
        @core(self.computeTileBN14_layer3_put, "bn14_conv2dk1_put.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_14_InH3):
                    elemIn = bn14_act_layer2_layer3_first.acquire(ObjectFifoPort.Consume, 1)
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit2):
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        # for WeightIndex in range (0,InputSplit//2):
                        for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN14_layer3_put.acquire(ObjectFifoPort.Consume, 1)
                            x_start=0  
                            call(
                                bn14_layer3_conv2dk1_put,
                                [
                                    elemIn,
                                    elemWts,
                                    arith.constant(bneck_14_InW3),
                                    arith.constant(bneck_14_OutC2),
                                    arith.constant(bneck_14_OutC3),
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast
                                ],
                            )
                            self.weightsInBN14_layer3_put.release(ObjectFifoPort.Consume,1) 

                            yield_([])
                        yield_([])
                    bn14_act_layer2_layer3_first.release(ObjectFifoPort.Consume,1) 
                    
                    yield_([])
                yield_([])

        # conv1x1_second get
        @core(self.computeTileBN14_layer3_get, "bn14_conv2dk1_skip_get.o")
        def core_body():
            for _ in for_(0xFFFFFFFF):
                
                for _ in for_(bneck_14_InH3):
                    
                    elemIn = bn14_act_layer2_layer3_second.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    elementSkipsIn = bn14_skip.acquire(ObjectFifoPort.Consume, 1)
                    
                    scale = self.bn14_scaleFactor3
                    scale_skip = self.bn14_scaleFactorAdd
                    # scale = memref.load(rtp04, [0])
                    # for oc in range(0,OutputSplit):
                    for oc in for_(OutputSplit2):
                        
                        oc_cast= arith.IndexCastOp(T.i32(), oc)
                        for WeightIndex in for_(0,InputSplit//2):
                            WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                            elemWts = self.weightsInBN14_layer3_get.acquire(ObjectFifoPort.Consume, 1)
                            x_start=0 

                            call(
                                bn14_layer3_conv2dk1_skip_get,
                                [
                                    elemIn,
                                    elemWts,
                                    elemOut0,
                                    elementSkipsIn,
                                    arith.constant(bneck_14_InW3),
                                    arith.constant(bneck_14_OutC2),
                                    arith.constant(bneck_14_OutC3),
                                    scale,
                                    scale_skip,
                                    InputSplit,
                                    WeightIndex_cast,
                                    x_start,
                                    oc_cast
                                ],
                            )
                            self.weightsInBN14_layer3_get.release(ObjectFifoPort.Consume,1) 
                            yield_([])
                        yield_([])
                    bn14_act_layer2_layer3_second.release(ObjectFifoPort.Consume,1) 
                    self.actOut.release(ObjectFifoPort.Produce,1) 
                    bn14_skip.release(ObjectFifoPort.Consume,1) 
                    yield_([])
                yield_([])

def mobilenetV3_bn_13_14(start_row = 2, start_col = 0, 
                        bn13_scaleFactor1=10,bn13_scaleFactor2=7,bn13_scaleFactor3=9,bn13_scaleFactorAdd=1,
                        bn14_scaleFactor1=9,bn14_scaleFactor2=8,bn14_scaleFactor3=12,bn14_scaleFactorAdd=1):

            bneck_13_InW1 = 7
            bneck_13_InH1 = 7
            bneck_13_InC1 = 160
            bneck_13_OutC1 = 960
            InputSplit=2
            OutputSplit=bneck_13_OutC1//8 #calculate 8 OCs at a time, should bneck_13_InC1rease to more


            RepeatChannels=math.floor(bneck_13_InH1)

            bneck_13_InW2 = bneck_13_InW1
            bneck_13_InH2 = bneck_13_InH1
            bneck_13_OutC2 = bneck_13_OutC1

            bneck_13_InW3 = bneck_13_InW2
            bneck_13_InH3 = bneck_13_InH2
            bneck_13_OutC3 = 160
            OutputSplit2=bneck_13_OutC3//8 #calculate 8 OCs at a time, should bneck_13_InC1rease to more

            # second block
            bneck_14_InW1 = bneck_13_InW1
            bneck_14_InH1 = bneck_13_InH1
            bneck_14_InC1 = bneck_13_OutC3
            bneck_14_OutC1 = 960

            bneck_14_InW2 = bneck_14_InW1
            bneck_14_InH2 = bneck_14_InH1
            bneck_14_OutC2 = bneck_14_OutC1

            bneck_14_InW3 = bneck_14_InW2
            bneck_14_InH3 = bneck_14_InH2
            bneck_14_OutC3 = 160
            OutputSplit3 = bneck_14_OutC3 // 8  # Calculate 8 OCs at a time, should increase to more


            @device(AIEDevice.npu1_4col)
            def device_body():
                    # define types
                    uint8_ty = IntegerType.get_unsigned(8)
                    int8_ty = IntegerType.get_signless(8)
                    int32_ty = IntegerType.get_signless(32)
                    uint32_ty = IntegerType.get_unsigned(32)
                    # ************************ bneck13 ************************
  
                    ty_bneck_13_layer1_in = MemRefType.get((bneck_13_InW1,1,bneck_13_InC1,),int8_ty,)
                    ty_bneck_13_layer2_in = MemRefType.get((bneck_13_InW2,1,bneck_13_OutC1,),uint8_ty,)
                
                    # define wts
                    #layer1
                    ty_bneck_13_layer1_wts_split = MemRefType.get(((bneck_13_InC1 * 8)//(InputSplit),), int8_ty)
                    ty_bneck_13_layer1_wts_full= MemRefType.get((bneck_13_InC1 * bneck_13_OutC1,),int8_ty,)
                    # layer2
                    ty_bneck_13_layer2_wts = MemRefType.get((3 * 3 * bneck_13_OutC2 * 1,), int8_ty )
                    # layer3
                    ty_bneck_13_layer3_wts_split = MemRefType.get(((bneck_13_OutC2 * 8)//(InputSplit),), int8_ty)
                    ty_bneck_13_layer3_wts_full= MemRefType.get((bneck_13_OutC2 * bneck_13_OutC3,),int8_ty,)

                    # OUTPUT
                    ty_bneck_13_layer1_out = MemRefType.get((bneck_13_InW1,1,bneck_13_OutC1,),uint8_ty,)
                    ty_bneck_13_layer2_out = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC2, ), uint8_ty, )
                    ty_bneck_13_layer2_out_split = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC2//InputSplit, ), uint8_ty, )
                    # layer3
                    ty_bneck_13_layer3_out = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC3, ), int8_ty, )
                    
        # HERE

        # ************************ bneck14 ************************

                    ty_bneck_14_layer1_in = MemRefType.get((bneck_14_InW1, 1, bneck_14_InC1,), int8_ty,)
                    ty_bneck_14_layer2_in = MemRefType.get((bneck_14_InW2, 1, bneck_14_OutC1,), uint8_ty,)

                    # define wts
                    # layer1
                    ty_bneck_14_layer1_wts_split = MemRefType.get(((bneck_14_InC1 * 8) // (InputSplit),), int8_ty)
                    ty_bneck_14_layer1_wts_full = MemRefType.get((bneck_14_InC1 * bneck_14_OutC1,), int8_ty,)
                    # layer2
                    ty_bneck_14_layer2_wts = MemRefType.get((3 * 3 * bneck_14_OutC2 * 1,), int8_ty)
                    # layer3
                    ty_bneck_14_layer3_wts_split = MemRefType.get(((bneck_14_OutC2 * 8) // (InputSplit),), int8_ty)
                    ty_bneck_14_layer3_wts_full = MemRefType.get((bneck_14_OutC2 * bneck_14_OutC3,), int8_ty,)

                    # OUTPUT
                    ty_bneck_14_layer1_out = MemRefType.get((bneck_14_InW1, 1, bneck_14_OutC1,), uint8_ty,)
                    ty_bneck_14_layer2_out = MemRefType.get((bneck_14_InW3, 1, bneck_14_OutC2,), uint8_ty,)
                    ty_bneck_14_layer2_out_split = MemRefType.get((bneck_14_InW3, 1, bneck_14_OutC2 // InputSplit,), uint8_ty,)
                    # layer3
                    ty_bneck_14_layer3_out = MemRefType.get((bneck_14_InW3, 1, bneck_14_OutC3,), int8_ty,)


                    # Tile declarations
                    ShimTile00 = tile(0, 0)
                    ShimTile10 = tile(1, 0)
                    ShimTile20 = tile(2, 0)
                    ShimTile30 = tile(3, 0)

                    MemTile01 = tile(0, 1)
                    MemTile11 = tile(1, 1)
                    MemTile21 = tile(2, 1)


                    bn13_tile_layer1_put = tile(0, 5)
                    bn13_tile_layer1_get = tile(0, 4)
                    bn13_tile_layer2 = tile(1, 4)
                    # ComputeTile15 = tile(1, 5)
                    
                    
                    bn13_tile_layer3_get = tile(1, 3)
                    bn13_tile_layer3_put = tile(0, 3)
                    

                    cascade_flow(bn13_tile_layer1_put, bn13_tile_layer1_get)
                    cascade_flow(bn13_tile_layer3_put, bn13_tile_layer3_get)


                    # tiles bn14

                    #  conv1
                    bn14_tile_layer1_put = tile(1, 2) #put
                    bn14_tile_layer1_get = tile(2, 2) #get

                    cascade_flow(bn14_tile_layer1_put, bn14_tile_layer1_get)

                    #conv3
                    bn14_tile_layer2 = tile(2, 3)

                    # conv
                    bn14_tile_layer3_put = tile(2, 5) #put
                    bn14_tile_layer3_get = tile(2, 4) #get
                    cascade_flow(bn14_tile_layer3_put, bn14_tile_layer3_get)


                                # Input
                    act_in = object_fifo("act_in", ShimTile00, [bn13_tile_layer1_put,bn13_tile_layer1_get,MemTile01], [2, 2, 2, 6], ty_bneck_13_layer1_in,)
                    bn13_skip = object_fifo("bn13_skip", MemTile01, bn13_tile_layer3_get, 2, ty_bneck_13_layer1_in)
                    object_fifo_link(act_in, bn13_skip)
                
                    # ************ wts ************
                        #LAYER1 
                    bn13_wts_L3L2_layer1 = object_fifo("bn13_wts_L3L2_layer1", ShimTile00, MemTile01, 1, ty_bneck_13_layer1_wts_full)
                    bn13_wts_memtile_layer1_put = object_fifo("bn13_wts_memtile_layer1_put", MemTile01, bn13_tile_layer1_put, [1,1], ty_bneck_13_layer1_wts_split)
                    bn13_wts_memtile_layer1_get = object_fifo("bn13_wts_memtile_layer1_get",MemTile01,bn13_tile_layer1_get,[1,1],ty_bneck_13_layer1_wts_split,)
                    object_fifo_link(bn13_wts_L3L2_layer1, [bn13_wts_memtile_layer1_put,bn13_wts_memtile_layer1_get],[],[0,(bneck_13_InC1 * bneck_13_OutC1)//2])
                    bn13_wts_memtile_layer1_put.set_memtile_repeat(RepeatChannels)
                    bn13_wts_memtile_layer1_get.set_memtile_repeat(RepeatChannels)
                        # LAYER2
                    bn13_wts_L3L2_layer2 = object_fifo("bn13_wts_L3L2_layer2", ShimTile10, MemTile11, 1, ty_bneck_13_layer2_wts )
                    bn13_wts_memtile_layer2 = object_fifo("bn13_wts_memtile_layer2",MemTile11,bn13_tile_layer2,1,ty_bneck_13_layer2_wts,)
                    object_fifo_link(bn13_wts_L3L2_layer2, [bn13_wts_memtile_layer2],[],[0])

                    #LAYER3
                    bn13_wts_L3L2_layer3 = object_fifo("bn13_wts_L3L2_layer3", ShimTile10, MemTile01, 1, ty_bneck_13_layer3_wts_full)
                    bn13_wts_memtile_layer3_put = object_fifo("bn13_wts_memtile_layer3_put", MemTile01, bn13_tile_layer3_put, 1, ty_bneck_13_layer3_wts_split)
                    bn13_wts_memtile_layer3_get = object_fifo("bn13_wts_memtile_layer3_get",MemTile01,bn13_tile_layer3_get,1,ty_bneck_13_layer3_wts_split)
                    object_fifo_link(bn13_wts_L3L2_layer3, [bn13_wts_memtile_layer3_put,bn13_wts_memtile_layer3_get],[],[0,(bneck_13_OutC2 * bneck_13_OutC3)//2])
                    bn13_wts_memtile_layer3_put.set_memtile_repeat(RepeatChannels)
                    bn13_wts_memtile_layer3_get.set_memtile_repeat(RepeatChannels)

                    # ************ wts ************
                    # wts for new block
                    bn14_wts_L3L2_layer1 = object_fifo("bn14_wts_L3L2_layer1", ShimTile20, MemTile21, 1, ty_bneck_14_layer1_wts_full)
                    bn14_wts_memtile_layer1_put = object_fifo("bn14_wts_memtile_layer1_put", MemTile21, bn14_tile_layer1_put, [1,1], ty_bneck_14_layer1_wts_split)
                    bn14_wts_memtile_layer1_get = object_fifo("bn14_wts_memtile_layer1_get", MemTile21, bn14_tile_layer1_get, [1,1], ty_bneck_14_layer1_wts_split,)
                    object_fifo_link(bn14_wts_L3L2_layer1, [bn14_wts_memtile_layer1_put, bn14_wts_memtile_layer1_get], [], [0, (bneck_14_InC1 * bneck_14_OutC1) // 2])
                    bn14_wts_memtile_layer1_put.set_memtile_repeat(RepeatChannels)
                    bn14_wts_memtile_layer1_get.set_memtile_repeat(RepeatChannels)
                    # LAYER2
                    bn14_wts_L3L2_layer2 = object_fifo("bn14_wts_L3L2_layer2", ShimTile20, MemTile11, 1, ty_bneck_14_layer2_wts)
                    bn14_wts_memtile_layer2 = object_fifo("bn14_wts_memtile_layer2", MemTile11, bn14_tile_layer2, 1, ty_bneck_14_layer2_wts)
                    object_fifo_link(bn14_wts_L3L2_layer2, bn14_wts_memtile_layer2, [], [0])
                    # LAYER3
                    bn14_wts_L3L2_layer3 = object_fifo("bn14_wts_L3L2_layer3", ShimTile30, MemTile21, 1, ty_bneck_14_layer3_wts_full)
                    bn14_wts_memtile_layer3_put = object_fifo("bn14_wts_memtile_layer3_put", MemTile21, bn14_tile_layer3_put, [1,1], ty_bneck_14_layer3_wts_split)
                    bn14_wts_memtile_layer3_get = object_fifo("bn14_wts_memtile_layer3_get", MemTile21, bn14_tile_layer3_get, [1,1], ty_bneck_14_layer3_wts_split,)
                    object_fifo_link(bn14_wts_L3L2_layer3, [bn14_wts_memtile_layer3_put, bn14_wts_memtile_layer3_get], [], [0, (bneck_14_OutC2 * bneck_14_OutC3) // 2])
                    bn14_wts_memtile_layer3_put.set_memtile_repeat(RepeatChannels)
                    bn14_wts_memtile_layer3_get.set_memtile_repeat(RepeatChannels)

                    act_out = object_fifo("act_out", bn14_tile_layer3_get, ShimTile30, 2, ty_bneck_14_layer3_out)

                                # Set up compute tiles
                    rtp_bn13_tile_layer1_get = Buffer(bn13_tile_layer1_get, [16], T.i32(), "rtp_bn13_tile_layer1_get")
                    rtp_bn13_tile_layer3_get = Buffer(bn13_tile_layer3_get, [16], T.i32(), "rtp_bn13_tile_layer3_get")

                    bottleneckCCore(bn13_tile_layer1_put,bn13_tile_layer1_get,bn13_tile_layer2,bn13_tile_layer3_put,bn13_tile_layer3_get,
                                    bn14_tile_layer1_put,bn14_tile_layer1_get,bn14_tile_layer2,bn14_tile_layer3_put,bn14_tile_layer3_get,
                                    bn13_wts_memtile_layer1_put,bn13_wts_memtile_layer1_get,bn13_wts_memtile_layer2,bn13_wts_memtile_layer3_put,bn13_wts_memtile_layer3_get,
                                    bn14_wts_memtile_layer1_put,bn14_wts_memtile_layer1_get,bn14_wts_memtile_layer2,bn14_wts_memtile_layer3_put,bn14_wts_memtile_layer3_get,
                                    rtp_bn13_tile_layer1_get,rtp_bn13_tile_layer3_get,bn13_scaleFactor1,bn13_scaleFactor2,bn13_scaleFactor3,bn13_scaleFactorAdd,
                                    bn14_scaleFactor1,bn14_scaleFactor2,bn14_scaleFactor3,bn14_scaleFactorAdd,MemTile21,act_in,act_out,bn13_skip )

                    # # instruction stream generation
                    activationsInSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_InC1) // 4
                    acitivationsOutSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_14_OutC3) // 4

                    bneck_13_totalWeightsSize32b_layer1 = (
                    bneck_13_InC1*bneck_13_OutC1
                    ) // 4

                    bneck_13_totalWeightsSize32b_layer2 = (
                    3 * 3 * bneck_13_OutC2 * 1
                    #    +bneck_13_OutC2*bneck_13_OutC3
                    ) // 4

                    bneck_13_totalWeightsSize32b_layer3 = (
                    bneck_13_OutC2*bneck_13_OutC3
                    ) // 4

                    bneck_13_layer3_offset=(
                        bneck_13_totalWeightsSize32b_layer1
                        +bneck_13_totalWeightsSize32b_layer2

                    )

                    bneck_13_totalWeightsSize32b_complete = (
                        bneck_13_totalWeightsSize32b_layer1
                        +bneck_13_totalWeightsSize32b_layer2
                        +bneck_13_totalWeightsSize32b_layer3
                    )

                    bneck_14_layer1_offset=(
                        bneck_13_totalWeightsSize32b_layer1
                        +bneck_13_totalWeightsSize32b_layer2
                        +bneck_13_totalWeightsSize32b_layer3

                    )
                    bneck_14_layer2_offset=(
                        2*bneck_13_totalWeightsSize32b_layer1
                        +bneck_13_totalWeightsSize32b_layer2
                        +bneck_13_totalWeightsSize32b_layer3

                    )

                    bneck_14_layer3_offset=(
                        2*bneck_13_totalWeightsSize32b_layer1
                        +2*bneck_13_totalWeightsSize32b_layer2
                        +bneck_13_totalWeightsSize32b_layer3

                    )


                    totalWeightsSize32b_complete = (
                        2*(bneck_13_totalWeightsSize32b_layer1
                        +bneck_13_totalWeightsSize32b_layer2
                        +bneck_13_totalWeightsSize32b_layer3)
                    )





                    activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
                    weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
                    activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

                    @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
                    def sequence(inputFromL3, weightsFromL3, outputToL3):
                        # NpuWriteRTPOp("rtp04", col=0, row=4, index=0, value=9)
                        # NpuWriteRTPOp("rtp13", col=1, row=3, index=0, value=11)

                            
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
                            metadata="bn13_wts_L3L2_layer1",
                            bd_id=1,
                            mem=weightsFromL3,
                            sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer1],
                        )
                        npu_dma_memcpy_nd(
                            metadata="bn13_wts_L3L2_layer2",
                            bd_id=1,
                            mem=weightsFromL3,
                            offsets=[0, 0, 0, bneck_13_totalWeightsSize32b_layer1],
                            sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer2],
                        )

                        npu_dma_memcpy_nd(
                            metadata="bn13_wts_L3L2_layer3",
                            bd_id=1,
                            mem=weightsFromL3,
                            offsets=[0, 0, 0, bneck_13_layer3_offset],
                            sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer3],
                        )


                        npu_dma_memcpy_nd(
                            metadata="bn14_wts_L3L2_layer1",
                            bd_id=1,
                            mem=weightsFromL3,
                            offsets=[0, 0, 0, bneck_14_layer1_offset],
                            sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer1],
                        )

                        npu_dma_memcpy_nd(
                            metadata="bn14_wts_L3L2_layer2",
                            bd_id=1,
                            mem=weightsFromL3,
                            offsets=[0, 0, 0, bneck_14_layer2_offset],
                            sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer2],
                        )

                        npu_dma_memcpy_nd(
                            metadata="bn14_wts_L3L2_layer3",
                            bd_id=1,
                            mem=weightsFromL3,
                            offsets=[0, 0, 0, bneck_14_layer3_offset],
                            sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer3],
                        )


                        npu_sync(column=3, row=0, direction=0, channel=0)

with mlir_mod_ctx() as ctx:
    mobilenetV3_bn_13_14(bn13_scaleFactor1=scale_factors["BN13"]["conv1x1_1"],bn13_scaleFactor2=scale_factors["BN13"]["conv3x3"],bn13_scaleFactor3=scale_factors["BN13"]["conv1x1_2"],bn13_scaleFactorAdd=scale_factors["BN13"]["skip_add"],
                           bn14_scaleFactor1=scale_factors["BN14"]["conv1x1_1"],bn14_scaleFactor2=scale_factors["BN14"]["conv3x3"],bn14_scaleFactor3=scale_factors["BN14"]["conv1x1_2"],bn14_scaleFactorAdd=scale_factors["BN14"]["skip_add"])
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)