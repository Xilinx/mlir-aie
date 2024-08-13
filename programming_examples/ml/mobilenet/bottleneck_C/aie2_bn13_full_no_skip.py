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


bneck_13_InW1 = 7
bneck_13_InH1 = 6
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

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

def mobilenetBottleneckB():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_3col)
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
            ty_bneck_13_layer3_out = MemRefType.get((bneck_13_InW3, 1, bneck_13_OutC3, ), uint8_ty, )
            
# HERE
            
   
            
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

            bn13_layer3_conv2dk1_get = external_func(
                "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_get",
                inputs=[
                    ty_bneck_13_layer2_out_split,
                    ty_bneck_13_layer3_wts_split,
                    ty_bneck_13_layer3_out,
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
            ShimTile00 = tile(0, 0)
            ShimTile10 = tile(1, 0)
            ShimTile20 = tile(2, 0)

            MemTile01 = tile(0, 1)
            MemTile11 = tile(1, 1)
            MemTile21 = tile(2, 1)


            ComputeTile05 = tile(0, 5)
            ComputeTile04 = tile(0, 4)
            ComputeTile14 = tile(1, 4)
            # ComputeTile15 = tile(1, 5)

            ComputeTile13 = tile(1, 3)
            ComputeTile03 = tile(0, 3)

            cascade_flow(ComputeTile05, ComputeTile04)
            cascade_flow(ComputeTile03, ComputeTile13)
            # AIE-array data movement with object fifos
            # ************************ bneck13 ************************
            # Input
            inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile00, [ComputeTile05,ComputeTile04], [2, 2, 2], ty_bneck_13_layer1_in,)
        
            # ************ wts ************
                #LAYER1 
            OF_bneck_13_wts_L3L2_layer1 = object_fifo("OF_bneck_13_wts_L3L2_layer1", ShimTile00, MemTile01, 1, ty_bneck_13_layer1_wts_full)
            OF_bneck_13_wts_memtile_layer1_put = object_fifo("OF_bneck_13_wts_memtile_layer1_put", MemTile01, ComputeTile05, [1,1], ty_bneck_13_layer1_wts_split)
            OF_bneck_13_wts_memtile_layer1_get = object_fifo("OF_bneck_13_wts_memtile_layer1_get",MemTile01,ComputeTile04,[1,1],ty_bneck_13_layer1_wts_split,)
            object_fifo_link(OF_bneck_13_wts_L3L2_layer1, [OF_bneck_13_wts_memtile_layer1_put,OF_bneck_13_wts_memtile_layer1_get],[],[0,(bneck_13_InC1 * bneck_13_OutC1)//2])
            OF_bneck_13_wts_memtile_layer1_put.set_memtile_repeat(RepeatChannels)
            OF_bneck_13_wts_memtile_layer1_get.set_memtile_repeat(RepeatChannels)
                # LAYER2
            OF_bneck_13_wts_L3L2_layer2 = object_fifo("OF_bneck_13_wts_L3L2_layer2", ShimTile10, MemTile11, 1, ty_bneck_13_layer2_wts )
            OF_bneck_13_wts_memtile_layer2 = object_fifo("OF_bneck_13_wts_memtile_layer2",MemTile11,ComputeTile14,1,ty_bneck_13_layer2_wts,)
            object_fifo_link(OF_bneck_13_wts_L3L2_layer2, [OF_bneck_13_wts_memtile_layer2],[],[0])

             #LAYER3
            OF_bneck_13_wts_L3L2_layer3 = object_fifo("OF_bneck_13_wts_L3L2_layer3", ShimTile20, MemTile21, 1, ty_bneck_13_layer3_wts_full)
            OF_bneck_13_wts_memtile_layer3_put = object_fifo("OF_bneck_13_wts_memtile_layer3_put", MemTile21, ComputeTile03, 1, ty_bneck_13_layer3_wts_split)
            OF_bneck_13_wts_memtile_layer3_get = object_fifo("OF_bneck_13_wts_memtile_layer3_get",MemTile21,ComputeTile13,1,ty_bneck_13_layer3_wts_split)
            object_fifo_link(OF_bneck_13_wts_L3L2_layer3, [OF_bneck_13_wts_memtile_layer3_put,OF_bneck_13_wts_memtile_layer3_get],[],[0,(bneck_13_OutC2 * bneck_13_OutC3)//2])
            OF_bneck_13_wts_memtile_layer3_put.set_memtile_repeat(RepeatChannels)
            OF_bneck_13_wts_memtile_layer3_get.set_memtile_repeat(RepeatChannels)
        
            # Set up compute tiles
            rtp04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")
            rtp13 = Buffer(ComputeTile13, [16], T.i32(), "rtp13")

            # OUTPUT
            OF_bneck_13_act_layer1_layer2 = object_fifo("OF_bneck_13_act_layer1_layer2", ComputeTile04, [ComputeTile14], 4,ty_bneck_13_layer2_in,via_DMA=True)
            
            OF_bneck_13_act_layer2_layer3_first = object_fifo("OF_bneck_13_act_layer2_layer3_first", ComputeTile14, [ComputeTile03], 2, ty_bneck_13_layer2_out_split)
            OF_bneck_13_act_layer2_layer3_second = object_fifo("OF_bneck_13_act_layer2_layer3_second", ComputeTile14, [ComputeTile13], 2, ty_bneck_13_layer2_out_split)


            OF_bneck_13_act_layer3_out = object_fifo("OF_bneck_13_act_layer3_out", ComputeTile13, [MemTile01], 2, ty_bneck_13_layer3_out)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile01, [ShimTile00], 2, ty_bneck_13_layer3_out)
            object_fifo_link(OF_bneck_13_act_layer3_out, [OF_outOFL2L3],[],[0])
            
            # object_fifo_link(OF_bneck_13_act_layer2_layer3_first, [OF_outOFL2L3],[],[0])
            # object_fifo_link([OF_bneck_13_act_layer2_layer3_first,OF_bneck_13_act_layer2_layer3_second],[OF_outOFL2L3],[0,(bneck_13_InW3 *  bneck_13_OutC2//2)])



            # conv1x1_first put
            @core(ComputeTile05, "bn13_1_conv2dk1_put.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(bneck_13_InH1):
                        elemIn = inOF_act_L3L2.acquire(ObjectFifoPort.Consume, 1)
                        # for oc in range(0,OutputSplit):
                        for oc in for_(OutputSplit):
                            oc_cast= arith.IndexCastOp(T.i32(), oc)
                            for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                                WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                                elemWts = OF_bneck_13_wts_memtile_layer1_put.acquire(ObjectFifoPort.Consume, 1)
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
                                objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer1_put", 1)
                                yield_([])
                            yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)
                        
                        yield_([])
                    yield_([])

            # conv1x1_first get
            @core(ComputeTile04, "bn13_1_conv2dk1_get.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(bneck_13_InH1):
                        elemIn = inOF_act_L3L2.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_13_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)
                        
                        # scale = memref.load(rtp04, [0])
                        scale = 9
                        # for oc in range(0,OutputSplit):
                        for oc in for_(OutputSplit):
                            oc_cast= arith.IndexCastOp(T.i32(), oc)
                            for WeightIndex in for_(InputSplit//2, InputSplit):
                                WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                                elemWts = OF_bneck_13_wts_memtile_layer1_get.acquire(ObjectFifoPort.Consume, 1)
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
                                objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer1_get", 1)
                                yield_([])
                            yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer1_layer2", 1)
                        
                        yield_([])
                    yield_([])
            
            # conv3x3
            @core(ComputeTile14, "bn13_conv2dk3_dw.o")
            def core_body():
                scale = 8
                for _ in for_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = OF_bneck_13_wts_memtile_layer2.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile04, 0)

                    # pre-amble: top row
                    elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_13_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                    element1ActivactionsOut = OF_bneck_13_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
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
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3_first", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3_second", 1)

                    # middle
                    for _ in for_(bneck_13_InH2 - 2):
                        elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = OF_bneck_13_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                        element1ActivactionsOut = OF_bneck_13_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
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

                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer1_layer2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3_first", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3_second", 1)
                        yield_([])

                    # last part
                    elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = OF_bneck_13_act_layer2_layer3_first.acquire(ObjectFifoPort.Produce, 1)
                    element1ActivactionsOut = OF_bneck_13_act_layer2_layer3_second.acquire(ObjectFifoPort.Produce, 1)
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

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer1_layer2", 2)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3_first", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer2_layer3_second", 1)

                    objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer2", 1)
                    yield_([])
            
            # conv1x1_second put
            @core(ComputeTile03, "conv2dk1_put.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(bneck_13_InH3):
                        elemIn = OF_bneck_13_act_layer2_layer3_first.acquire(ObjectFifoPort.Consume, 1)
                        # for oc in range(0,OutputSplit):
                        for oc in for_(OutputSplit2):
                            oc_cast= arith.IndexCastOp(T.i32(), oc)
                            # for WeightIndex in range (0,InputSplit//2):
                            for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                                WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                                elemWts = OF_bneck_13_wts_memtile_layer3_put.acquire(ObjectFifoPort.Consume, 1)
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
                                objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer3_put", 1)
                                yield_([])
                            yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer2_layer3_first", 1)
                        
                        yield_([])
                    yield_([])

            # conv1x1_second get
            @core(ComputeTile13, "conv2dk1_get.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(bneck_13_InH3):
                        
                        elemIn = OF_bneck_13_act_layer2_layer3_second.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_13_act_layer3_out.acquire(ObjectFifoPort.Produce, 1)
                        
                        scale = 11
                        # scale = memref.load(rtp04, [0])
                        # for oc in range(0,OutputSplit):
                        for oc in for_(OutputSplit2):
                            
                            oc_cast= arith.IndexCastOp(T.i32(), oc)
                            for WeightIndex in for_(0,InputSplit//2):
                                WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                                elemWts = OF_bneck_13_wts_memtile_layer3_get.acquire(ObjectFifoPort.Consume, 1)
                                x_start=0 

                                call(
                                    bn13_layer3_conv2dk1_get,
                                    [
                                        elemIn,
                                        elemWts,
                                        elemOut0,
                                        arith.constant(bneck_13_InW3),
                                        arith.constant(bneck_13_OutC2),
                                        arith.constant(bneck_13_OutC3),
                                        scale,
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer3_get", 1)
                                yield_([])
                            yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer2_layer3_second", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer3_out", 1)
                        
                        yield_([])
                    yield_([])

            # # instruction stream generation
            activationsInSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_InC1) // 4

            acitivationsOutSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_OutC3) // 4

        
            totalWeightsSize32b_layer1 = (
               bneck_13_InC1*bneck_13_OutC1
            ) // 4

            totalWeightsSize32b_layer2 = (
               3 * 3 * bneck_13_OutC2 * 1
            #    +bneck_13_OutC2*bneck_13_OutC3
            ) // 4

            totalWeightsSize32b_layer3 = (
               bneck_13_OutC2*bneck_13_OutC3
            ) // 4

            layer3_offset=(
                totalWeightsSize32b_layer1
                +totalWeightsSize32b_layer2

            )

            totalWeightsSize32b_complete = (
                totalWeightsSize32b_layer1
                +totalWeightsSize32b_layer2
                +totalWeightsSize32b_layer3
            )

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

            @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                NpuWriteRTPOp("rtp04", index=0, value=9)
                NpuWriteRTPOp("rtp13", index=0, value=11)

                    
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
                    metadata="OF_bneck_13_wts_L3L2_layer1",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeightsSize32b_layer1],
                )
                npu_dma_memcpy_nd(
                    metadata="OF_bneck_13_wts_L3L2_layer2",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[0, 0, 0, totalWeightsSize32b_layer1],
                    sizes=[1, 1, 1, totalWeightsSize32b_layer2],
                )

                npu_dma_memcpy_nd(
                    metadata="OF_bneck_13_wts_L3L2_layer3",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[0, 0, 0, layer3_offset],
                    sizes=[1, 1, 1, totalWeightsSize32b_layer3],
                )


                npu_sync(column=0, row=0, direction=0, channel=0)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


mobilenetBottleneckB()