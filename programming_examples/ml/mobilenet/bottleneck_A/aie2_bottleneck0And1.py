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
from aie.extras.context import mlir_mod_ctx
#from aie.dialects.memref import *
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view

import aie.utils.trace as trace_utils

def mobilenetV3Bottleneck0And1(tileRowIndex = 2, tileColIndex = 0, tensorInW = 112, tensorInH = 112, tensorInC = 16, depthWiseStride = 2, depthWiseChannels = 64, tensorOutC = 16, scaleFactor0_2 = 8, scaleFactor0_3 = 9, scaleFactorAdd0 = 1, scaleFactor1_1 = 8, scaleFactor1_2 = 9, scaleFactor1_3 = 11, enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    tensorOutW = tensorInW // depthWiseStride
    tensorOutH = tensorInH // depthWiseStride

    tensorL0_2InC = tensorInC 
    tensorL0_2OutC = tensorL0_2InC

    tensorL0_3InC = tensorL0_2InC
    tensorL0_3OutC = tensorL0_3InC

    tensorL1_1InC = tensorL0_3OutC
    tensorL1_1OutC = depthWiseChannels

    tensorL1_2InC = tensorL1_1OutC
    tensorL1_2OutC = tensorL1_2InC

    tensorL1_3InC = tensorL1_2InC
    tensorL1_3OutC = tensorOutC

    @device(AIEDevice.npu1_1col)
    def device_body():
        
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)
        tensorLayer0_2In_ty = MemRefType.get((tensorInW, 1, tensorL0_2InC), uint8_ty)
        weightsLayer0_2_ty = MemRefType.get((3 * 3 * tensorL0_2OutC * 1,), int8_ty)
        tensorLayer0_2Out_ty = MemRefType.get((tensorInW, 1, tensorL0_2OutC),uint8_ty)
        tensorLayer0_3In_ty = MemRefType.get((tensorInW, 1, tensorL0_3InC), uint8_ty)
        weightsLayer0_3_ty = MemRefType.get((1 * 1 * tensorL0_3OutC * tensorL0_3InC,), int8_ty)
        tensorLayer0_3Out_ty = MemRefType.get((tensorInW, 1, tensorL0_3OutC),int8_ty)

        tensorLayer1_1In_ty = MemRefType.get((tensorInW, 1, tensorL1_1InC), int8_ty)
        weightsLayer1_1_ty = MemRefType.get((1 * 1 * tensorL1_1OutC * tensorL1_1InC,), int8_ty)
        tensorLayer1_1Out_ty = MemRefType.get((tensorInW, 1, tensorL1_1OutC),uint8_ty)
        tensorLayer1_2In_ty = MemRefType.get((tensorInW, 1, tensorL1_2InC), uint8_ty)
        weightsLayer1_2_ty = MemRefType.get((3 * 3 * tensorL1_2OutC * 1,), int8_ty)
        tensorLayer1_2Out_ty = MemRefType.get((tensorOutW, 1, tensorL1_2OutC),uint8_ty)
        tensorLayer1_3In_ty = MemRefType.get((tensorOutW, 1, tensorL1_3InC), uint8_ty)
        weightsLayer1_3_ty = MemRefType.get((1 * 1 * tensorL1_3OutC * tensorL1_3InC,), int8_ty)
        tensorLayer1_3Out_ty = MemRefType.get((tensorOutW, 1, tensorL1_3OutC),int8_ty)

        weightsAllLayers_ty = MemRefType.get((3 * 3 * tensorL0_2OutC * 1 + 1 * 1 * tensorL0_3OutC * tensorL0_3InC + 1 * 1 * tensorL1_1OutC * tensorL1_1InC + 3 * 3 * tensorL1_2OutC * 1 + 1 * 1 * tensorL1_3OutC * tensorL1_3InC,), int8_ty)
        
        # AIE Core Function declarations
        conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[tensorLayer0_2In_ty,tensorLayer0_2In_ty,tensorLayer0_2In_ty, weightsLayer0_2_ty, tensorLayer0_2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_skip_ui8_ui8_i8 = external_func("conv2dk1_skip_ui8_ui8_i8",inputs=[tensorLayer0_3In_ty, weightsLayer0_3_ty, tensorLayer0_3Out_ty, tensorLayer0_2In_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        
        conv2dk1_relu_i8_ui8 = external_func("conv2dk1_relu_i8_ui8",inputs=[tensorLayer1_1In_ty, weightsLayer1_1_ty, tensorLayer1_1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[tensorLayer1_2In_ty,tensorLayer1_2In_ty,tensorLayer1_2In_ty, weightsLayer1_2_ty, tensorLayer1_2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_ui8_i8 = external_func("conv2dk1_ui8_i8",inputs=[tensorLayer1_3In_ty, weightsLayer1_3_ty, tensorLayer1_3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Tile declarations
        ShimTile = tile(tileColIndex, 0)
        # MemTile = tile(tileColIndex, 1)
        ComputeTile = tile(tileColIndex, tileRowIndex)
        
        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile, ComputeTile, [3,3], tensorLayer0_2In_ty )
        
        # wts
        wts_OF_L3L1 = object_fifo("wts_OF_L3L1", ShimTile, ComputeTile, 1, weightsAllLayers_ty)
        
        # Output
        act_out = object_fifo("act_out", ComputeTile, [ShimTile], 1, tensorLayer1_3Out_ty)
        
        # Intermediate
        of_act_bn0_2_3 = object_fifo("act_bn0_2_3", ComputeTile, ComputeTile, 1, tensorLayer0_2Out_ty)
        of_act_bn0_bn1 = object_fifo("act_bn0_bn1", ComputeTile, ComputeTile, 1, tensorLayer0_3Out_ty)
        of_act_bn1_1_2 = object_fifo("act_bn1_1_2", ComputeTile, ComputeTile, 3, tensorLayer1_1Out_ty)
        of_act_bn1_2_3 = object_fifo("act_bn1_2_3", ComputeTile, ComputeTile, 1, tensorLayer1_2Out_ty)
        
        # Set up compute tiles
        rtpComputeTile = Buffer(ComputeTile, [16], T.i32(), "rtp")
        
        # Compute tile
        objectArchiveName = "combined_bn0_bn1.a"
        @core(ComputeTile, objectArchiveName)
        def core_body():

            for _ in for_(1): #for _ in for_(sys.maxsize):
                
                # acquire weights and rtps NOTE: needs to become once so outside for loop
                weightsAllLayers = wts_OF_L3L1.acquire(ObjectFifoPort.Consume, 1)
            
                weightsLayer0_2 = memref_view(weightsAllLayers.output, [3 * 3 * tensorL0_2OutC * 1], shift=0)
                weightsLayer0_3 = memref_view(weightsAllLayers.output, [1 * 1 * tensorL0_3OutC * tensorL0_3InC], shift=(3 * 3 * tensorL0_2OutC * 1))
                weightsLayer1_1 = memref_view(weightsAllLayers.output, [1 * 1 * tensorL1_1OutC * tensorL1_1InC], shift=(3 * 3 * tensorL0_2OutC * 1 + 1 * 1 * tensorL0_3OutC * tensorL0_3InC))
                weightsLayer1_2 = memref_view(weightsAllLayers.output, [3 * 3 * tensorL1_2OutC * 1], shift=(3 * 3 * tensorL0_2OutC * 1 + 1 * 1 * tensorL0_3OutC * tensorL0_3InC + 1 * 1 * tensorL1_1OutC * tensorL1_1InC))
                weightsLayer1_3 = memref_view(weightsAllLayers.output, [1 * 1 * tensorL1_3OutC * tensorL1_3InC], shift=(3 * 3 * tensorL0_2OutC * 1 + 1 * 1 * tensorL0_3OutC * tensorL0_3InC + 1 * 1 * tensorL1_1OutC * tensorL1_1InC + 3 * 3 * tensorL1_2OutC * 1))
                scaleLayer0_2 = memref.load(rtpComputeTile, [0]) # bn0 scaleFactor2
                scaleLayer0_3 = memref.load(rtpComputeTile, [1]) # bn0 scaleFactor3
                skipScaleLayer0_3 = memref.load(rtpComputeTile, [2]) # bn0 scaleFactorAdd
                scaleLayer1_1 = memref.load(rtpComputeTile, [3]) # bn1 scaleFactor1
                scaleLayer1_2 = memref.load(rtpComputeTile, [4]) # bn1 scaleFactor2
                scaleLayer1_3 = memref.load(rtpComputeTile, [5]) # bn1 scaleFactor3

                # pre-amble 0: row 0 in layer 0_2 3x3 dw; row 0 in layer 0_3 1x1 conv; row 0 on layer 1_1 1x1 conv
                actInLayer0_2Rows = act_in.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer0_2Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_stride1_relu_ui8_ui8, [actInLayer0_2Rows[0], actInLayer0_2Rows[0], actInLayer0_2Rows[1], weightsLayer0_2, actOutLayer0_2Row, tensorInW, 1, tensorL0_2OutC, 3, 3, 0, scaleLayer0_2, 0])
                of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer0_3Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_skip_ui8_ui8_i8, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[0], tensorInW, tensorL0_3InC, tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_relu_i8_ui8, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, tensorInW, tensorL1_1InC, tensorL1_1OutC, scaleLayer1_1])
                of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                # pre-amble 1: row 1 in layer 0_2 3x3 dw; row 1 in layer 0_3 1x1 conv; row 1 on layer 1_1 1x1 conv; row 0 on layer 1_2 3x3 dw
                actInLayer0_2Rows = act_in.acquire(ObjectFifoPort.Consume, 3)
                actOutLayer0_2Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_stride1_relu_ui8_ui8, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[2], weightsLayer0_2, actOutLayer0_2Row, tensorInW, 1, tensorL0_2OutC, 3, 3, 1, scaleLayer0_2, 0]) 
                of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer0_3Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_skip_ui8_ui8_i8, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], tensorInW, tensorL0_3InC, tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                act_in.release(ObjectFifoPort.Consume, 1)
                of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_relu_i8_ui8, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, tensorInW, tensorL1_1InC, tensorL1_1OutC, scaleLayer1_1])
                of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer1_2Rows = of_act_bn1_1_2.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer1_2Row = of_act_bn1_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_stride2_relu_ui8_ui8, [actInLayer1_2Rows[0], actInLayer1_2Rows[0], actInLayer1_2Rows[1], weightsLayer1_2, actOutLayer1_2Row, tensorInW, 1, tensorL1_2OutC, 3, 3, 0, scaleLayer1_2, 0]) 
                of_act_bn1_1_2.release(ObjectFifoPort.Consume, 1)
                of_act_bn1_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer1_3Row = of_act_bn1_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_ui8_i8, [actInLayer1_3Row, weightsLayer1_3, actOutLayer1_3Row, tensorOutW, tensorL1_3InC, tensorL1_3OutC, scaleLayer1_3])
                of_act_bn1_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                
                for _ in for_(tensorOutH - 2):
                    for _ in for_(2):
                        actInLayer0_2Rows = act_in.acquire(ObjectFifoPort.Consume, 3)
                        actOutLayer0_2Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                        call(conv2dk3_dw_stride1_relu_ui8_ui8, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[2], weightsLayer0_2, actOutLayer0_2Row, tensorInW, 1, tensorL0_2OutC, 3, 3, 1, scaleLayer0_2, 0]) 
                        of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)
                
                        actInLayer0_3Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                        actOutLayer0_3Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                        call(conv2dk1_skip_ui8_ui8_i8, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], tensorInW, tensorL0_3InC, tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                        act_in.release(ObjectFifoPort.Consume, 1)
                        of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                        of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                        actInLayer1_1Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                        actOutLayer1_1Row = of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(conv2dk1_relu_i8_ui8, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, tensorInW, tensorL1_1InC, tensorL1_1OutC, scaleLayer1_1])
                        of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                        of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                        yield_([])

                    actInLayer1_2Rows = of_act_bn1_1_2.acquire(ObjectFifoPort.Consume, 3)
                    actOutLayer1_2Row = of_act_bn1_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk3_dw_stride2_relu_ui8_ui8, [actInLayer1_2Rows[0], actInLayer1_2Rows[1], actInLayer1_2Rows[2], weightsLayer1_2, actOutLayer1_2Row, tensorInW, 1, tensorL1_2OutC, 3, 3, 1, scaleLayer1_2, 0]) 
                    of_act_bn1_1_2.release(ObjectFifoPort.Consume, 2)
                    of_act_bn1_2_3.release(ObjectFifoPort.Produce, 1)

                    actInLayer1_3Row = of_act_bn1_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer1_3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk1_ui8_i8, [actInLayer1_3Row, weightsLayer1_3, actOutLayer1_3Row, tensorOutW, tensorL1_3InC, tensorL1_3OutC, scaleLayer1_3])
                    of_act_bn1_2_3.release(ObjectFifoPort.Consume, 1)
                    act_out.release(ObjectFifoPort.Produce, 1)
                    
                    yield_([])
                
                # last part

                actInLayer0_2Rows = act_in.acquire(ObjectFifoPort.Consume, 3)
                actOutLayer0_2Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_stride1_relu_ui8_ui8, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[2], weightsLayer0_2, actOutLayer0_2Row, tensorInW, 1, tensorL0_2OutC, 3, 3, 1, scaleLayer0_2, 0]) 
                of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer0_3Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_skip_ui8_ui8_i8, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], tensorInW, tensorL0_3InC, tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                act_in.release(ObjectFifoPort.Consume, 1)
                of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_relu_i8_ui8, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, tensorInW, tensorL1_1InC, tensorL1_1OutC, scaleLayer1_1])
                of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)
                
                actInLayer0_2Rows = act_in.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer0_2Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_stride1_relu_ui8_ui8, [actInLayer0_2Rows[0], actInLayer0_2Rows[1], actInLayer0_2Rows[1], weightsLayer0_2, actOutLayer0_2Row, tensorInW, 1, tensorL0_2OutC, 3, 3, 2, scaleLayer0_2, 0]) 
                of_act_bn0_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer0_3Row = of_act_bn0_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer0_3Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_skip_ui8_ui8_i8, [actInLayer0_3Row, weightsLayer0_3, actOutLayer0_3Row, actInLayer0_2Rows[1], tensorInW, tensorL0_3InC, tensorL0_3OutC, scaleLayer0_3, skipScaleLayer0_3])
                act_in.release(ObjectFifoPort.Consume, 2)
                of_act_bn0_2_3.release(ObjectFifoPort.Consume, 1)
                of_act_bn0_bn1.release(ObjectFifoPort.Produce, 1)

                actInLayer1_1Row = of_act_bn0_bn1.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_1Row = of_act_bn1_1_2.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_relu_i8_ui8, [actInLayer1_1Row, weightsLayer1_1, actOutLayer1_1Row, tensorInW, tensorL1_1InC, tensorL1_1OutC, scaleLayer1_1])
                of_act_bn0_bn1.release(ObjectFifoPort.Consume, 1)
                of_act_bn1_1_2.release(ObjectFifoPort.Produce, 1)

                actInLayer1_2Rows = of_act_bn1_1_2.acquire(ObjectFifoPort.Consume, 3)
                actOutLayer1_2Row = of_act_bn1_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_stride2_relu_ui8_ui8, [actInLayer1_2Rows[0], actInLayer1_2Rows[1], actInLayer1_2Rows[2], weightsLayer1_2, actOutLayer1_2Row, tensorInW, 1, tensorL1_2OutC, 3, 3, 1, scaleLayer1_2, 0]) 
                of_act_bn1_1_2.release(ObjectFifoPort.Consume, 3)
                of_act_bn1_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer1_3Row = of_act_bn1_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer1_3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_ui8_i8, [actInLayer1_3Row, weightsLayer1_3, actOutLayer1_3Row, tensorOutW, tensorL1_3InC, tensorL1_3OutC, scaleLayer1_3])
                of_act_bn1_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                wts_OF_L3L1.release(ObjectFifoPort.Consume, 1)
                yield_([])
            
        
        # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        totalWeightsSize32b = (3*3*tensorL0_2OutC*1 + 1*1*tensorL0_3InC*tensorL0_3OutC + 1*1*tensorL1_1InC*tensorL1_1OutC + 3*3*tensorL1_2OutC + 1*1*tensorL1_3InC*tensorL1_3OutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp", index=0, value=scaleFactor0_2)
            NpuWriteRTPOp("rtp", index=1, value=scaleFactor0_3)
            NpuWriteRTPOp("rtp", index=2, value=scaleFactorAdd0)
            NpuWriteRTPOp("rtp", index=3, value=scaleFactor1_1)
            NpuWriteRTPOp("rtp", index=4, value=scaleFactor1_2)
            NpuWriteRTPOp("rtp", index=5, value=scaleFactor1_3)
            
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
                sizes=[1, 1, 1, activationsOutSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_OF_L3L1",
                bd_id=1,
                mem=weightsFromL3,
                sizes=[1, 1, 1, totalWeightsSize32b],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)