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


def mobilenetV3Bottleneck0(tileRowIndex = 2, tileColIndex = 0, tensorInW = 112, tensorInH = 112, tensorInC = 16, tensorOutC = 16, scaleFactor2 = 7, scaleFactor3 = 10, scaleFactorAdd = 0, enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    tensorOutW = tensorInW
    tensorOutH = tensorInH

    tensorL2InC = tensorInC
    tensorL2OutC = tensorL2InC

    tensorL3InC = tensorL2InC
    tensorL3OutC = tensorOutC

    @device(AIEDevice.npu1_1col)
    def device_body():
        
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)
        tensorLayer2In_ty = MemRefType.get((tensorInW, 1, tensorL2InC), uint8_ty)
        weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL2OutC * 1,), int8_ty)
        tensorLayer2Out_ty = MemRefType.get((tensorOutW, 1, tensorL2OutC),uint8_ty)
        tensorLayer3In_ty = MemRefType.get((tensorOutW, 1, tensorL3InC), uint8_ty)
        weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
        tensorLayer3Out_ty = MemRefType.get((tensorOutW, 1, tensorL3OutC),int8_ty)

        weightsAllLayers_ty = MemRefType.get((3 * 3 * tensorL2OutC * 1 + 1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
        
        # AIE Core Function declarations
        conv2dk3_dw_relu_ui8_ui8 = external_func("conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_skip_ui8_ui8_i8 = external_func("conv2dk1_skip_ui8_ui8_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, tensorLayer2In_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        
        # Tile declarations
        ShimTile = tile(tileColIndex, 0)
        # MemTile = tile(tileColIndex, 1)
        ComputeTile = tile(tileColIndex, tileRowIndex)
        
        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile, ComputeTile, 4, tensorLayer2In_ty)
        
        # wts
        wts_OF_L3L1 = object_fifo("wts_OF_L3L1", ShimTile, ComputeTile, 1, weightsAllLayers_ty)
        
        # Output
        act_out = object_fifo("act_out", ComputeTile, [ShimTile], 1, tensorLayer3Out_ty)
        
        # Intermediate
        of_act_2_3 = object_fifo("act_2_3", ComputeTile, ComputeTile, 1, tensorLayer2Out_ty)
        
        # Set up compute tiles
        rtpComputeTile = Buffer(ComputeTile, [16], T.i32(), "rtp")
        
        # Compute tile
        objectArchiveName = "combined_conv2dk3dwstride1_conv2dk1skip.a"
        @core(ComputeTile, objectArchiveName)
        def core_body():

            for _ in for_(1): #for _ in for_(sys.maxsize):
                
                # acquire weights and rtps NOTE: needs to become once so outside for loop
                weightsAllLayers = wts_OF_L3L1.acquire(ObjectFifoPort.Consume, 1)
            
                weightsLayer2 = memref_view(weightsAllLayers.output, [3 * 3 * tensorL2OutC * 1], shift=0)
                weightsLayer3 = memref_view(weightsAllLayers.output, [1 * 1 * tensorL3OutC * tensorL3InC], shift=3 * 3 * tensorL2OutC * 1)
                scaleLayer2 = memref.load(rtpComputeTile, [0]) # scaleFactor2
                scaleLayer3 = memref.load(rtpComputeTile, [1]) # scaleFactor3
                skipScaleLayer3 = memref.load(rtpComputeTile, [2]) # scaleFactorAdd

                # pre-amble 0: row 0 in layer 2 3x3 dw; row 0 in layer 3 1x1 conv
                actInLayer2Rows = act_in.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[0], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 0, scaleLayer2, 0]) # where do we plug in stride
                of_act_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_skip_ui8_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer2Rows[0], tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3, skipScaleLayer3])
                of_act_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                
                for _ in for_(tensorOutH - 2):                    
                    actInLayer2Rows = act_in.acquire(ObjectFifoPort.Consume, 3)
                    actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[1], actInLayer2Rows[2], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 1, scaleLayer2, 0]) # where do we plug in stride
                    of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                    actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk1_skip_ui8_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer2Rows[1], tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3, skipScaleLayer3])
                    act_in.release(ObjectFifoPort.Consume, 1)
                    of_act_2_3.release(ObjectFifoPort.Consume, 1)
                    act_out.release(ObjectFifoPort.Produce, 1)
                    
                    yield_([])
                
                # last part
                
                actInLayer2Rows = act_in.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[1], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 2, scaleLayer2, 0]) # where do we plug in stride
                of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_skip_ui8_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer2Rows[1], tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3, skipScaleLayer3])
                act_in.release(ObjectFifoPort.Consume, 2)
                of_act_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                wts_OF_L3L1.release(ObjectFifoPort.Consume, 1)
                yield_([])
            
        
        # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        totalWeightsSize32b = (3*3*tensorL2OutC*1 + 1*1*tensorL3InC*tensorL3OutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp", index=0, value=scaleFactor2)
            NpuWriteRTPOp("rtp", index=1, value=scaleFactor3)
            NpuWriteRTPOp("rtp", index=2, value=scaleFactorAdd)
            
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

