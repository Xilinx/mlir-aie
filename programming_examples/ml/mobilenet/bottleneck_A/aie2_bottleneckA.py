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


def mobilenetV3BottleneckA(tileRowIndex = 2, tileColIndex = 0, tensorInW = 112, tensorInH = 112, tensorInC = 16, depthWiseStride = 2, depthWiseChannels = 64, tensorOutC = 24, withSkip = False, enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    tensorOutW = tensorInW // depthWiseStride
    tensorOutH = tensorInH // depthWiseStride

    tensorL1InC = tensorInC
    tensorL1OutC = depthWiseChannels

    tensorL2InC = tensorL1OutC
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
        tensorLayer1In_ty = MemRefType.get((tensorInW, 1, tensorL1InC), int8_ty)
        weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL1OutC * tensorL1InC,), int8_ty)
        tensorLayer1Out_ty = MemRefType.get((tensorInW, 1, tensorL1OutC),uint8_ty)
        tensorLayer2In_ty = MemRefType.get((tensorInW, 1, tensorL2InC), uint8_ty)
        weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL2OutC * 1,), int8_ty)
        tensorLayer2Out_ty = MemRefType.get((tensorOutW, 1, tensorL2OutC),uint8_ty)
        tensorLayer3In_ty = MemRefType.get((tensorOutW, 1, tensorL3InC), uint8_ty)
        weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
        tensorLayer3Out_ty = tensorLayer3Out_ty = MemRefType.get((tensorOutW, 1, tensorL3OutC),int8_ty)

        weightsAllLayers_ty = MemRefType.get((1 * 1 * tensorL1OutC * tensorL1InC + 3 * 3 * tensorL2OutC * 1 + 1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
        
        # AIE Core Function declarations
        conv2dk1_relu_i8_ui8 = external_func("conv2dk1_relu_i8_ui8",inputs=[tensorLayer1In_ty, weightsLayer1_ty, tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        if depthWiseStride == 2:
            conv2dk3_dw_relu_ui8_ui8 = external_func("conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        else:
            conv2dk3_dw_relu_ui8_ui8 = external_func("conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        
        if (withSkip):
            conv2dk1_skip_ui8_i8_i8 = external_func("conv2dk1_skip_ui8_i8_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        else:
            conv2dk1_ui8_i8 = external_func("conv2dk1_ui8_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        
        # Tile declarations
        ShimTile = tile(tileColIndex, 0)
        # MemTile = tile(tileColIndex, 1)
        ComputeTile = tile(tileColIndex, tileRowIndex)
        
        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile, ComputeTile, 2, tensorLayer1In_ty)
        
        # wts
        wts_OF_L3L1 = object_fifo("wts_OF_L3L1", ShimTile, ComputeTile, 1, weightsAllLayers_ty)
        
        # Output
        act_out = object_fifo("act_out", ComputeTile, [ShimTile], 1, tensorLayer3Out_ty)
        
        # Intermediate
        of_act_1_2 = object_fifo("act_1_2", ComputeTile, ComputeTile, 3, tensorLayer1Out_ty)
        of_act_2_3 = object_fifo("act_2_3", ComputeTile, ComputeTile, 1, tensorLayer2Out_ty)
        
        # Set up compute tiles
        rtpComputeTile = Buffer(ComputeTile, [16], T.i32(), "rtp")
        
        # Compute tile
        objectArchiveName = "combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (depthWiseStride, "skip" if (withSkip) else "")
        @core(ComputeTile, objectArchiveName)
        def core_body():

            for _ in for_(1): #for _ in for_(sys.maxsize):
                
                # acquire weights and rtps NOTE: needs to become once so outside for loop
                weightsAllLayers = wts_OF_L3L1.acquire(ObjectFifoPort.Consume, 1)
            
                weightsLayer1 = memref_view(weightsAllLayers.output, [1 * 1 * tensorL1OutC * tensorL1InC], shift=0)
                weightsLayer2 = memref_view(weightsAllLayers.output, [3 * 3 * tensorL2OutC * 1], shift=1 * 1 * tensorL1OutC * tensorL1InC)
                weightsLayer3 = memref_view(weightsAllLayers.output, [1 * 1 * tensorL3OutC * tensorL3InC], shift=(1 * 1 * tensorL1OutC * tensorL1InC + 3 * 3 * tensorL2OutC * 1))
                scaleLayer1 = 7 #memref.load(rtpComputeTile, [0])
                scaleLayer2 = 8 #memref.load(rtpComputeTile, [1])
                scaleLayer3 = 9 # memref.load(rtpComputeTile, [2])
                if (withSkip):
                    skipScaleLayer3 = 0 # memref.load(rtpComputeTile, [3])

                # pre-amble 0: rows 0, 1 in layer 1 1x1 conv; row 0 in layer 2 3x3 dw; row 0 in layer 3 1x1 conv
                actInLayer1Rows = act_in.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer1Rows = of_act_1_2.acquire(ObjectFifoPort.Produce, 2)
                call(conv2dk1_relu_i8_ui8, [actInLayer1Rows[0], weightsLayer1, actOutLayer1Rows[0], tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                call(conv2dk1_relu_i8_ui8, [actInLayer1Rows[1], weightsLayer1, actOutLayer1Rows[1], tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                of_act_1_2.release(ObjectFifoPort.Produce, 2)
                if not (withSkip):
                    act_in.release(ObjectFifoPort.Consume, 2)

                actInLayer2Rows = of_act_1_2.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[0], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 0, scaleLayer2, 0]) # where do we plug in stride
                if (depthWiseStride == 2):
                    of_act_1_2.release(ObjectFifoPort.Consume, 1) # if (depthWiseStride == 2) : 1 else 0 
                of_act_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                if (withSkip):
                    call(conv2dk1_skip_ui8_i8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer1Rows[0], tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3, skipScaleLayer3])
                    act_in.release(ObjectFifoPort.Consume, depthWiseStride)
                else:
                    call(conv2dk1_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3])
                of_act_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                
                for _ in for_(tensorOutH - (2 if (depthWiseStride == 1) else 1)):    
                    if (withSkip):
                        actInLayer1Rows = act_in.acquire(ObjectFifoPort.Consume, 2)
                        actOutLayer1Row = of_act_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(conv2dk1_relu_i8_ui8, [actInLayer1Rows[1], weightsLayer1, actOutLayer1Row, tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                        of_act_1_2.release(ObjectFifoPort.Produce, 1)
                    else:
                        actInLayer1Rows = act_in.acquire(ObjectFifoPort.Consume, depthWiseStride)
                        actOutLayer1Rows = of_act_1_2.acquire(ObjectFifoPort.Produce, depthWiseStride)
                        call(conv2dk1_relu_i8_ui8, [actInLayer1Rows[0], weightsLayer1, actOutLayer1Rows[0], tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                        if (depthWiseStride==2):
                            call(conv2dk1_relu_i8_ui8, [actInLayer1Rows[1], weightsLayer1, actOutLayer1Rows[1], tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                        of_act_1_2.release(ObjectFifoPort.Produce, depthWiseStride)
                        act_in.release(ObjectFifoPort.Consume, depthWiseStride)
                    
                    actInLayer2Rows = of_act_1_2.acquire(ObjectFifoPort.Consume, 3)
                    actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[1], actInLayer2Rows[2], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 1, scaleLayer2, 0]) # where do we plug in stride
                    of_act_1_2.release(ObjectFifoPort.Consume, 2 if (depthWiseStride == 2) else 1) #if (depthWiseStride == 2) : 2 else 1
                    of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                    actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                    if (withSkip):
                        call(conv2dk1_skip_ui8_i8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer1Rows[0], tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3, skipScaleLayer3])
                        act_in.release(ObjectFifoPort.Consume, depthWiseStride)
                    else:
                        call(conv2dk1_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3])
                    of_act_2_3.release(ObjectFifoPort.Consume, 1)
                    act_out.release(ObjectFifoPort.Produce, 1)
                    
                    yield_([])
                
                # last part
                if (depthWiseStride == 1):
                    actInLayer2Rows = of_act_1_2.acquire(ObjectFifoPort.Consume, 2)
                    actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[1], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 2, scaleLayer2, 0]) # where do we plug in stride
                    of_act_1_2.release(ObjectFifoPort.Consume, 3 if (depthWiseStride == 2) else 2) #if (depthWiseStride == 2) : 2 else 1
                    of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                    actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                    if (withSkip):
                        actInLayer1Row = act_in.acquire(ObjectFifoPort.Consume, 1)
                        call(conv2dk1_skip_ui8_i8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer1Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3, skipScaleLayer3])
                        act_in.release(ObjectFifoPort.Consume, 1)
                    else:
                        call(conv2dk1_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3])
                    of_act_2_3.release(ObjectFifoPort.Consume, 1)
                    act_out.release(ObjectFifoPort.Produce, 1)
                
                wts_OF_L3L1.release(ObjectFifoPort.Consume, 1)
                yield_([])
            
        
        # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        totalWeightsSize32b = (1*1*tensorL1InC*tensorL1OutC + 3*3*tensorL2OutC*1 + 1*1*tensorL3InC*tensorL3OutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=0, value=8)
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=0, value=8)
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=0, value=11)
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=0, value=0)
            
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


with mlir_mod_ctx() as ctx:
    mobilenetV3BottleneckA(withSkip=False, depthWiseStride=2, tensorInW=8, tensorInH=8, tensorInC=8,tensorOutC=8,depthWiseChannels=8) # bottleneck 1
    # mobilenetV3BottleneckA(withSkip=True, depthWiseStride=1, tensorInW=56, tensorInH=56 ,tensorInC=24,tensorOutC=24,depthWiseChannels=72) # bottleneck 2
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

