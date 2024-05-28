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
from aie.extras.dialects.ext.memref import view as memref_view

import aie.utils.trace as trace_utils

tensorInW = 112
tensorInH = 112 // 8
tensorInC = 16

depthWiseStride = 2
depthWiseChannels = 64 
tensorOutC = 24

tensorOutW = tensorInW // depthWiseStride
tensorOutH = tensorInH // depthWiseStride

tensorL1InC = tensorInC
tensorL1OutC = depthWiseChannels

tensorL2InC = tensorL1OutC
tensorL2OutC = tensorL2InC

tensorL3InC = tensorL2InC
tensorL3OutC = tensorOutC

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4

tileRowIndex = 2
tileColIndex = 0

def bottleneck1_mobilenetv3():

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

        weightsAllLayers_ty = MemRefType.get((1 * 1 * tensorL1OutC * tensorL1InC + 3 * 3 * tensorL2OutC * 1 + 1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
        
        # AIE Core Function declarations
        conv2dk1_relu_i8_ui8 = external_func("conv2dk1_relu_i8_ui8",inputs=[tensorLayer1In_ty, weightsAllLayers_ty, tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        if depthWiseStride == 2:
            conv2dk3_dw_relu_ui8_ui8 = external_func("conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsAllLayers_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        else:
            conv2dk3_dw_relu_ui8_ui8 = external_func("conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsAllLayers_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_ui8_i8 = external_func("conv2dk1_ui8_i8",inputs=[tensorLayer3In_ty, weightsAllLayers_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        
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
        wts_OF_L3L2 = object_fifo("wts_OF_L3L2", ShimTile, ComputeTile, 3, weightsAllLayers_ty)
        # wts_buf_01 = object_fifo("wts_buf_01", ComputeTile, [ComputeTile], 1, weightsLayer1_ty)
        # wts_buf_02 = object_fifo("wts_buf_02", ComputeTile, [ComputeTile], 1, weightsLayer2_ty)
        # wts_buf_03 = object_fifo("wts_buf_03", ComputeTile, [ComputeTile], 1, weightsLayer3_ty)
        # object_fifo_link(wts_OF_L3L2, [wts_buf_01, wts_buf_02, wts_buf_03])
        
        # Output
        act_out = object_fifo("act_out", ComputeTile, [ShimTile], 2, tensorLayer3Out_ty)
        #of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, tensorLayer3Out_ty)
        #object_fifo_link(act_3, of_outOFL2L3)
        
        # Intermediate
        of_act_1_2 = object_fifo("act_1_2", ComputeTile, ComputeTile, 3, tensorLayer1Out_ty)
        of_act_2_3 = object_fifo("act_2_3", ComputeTile, ComputeTile, 1, tensorLayer2Out_ty)
        
        # Set up compute tiles
        rtpComputeTile = Buffer(ComputeTile, [16], T.i32(), "rtp")
        
        # Compute tile
        @core(ComputeTile, "combined_con2dk1fusedrelu_conv2dk3dw_conv2dk1i8.a")
        def core_body():

            # acquire weights and rtps once
            weightsAllLayers = wts_OF_L3L2.acquire(ObjectFifoPort.Consume, 3)
            #weightsLayer1 = memref_view(weightsAllLayers.subview, [1 * 1 * tensorL1OutC * tensorL1InC], None, 0)
            #memref.view(weightsLayer2, weightsAllLayers,1 * 1 * tensorL1OutC * tensorL1InC, 3*3*tensorL2OutC*1)
            #memref.view(weightsLayer2, weightsAllLayers,1 * 1 * tensorL1OutC * tensorL1InC + 3*3*tensorL2OutC*1, 1*1*tensorL3OutC*tensorL3InC)
            weightsLayer1 = weightsAllLayers[0]
            weightsLayer2 = weightsAllLayers[1]
            weightsLayer3 = weightsAllLayers[2]
            scaleLayer1 = memref.load(rtpComputeTile, [0])
            scaleLayer2 = memref.load(rtpComputeTile, [1])
            scaleLayer3 = memref.load(rtpComputeTile, [2])

            for _ in for_(sys.maxsize):
                
                # pre-amble 0: rows 0, 1 in layer 1 1x1 conv; row 0 in layer 2 3x3 dw; row 0 in layer 3 1x1 conv
                for _ in for_(2):
                    actInLayer1Row = act_in.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer1Row = of_act_1_2.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk1_relu_i8_ui8, [actInLayer1Row, weightsLayer1, actOutLayer1Row, tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                    act_in.release(ObjectFifoPort.Consume, 1)
                    of_act_1_2.release(ObjectFifoPort.Produce, 1)
                    yield_([])

                actInLayer2Rows = of_act_1_2.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[0], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 1, scaleLayer2, 0]) # where do we plug in stride
                of_act_1_2.release(ObjectFifoPort.Consume, 1 if (depthWiseStride == 2) else 0) # if (depthWiseStride == 2) : 1 else 0 
                of_act_2_3.release(ObjectFifoPort.Produce, 1)

                actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3])
                of_act_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                for _ in for_(tensorOutH - 2):
                    for _ in for_(depthWiseStride):
                        actInLayer1Row = act_in.acquire(ObjectFifoPort.Consume, 1)
                        actOutLayer1Row = of_act_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(conv2dk1_relu_i8_ui8, [actInLayer1Row, weightsLayer1, actOutLayer1Row, tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                        act_in.release(ObjectFifoPort.Consume, 1)
                        of_act_1_2.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    
                    actInLayer2Rows = of_act_1_2.acquire(ObjectFifoPort.Consume, 3)
                    actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[0], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 1, scaleLayer2, 0]) # where do we plug in stride
                    of_act_1_2.release(ObjectFifoPort.Consume, 2 if (depthWiseStride == 2) else 1) #if (depthWiseStride == 2) : 2 else 1
                    of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                    actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk1_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3])
                    of_act_2_3.release(ObjectFifoPort.Consume, 1)
                    act_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                
                # last part
                for _ in for_(depthWiseStride):
                    actInLayer1Row = act_in.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer1Row = of_act_1_2.acquire(ObjectFifoPort.Produce, 1)
                    call(conv2dk1_relu_i8_ui8, [actInLayer1Row, weightsLayer1, actOutLayer1Row, tensorInW, tensorL1InC, tensorL1OutC, scaleLayer1])
                    act_in.release(ObjectFifoPort.Consume, 1)
                    of_act_1_2.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                
                actInLayer2Rows = of_act_1_2.acquire(ObjectFifoPort.Consume, 3 if (depthWiseStride == 2) else 2)
                actOutLayer2Row = of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk3_dw_relu_ui8_ui8, [actInLayer2Rows[0], actInLayer2Rows[0], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, tensorInW, 1, tensorL2OutC, 3, 3, 1, scaleLayer2, 0]) # where do we plug in stride
                of_act_1_2.release(ObjectFifoPort.Consume, 3 if (depthWiseStride == 2) else 2) #if (depthWiseStride == 2) : 2 else 1
                of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer3Row = of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer3Row = act_out.acquire(ObjectFifoPort.Produce, 1)
                call(conv2dk1_ui8_i8, [actInLayer3Row, weightsLayer3, actOutLayer3Row, tensorOutW, tensorL3InC, tensorL3OutC, scaleLayer3])
                of_act_2_3.release(ObjectFifoPort.Consume, 1)
                act_out.release(ObjectFifoPort.Produce, 1)
                
                yield_([])
        
        # # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        totalWeightsSize32b = (1*1*tensorL1InC*tensorL1OutC + 3*3*tensorL2OutC*1 + 1*1*tensorL3InC*tensorL3OutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=0, value=8)
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
                metadata="wts_OF_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                sizes=[1, 1, 1, totalWeightsSize32b],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    bottleneck1_mobilenetv3()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

