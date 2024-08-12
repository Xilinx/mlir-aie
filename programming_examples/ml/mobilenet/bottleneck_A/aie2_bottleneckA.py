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

class bottleneckACoreClass:
    def __init__(self, bottleneckName, computeTile, tensorLayer_ty):
        self.testOF = object_fifo(bottleneckName+"_act_1_2", computeTile, computeTile, 3, tensorLayer_ty)

class bottleneckACore:
    def __init__(self, _bottleneckName, _computeTile, _actIn, _weightsIn, _actOut, _rtpsIn,
                    _objectArchive, _f1x1Relu, _f3x3dwStride1Relu, _f3x3dwStride2Relu, _f1x1, _f1x1Skip,
                    _layer1OutType, _layer2OutType,
                    _tensorInW = 112, _tensorInH = 112, _tensorInC = 16, _depthWiseStride = 2, _depthWiseChannels = 64, _tensorOutC = 24, _withSkip = False,_scaleFactor1 = 8, _scaleFactor2 = 8, _scaleFactor3 = 8,  _scaleFactorAdd = 0, ):

        self.bottleneckName = _bottleneckName
        self.computeTile = _computeTile
        
        self.actIn = _actIn
        self.weightsIn = _weightsIn
        self.actOut = _actOut
        self.rtpsIn = _rtpsIn
        
        self.objectArchive = _objectArchive
        self.f1x1Relu = _f1x1Relu
        self.f3x3dwRelu = _f3x3dwStride2Relu if (_depthWiseStride == 2) else _f3x3dwStride1Relu
        self.f1x1 = _f1x1
        self.f1x1Skip = _f1x1Skip
        
        self.layer1OutType = _layer1OutType
        self.layer2OutType = _layer2OutType

        self.tensorInW = _tensorInW
        self.tensorInH = _tensorInH
        self.tensorInC = _tensorInC
        self.depthWiseStride = _depthWiseStride
        self.depthWiseChannels = _depthWiseChannels
        self.tensorOutC = _tensorOutC
        self.withSkip = _withSkip
        
        self.tensorOutH = self.tensorInH // self.depthWiseStride
        self.tensorOutW = self.tensorInW // self.depthWiseStride

        self.tensorL1InC = self.tensorInC
        self.tensorL1OutC = self.depthWiseChannels

        self.tensorL2InC = self.tensorL1OutC
        self.tensorL2OutC = self.tensorL2InC

        self.tensorL3InC = self.tensorL2InC
        self.tensorL3OutC = self.tensorOutC

        self.scaleFactor1=_scaleFactor1
        self.scaleFactor2=_scaleFactor2
        self.scaleFactor3=_scaleFactor3
        self.scaleFactorAdd=_scaleFactorAdd
    
        # Intermediate
        self.of_act_1_2 = object_fifo(self.bottleneckName+"_act_1_2", self.computeTile, self.computeTile, 3, self.layer1OutType)
        self.of_act_2_3 = object_fifo(self.bottleneckName+"_act_2_3", self.computeTile, self.computeTile, 1, self.layer2OutType)

        # Compute tile
        @core(self.computeTile, self.objectArchive)
        def core_body():
            for _ in for_(1): #for _ in for_(sys.maxsize):
            
                # acquire weights and rtps NOTE: needs to become once so outside for loop
                weightsAllLayers = self.weightsIn.acquire(ObjectFifoPort.Consume, 1)
        
                weightsLayer1 = memref_view(weightsAllLayers.output, [1 * 1 * self.tensorL1OutC * self.tensorL1InC], shift=0)
                weightsLayer2 = memref_view(weightsAllLayers.output, [3 * 3 * self.tensorL2OutC * 1], shift=1 * 1 * self.tensorL1OutC * self.tensorL1InC)
                weightsLayer3 = memref_view(weightsAllLayers.output, [1 * 1 * self.tensorL3OutC * self.tensorL3InC], shift=(1 * 1 * self.tensorL1OutC * self.tensorL1InC + 3 * 3 * self.tensorL2OutC * 1))
                scaleLayer1 = memref.load(self.rtpsIn, [0]) # scaleFactor1
                scaleLayer2 = memref.load(self.rtpsIn, [1]) # scaleFactor2
                scaleLayer3 = memref.load(self.rtpsIn, [2]) # scaleFactor3
                if (self.withSkip):
                    skipScaleLayer3 = memref.load(self.rtpsIn, [3]) # scaleFactorAdd
            
                # pre-amble 0: rows 0, 1 in layer 1 1x1 conv; row 0 in layer 2 3x3 dw; row 0 in layer 3 1x1 conv
                actInLayer1Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer1Rows = self.of_act_1_2.acquire(ObjectFifoPort.Produce, 2)
                call(self.f1x1Relu, [actInLayer1Rows[0], weightsLayer1, actOutLayer1Rows[0], self.tensorInW, self.tensorL1InC, self.tensorL1OutC,scaleLayer1])
                call(self.f1x1Relu, [actInLayer1Rows[1], weightsLayer1, actOutLayer1Rows[1], self.tensorInW, self.tensorL1InC, self.tensorL1OutC,scaleLayer1])
                self.of_act_1_2.release(ObjectFifoPort.Produce, 2)
                if not (self.withSkip):
                    self.actIn.release(ObjectFifoPort.Consume, 2)
                
                actInLayer2Rows = self.of_act_1_2.acquire(ObjectFifoPort.Consume, 2)
                actOutLayer2Row = self.of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                call(self.f3x3dwRelu, [actInLayer2Rows[0], actInLayer2Rows[0], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, self.tensorInW, 1, self.tensorL2OutC, 3, 3, 0, scaleLayer2, 0]) # where do we plug in stride
                if (self.depthWiseStride == 2):
                    self.of_act_1_2.release(ObjectFifoPort.Consume, 1) # if (depthWiseStride == 2) : 1 else 0 
                self.of_act_2_3.release(ObjectFifoPort.Produce, 1)
                
                actInLayer3Row = self.of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                actOutLayer3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                if (self.withSkip):
                    call(self.f1x1Skip, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer1Rows[0], self.tensorOutW, self.tensorL3InC, self.tensorL3OutC, scaleLayer3, skipScaleLayer3])
                    self.actIn.release(ObjectFifoPort.Consume, self.depthWiseStride)
                else:
                    call(self.f1x1, [actInLayer3Row, weightsLayer3, actOutLayer3Row, self.tensorOutW, self.tensorL3InC, self.tensorL3OutC, scaleLayer3])
                self.of_act_2_3.release(ObjectFifoPort.Consume, 1)
                self.actOut.release(ObjectFifoPort.Produce, 1)
            
                # middle: layer 3 1x1 conv and layer 2 3x3 dw and layer 1 1x1 conv
                for _ in for_(self.tensorOutH - (2 if (self.depthWiseStride == 1) else 1)):    
                    if (self.withSkip):
                        actInLayer1Rows = self.actIn.acquire(ObjectFifoPort.Consume, 2)
                        actOutLayer1Row = self.of_act_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(self.f1x1Relu, [actInLayer1Rows[1], weightsLayer1, actOutLayer1Row, self.tensorInW, self.tensorL1InC, self.tensorL1OutC, scaleLayer1])
                        self.of_act_1_2.release(ObjectFifoPort.Produce, 1)
                    else:
                        actInLayer1Rows = self.actIn.acquire(ObjectFifoPort.Consume, self.depthWiseStride)
                        actOutLayer1Rows = self.of_act_1_2.acquire(ObjectFifoPort.Produce, self.depthWiseStride)
                        if (self.depthWiseStride==1):
                            call(self.f1x1Relu, [actInLayer1Rows, weightsLayer1, actOutLayer1Rows, self.tensorInW, self.tensorL1InC, self.tensorL1OutC, scaleLayer1])
                        if (self.depthWiseStride==2):
                            call(self.f1x1Relu, [actInLayer1Rows[0], weightsLayer1, actOutLayer1Rows[0], self.tensorInW, self.tensorL1InC, self.tensorL1OutC, scaleLayer1])
                            call(self.f1x1Relu, [actInLayer1Rows[1], weightsLayer1, actOutLayer1Rows[1], self.tensorInW, self.tensorL1InC, self.tensorL1OutC, scaleLayer1])
                        self.of_act_1_2.release(ObjectFifoPort.Produce, self.depthWiseStride)
                        self.actIn.release(ObjectFifoPort.Consume, self.depthWiseStride)

                    actInLayer2Rows = self.of_act_1_2.acquire(ObjectFifoPort.Consume, 3)
                    actOutLayer2Row = self.of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(self.f3x3dwRelu, [actInLayer2Rows[0], actInLayer2Rows[1], actInLayer2Rows[2], weightsLayer2, actOutLayer2Row, self.tensorInW, 1, self.tensorL2OutC, 3, 3, 1, scaleLayer2, 0]) # where do we plug in stride
                    self.of_act_1_2.release(ObjectFifoPort.Consume, 2 if (self.depthWiseStride == 2) else 1) #if (depthWiseStride == 2) : 2 else 1
                    self.of_act_2_3.release(ObjectFifoPort.Produce, 1)
            
                    actInLayer3Row = self.of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    if (self.withSkip):
                        call(self.f1x1Skip, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer1Rows[0], self.tensorOutW, self.tensorL3InC, self.tensorL3OutC, scaleLayer3, skipScaleLayer3])
                        self.actIn.release(ObjectFifoPort.Consume, self.depthWiseStride)
                    else:
                        call(self.f1x1, [actInLayer3Row, weightsLayer3, actOutLayer3Row, self.tensorOutW, self.tensorL3InC, self.tensorL3OutC, scaleLayer3])
                    self.of_act_2_3.release(ObjectFifoPort.Consume, 1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)
                
                    yield_([])
            
                # last part
                if (self.depthWiseStride == 1):
                    actInLayer2Rows = self.of_act_1_2.acquire(ObjectFifoPort.Consume, 2)
                    actOutLayer2Row = self.of_act_2_3.acquire(ObjectFifoPort.Produce, 1)
                    call(self.f3x3dwRelu, [actInLayer2Rows[0], actInLayer2Rows[1], actInLayer2Rows[1], weightsLayer2, actOutLayer2Row, self.tensorInW, 1, self.tensorL2OutC, 3, 3, 2, scaleLayer2, 0]) # where do we plug in stride
                    self.of_act_1_2.release(ObjectFifoPort.Consume, 3 if (self.depthWiseStride == 2) else 2) #if (depthWiseStride == 2) : 2 else 1
                    self.of_act_2_3.release(ObjectFifoPort.Produce, 1)

                    actInLayer3Row = self.of_act_2_3.acquire(ObjectFifoPort.Consume, 1)
                    actOutLayer3Row = self.actOut.acquire(ObjectFifoPort.Produce, 1)
                    if (self.withSkip):
                        actInLayer1Row = self.actIn.acquire(ObjectFifoPort.Consume, 1)
                        call(self.f1x1Skip, [actInLayer3Row, weightsLayer3, actOutLayer3Row, actInLayer1Row, self.tensorOutW, self.tensorL3InC, self.tensorL3OutC, scaleLayer3, skipScaleLayer3])
                        self.actIn.release(ObjectFifoPort.Consume, 1)
                    else:
                        call(self.f1x1, [actInLayer3Row, weightsLayer3, actOutLayer3Row, self.tensorOutW, self.tensorL3InC, self.tensorL3OutC, scaleLayer3])
                    self.of_act_2_3.release(ObjectFifoPort.Consume, 1)
                    self.actOut.release(ObjectFifoPort.Produce, 1)
            
                self.weightsIn.release(ObjectFifoPort.Consume, 1)
                yield_([])    

def mobilenetV3BottleneckA(bottleneckName, tileRowIndex = 2, tileColIndex = 0, tensorInW = 112, tensorInH = 112, tensorInC = 16, depthWiseStride = 2, depthWiseChannels = 64, tensorOutC = 24, withSkip = False, scaleFactor1 = 8, scaleFactor2 = 9, scaleFactor3 = 11, scaleFactorAdd = 0, enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

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
        tensorLayer3Out_ty = MemRefType.get((tensorOutW, 1, tensorL3OutC),int8_ty)

        weightsAllLayers_ty = MemRefType.get((1 * 1 * tensorL1OutC * tensorL1InC + 3 * 3 * tensorL2OutC * 1 + 1 * 1 * tensorL3OutC * tensorL3InC,), int8_ty)
        
        # AIE Core Function declarations
        conv2dk1_relu_i8_ui8 = external_func("conv2dk1_relu_i8_ui8",inputs=[tensorLayer1In_ty, weightsLayer1_ty, tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[tensorLayer2In_ty,tensorLayer2In_ty,tensorLayer2In_ty, weightsLayer2_ty, tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        conv2dk1_skip_ui8_i8_i8 = external_func("conv2dk1_skip_ui8_i8_i8",inputs=[tensorLayer3In_ty, weightsLayer3_ty, tensorLayer3Out_ty, tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
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
                
        # Set up compute tiles
        rtpComputeTile = Buffer(ComputeTile, [16], T.i32(), "rtp")
        
        # Compute tile
        objectArchiveName = "combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (depthWiseStride, "skip" if (withSkip) else "")

        bottleneckACore(bottleneckName, ComputeTile, act_in, wts_OF_L3L1, act_out, rtpComputeTile, objectArchiveName, conv2dk1_relu_i8_ui8, conv2dk3_dw_stride1_relu_ui8_ui8, conv2dk3_dw_stride2_relu_ui8_ui8, conv2dk1_ui8_i8, conv2dk1_skip_ui8_i8_i8, tensorLayer1Out_ty, tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, depthWiseStride, depthWiseChannels, tensorOutC, withSkip)
        
        # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        totalWeightsSize32b = (1*1*tensorL1InC*tensorL1OutC + 3*3*tensorL2OutC*1 + 1*1*tensorL3InC*tensorL3OutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=0, value=scaleFactor1)
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=1, value=scaleFactor2)
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=2, value=scaleFactor3)
            NpuWriteRTPOp("rtp", col=tileColIndex, row=tileRowIndex, index=3, value=scaleFactorAdd)
            
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
